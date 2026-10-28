"""Entropy-Bounded certificate-gated commitment (EB gate) for masked diffusion sampling.

A per-step commitment selector: given one probability distribution per still-masked
position, decide which positions to commit *in this step* and which to leave masked.

Prior art (absorbed, not copied):
- EB-Sampler (NeurIPS 2025): unmask in ascending-error order but halt the step
  once a joint entropy bound is exceeded: for the committed set U,
  ``sum(H(U)) - max(H(U)) <= gamma``. The bound, not a fixed count, sizes the
  batch: high-entropy steps commit few tokens, low-entropy steps commit many.
- CCD / "Beyond Confidence" (arXiv:2512.02044): trajectory consistency over
  immediate-step confidence; an adaptive per-step budget instead of a rigid one.
- NTU learned unmasking policy (2025): trajectory-contraction targets beat
  match-the-final-output targets at low step budgets; confidence alone is weak.

What is new here relative to in-repo CGAU (xiiv/cgau.py): CGAU gates each
position individually by certificate + margin, with recovery and re-masking, but
it places NO bound on the *joint* entropy of the tokens committed in one step.
A step can therefore commit many individually-certifiable but jointly-uncertain
tokens, and their errors compound in the next denoising pass. The EB gate bounds
exactly that joint risk, at the same semantic certificate level: a position is
committed only if its top-1 token is certifiable, and the committed set as a
whole must satisfy the entropy bound. This is a complement to CGAU, not a
replacement: CGAU's recovery / re-masking machinery is untouched.

Design decisions (deliberate, and limitations):
- Top-1 only (v1). CGAU already covers the top-k certifiable scan; duplicating
  it here would blur which mechanism earned a commit. A position whose top-1
  is uncertifiable is left masked (fail-closed), never downgraded silently.
- Pure stdlib, no model, no torch: the selector is a function of distributions
  plus a certificate predicate, so it is testable on a bare CPU box and its
  falsifiers run without loading any checkpoint.
- NOT wired into any sampler yet. Wiring changes generation behaviour and is
  owed a real-model evaluation (assembled-1B, GPU) before it ships; this module
  is the tested component, not a generation claim.

Pre-registered falsifiers (see tests/test_eb_gate.py and
research/eb_gate_2026-10-28.md):
- F1 certificate soundness: no committed token is uncertifiable.
- F2 bound: sum(H(committed)) - max(H(committed)) <= gamma (gamma >= 0).
- F3 progress: if any certifiable top-1 exists and gamma >= 0, at least one
  commit is returned (the first commit always has sum - max == 0).
- F4 prefix property: committed positions are the lowest-entropy certifiable
  positions in ascending-entropy order, with no gaps.
- KILL: any falsifier failure on the deterministic + fuzz suite -> the module
  does not ship; a joint-entropy reduction vs "commit all certifiable" is
  reported as a measurement on synthetic distributions only, never as a
  model-quality claim.
"""

import math
from dataclasses import dataclass, field

__all__ = ["Commit", "CommitResult", "entropy", "select_commits"]


def entropy(probs):
    """Shannon entropy (nats) of a discrete distribution.

    Inputs are validated fail-closed: negative entries or a zero/negative total
    raise ValueError. The distribution is normalised internally so float
    rounding in a caller's softmax output cannot silently skew the bound.
    """
    total = 0.0
    for p in probs:
        if p < 0.0:
            raise ValueError("distribution contains a negative probability")
        total += p
    if total <= 0.0:
        raise ValueError("distribution sums to zero")
    h = 0.0
    for p in probs:
        if p > 0.0:
            q = p / total
            h -= q * math.log(q)
    return h


@dataclass(frozen=True)
class Commit:
    """One committed position for this step."""

    position: int
    token: int
    entropy: float


@dataclass(frozen=True)
class CommitResult:
    """Outcome of one select_commits call."""

    commits: tuple = ()
    skipped_uncertifiable: tuple = ()
    skipped_by_bound: tuple = ()
    bound_value: float = 0.0
    gamma: float = 0.0


def select_commits(distributions, certifiable, gamma):
    """Select this step's commits under the entropy bound.

    Args:
        distributions: sequence of per-position probability distributions
            (one per still-masked position, in position order). Position i in
            this sequence is reported as position i in the result.
        certifiable: callable ``(position, token) -> bool``. The certificate
            predicate; anything it rejects is never committed.
        gamma: entropy-bound budget (>= 0 to commit anything). gamma < 0
            returns an empty result (fail-closed, no partial commit).

    Returns:
        CommitResult with commits in ascending-entropy order.
    """
    if gamma < 0.0:
        return CommitResult(gamma=gamma)
    candidates = []  # (entropy, position, token)
    skipped_uncertifiable = []
    for pos, probs in enumerate(distributions):
        probs = list(probs)
        if not probs:
            raise ValueError(f"empty distribution at position {pos}")
        top_token = max(range(len(probs)), key=lambda t: (probs[t], -t))
        if not certifiable(pos, top_token):
            skipped_uncertifiable.append(pos)
            continue
        candidates.append((entropy(probs), pos, top_token))
    # Ascending entropy; ties broken by position so the result is deterministic.
    candidates.sort(key=lambda c: (c[0], c[1]))

    commits = []
    skipped_by_bound = []
    running_sum = 0.0
    running_max = 0.0
    for h, pos, token in candidates:
        new_sum = running_sum + h
        new_max = running_max if running_max > h else h
        # Bound is monotone in this ascending order: once it is exceeded, every
        # later candidate can only keep or raise it, so stopping is exact.
        if new_sum - new_max > gamma + 1e-12:
            skipped_by_bound.append(pos)
            continue
        commits.append(Commit(position=pos, token=token, entropy=h))
        running_sum, running_max = new_sum, new_max
    # Note: after the first skip, later candidates are also skipped by the same
    # monotone argument only if their entropies are >= the skipped one's; the
    # ascending sort guarantees that, so skipped_by_bound is a clean suffix.
    bound_value = running_sum - running_max if commits else 0.0
    return CommitResult(
        commits=tuple(commits),
        skipped_uncertifiable=tuple(skipped_uncertifiable),
        skipped_by_bound=tuple(skipped_by_bound),
        bound_value=bound_value,
        gamma=gamma,
    )
