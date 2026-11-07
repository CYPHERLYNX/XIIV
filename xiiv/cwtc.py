"""CWTC -- Certificate-Weighted Trace Credit for masked-diffusion decoding.

Every sampler in this family (DCMR, CGEC, CDR, PTCS, SCV, CBRS, CCCA)
decides commits from the CURRENT step alone: top-1 confidence, a margin,
a planned budget, a verifier gate, or cross-chain agreement. None of
them remembers that a token was already endorsed -- safely -- on earlier
steps. The literature now says that is wasted work:

- CreditDecoding (arXiv:2510.06133v3): models often predict the correct
  token several steps before its confidence is high enough to decode;
  the gap forces repeated remasking of already-correct tokens. Their
  fix fuses accumulated historical logits into the current logits.

CWTC takes the *diagnosis* (a token can be right long before it is
confident enough to commit) and builds a different mechanism around it,
one that fits the guarantee-native thesis:

A per-position CREDIT LEDGER. At each step, the leading
certificate-safe candidate at a masked position earns credit equal to
its current probability -- but ONLY if it passes the independent
no-repeat gate against the current canvas. A candidate that fails the
gate earns exactly zero, however confident. No logits are ever modified
or fused: the model distribution is untouched, so the guarantee stays
clean -- every committed token was certificate-safe at commit time AND
repeatedly certificate-safe before it.

A position commits on either of two keys:

- CONFIDENCE path: the leading safe candidate's probability is at or
  above ``conf_threshold`` (the usual rule).
- CREDIT path: the leading safe candidate's accumulated credit is at or
  above ``credit_threshold`` while its probability is still below the
  confidence threshold. A credit-driven commit is a token the model
  kept endorsing, safely, across steps -- temporal endorsement, not a
  single-pass spike.

Credit DECAYS (multiplicative, per step) and is per (position, token):
a token endorsed once then abandoned can never commit later on stale
credit. The ledger is the audit trail: the cwtc-v1 certificate
itemizes every commit with its credit at commit, steps endorsed, and
whether it was credit-driven or confidence-driven.

A step counts as progress when it commits something OR earns any
credit; only a step with neither is a stall. Fail-closed like the rest
of the family: a stall budget plus a max-steps cap, and an incomplete
run reports ``completed=False`` with masks remaining rather than
emitting an uncertified canvas.

Prior art (absorbed, not copied):
- CreditDecoding (arXiv:2510.06133v3): right-before-confident tokens
  and wasted re-proving. CWTC takes the diagnosis but not the
  mechanism: no logit fusion, no boosted distribution -- credit is a
  certificate-gated temporal endorsement that only gates the commit
  decision, and credit accrues solely from certificate-safe history.
- CoDR (arXiv:2610.08833): endorsement is temporal; a commit-time
  snapshot is not the whole story. CWTC prices that into the commit
  decision itself via the ledger instead of remasking after the fact.
  (CoDR tracks confidence drift, XIIV's own CDR tracks certificate
  drift; CWTC tracks endorsement over time.)
- In-repo: dcmr._would_repeat / _topk_for_pos (the certificate gate),
  scv.verify_canvas (independent re-check), the CDR/CBRS fail-closed
  pattern.

Pre-registered falsifiers (see xiiv/tests/test_cwtc.py):
- F1 credit convergence: on the stubborn stub (stable top-1 prob
  ~0.29, below the 0.5 confidence threshold), cwtc completes with
  every commit credit-driven, while the confidence-only vanilla
  policy in the same module cannot complete within max_steps.
- F2 gate on credit: a token that is certificate-unsafe at step start
  earns zero credit that step; the blocked repeat token never reaches
  the credit threshold and never commits; the run completes via the
  safe runner-up and the final canvas is clean.
- F3 decay: a token endorsed on step 0 then abandoned never commits;
  its ledger credit stays below threshold at the end and the final
  canvas carries the replacement token.
- F4 final soundness: completed=True implies the independent
  verify_canvas(final) is clean (recomputed).
- F5 determinism: identical runs -> identical canvas and commit log.
- F6 budget: forward_passes <= max_steps and at most ``budget``
  commits in any step.
- KILL: F1 fails -> the module does not ship. F2 failing means the
  credit is not actually certificate-gated -> kill.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.
"""

import torch

from .dcmr import _topk_for_pos, _would_repeat
from .scv import verify_canvas

__all__ = [
    "CWTCConfig",
    "CWTCTrace",
    "cwtc_sample",
    "cwtc_certificate",
    "POLICIES",
]

POLICIES = ("vanilla", "cwtc")


class CWTCConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        topk: int = 4,
        budget: int = 4,
        max_steps: int = 64,
        conf_threshold: float = 0.5,
        credit_threshold: float = 1.0,
        decay: float = 0.9,
        stall_budget: int = 3,
    ):
        self.n = n                                # n-gram order of the certificate
        self.topk = topk                          # candidates scanned per position
        self.budget = budget                      # commits per step
        self.max_steps = max_steps                # hard termination cap
        self.conf_threshold = conf_threshold      # confidence-path bar
        self.credit_threshold = credit_threshold  # credit-path bar
        self.decay = decay                        # per-step credit decay
        self.stall_budget = stall_budget          # no-progress steps before giving up


class CWTCTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.steps = 0
        self.forward_passes = 0
        self.commits = 0
        self.confidence_commits = 0
        self.credit_driven_commits = 0
        self.gate_rejects = 0       # top-1 candidates refused by the gate
        self.stalls = 0
        self.completed = False
        self.final_no_repeat = False
        self.commit_log = []        # per-commit audit entries
        self.final_ledger = {}      # "pos:tok" -> credit left at end

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def cwtc_certificate(trace: CWTCTrace) -> dict:
    """The cwtc-v1 certificate: what was committed, on which key, and
    the independent re-check of the final canvas."""
    return {
        "version": "cwtc-v1",
        "policy": trace.policy,
        "commits": trace.commits,
        "confidence_commits": trace.confidence_commits,
        "credit_driven_commits": trace.credit_driven_commits,
        "gate_rejects": trace.gate_rejects,
        "steps": trace.steps,
        "forward_passes": trace.forward_passes,
        "stalls": trace.stalls,
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "commit_log": trace.commit_log,
    }


def _leading_safe_candidate(canvas, pos, probs_row, cfg, mask_id):
    """First certificate-safe candidate in the top-k at pos, else None.

    Returns (tok, prob, top1_was_rejected). The top-1 rejection flag is
    the gate-reject count input: a top-1 that fails the gate earns no
    credit, however confident.
    """
    cands = _topk_for_pos(probs_row, cfg.topk, {mask_id})
    if not cands:
        return None
    top1_tok = cands[0][0]
    top1_rejected = _would_repeat(canvas, pos, top1_tok, cfg.n, mask_id)
    for tok, prob in cands:
        if not _would_repeat(canvas, pos, tok, cfg.n, mask_id):
            return (tok, prob, top1_rejected)
    return None


def cwtc_sample(model, length: int, mask_id: int, config=None,
                policy: str = "cwtc", frozen=None):
    """Run the CWTC sampler.

    ``frozen`` maps position -> token for prompt tokens that are never
    remasked. Returns (ids, trace, certificate); ``ids`` is a LongTensor
    of shape (length,) with ``mask_id`` where the run did not complete.
    Deterministic (greedy); no sampling anywhere.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; want one of {POLICIES}")
    cfg = config if config is not None else CWTCConfig()
    trace = CWTCTrace(policy)
    canvas = [mask_id] * length
    if frozen:
        for pos, tok in frozen.items():
            canvas[pos] = int(tok)
    ledger = {}    # (pos, tok) -> credit
    endorsed = {}  # (pos, tok) -> steps endorsed

    for step in range(cfg.max_steps):
        masked = [p for p, t in enumerate(canvas) if t == mask_id]
        if not masked:
            break
        with torch.no_grad():
            logits = model(torch.tensor([canvas], dtype=torch.long))
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)

        # Decay every ledger entry; prune dust.
        for key in list(ledger):
            ledger[key] *= cfg.decay
            if ledger[key] < 1e-6:
                del ledger[key]

        # Leading safe candidate per masked position; credit accrues
        # ONLY for certificate-safe candidates (vanilla earns none).
        candidates = {}
        earned_any = False
        for pos in masked:
            got = _leading_safe_candidate(canvas, pos, probs[pos], cfg, mask_id)
            if got is None:
                continue
            tok, prob, top1_rejected = got
            if top1_rejected:
                trace.gate_rejects += 1
            candidates[pos] = (tok, prob)
            if policy == "cwtc":
                key = (pos, tok)
                ledger[key] = ledger.get(key, 0.0) + prob
                endorsed[key] = endorsed.get(key, 0) + 1
                earned_any = True

        # Commit in descending (credit, prob) order; re-gate against the
        # evolving canvas so same-step commits cannot jointly repeat.
        order = sorted(
            candidates,
            key=lambda p: (-ledger.get((p, candidates[p][0]), 0.0),
                           -candidates[p][1], p),
        )
        committed_this_step = 0
        for pos in order:
            if committed_this_step >= cfg.budget:
                break
            tok, prob = candidates[pos]
            credit = ledger.get((pos, tok), 0.0)
            credit_driven = (
                policy == "cwtc"
                and credit >= cfg.credit_threshold
                and prob < cfg.conf_threshold
            )
            if prob >= cfg.conf_threshold or (
                policy == "cwtc" and credit >= cfg.credit_threshold
            ):
                if _would_repeat(canvas, pos, tok, cfg.n, mask_id):
                    continue
                canvas[pos] = tok
                trace.commits += 1
                if credit_driven:
                    trace.credit_driven_commits += 1
                else:
                    trace.confidence_commits += 1
                trace.commit_log.append({
                    "pos": pos,
                    "tok": tok,
                    "step": step,
                    "credit_driven": credit_driven,
                    "credit_at_commit": round(credit, 4),
                    "steps_endorsed": endorsed.get((pos, tok), 0),
                })
                committed_this_step += 1
                ledger.pop((pos, tok), None)
                endorsed.pop((pos, tok), None)

        trace.steps += 1
        if committed_this_step == 0 and not earned_any:
            trace.stalls += 1
            if trace.stalls > cfg.stall_budget:
                break
        else:
            trace.stalls = 0

    trace.completed = all(t != mask_id for t in canvas)
    check = verify_canvas(canvas, mask_id, cfg.n)
    trace.final_no_repeat = bool(check["final_no_repeat"])
    trace.final_ledger = {
        f"{pos}:{tok}": round(credit, 4)
        for (pos, tok), credit in sorted(ledger.items())
    }
    cert = cwtc_certificate(trace)
    return torch.tensor(canvas, dtype=torch.long), trace, cert
