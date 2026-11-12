"""CGBS -- Certificate-Guided Beam Search for masked-diffusion decoding.

Every sampler in this family so far commits greedily along a single
trajectory (DCMR, CGEC, CDR, PTCS, SCV, CBRS, CCCA, CWTC) or retreats
along it (CSBR). None of them *searches*: at each step the order in
which positions are committed is whatever the model's confidence
happens to dictate, and a single attractive-but-wrong commit can strand
the rest of the canvas (the CSBR trap). The literature now says the
commit order is the decision that matters, and that verification can
guide it:

- CARVE (arXiv:2608.30922, Aug 2026): Counterfactual-Aware Reveal
  with Verified Expansion. A training-free variable-length decoder for
  masked diffusion LMs that verifies each proposed canvas expansion by
  measuring its effect on the model's remaining predictive state
  (Jensen-Shannon divergence). Gains on Dream/LLaDA (up to +4.27
  HumanEval on Dream) at roughly half the FLOPs in some settings.
- Where-to-Unmask / Gt-Margin (arXiv:2602.09501): the unmasking order
  is a decision in its own right, not a side effect of confidence.
  A ground-truth-margin oracle order improves reasoning accuracy, and
  a supervised planner can imitate it -- but that needs training.
- EB-Sampler (Aug 2026): entropy-bounded unmasking, 2-3x faster
  sampling. Speed, not search: it still walks one trajectory.

CGBS takes the *diagnoses* (commit order is the decision; a verifier
can score candidate expansions) and builds a different mechanism
around them, one that fits the guarantee-native thesis and needs no
training:

- BEAM OVER COMMIT ORDERS, not over tokens. Each beam state is a
  partial canvas; each successor commits exactly one (position,
  token). The beam therefore explores different *orders* of filling
  the same canvas, keeping the branches whose partial assignments
  look best -- including branches the greedy path would have
  discarded at the first step.
- THE CERTIFICATE IS A HARD FILTER, NOT A SOFT SCORE. A successor
  is admitted only if the independent verifier (scv.verify_canvas)
  accepts its canvas. The beam score is the sum of log-probabilities
  of the commits so far; certificate validity never trades off
  against likelihood, it gates it. This is the deliberate contrast
  with CARVE: CARVE scores expansions by Jensen-Shannon divergence
  of the model's *own* predictive distribution -- self-referential,
  a confidently-wrong model verifies its own mistakes. CGBS scores
  branches with a verifier the model cannot game, and the
  certificate is replayable from the token list alone.
- CERTIFICATE PRUNING WITH MONOTONE TABOO (VRC doctrine). A
  successor the verifier rejects is pruned and its (position, token)
  is tabooed run-wide, so no other branch re-proposes the same
  pruned commit. The gate (dcmr._would_repeat) already filters the
  same property, so pruning is normally zero -- it is the
  independent re-check doing its job, counted, not assumed.
- TERMINATION: one commit per state per round, so a run needs
  exactly as many rounds as there are unfilled positions.
  ``max_rounds`` must cover that; a run that cannot finish in the
  budget, or whose beam empties (every branch stalled), reports
  ``completed=False`` with masks remaining rather than emitting an
  uncertified canvas. Fail-closed like the rest of the family.
- The cgbs-v1 certificate itemizes the beam (width, rounds,
  expansions, prunes, verified successors, gate rejects, final
  score) alongside the commit log and taboo list, so a third party
  can replay the claim from the token list alone.

A beam never touches a certificate-verified commit except by
abandoning the branch that holds it, and any commit on the winning
branch re-passed the gate at generation time. No logits are ever
modified or fused.

Prior art (absorbed, not copied):
- CARVE (arXiv:2608.30922): verified expansion as the guiding
  signal. CGBS takes the diagnosis but not the mechanism: the
  verifier is the independent certificate (not the model's own
  distribution), and the search is a beam over commit orders (not
  canvas-length expansion).
- Where-to-Unmask / Gt-Margin: order matters. CGBS searches orders
  at inference with no training and no learned planner.
- EB-Sampler: adaptive unmasking. CGBS is not a speed device; it
  spends *more* forward passes (up to beam_width per round) to buy
  search, and says so in the certificate.
- In-repo: dcmr._would_repeat / _topk_for_pos (the certificate
  gate), scv.verify_canvas (independent re-check), the CDR/CBRS/
  CWTC/CSBR fail-closed pattern, VRC's monotone taboo.

Pre-registered falsifiers (see xiiv/tests/test_cgbs.py):
- F1 beam rescues a greedy trap: on the trap stub (an attractive
  0.57 commit at position 4 strands positions 5-7 below threshold),
  cgbs with beam_width=4 completes while the vanilla policy cannot
  complete; the final canvas holds the safe alternative (51) at
  position 4 and the trap token (50) is absent there.
- F2 certificate gate is live: on the repeat stub, gate_rejects >=
  1, successors_verified == expansions - certificate_prunes, and
  the final canvas is clean under an independent verify_canvas.
- F3 final soundness: completed=True implies the independent
  verify_canvas(final) is clean (recomputed).
- F4 determinism: identical runs -> identical canvas, commit log
  and score.
- F5 budget: forward_passes <= max_rounds * beam_width, rounds <=
  max_rounds, expansions <= rounds * beam_width *
  branch_positions * branch_tokens.
- F6 width-1 greedy equivalence: cgbs with beam_width=1 produces
  the same canvas as an independent greedy reference loop in the
  test, on the well-behaved stub.
- KILL: F1 fails -> the module does not ship. F2 or F3 failing
  means the beam is not actually certificate-gated -> kill.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.
"""

import math

import torch

from .dcmr import _topk_for_pos, _would_repeat
from .scv import verify_canvas

__all__ = [
    "CGBSConfig",
    "CGBSTrace",
    "cgbs_sample",
    "cgbs_certificate",
    "POLICIES",
]

POLICIES = ("vanilla", "cgbs")


class CGBSConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        topk: int = 4,
        beam_width: int = 4,
        branch_positions: int = 2,
        branch_tokens: int = 2,
        max_rounds: int = 64,
        conf_threshold: float = 0.35,
    ):
        self.n = n                                # n-gram order of the certificate
        self.topk = topk                          # candidates scanned per position
        self.beam_width = beam_width              # states kept per round
        self.branch_positions = branch_positions  # positions expanded per state
        self.branch_tokens = branch_tokens        # candidates at the best position
        self.max_rounds = max_rounds              # hard termination cap
        self.conf_threshold = conf_threshold      # commit bar


class CGBSTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.rounds = 0
        self.forward_passes = 0
        self.expansions = 0         # successors generated (pre-prune)
        self.successors_verified = 0  # successors passing verify_canvas
        self.certificate_prunes = 0   # successors rejected by verify_canvas
        self.gate_rejects = 0       # top-1 candidates refused by the gate
        self.commits = 0            # commits on the winning branch
        self.completed = False
        self.final_no_repeat = False
        self.score = 0.0            # sum log-prob of the winning branch
        self.commit_log = []        # per-commit audit entries (final canvas)
        self.taboo = []             # "pos:tok" pairs, final

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def cgbs_certificate(trace: CGBSTrace) -> dict:
    """The cgbs-v1 certificate: the beam, the prunes, and the
    independent re-check of the final canvas."""
    return {
        "version": "cgbs-v1",
        "policy": trace.policy,
        "beam_width": trace.beam_width if hasattr(trace, "beam_width") else None,
        "rounds": trace.rounds,
        "forward_passes": trace.forward_passes,
        "expansions": trace.expansions,
        "successors_verified": trace.successors_verified,
        "certificate_prunes": trace.certificate_prunes,
        "gate_rejects": trace.gate_rejects,
        "commits": trace.commits,
        "score": round(trace.score, 4),
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "commit_log": trace.commit_log,
        "taboo": trace.taboo,
    }


def _safe_candidates(canvas, pos, probs_row, cfg, mask_id, taboo):
    """Gate-safe candidates at pos, best-first.

    Returns (cands, top1_rejected): cands is a list of (tok, prob)
    that pass the no-repeat gate and clear conf_threshold, in
    descending probability order; top1_rejected is True when the
    highest-probability candidate was refused by the gate.
    """
    banned = {mask_id}
    banned.update(tok for (p, tok) in taboo if p == pos)
    cands = _topk_for_pos(probs_row, cfg.topk, banned)
    if not cands:
        return [], False
    top1_tok = cands[0][0]
    top1_rejected = _would_repeat(canvas, pos, top1_tok, cfg.n, mask_id)
    safe = [(tok, prob) for tok, prob in cands
            if prob >= cfg.conf_threshold
            and not _would_repeat(canvas, pos, tok, cfg.n, mask_id)]
    return safe, top1_rejected


class _BeamState:
    """One partial canvas in the beam."""

    __slots__ = ("canvas", "commit_log", "score")

    def __init__(self, canvas, commit_log, score):
        self.canvas = canvas
        self.commit_log = commit_log
        self.score = score


def cgbs_sample(model, length: int, mask_id: int, config=None,
                policy: str = "cgbs", frozen=None):
    """Run the CGBS sampler.

    ``frozen`` maps position -> token for prompt tokens that are never
    committed by the beam (and never tabooed). Returns (ids, trace,
    certificate); ``ids`` is a LongTensor of shape (length,) with
    ``mask_id`` where the run did not complete. Deterministic (greedy);
    no sampling anywhere.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; want one of {POLICIES}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    cfg = config if config is not None else CGBSConfig()
    if policy == "vanilla":
        # The degenerate beam: one state, one successor per round --
        # the greedy one-commit-per-round baseline the beam must beat.
        cfg = CGBSConfig(
            n=cfg.n, topk=cfg.topk, beam_width=1, branch_positions=1,
            branch_tokens=1, max_rounds=cfg.max_rounds,
            conf_threshold=cfg.conf_threshold)
    trace = CGBSTrace(policy)
    trace.beam_width = cfg.beam_width
    canvas0 = [mask_id] * length
    frozen = dict(frozen) if frozen else {}
    for pos, tok in frozen.items():
        canvas0[pos] = int(tok)
    taboo = set()           # (pos, tok) pairs pruned by the verifier
    beam = [_BeamState(list(canvas0), [], 0.0)]

    for round_no in range(cfg.max_rounds):
        # A complete state ends the run: the beam is scored, so the
        # first complete state at the top of the order is the winner.
        complete = [s for s in beam
                    if all(t != mask_id for t in s.canvas)]
        if complete:
            break
        successors = []
        for state in beam:
            masked = [p for p, t in enumerate(state.canvas)
                      if t == mask_id]
            if not masked:
                continue
            with torch.no_grad():
                logits = model(torch.tensor([state.canvas],
                                            dtype=torch.long))
            trace.forward_passes += 1
            probs = torch.softmax(logits[0].float(), dim=-1)

            # Rank masked positions by their best safe candidate.
            ranked = []
            for pos in masked:
                safe, top1_rejected = _safe_candidates(
                    state.canvas, pos, probs[pos], cfg, mask_id, taboo)
                if top1_rejected:
                    trace.gate_rejects += 1
                if safe:
                    ranked.append((pos, safe))
            if not ranked:
                continue
            ranked.sort(key=lambda ps: (-ps[1][0][1], ps[0]))
            for rank, (pos, safe) in enumerate(ranked):
                if rank >= cfg.branch_positions:
                    break
                take = cfg.branch_tokens if rank == 0 else 1
                for tok, prob in safe[:take]:
                    new_canvas = list(state.canvas)
                    new_canvas[pos] = tok
                    trace.expansions += 1
                    # Independent re-check: the successor is admitted
                    # only if the verifier accepts its canvas.
                    check = verify_canvas(new_canvas, mask_id, cfg.n)
                    if not check["final_no_repeat"]:
                        trace.certificate_prunes += 1
                        taboo.add((pos, tok))
                        continue
                    trace.successors_verified += 1
                    successors.append(_BeamState(
                        new_canvas,
                        state.commit_log + [{
                            "pos": pos,
                            "tok": tok,
                            "round": round_no,
                            "prob": round(prob, 4),
                        }],
                        state.score + math.log(prob),
                    ))
        trace.rounds += 1
        if not successors:
            # Every branch stalled: the beam is empty, fail closed.
            beam = []
            break
        # Dedup identical canvases (keep the best score), then keep
        # the top beam_width by score with a deterministic tie-break
        # on commit order: on equal scores the branch that committed
        # the earlier position first wins, so the degenerate beam
        # (width 1) reduces exactly to greedy one-commit-per-round.
        best = {}
        for s in successors:
            key = tuple(s.canvas)
            if key not in best or s.score > best[key].score:
                best[key] = s
        ordered = sorted(
            best.values(),
            key=lambda s: (-s.score,
                           tuple((e["pos"], e["tok"])
                                 for e in s.commit_log)))
        beam = ordered[: cfg.beam_width]

    winner = None
    for state in beam:
        if all(t != mask_id for t in state.canvas):
            winner = state
            break
    if winner is None:
        # Incomplete: report the best partial canvas, masks intact.
        canvas = beam[0].canvas if beam else canvas0
        trace.completed = False
        trace.commits = 0
        trace.commit_log = []
        trace.score = 0.0
    else:
        canvas = winner.canvas
        trace.completed = True
        trace.commits = len(winner.commit_log)
        trace.commit_log = winner.commit_log
        trace.score = winner.score
    check = verify_canvas(canvas, mask_id, cfg.n)
    trace.final_no_repeat = bool(check["final_no_repeat"])
    trace.taboo = [f"{pos}:{tok}" for (pos, tok) in sorted(taboo)]
    cert = cgbs_certificate(trace)
    return torch.tensor(canvas, dtype=torch.long), trace, cert
