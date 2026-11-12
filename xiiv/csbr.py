"""CSBR -- Certificate-Snapshot Backtracking Rollback for masked-diffusion decoding.

Every sampler in this family (DCMR, CGEC, CDR, PTCS, SCV, CBRS, CCCA,
CWTC) commits monotonically: once a token is on the canvas it stays
there, and a run that stops making progress can only stall out or give
up. The literature now says that is the wrong shape for the failure:

- RDD (arXiv:2602.00150): irreversible block commitments can lead to
  stagnation -- the reverse process fails to make further progress
  under a suboptimal context. Their fix backtracks to earlier blocks
  via cached model states, with confidence-guided re-masking.
- ProSeCo (arXiv:2602.11590): already-decoded tokens are the
  error-accumulation surface; corrective refinement steps between
  unmasking steps can change them -- but only via a trained corrector.
- BackPlay (arXiv:2601.06428): later context exposes earlier mistakes;
  periodically revisit generated tokens through selective remasking --
  but via a trained correction head, on a schedule, whether or not
  anything is wrong.

CSBR takes the *diagnoses* (stagnation is a state failure; move
backwards. Already-decoded tokens are the error surface; later context
exposes earlier mistakes) and builds a different mechanism around
them, one that fits the guarantee-native thesis and needs no training:

- SNAPSHOTS ONLY AT CERTIFICATE-CLEAN STATES. After every step that
  commits >= 1 token, the canvas is re-checked by the independent
  verifier; only a passing canvas is pushed on the snapshot stack
  (bounded, last ``snapshot_keep``). A rollback can therefore only
  ever restore a canvas the verifier already accepted.
- STAGNATION TRIGGER ON CERTIFICATE PROGRESS, not confidence.
  ``stall_window`` consecutive steps with zero commits (all remaining
  candidates below threshold, or all gate/taboo-blocked) is the trap
  signal.
- ROLLBACK WITH MONOTONE TABOO (VRC doctrine). Restore the most
  recent snapshot; every (position, token) committed *after* that
  snapshot is tabooed, and the ``reopen`` lowest-margin committed
  positions inside the snapshot are re-masked with their tokens
  tabooed too. Taboo only ever grows, so a rollback can never replay
  the trajectory that stalled. The restored snapshot is consumed, so a
  second stall rolls back one snapshot further (escalation).
- TERMINATION: monotone taboo growth + ``max_rollbacks`` cap +
  ``max_steps`` cap. Fail-closed like the rest of the family: an
  incomplete run reports ``completed=False`` with masks remaining
  rather than emitting an uncertified canvas.
- The csbr-v1 certificate itemizes every rollback (step, snapshot
  step, taboo added, positions re-opened, reason) alongside every
  commit, so a third party can replay the claim from the token list
  alone.

A rollback never touches a certificate-verified commit except by
re-opening it, and any re-commit must re-pass the gate. No logits are
ever modified or fused.

Prior art (absorbed, not copied):
- RDD (arXiv:2602.00150): stagnation as a state-dependent failure and
  rollback as the response. CSBR takes the diagnosis but not the
  mechanism: rollback target is a certificate-verified snapshot, not
  a cached model state; re-opening is margin-guided, not
  confidence-guided; repeated failure trajectories are prevented by
  monotone taboo, not by heuristics.
- ProSeCo / BackPlay: revisiting already-decoded tokens. CSBR revisits
  only on stagnation (never on a schedule) and re-opens by certificate
  margin, with no trained corrector head anywhere in the loop.
- In-repo: dcmr._would_repeat / _topk_for_pos (the certificate gate),
  scv.verify_canvas (independent re-check), the CDR/CBRS/CWTC
  fail-closed pattern, VRC's monotone taboo.

Pre-registered falsifiers (see xiiv/tests/test_csbr.py):
- F1 rollback rescues a stall: on the trap stub (an attractive 0.6
  commit at position 4 strands positions 5-7 below threshold), csbr
  completes with exactly 1 rollback while the vanilla policy in the
  same module cannot complete.
- F2 taboo prevents the repeat trap: the trap (position, token) pair
  is in the final taboo list, never re-committed, and absent from the
  final canvas at that position.
- F3 snapshot integrity: every snapshot pushed passed the independent
  verifier (snapshots_verified == snapshots_taken >= 1).
- F4 final soundness: completed=True implies the independent
  verify_canvas(final) is clean (recomputed).
- F5 determinism: identical runs -> identical canvas, commit log and
  rollback log.
- F6 budget: forward_passes <= max_steps, rollbacks <= max_rollbacks,
  at most ``budget`` commits in any step.
- F7 no-rollback inertness: on a well-behaved stub csbr completes
  with zero rollbacks and a canvas identical to the vanilla policy.
- KILL: F1 fails -> the module does not ship. F2 or F3 failing means
  the rollback is not actually certificate-gated -> kill.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.
"""

import torch

from .dcmr import _topk_for_pos, _would_repeat
from .scv import verify_canvas

__all__ = [
    "CSBRConfig",
    "CSBRTrace",
    "csbr_sample",
    "csbr_certificate",
    "POLICIES",
]

POLICIES = ("vanilla", "csbr")


class CSBRConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        topk: int = 4,
        budget: int = 4,
        max_steps: int = 64,
        conf_threshold: float = 0.5,
        stall_window: int = 3,
        max_rollbacks: int = 3,
        reopen: int = 2,
        snapshot_keep: int = 8,
    ):
        self.n = n                                # n-gram order of the certificate
        self.topk = topk                          # candidates scanned per position
        self.budget = budget                      # commits per step
        self.max_steps = max_steps                # hard termination cap
        self.conf_threshold = conf_threshold      # commit bar
        self.stall_window = stall_window          # no-commit steps before rollback
        self.max_rollbacks = max_rollbacks        # hard rollback cap
        self.reopen = reopen                      # lowest-margin positions re-masked
        self.snapshot_keep = snapshot_keep        # bounded snapshot stack


class CSBRTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.steps = 0
        self.forward_passes = 0
        self.commits = 0
        self.confidence_commits = 0
        self.gate_rejects = 0       # top-1 candidates refused by the gate
        self.stalls = 0             # current consecutive no-commit steps
        self.rollbacks = 0
        self.rollback_log = []      # per-rollback audit entries
        self.snapshots_taken = 0
        self.snapshots_verified = 0
        self.completed = False
        self.final_no_repeat = False
        self.commit_log = []        # per-commit audit entries (final canvas)
        self.taboo = []             # "pos:tok" pairs, final

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def csbr_certificate(trace: CSBRTrace) -> dict:
    """The csbr-v1 certificate: what was committed, what rolled back,
    and the independent re-check of the final canvas."""
    return {
        "version": "csbr-v1",
        "policy": trace.policy,
        "commits": trace.commits,
        "confidence_commits": trace.confidence_commits,
        "gate_rejects": trace.gate_rejects,
        "steps": trace.steps,
        "forward_passes": trace.forward_passes,
        "stalls": trace.stalls,
        "rollbacks": trace.rollbacks,
        "rollback_log": trace.rollback_log,
        "snapshots_taken": trace.snapshots_taken,
        "snapshots_verified": trace.snapshots_verified,
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "commit_log": trace.commit_log,
        "taboo": trace.taboo,
    }


def _leading_safe_candidate(canvas, pos, probs_row, cfg, mask_id, taboo):
    """First certificate-safe, non-taboo candidate in the top-k at pos.

    Returns (tok, prob, top1_was_rejected). Tabooed tokens are banned
    from the row before ranking, so a tabooed top-1 can never block its
    runner-up. The top-1 rejection flag is the gate-reject count input.
    """
    banned = {mask_id}
    banned.update(tok for (p, tok) in taboo if p == pos)
    cands = _topk_for_pos(probs_row, cfg.topk, banned)
    if not cands:
        return None
    top1_tok = cands[0][0]
    top1_rejected = _would_repeat(canvas, pos, top1_tok, cfg.n, mask_id)
    for tok, prob in cands:
        if not _would_repeat(canvas, pos, tok, cfg.n, mask_id):
            return (tok, prob, top1_rejected)
    return None


def csbr_sample(model, length: int, mask_id: int, config=None,
                policy: str = "csbr", frozen=None):
    """Run the CSBR sampler.

    ``frozen`` maps position -> token for prompt tokens that are never
    remasked (and never re-opened by a rollback). Returns (ids, trace,
    certificate); ``ids`` is a LongTensor of shape (length,) with
    ``mask_id`` where the run did not complete. Deterministic (greedy);
    no sampling anywhere.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; want one of {POLICIES}")
    cfg = config if config is not None else CSBRConfig()
    trace = CSBRTrace(policy)
    canvas = [mask_id] * length
    frozen = dict(frozen) if frozen else {}
    for pos, tok in frozen.items():
        canvas[pos] = int(tok)
    taboo = set()           # (pos, tok) pairs that may never commit again
    # Snapshot stack: each entry is (canvas_copy, commit_log_copy, step).
    # Only certificate-clean canvases are ever pushed.
    snapshots = []

    for step in range(cfg.max_steps):
        masked = [p for p, t in enumerate(canvas) if t == mask_id]
        if not masked:
            break
        with torch.no_grad():
            logits = model(torch.tensor([canvas], dtype=torch.long))
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)

        candidates = {}
        for pos in masked:
            got = _leading_safe_candidate(canvas, pos, probs[pos], cfg,
                                          mask_id, taboo)
            if got is None:
                continue
            tok, prob, top1_rejected = got
            if top1_rejected:
                trace.gate_rejects += 1
            candidates[pos] = (tok, prob)

        # Commit in descending probability order; re-gate against the
        # evolving canvas so same-step commits cannot jointly repeat.
        order = sorted(candidates,
                       key=lambda p: (-candidates[p][1], p))
        committed_this_step = 0
        for pos in order:
            if committed_this_step >= cfg.budget:
                break
            tok, prob = candidates[pos]
            if prob < cfg.conf_threshold:
                continue
            if (pos, tok) in taboo:
                continue
            if _would_repeat(canvas, pos, tok, cfg.n, mask_id):
                continue
            canvas[pos] = tok
            trace.commits += 1
            trace.confidence_commits += 1
            trace.commit_log.append({
                "pos": pos,
                "tok": tok,
                "step": step,
                "prob": round(prob, 4),
            })
            committed_this_step += 1

        trace.steps += 1
        if committed_this_step == 0:
            trace.stalls += 1
            if trace.stalls < cfg.stall_window:
                continue
            # Stagnation: the trap signal. Roll back if the policy
            # allows it and a certificate-clean snapshot exists.
            if policy != "csbr":
                break
            if trace.rollbacks >= cfg.max_rollbacks or not snapshots:
                break
            snap_canvas, snap_log, snap_step = snapshots.pop()
            # Taboo every (pos, tok) committed after the snapshot.
            taboo_added = []
            for entry in trace.commit_log[len(snap_log):]:
                key = (entry["pos"], entry["tok"])
                if key not in taboo:
                    taboo.add(key)
                    taboo_added.append([entry["pos"], entry["tok"]])
            # Re-open the lowest-margin committed positions inside the
            # snapshot (never frozen positions): re-mask them and taboo
            # the tokens they held, so the model must decide them anew.
            reopened = []
            reopen_cands = sorted(
                (e for e in snap_log if e["pos"] not in frozen),
                key=lambda e: (e["prob"], e["pos"]),
            )[: cfg.reopen]
            for entry in reopen_cands:
                key = (entry["pos"], entry["tok"])
                if key not in taboo:
                    taboo.add(key)
                    taboo_added.append([entry["pos"], entry["tok"]])
                reopened.append(entry["pos"])
            canvas = list(snap_canvas)
            for pos in reopened:
                canvas[pos] = mask_id
            trace.commit_log = [
                dict(e) for e in snap_log
                if e["pos"] not in reopened
            ]
            trace.rollbacks += 1
            trace.rollback_log.append({
                "step": step,
                "snapshot_step": snap_step,
                "taboo_added": taboo_added,
                "reopened": reopened,
                "reason": "stall",
            })
            trace.stalls = 0
            continue

        trace.stalls = 0
        # Snapshot the new state -- but only if the independent
        # verifier accepts it. A snapshot that fails verification is
        # never pushed (loud by omission: snapshots_verified counts it).
        trace.snapshots_taken += 1
        check = verify_canvas(canvas, mask_id, cfg.n)
        if check["final_no_repeat"]:
            trace.snapshots_verified += 1
            snapshots.append((list(canvas),
                              [dict(e) for e in trace.commit_log], step))
            if len(snapshots) > cfg.snapshot_keep:
                snapshots.pop(0)

    # Reconcile: commits counts the commits present in the FINAL
    # canvas; rolled-back commits are accounted in rollback_log, not
    # here (so commits == len(commit_log) always holds).
    trace.commits = len(trace.commit_log)
    trace.confidence_commits = len(trace.commit_log)
    trace.completed = all(t != mask_id for t in canvas)
    check = verify_canvas(canvas, mask_id, cfg.n)
    trace.final_no_repeat = bool(check["final_no_repeat"])
    trace.taboo = [f"{pos}:{tok}" for (pos, tok) in sorted(taboo)]
    cert = csbr_certificate(trace)
    return torch.tensor(canvas, dtype=torch.long), trace, cert
