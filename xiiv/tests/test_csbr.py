"""Tests for xiiv/csbr.py -- Certificate-Snapshot Backtracking Rollback.

Falsifiers (pre-registered in the module docstring):
- F1 rollback rescues a stall: on the trap stub (an attractive 0.6
  commit at position 4 strands positions 5-7 below threshold), csbr
  completes with exactly 1 rollback while the vanilla policy cannot
  complete.
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
"""

import torch

from xiiv.csbr import CSBRConfig, csbr_sample, csbr_certificate, POLICIES
from xiiv.scv import verify_canvas

MASK = 0


class ScriptedModel:
    """Canned logits: table maps pos -> [(tok, logit), ...]."""

    def __init__(self, length, vocab, table):
        self.length = length
        self.vocab = vocab
        self.table = table

    def __call__(self, x):
        logits = torch.full((1, self.length, self.vocab), -1e9)
        for p, cands in self.table.items():
            for tok, lg in cands:
                logits[0, p, tok] = lg
        return logits


class TrapModel:
    """State-dependent trap stub (length 8, vocab 128).

    Positions 0-3 commit at prob ~0.95. Position 4 proposes the trap
    token 50 at ~0.57 with the safe alternative 51 at ~0.43. Positions
    5-7 propose at ~0.90 ONLY when the canvas holds 51 at position 4;
    while position 4 holds 50 (or is still masked) they sit at ~0.29.
    With conf_threshold=0.35: the trap commits, the tail stalls, and
    only a rollback + taboo of (4, 50) lets 51 through (0.43 clears
    the bar) and unblocks the tail.
    """

    def __init__(self, length=8, vocab=128):
        self.length = length
        self.vocab = vocab

    def __call__(self, x):
        canvas = x[0].tolist()
        logits = torch.full((1, self.length, self.vocab), -1e9)
        for p in range(4):
            logits[0, p, 10 + p] = 0.0
            logits[0, p, 100 + p] = -3.0
        logits[0, 4, 50] = 0.0
        logits[0, 4, 51] = -0.3
        tail_high = canvas[4] == 51
        for p in range(5, 8):
            logits[0, p, 20 + p] = 0.0
            if tail_high:
                logits[0, p, 120 + p] = -2.2
            else:
                logits[0, p, 120 + p] = -0.2
                logits[0, p, 60 + p] = -0.2
                logits[0, p, 70 + p] = -0.2
        return logits


def trap_config():
    return CSBRConfig(conf_threshold=0.35, stall_window=2,
                      max_rollbacks=3, reopen=1, budget=4, topk=4)


def test_f1_rollback_rescues_stall():
    cfg = trap_config()
    model = TrapModel()
    ids, trace, cert = csbr_sample(model, 8, MASK, config=cfg,
                                   policy="csbr")
    assert trace.completed, "csbr must complete on the trap stub"
    assert trace.final_no_repeat
    assert trace.rollbacks == 1, (
        f"exactly one rollback expected, got {trace.rollbacks}")
    assert ids.tolist() == [10, 11, 12, 13, 51, 25, 26, 27], (
        f"unexpected final canvas {ids.tolist()}")
    assert len(trace.rollback_log) == 1
    assert trace.rollback_log[0]["reopened"] == [4]
    assert trace.rollback_log[0]["reason"] == "stall"
    assert cert["version"] == "csbr-v1"
    assert cert["completed"] is True

    # Same stub, vanilla policy: must NOT complete (stalls, no rollback).
    model2 = TrapModel()
    _, vtrace, _ = csbr_sample(model2, 8, MASK, config=cfg,
                              policy="vanilla")
    assert not vtrace.completed
    assert vtrace.rollbacks == 0
    assert vtrace.commits == 5  # positions 0-4 only; tail never commits


def test_f2_taboo_prevents_repeat_trap():
    cfg = trap_config()
    model = TrapModel()
    ids, trace, cert = csbr_sample(model, 8, MASK, config=cfg,
                                   policy="csbr")
    assert trace.completed
    assert "4:50" in trace.taboo, (
        f"trap pair (4,50) must be tabooed, taboo={trace.taboo}")
    assert [4, 50] in trace.rollback_log[0]["taboo_added"]
    assert ids.tolist()[4] == 51
    assert all(not (e["pos"] == 4 and e["tok"] == 50)
               for e in trace.commit_log)
    assert cert["taboo"] == trace.taboo


def test_f3_snapshot_integrity():
    cfg = trap_config()
    model = TrapModel()
    _, trace, _ = csbr_sample(model, 8, MASK, config=cfg, policy="csbr")
    assert trace.completed
    assert trace.snapshots_taken >= 3, (
        f"expected >= 3 snapshots, got {trace.snapshots_taken}")
    assert trace.snapshots_verified == trace.snapshots_taken, (
        "every pushed snapshot must have passed the independent verifier")


def test_f4_final_soundness():
    cfg = trap_config()
    model = TrapModel()
    ids, trace, _ = csbr_sample(model, 8, MASK, config=cfg, policy="csbr")
    assert trace.completed
    check = verify_canvas(ids.tolist(), MASK, cfg.n)
    assert check["final_no_repeat"]


def test_f5_determinism():
    cfg = trap_config()
    ids1, trace1, _ = csbr_sample(TrapModel(), 8, MASK, config=cfg,
                                  policy="csbr")
    ids2, trace2, _ = csbr_sample(TrapModel(), 8, MASK, config=cfg,
                                  policy="csbr")
    assert ids1.tolist() == ids2.tolist()
    assert trace1.commit_log == trace2.commit_log
    assert trace1.rollback_log == trace2.rollback_log
    assert trace1.taboo == trace2.taboo


def test_f6_budget():
    cfg = trap_config()
    model = TrapModel()
    _, trace, _ = csbr_sample(model, 8, MASK, config=cfg, policy="csbr")
    assert trace.completed
    assert trace.forward_passes <= cfg.max_steps
    assert trace.rollbacks <= cfg.max_rollbacks
    per_step = {}
    for e in trace.commit_log:
        per_step[e["step"]] = per_step.get(e["step"], 0) + 1
    assert all(c <= cfg.budget for c in per_step.values()), (
        f"per-step commits {per_step} exceed budget {cfg.budget}")
    assert trace.commits == len(trace.commit_log)


def test_f7_no_rollback_inertness():
    # Well-behaved stub: every position commits at ~0.90 on step 0.
    # Rollback must be inert: zero rollbacks, canvas identical to
    # the vanilla policy on the same stub.
    table = {}
    for p in range(8):
        table[p] = [(10 + p, 0.0), (100 + p, -2.2)]
    cfg = CSBRConfig()
    ids_c, trace_c, _ = csbr_sample(ScriptedModel(8, 128, table), 8, MASK,
                                    config=cfg, policy="csbr")
    ids_v, trace_v, _ = csbr_sample(ScriptedModel(8, 128, table), 8, MASK,
                                    config=cfg, policy="vanilla")
    assert trace_c.completed
    assert trace_c.rollbacks == 0
    assert trace_c.rollback_log == []
    assert ids_c.tolist() == ids_v.tolist()
    assert trace_c.commit_log == trace_v.commit_log
