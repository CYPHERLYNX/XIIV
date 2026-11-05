"""Tests for xiiv/scv.py -- Speculative Certificate Verification.

Falsifiers (pre-registered in the module docstring):
- F1 speculative dividend: deceptive stub -> SCV completes in fewer
  forward passes than gated top-1 vanilla, which stalls.
- F2 acceptance soundness: every accepted candidate was certificate-clean
  at accept time (replayed independently); rejects > 0.
- F3 verifier-over-model: the fork stub's accepted token is the model's
  rank-1 candidate, not rank-0.
- F4 prefix property: a speculation-exhausted position stops the round;
  safe tail positions wait.
- F5 fail-closed: the repeater stub never completes; nothing violating
  is committed; the certificate claim is False.
- F6 batching: clean stub -> forward_passes == ceil(n_masked / frontier).
- F7 determinism: identical runs -> identical accept order.
"""

import torch

from xiiv.scv import (SCVConfig, scv_sample, scv_certificate,
                      verify_canvas, POLICIES)
from xiiv.dcmr import _would_repeat

MASK = 0


class ScriptedModel:
    """Canned logits: table maps pos -> [(tok, logit), ...] (rank order)."""

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


def _deceptive():
    """F1/F2/F3 stub: model top-1 is certificate-violating at pos 3 and
    pos 5 (frozen trigrams on both sides); top-2 is safe."""
    frozen = {0: 7, 1: 7, 2: 7, 6: 8, 7: 8, 8: 8}
    table = {
        3: [(7, 10.0), (13, 9.0), (23, 8.0)],
        4: [(9, 10.0), (19, 9.0), (29, 8.0)],
        5: [(8, 10.0), (15, 9.0), (25, 8.0)],
    }
    return ScriptedModel(9, 32, table), frozen


def _prefix_stub():
    """F4 stub: pos 4's only candidate completes a frozen repeat;
    pos 5 and pos 9 are safe-if-alone but sit behind it."""
    frozen = {0: 7, 1: 7, 2: 7, 6: 7, 7: 6, 8: 7}
    table = {
        3: [(6, 10.0)],
        4: [(7, 9.5)],
        5: [(5, 9.0)],
        9: [(5, 8.0)],
    }
    return ScriptedModel(10, 32, table), frozen


def _repeater():
    """F5 stub: the model only ever predicts the repeating token."""
    frozen = {0: 7, 1: 7, 2: 7}
    table = {p: [(7, 10.0)] for p in (3, 4, 5, 6)}
    return ScriptedModel(7, 32, table), frozen


def _clean():
    """F6/F7 stub: all-distinct tokens, no repeat possible."""
    table = {p: [(10 + p, 10.0), (20 + p, 9.0)] for p in range(10)}
    return ScriptedModel(10, 64, table), {}


def test_policies_known():
    assert POLICIES == ("vanilla", "scv")


def test_f1_speculative_dividend():
    model, frozen = _deceptive()
    cfg = SCVConfig(n=3, frontier=8, draft_depth=3, max_steps=6,
                    stall_budget=3)
    _, t_scv, _ = scv_sample(model, 9, MASK, config=cfg, frozen=frozen,
                             policy="scv")
    _, t_van, _ = scv_sample(model, 9, MASK, config=cfg, frozen=frozen,
                             policy="vanilla")
    assert t_scv.completed, "SCV must recover via candidate depth"
    assert t_scv.final_no_repeat
    assert not t_van.completed, "gated top-1 vanilla must stall"
    assert t_scv.forward_passes < t_van.forward_passes, (
        t_scv.forward_passes, t_van.forward_passes)
    assert t_scv.rejects >= 2, "verifier must actually arbitrate"


def test_f2_acceptance_soundness_replay():
    model, frozen = _deceptive()
    cfg = SCVConfig(n=3, frontier=8, draft_depth=3)
    _, trace, _ = scv_sample(model, 9, MASK, config=cfg, frozen=frozen,
                             policy="scv")
    canvas = [MASK] * 9
    for p, t in frozen.items():
        canvas[p] = t
    for (rnd, p, tok, rank) in trace.accept_order:
        assert not _would_repeat(canvas, p, tok, 3, MASK), (rnd, p, tok)
        canvas[p] = tok
    assert trace.rejects > 0


def test_f3_verifier_overrules_model_ranking():
    model, frozen = _deceptive()
    cfg = SCVConfig(n=3, frontier=8, draft_depth=3)
    _, trace, _ = scv_sample(model, 9, MASK, config=cfg, frozen=frozen,
                             policy="scv")
    by_pos = {p: (tok, rank) for (_, p, tok, rank) in trace.accept_order}
    tok, rank = by_pos[3]
    assert (tok, rank) == (13, 1), (tok, rank)  # model's rank-0 (7) rejected
    tok5, rank5 = by_pos[5]
    assert (tok5, rank5) == (15, 1), (tok5, rank5)


def test_f4_prefix_property():
    model, frozen = _prefix_stub()
    cfg = SCVConfig(n=3, frontier=8, draft_depth=1, max_steps=8,
                    stall_budget=3)
    _, trace, _ = scv_sample(model, 10, MASK, config=cfg, frozen=frozen,
                             policy="scv")
    assert trace.tail_stops >= 1, "pos 4 must exhaust and stop the round"
    assert trace.round_accepts[1] == [], (
        "safe tail positions must wait after a prefix stop")
    committed = {p for (_, p, _, _) in trace.accept_order}
    assert committed == {3}, committed
    assert not trace.completed, "fail-closed: never strand, never violate"


def test_f5_fail_closed():
    model, frozen = _repeater()
    cfg = SCVConfig(n=3, frontier=8, draft_depth=1, max_steps=8,
                    stall_budget=3)
    x, trace, cert = scv_sample(model, 7, MASK, config=cfg, frozen=frozen,
                                policy="scv")
    assert not trace.completed
    assert trace.accepts == 0 and trace.commits == 0
    assert cert["completed"] is False
    assert x[0].tolist() == [7, 7, 7, 0, 0, 0, 0], x[0].tolist()


def test_f6_verifier_decided_batching():
    model, frozen = _clean()
    cfg = SCVConfig(n=4, frontier=4, draft_depth=2)
    _, trace, cert = scv_sample(model, 10, MASK, config=cfg, frozen=frozen,
                                policy="scv")
    assert trace.completed and trace.final_no_repeat
    assert trace.forward_passes == 3, trace.forward_passes  # ceil(10/4)
    assert trace.rejects == 0
    assert all(rank == 0 for (_, _, _, rank) in trace.accept_order)
    assert cert["completed"] is True


def test_f7_determinism():
    model, frozen = _clean()
    cfg = SCVConfig(n=4, frontier=4, draft_depth=2)
    _, t1, _ = scv_sample(model, 10, MASK, config=cfg, frozen=frozen,
                          policy="scv")
    _, t2, _ = scv_sample(model, 10, MASK, config=cfg, frozen=frozen,
                          policy="scv")
    assert t1.accept_order == t2.accept_order


def test_verify_canvas_independent():
    ok = verify_canvas([1, 2, 3, 4, 5], MASK, 3)
    assert ok["final_no_repeat"] and ok["n_repeated_windows"] == 0
    bad = verify_canvas([7, 7, 7, 7, 2], MASK, 3)
    assert not bad["final_no_repeat"] and bad["n_repeated_windows"] > 0
    masked = verify_canvas([7, 7, MASK, 1, 2], MASK, 3)
    assert masked["final_no_repeat"], "masked windows are not complete"


def test_certificate_fail_closed_without_completion():
    model, frozen = _repeater()
    cfg = SCVConfig(n=3, frontier=8, draft_depth=1, stall_budget=1,
                    max_steps=4)
    _, trace, _ = scv_sample(model, 7, MASK, config=cfg, frozen=frozen,
                             policy="scv")
    cert = scv_certificate(trace)
    assert cert["cert"] == "scv-v1"
    assert cert["completed"] is False
