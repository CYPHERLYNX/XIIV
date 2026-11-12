"""Tests for xiiv/cgbs.py -- Certificate-Guided Beam Search.

Falsifiers (pre-registered in the module docstring):
- F1 beam rescues a greedy trap: cgbs beam_width=4 completes on the
  trap stub while vanilla cannot; final canvas holds 51 at position
  4, trap token 50 absent there.
- F2 certificate gate is live: on the repeat stub, gate_rejects >= 1,
  successors_verified == expansions - certificate_prunes, and the
  final canvas is clean under an independent verify_canvas.
- F3 final soundness: completed=True implies the independent
  verify_canvas(final) is clean (recomputed).
- F4 determinism: identical runs -> identical canvas, commit log
  and score.
- F5 budget: forward_passes <= max_rounds * beam_width, rounds <=
  max_rounds, expansions <= rounds * beam_width *
  branch_positions * branch_tokens.
- F6 width-1 greedy equivalence: cgbs beam_width=1 produces the same
  canvas as an independent greedy reference loop, on the
  well-behaved stub.
"""

import math

import torch

from xiiv.cgbs import CGBSConfig, cgbs_sample, cgbs_certificate, POLICIES
from xiiv.dcmr import _topk_for_pos, _would_repeat
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
    With conf_threshold=0.35: the greedy path commits 50, the tail
    stalls, and only a beam that kept the 51-branch alive can
    complete.
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


def well_behaved_model():
    """Length 6, n=2. Positions 0/1 pin (10, 11); position 2's top
    candidate 10 and position 3's top candidate 11 would jointly
    repeat the bigram (10, 11) -- the gate must route around one of
    them via the 0.43-probability alternatives."""
    table = {
        0: [(10, 0.0)],
        1: [(11, 0.0)],
        2: [(10, 0.0), (12, -0.3)],
        3: [(11, 0.0), (13, -0.3)],
        4: [(14, 0.0)],
        5: [(15, 0.0)],
    }
    return ScriptedModel(6, 128, table)


def greedy_reference(model, length, mask_id, cfg):
    """Independent greedy loop: one commit per round, best safe
    candidate at the best-ranked position. Mirrors the degenerate
    beam (width 1) without sharing its code."""
    canvas = [mask_id] * length
    for _ in range(cfg.max_rounds):
        masked = [p for p, t in enumerate(canvas) if t == mask_id]
        if not masked:
            break
        with torch.no_grad():
            logits = model(torch.tensor([canvas], dtype=torch.long))
        probs = torch.softmax(logits[0].float(), dim=-1)
        ranked = []
        for pos in masked:
            cands = _topk_for_pos(probs[pos], cfg.topk, {mask_id})
            safe = [(t, pr) for t, pr in cands
                    if pr >= cfg.conf_threshold
                    and not _would_repeat(canvas, pos, t, cfg.n, mask_id)]
            if safe:
                ranked.append((pos, safe))
        if not ranked:
            break
        ranked.sort(key=lambda ps: (-ps[1][0][1], ps[0]))
        pos, safe = ranked[0]
        canvas[pos] = safe[0][0]
    return canvas


def test_f1_beam_rescues_trap():
    model = TrapModel()
    cfg = CGBSConfig(n=4, beam_width=4, max_rounds=64)
    ids, trace, cert = cgbs_sample(model, 8, MASK, config=cfg)
    assert trace.completed, "cgbs did not complete the trap stub"
    canvas = ids.tolist()
    assert canvas[4] == 51, f"expected safe 51 at pos 4, got {canvas}"
    assert 50 not in canvas, f"trap token 50 leaked into {canvas}"
    assert cert["version"] == "cgbs-v1"
    assert cert["completed"] is True


def test_f1_vanilla_stalls_on_trap():
    model = TrapModel()
    cfg = CGBSConfig(n=4, beam_width=4, max_rounds=64)
    ids, trace, cert = cgbs_sample(model, 8, MASK, config=cfg,
                                   policy="vanilla")
    assert not trace.completed, "vanilla unexpectedly completed the trap"
    assert MASK in ids.tolist(), "incomplete run must keep masks"
    assert cert["completed"] is False


def test_f2_certificate_gate_live():
    model = well_behaved_model()
    cfg = CGBSConfig(n=2, beam_width=4, max_rounds=64)
    ids, trace, cert = cgbs_sample(model, 6, MASK, config=cfg)
    assert trace.completed, "cgbs did not complete the repeat stub"
    assert trace.gate_rejects >= 1, (
        f"gate never fired; rejects={trace.gate_rejects}")
    assert trace.successors_verified == (
        trace.expansions - trace.certificate_prunes), (
        "verified + pruned must equal expansions")
    check = verify_canvas(ids.tolist(), MASK, 2)
    assert check["final_no_repeat"], (
        f"final canvas repeats: {check}")
    assert trace.final_no_repeat is True


def test_f3_final_soundness():
    model = well_behaved_model()
    cfg = CGBSConfig(n=2, beam_width=4, max_rounds=64)
    ids, trace, cert = cgbs_sample(model, 6, MASK, config=cfg)
    assert trace.completed
    check = verify_canvas(ids.tolist(), MASK, 2)
    assert check["final_no_repeat"] is True
    assert cert["final_no_repeat"] is True
    assert cert["commits"] == len(cert["commit_log"]) == 6


def test_f4_determinism():
    model = TrapModel()
    cfg = CGBSConfig(n=4, beam_width=4, max_rounds=64)
    ids1, tr1, _ = cgbs_sample(model, 8, MASK, config=cfg)
    ids2, tr2, _ = cgbs_sample(model, 8, MASK, config=cfg)
    assert ids1.tolist() == ids2.tolist()
    assert tr1.commit_log == tr2.commit_log
    assert tr1.score == tr2.score
    assert tr1.expansions == tr2.expansions


def test_f5_budget():
    model = TrapModel()
    cfg = CGBSConfig(n=4, beam_width=4, max_rounds=64)
    _, trace, _ = cgbs_sample(model, 8, MASK, config=cfg)
    assert trace.rounds <= cfg.max_rounds
    assert trace.forward_passes <= cfg.max_rounds * cfg.beam_width, (
        f"forward_passes={trace.forward_passes}")
    assert trace.expansions <= (
        trace.rounds * cfg.beam_width
        * cfg.branch_positions * cfg.branch_tokens), (
        f"expansions={trace.expansions} rounds={trace.rounds}")


def test_f6_width1_matches_greedy_reference():
    model = well_behaved_model()
    cfg = CGBSConfig(n=2, beam_width=1, max_rounds=64)
    ids, trace, _ = cgbs_sample(model, 6, MASK, config=cfg)
    assert trace.completed
    ref = greedy_reference(model, 6, MASK, cfg)
    assert ids.tolist() == ref, (
        f"beam_width=1 {ids.tolist()} != greedy reference {ref}")


def test_verifier_sanity_flags_repeats():
    # Hand-built canvas with a repeated bigram: the verifier itself
    # must flag it, or the F2/F3 checks above would be vacuous.
    assert verify_canvas([10, 11, 10, 11], MASK, 2)["final_no_repeat"] is False
    assert verify_canvas([10, 11, 12, 13], MASK, 2)["final_no_repeat"] is True


def test_frozen_positions_never_recommitted():
    model = well_behaved_model()
    cfg = CGBSConfig(n=2, beam_width=4, max_rounds=64)
    ids, trace, _ = cgbs_sample(model, 6, MASK, config=cfg,
                                frozen={0: 10, 1: 11})
    assert trace.completed
    canvas = ids.tolist()
    assert canvas[0] == 10 and canvas[1] == 11
    assert all(e["pos"] not in (0, 1) for e in trace.commit_log)


def test_bad_policy_and_length_rejected():
    model = well_behaved_model()
    cfg = CGBSConfig(n=2)
    try:
        cgbs_sample(model, 6, MASK, config=cfg, policy="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown policy must raise ValueError")
    try:
        cgbs_sample(model, 0, MASK, config=cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("length 0 must raise ValueError")
    assert POLICIES == ("vanilla", "cgbs")


def test_certificate_shape():
    model = well_behaved_model()
    cfg = CGBSConfig(n=2, beam_width=2, max_rounds=64)
    _, trace, cert = cgbs_sample(model, 6, MASK, config=cfg)
    assert cert["version"] == "cgbs-v1"
    assert cert["score"] == round(trace.score, 4)
    assert cert["score"] < 0.0  # sum of log-probs
    assert abs(trace.score - sum(
        math.log(e["prob"]) for e in trace.commit_log)) < 1e-3
