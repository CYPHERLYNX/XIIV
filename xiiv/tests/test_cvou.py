"""Tests for xiiv/cvou.py -- Certificate-Verified Ordered Unmasking.

Falsifiers (pre-registered in the module docstring):
- F1 L2R order: on a fully-masked canvas, cvou commit positions are
  strictly increasing (0, 1, 2, ...).
- F2 same canvas, same cost: vanilla and cvou produce the identical
  final canvas on the distinct-position-stable stub, in the same
  number of forward passes.
- F3 prefix finality: on the adversarial right-first stub, vanilla's
  certified prefix is 0 until the final round (prefix_area 8 on
  length 8) while cvou's grows every round (prefix_area 36); cvou
  closes each window exactly once, in order.
- F4 segment interleave: on the two-segment prompted stub, cvou
  commits are L2R within each segment and every commit is a segment
  head at its round.
- F5 certificate replay: a real run verifies; tampering with the
  final canvas or with any round's chosen position fails
  verification.
- F6 fail-closed: a model offering no candidate above the confidence
  bar stalls with completed=False and masks remaining; a frozen
  prompt that already repeats fails closed at round zero with zero
  forward passes and preexisting_repeat=True.
- F7 determinism: identical runs -> identical canvas and certificate.
"""

import copy

import pytest
import torch

from xiiv.cvou import (
    CVOUConfig, cvou_sample, cvou_certificate, cvou_verify_certificate,
    order_admissible, segment_heads, closed_windows, certified_prefix,
    POLICIES,
)
from xiiv.scv import verify_canvas

MASK = 0


class TableModel:
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


class FlatModel:
    """All logits equal: every token's prob is 1/vocab, below any
    realistic confidence bar -- no safe candidate anywhere."""

    def __init__(self, length, vocab):
        self.length = length
        self.vocab = vocab

    def __call__(self, x):
        return torch.zeros((1, self.length, self.vocab))


def distinct_model(length=16, vocab=512):
    """Every position proposes its own distinct token (100 + pos),
    stable across rounds, so vanilla and cvou must agree on the
    final canvas."""
    table = {p: [(100 + p, 0.0)] for p in range(length)}
    return TableModel(length, vocab, table)


def right_first_model(length=8, vocab=512):
    """Every position proposes its own distinct token, but the top
    probability INCREASES with position: vanilla commits right-first
    (7, 6, ..., 0) while cvou must still commit left-to-right.

    Each position offers its distinct token at logit p and a shared
    decoy at logit 0, so top prob = exp(p) / (exp(p) + 1), strictly
    increasing in p.
    """
    table = {p: [(100 + p, float(p)), (500, 0.0)] for p in range(length)}
    return TableModel(length, vocab, table)


def two_segment_model(vocab=512):
    """Two masked segments separated by frozen prompt.

    Frozen: 0-3 hold [10, 11, 12, 13], 8-11 hold [20, 21, 22, 23].
    Masked: 4-7 propose [30, 31, 32, 33], 12-15 propose [40, 41, 42,
    43]. The second segment's head outranks the first's, so the
    model interleaves segment heads by rank.
    """
    table = {}
    for p in range(4, 8):
        # Shared decoy at equal logit -> top prob 0.5, outranked by
        # the second segment's heads (~1.0).
        table[p] = [(30 + (p - 4), 0.0), (500, 0.0)]
    for p in range(12, 16):
        # No decoy -> top prob ~1.0, ranked first.
        table[p] = [(40 + (p - 12), 5.0)]
    return TableModel(16, vocab, table)


# ---------------------------------------------------------------------------
# Geometry unit checks
# ---------------------------------------------------------------------------

def test_order_admissible_blank_canvas_is_leftmost_only():
    canvas = [MASK] * 8
    assert segment_heads(canvas, 4, MASK) == [0]
    assert order_admissible(canvas, 0, 4, MASK)
    assert not order_admissible(canvas, 1, 4, MASK)
    assert not order_admissible(canvas, 7, 4, MASK)


def test_order_admissible_advances_after_commit():
    canvas = [MASK] * 8
    canvas[0] = 100
    assert segment_heads(canvas, 4, MASK) == [1]
    canvas[1] = 101
    canvas[2] = 102
    assert segment_heads(canvas, 4, MASK) == [3]


def test_segment_heads_are_per_segment():
    # Frozen prompt splits the canvas into two masked runs.
    canvas = [10, 11, 12, 13, MASK, MASK, MASK, MASK,
              20, 21, 22, 23, MASK, MASK, MASK, MASK]
    assert segment_heads(canvas, 4, MASK) == [4, 12]


def test_closed_windows_counts_finalized_windows():
    # n=4, canvas [10, 11, 12, M, ...]: committing pos 3 closes the
    # window starting at 0 and no other.
    canvas = [10, 11, 12, MASK, MASK, MASK, MASK, MASK]
    assert closed_windows(canvas, 3, 4, MASK) == [0]
    # Committing pos 4 closes nothing yet (window [1..4] needs pos 4
    # plus 1,2,3 -- 1,2,3 are committed, so it DOES close [1]).
    canvas[3] = 13
    assert closed_windows(canvas, 4, 4, MASK) == [1]


def test_certified_prefix():
    assert certified_prefix([MASK, MASK], MASK) == 0
    assert certified_prefix([10, MASK, MASK], MASK) == 1
    assert certified_prefix([10, 11, 12], MASK) == 3


# ---------------------------------------------------------------------------
# F1: L2R order on a blank canvas
# ---------------------------------------------------------------------------

def test_f1_cvou_commits_left_to_right():
    ids, trace, cert = cvou_sample(distinct_model(), 16, MASK)
    assert trace.completed
    positions = [e["pos"] for e in trace.commit_log]
    assert positions == list(range(16))
    assert trace.forward_passes == 16
    assert trace.commits == 16
    assert trace.prefix_area == sum(range(1, 17))  # 136
    assert trace.windows_closed == 13  # windows [0..12] close one per
    # commit once the first 3 positions are in (16 - 4 + 1 = 13).


# ---------------------------------------------------------------------------
# F2: same canvas, same cost as vanilla
# ---------------------------------------------------------------------------

def test_f2_vanilla_and_cvou_agree_on_canvas():
    ids_v, trace_v, _ = cvou_sample(
        distinct_model(), 16, MASK, policy="vanilla")
    ids_c, trace_c, _ = cvou_sample(distinct_model(), 16, MASK)
    assert trace_v.completed and trace_c.completed
    assert ids_v.tolist() == ids_c.tolist()
    assert trace_v.forward_passes == trace_c.forward_passes == 16
    assert trace_v.commits == trace_c.commits == 16
    assert trace_c.final_no_repeat and trace_v.final_no_repeat


# ---------------------------------------------------------------------------
# F3: prefix finality on the adversarial right-first stub
# ---------------------------------------------------------------------------

def test_f3_prefix_finality_right_first():
    ids_v, trace_v, _ = cvou_sample(
        right_first_model(), 8, MASK, policy="vanilla")
    ids_c, trace_c, _ = cvou_sample(right_first_model(), 8, MASK)
    assert trace_v.completed and trace_c.completed
    assert ids_v.tolist() == ids_c.tolist()

    # Vanilla commits right-first: its certified prefix is 0 until
    # the final round, when position 0 commits.
    v_positions = [e["pos"] for e in trace_v.commit_log]
    assert v_positions == [7, 6, 5, 4, 3, 2, 1, 0]
    v_prefixes = [e["prefix_after"] for e in trace_v.commit_log]
    assert v_prefixes == [0, 0, 0, 0, 0, 0, 0, 8]
    assert trace_v.prefix_area == 8

    # CVOU commits left-to-right: its prefix grows every round.
    c_positions = [e["pos"] for e in trace_c.commit_log]
    assert c_positions == list(range(8))
    c_prefixes = [e["prefix_after"] for e in trace_c.commit_log]
    assert c_prefixes == list(range(1, 9))
    assert trace_c.prefix_area == 36
    assert trace_c.prefix_area > trace_v.prefix_area

    # Each window closes exactly once, in order, under CVOU.
    all_closed = [w for e in trace_c.commit_log for w in e["windows_closed"]]
    assert all_closed == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# F4: segment interleave on the prompted stub
# ---------------------------------------------------------------------------

def test_f4_two_segment_interleave():
    frozen = {p: tok for p, tok in
              zip(range(0, 4), [10, 11, 12, 13])}
    frozen.update({p: tok for p, tok in
                   zip(range(8, 12), [20, 21, 22, 23])})
    ids, trace, cert = cvou_sample(
        two_segment_model(), 16, MASK, frozen=frozen)
    assert trace.completed
    assert trace.final_no_repeat

    # Every commit was a segment head at its round.
    for entry in trace.commit_log:
        assert entry["pos"] in entry["admissible"]
    # Within each segment, commits are left-to-right.
    seg_a = [e["pos"] for e in trace.commit_log if e["pos"] < 8]
    seg_b = [e["pos"] for e in trace.commit_log if e["pos"] >= 8]
    assert seg_a == [4, 5, 6, 7]
    assert seg_b == [12, 13, 14, 15]
    # The higher-ranked second-segment head goes first overall.
    assert trace.commit_log[0]["pos"] == 12
    # Certificate verifies.
    assert cvou_verify_certificate(ids, cert, MASK)["valid"]


# ---------------------------------------------------------------------------
# F5: certificate replay
# ---------------------------------------------------------------------------

def test_f5_certificate_replay_and_tamper():
    ids, trace, cert = cvou_sample(distinct_model(), 16, MASK)
    assert trace.completed
    checks = cvou_verify_certificate(ids, cert, MASK)
    assert checks["valid"]
    assert checks["commits_admissible"]
    assert checks["chain_exact"]
    assert checks["accounting_consistent"]
    assert checks["completed_sound"]

    # Tamper with the final canvas: chain must break.
    bad_ids = ids.clone()
    bad_ids[0] = 999
    assert not cvou_verify_certificate(bad_ids, cert, MASK)["valid"]

    # Tamper with a round's chosen position: it was not a segment
    # head on its pre-round canvas.
    bad_cert = copy.deepcopy(cert)
    bad_cert["commit_log"][0]["pos"] = 5
    assert not cvou_verify_certificate(ids, bad_cert, MASK)["valid"]

    # Tamper with a round's prefix accounting.
    bad_cert2 = copy.deepcopy(cert)
    bad_cert2["commit_log"][0]["prefix_after"] = 99
    assert not cvou_verify_certificate(ids, bad_cert2, MASK)["valid"]


# ---------------------------------------------------------------------------
# F6: fail-closed
# ---------------------------------------------------------------------------

def test_f6_stall_when_no_safe_candidate():
    ids, trace, cert = cvou_sample(FlatModel(8, 512), 8, MASK)
    assert not trace.completed
    assert trace.stalled
    assert MASK in ids.tolist()
    assert trace.forward_passes == 1  # one pass, then stall


def test_f6_preexisting_repeat_fails_at_round_zero():
    frozen = {p: tok for p, tok in
              zip(range(8), [10, 11, 12, 13, 10, 11, 12, 13])}
    ids, trace, cert = cvou_sample(
        distinct_model(8), 8, MASK, frozen=frozen)
    assert not trace.completed
    assert trace.preexisting_repeat
    assert trace.forward_passes == 0
    assert trace.stalled


# ---------------------------------------------------------------------------
# F7: determinism
# ---------------------------------------------------------------------------

def test_f7_determinism():
    ids1, trace1, cert1 = cvou_sample(distinct_model(), 16, MASK)
    ids2, trace2, cert2 = cvou_sample(distinct_model(), 16, MASK)
    assert ids1.tolist() == ids2.tolist()
    assert cert1 == cert2


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        cvou_sample(distinct_model(), 8, MASK, policy="bogus")


def test_policies_listed():
    assert set(POLICIES) == {"vanilla", "cvou"}
