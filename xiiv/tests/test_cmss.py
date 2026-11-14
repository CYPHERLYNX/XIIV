"""Tests for xiiv/cmss.py -- Certificate Mutual-Stability Sampling.

Falsifiers (pre-registered in the module docstring):
- F1 stable stub: cmss completes in 2 forward passes (round 0
  observes, round 1 commits all 16), every commit persistence >= 2
  and jsd == 0.
- F2 cost vs vanilla: vanilla needs 16 forward passes on the same
  stub; cmss needs 2; final canvases identical.
- F3 flip-flop: a model whose top-1 alternates every round never
  stabilizes -- exhausts max_rounds, completed=False,
  patience_rounds == max_rounds.
- F4 reshuffle: a model whose top-1 is constant but whose runner-up
  mass swaps every round has inter-step JSD ~0.052 > 0.02 -- never
  stabilizes, exhausts, completed=False.
- F5 low confidence: a model whose top-1 prob is ~1/3 < 0.35 never
  stabilizes on the confidence leg -- exhausts, completed=False.
- F6 certificate replay: a real run verifies; tampering with a
  commit's jsd, or with any round's chosen token, fails verification.
- F7 fail-closed: a model whose stable top-1 would create a repeat
  stalls with completed=False and masks remaining; a frozen prompt
  that already repeats fails closed at round zero with zero forward
  passes and preexisting_repeat=True.
- F8 determinism: identical runs -> identical canvas and certificate.
"""

import copy
import math

import pytest
import torch

from xiiv.cmss import (
    CMSSConfig, cmss_sample, cmss_certificate, cmss_verify_certificate,
    jsd_topk, POLICIES,
)

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
    realistic confidence bar -- no stable position anywhere."""

    def __init__(self, length, vocab):
        self.length = length
        self.vocab = vocab

    def __call__(self, x):
        return torch.zeros((1, self.length, self.vocab))


def distinct_model(length=16, vocab=512):
    """Every position proposes its own distinct token (100 + pos),
    stable across rounds."""
    table = {p: [(100 + p, 0.0)] for p in range(length)}
    return TableModel(length, vocab, table)


class FlipFlopModel:
    """Top-1 alternates between two tokens every call: the prediction
    never persists, so nothing can stabilize."""

    def __init__(self, length=8, vocab=512):
        self.length = length
        self.vocab = vocab
        self.calls = 0

    def __call__(self, x):
        logits = torch.full((1, self.length, self.vocab), -1e9)
        for p in range(self.length):
            tok = (100 + p) if self.calls % 2 == 0 else (200 + p)
            logits[0, p, tok] = 5.0
        self.calls += 1
        return logits


class ReshuffleModel:
    """Top-1 constant, but the runner-up mass swaps every call:
    {0.6, 0.3, 0.1} <-> {0.6, 0.1, 0.3}. Inter-step JSD ~0.0523,
    above the 0.02 threshold, so nothing stabilizes even though the
    argmax never moves."""

    def __init__(self, length=8, vocab=512):
        self.length = length
        self.vocab = vocab
        self.calls = 0

    def __call__(self, x):
        logits = torch.full((1, self.length, self.vocab), -1e9)
        ln06, ln03, ln01 = (math.log(0.6), math.log(0.3), math.log(0.1))
        for p in range(self.length):
            if self.calls % 2 == 0:
                cands = [(100 + p, ln06), (300 + p, ln03), (400 + p, ln01)]
            else:
                cands = [(100 + p, ln06), (300 + p, ln01), (400 + p, ln03)]
            for tok, lg in cands:
                logits[0, p, tok] = lg
        self.calls += 1
        return logits


def low_conf_model(length=8, vocab=512):
    """Every position offers three tokens at equal logit: top-1 prob
    is 1/3 < 0.35, so the confidence leg never passes."""
    table = {p: [(100 + p, 0.0), (200 + p, 0.0), (300 + p, 0.0)]
             for p in range(length)}
    return TableModel(length, vocab, table)


# ---------------------------------------------------------------------------
# JSD unit checks
# ---------------------------------------------------------------------------

def test_jsd_topk_identical_is_zero():
    d = [(10, 0.6), (20, 0.3), (30, 0.1)]
    assert jsd_topk(d, d) == 0.0


def test_jsd_topk_reshuffle_value():
    a = [(10, 0.6), (20, 0.3), (30, 0.1)]
    b = [(10, 0.6), (20, 0.1), (30, 0.3)]
    # M = (0.6, 0.2, 0.2); KL = 0.3*ln(1.5) + 0.1*ln(0.5), symmetric.
    expected = 0.3 * math.log(1.5) + 0.1 * math.log(0.5)
    assert jsd_topk(a, b) == pytest.approx(expected, abs=1e-9)
    assert jsd_topk(a, b) > 0.02  # above the CMSS threshold


def test_jsd_topk_disjoint_is_ln2():
    assert jsd_topk([(1, 1.0)], [(2, 1.0)]) == pytest.approx(
        math.log(2), abs=1e-9)


# ---------------------------------------------------------------------------
# F1: stable stub completes in 2 forward passes
# ---------------------------------------------------------------------------

def test_f1_stable_stub_two_passes():
    ids, trace, cert = cmss_sample(distinct_model(), 16, MASK)
    assert trace.completed
    assert trace.final_no_repeat
    assert trace.forward_passes == 2
    assert trace.rounds == 1
    assert trace.commits == 16
    assert trace.patience_rounds == 1
    assert trace.batch_shrinks == 0
    # Every commit carries a stability triple: persistence >= 2,
    # jsd == 0 on the identical-distribution stub.
    assert len(trace.commit_log) == 2
    assert trace.commit_log[0]["patience"] is True
    assert trace.commit_log[0]["commits"] == []
    commits = trace.commit_log[1]["commits"]
    assert len(commits) == 16
    assert sorted(c["pos"] for c in commits) == list(range(16))
    for c in commits:
        assert c["persistence"] >= 2
        assert c["jsd"] == 0.0
        assert c["conf"] == pytest.approx(1.0, abs=1e-6)
    assert ids.tolist() == [100 + p for p in range(16)]


# ---------------------------------------------------------------------------
# F2: cost vs vanilla on the same stub
# ---------------------------------------------------------------------------

def test_f2_vanilla_needs_sixteen_passes():
    ids_v, trace_v, _ = cmss_sample(
        distinct_model(), 16, MASK, policy="vanilla")
    ids_c, trace_c, _ = cmss_sample(distinct_model(), 16, MASK)
    assert trace_v.completed and trace_c.completed
    assert ids_v.tolist() == ids_c.tolist()
    assert trace_v.forward_passes == 16
    assert trace_c.forward_passes == 2
    assert trace_v.commits == trace_c.commits == 16


# ---------------------------------------------------------------------------
# F3: flip-flop never stabilizes
# ---------------------------------------------------------------------------

def test_f3_flip_flop_exhausts():
    cfg = CMSSConfig(max_rounds=6)
    ids, trace, cert = cmss_sample(FlipFlopModel(), 8, MASK, config=cfg)
    assert not trace.completed
    assert trace.exhausted
    assert not trace.stalled
    assert MASK in ids.tolist()
    assert trace.forward_passes == 6
    assert trace.patience_rounds == 6
    assert trace.commits == 0


# ---------------------------------------------------------------------------
# F4: reshuffle never stabilizes (JSD leg)
# ---------------------------------------------------------------------------

def test_f4_reshuffle_exhausts():
    cfg = CMSSConfig(max_rounds=6)
    ids, trace, cert = cmss_sample(ReshuffleModel(), 8, MASK, config=cfg)
    assert not trace.completed
    assert trace.exhausted
    assert trace.forward_passes == 6
    assert trace.patience_rounds == 6
    assert trace.commits == 0


# ---------------------------------------------------------------------------
# F5: low confidence never stabilizes (confidence leg)
# ---------------------------------------------------------------------------

def test_f5_low_confidence_exhausts():
    cfg = CMSSConfig(max_rounds=6)
    ids, trace, cert = cmss_sample(low_conf_model(), 8, MASK, config=cfg)
    assert not trace.completed
    assert trace.exhausted
    assert trace.forward_passes == 6
    assert trace.commits == 0


# ---------------------------------------------------------------------------
# F6: certificate replay and tamper
# ---------------------------------------------------------------------------

def test_f6_certificate_replay_and_tamper():
    ids, trace, cert = cmss_sample(distinct_model(), 16, MASK)
    assert trace.completed
    checks = cmss_verify_certificate(ids, cert, MASK)
    assert checks["valid"]
    assert checks["commits_stable"]
    assert checks["commits_admissible"]
    assert checks["chain_exact"]
    assert checks["completed_sound"]

    # Tamper with the final canvas: chain must break.
    bad_ids = ids.clone()
    bad_ids[0] = 999
    assert not cmss_verify_certificate(bad_ids, cert, MASK)["valid"]

    # Tamper with a commit's recorded jsd: recomputation disagrees.
    bad_cert = copy.deepcopy(cert)
    bad_cert["commit_log"][1]["commits"][0]["jsd"] = 0.5
    assert not cmss_verify_certificate(ids, bad_cert, MASK)["valid"]

    # Tamper with a commit's persistence: recomputation disagrees.
    bad_cert2 = copy.deepcopy(cert)
    bad_cert2["commit_log"][1]["commits"][0]["persistence"] = 99
    assert not cmss_verify_certificate(ids, bad_cert2, MASK)["valid"]

    # Tamper with a commit's token: it no longer equals the recorded
    # top-1 observation.
    bad_cert3 = copy.deepcopy(cert)
    bad_cert3["commit_log"][1]["commits"][0]["tok"] = 777
    assert not cmss_verify_certificate(ids, bad_cert3, MASK)["valid"]


# ---------------------------------------------------------------------------
# F7: fail-closed
# ---------------------------------------------------------------------------

def test_f7_gate_stall_when_stable_top1_would_repeat():
    # Frozen [10,11,12,13,10,11,12] + mask at 7; the model stably
    # proposes 13 at pos 7, which would duplicate the [10,11,12,13]
    # window at [0..3]. The gate must refuse it -- stall, not commit.
    frozen = {p: tok for p, tok in
              zip(range(7), [10, 11, 12, 13, 10, 11, 12])}
    table = {7: [(13, 5.0)]}
    ids, trace, cert = cmss_sample(
        TableModel(8, 512, table), 8, MASK, frozen=frozen)
    assert not trace.completed
    assert trace.stalled
    assert not trace.exhausted
    assert MASK in ids.tolist()
    assert trace.forward_passes == 2  # observe, then stable+rejected
    assert trace.gate_rejects == 1
    assert trace.commits == 0


def test_f7_preexisting_repeat_fails_at_round_zero():
    frozen = {p: tok for p, tok in
              zip(range(8), [10, 11, 12, 13, 10, 11, 12, 13])}
    ids, trace, cert = cmss_sample(
        distinct_model(8), 8, MASK, frozen=frozen)
    assert not trace.completed
    assert trace.preexisting_repeat
    assert trace.forward_passes == 0
    assert trace.stalled


# ---------------------------------------------------------------------------
# F8: determinism
# ---------------------------------------------------------------------------

def test_f8_determinism():
    ids1, trace1, cert1 = cmss_sample(distinct_model(), 16, MASK)
    ids2, trace2, cert2 = cmss_sample(distinct_model(), 16, MASK)
    assert ids1.tolist() == ids2.tolist()
    assert cert1 == cert2


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        cmss_sample(distinct_model(), 8, MASK, policy="bogus")


def test_policies_listed():
    assert set(POLICIES) == {"vanilla", "cmss"}
