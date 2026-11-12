"""Tests for xiiv/cvab.py -- Certificate-Verified Adaptive Batching.

Falsifiers (pre-registered in the module docstring):
- F1 width-1 equivalence: policy="vanilla" and policy="cvab" with
  max_width=1 produce the same canvas on the same model.
- F2 batch speedup, same canvas: on a 16-mask stub with distinct
  position-stable proposals, cvab completes in 4 forward passes
  where vanilla needs 16, with an identical final canvas.
- F3 cross-commit safety: on the adjacent-conflict stub, the four
  top-ranked positions would jointly complete a repeated 4-gram; a
  naive batch violates the certificate, while cvab shrinks to width
  1 per round, records the shrinks, and finishes clean.
- F4 certificate replay: a real run verifies; tampering with the
  final canvas or with any round's batch fails verification.
- F5 fail-closed: a model offering no candidate above the confidence
  bar stalls with completed=False and masks remaining; a frozen
  prompt that already repeats fails closed at round zero with zero
  forward passes and preexisting_repeat=True.
- F6 determinism: identical runs -> identical canvas and certificate.
"""

import copy

import pytest
import torch

from xiiv.cvab import (
    CVABConfig, cvab_sample, cvab_certificate, cvab_verify_certificate,
    interaction_radius, min_separation, POLICIES,
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
    stable across rounds, so vanilla and cvab must agree on the
    final canvas."""
    table = {p: [(100 + p, 0.0)] for p in range(length)}
    return TableModel(length, vocab, table)


def conflict_model(vocab=512):
    """Adjacent-conflict stub (n=4, length 8).

    Frozen: positions 0-3 hold [10, 11, 12, 13] -- a committed
    4-gram. Masked: 4, 5, 6, 7 propose [10, 11, 12, 13]
    respectively. Each proposal is individually gate-safe, but
    committing all four jointly would repeat the 4-gram at 4-7.
    Position 7 also offers 99 as a second candidate (prob ~0.38),
    which the gate accepts once 13 would repeat.
    """
    table = {
        4: [(10, 0.0)],
        5: [(11, 0.0)],
        6: [(12, 0.0)],
        7: [(13, 0.0), (99, -0.5)],
    }
    return TableModel(8, vocab, table)


def test_width_one_equivalence():
    model = distinct_model()
    ids_v, trace_v, _ = cvab_sample(
        model, 16, MASK, config=CVABConfig(), policy="vanilla")
    ids_c, trace_c, _ = cvab_sample(
        model, 16, MASK, config=CVABConfig(max_width=1), policy="cvab")
    assert ids_v.tolist() == ids_c.tolist()
    assert trace_v.forward_passes == trace_c.forward_passes == 16
    assert all(len(e["batch"]) == 1 for e in trace_c.batch_log)


def test_batch_speedup_same_canvas():
    model = distinct_model()
    ids_v, trace_v, _ = cvab_sample(
        model, 16, MASK, config=CVABConfig(), policy="vanilla")
    ids_c, trace_c, cert_c = cvab_sample(
        model, 16, MASK, config=CVABConfig(max_width=4), policy="cvab")
    assert ids_c.tolist() == ids_v.tolist()
    assert ids_c.tolist() == [100 + p for p in range(16)]
    assert trace_c.forward_passes == 4
    assert trace_v.forward_passes == 16
    assert trace_c.commits == 16
    assert trace_c.batches == 4
    assert trace_c.completed
    assert trace_c.final_no_repeat
    assert trace_c.verify_alarms == 0
    assert all(e["admitted_width"] == 4 for e in trace_c.batch_log)
    assert all(e["min_separation"] == 4 for e in trace_c.batch_log)
    checks = cvab_verify_certificate(ids_c, cert_c, MASK)
    assert checks["valid"]


def test_cross_commit_conflict_shrinks():
    model = conflict_model()
    frozen = {0: 10, 1: 11, 2: 12, 3: 13}
    # A naive batch of all four would repeat the frozen 4-gram.
    naive = [10, 11, 12, 13, 10, 11, 12, 13]
    assert not verify_canvas(naive, MASK, 4)["final_no_repeat"]
    ids, trace, cert = cvab_sample(
        model, 8, MASK, config=CVABConfig(max_width=4), policy="cvab",
        frozen=frozen)
    assert trace.completed
    assert trace.final_no_repeat
    assert ids.tolist() == [10, 11, 12, 13, 10, 11, 12, 99]
    # Every round shrank: the four adjacent positions can never go
    # together, so each round admitted exactly one.
    assert trace.batches == 4
    assert all(e["admitted_width"] == 1 for e in trace.batch_log)
    assert trace.shrinks == 3 + 2 + 1 + 0
    assert trace.forward_passes == 4
    assert trace.verify_alarms == 0
    checks = cvab_verify_certificate(ids, cert, MASK)
    assert checks["valid"]


def test_certificate_valid_and_tamper_fails():
    model = distinct_model()
    ids, _, cert = cvab_sample(
        model, 16, MASK, config=CVABConfig(max_width=4), policy="cvab")
    assert cvab_verify_certificate(ids, cert, MASK)["valid"]
    # Tamper with the final canvas: the chain no longer replays.
    tampered = ids.clone()
    tampered[0] = 200
    assert not cvab_verify_certificate(tampered, cert, MASK)["valid"]
    # Tamper with a round's batch: admissibility or the chain fails.
    bad = copy.deepcopy(cert)
    bad["batch_log"][0]["batch"][1]["pos"] = 1  # 1 apart from pos 0
    assert not cvab_verify_certificate(ids, bad, MASK)["valid"]
    bad2 = copy.deepcopy(cert)
    bad2["batch_log"][0]["batch"][0]["tok"] = 201
    assert not cvab_verify_certificate(ids, bad2, MASK)["valid"]


def test_stall_fails_closed():
    model = FlatModel(8, 128)
    ids, trace, cert = cvab_sample(
        model, 8, MASK, config=CVABConfig(), policy="cvab")
    assert not trace.completed
    assert trace.stalled
    assert MASK in ids.tolist()
    assert trace.forward_passes == 1
    assert trace.rounds == 0
    assert trace.commits == 0
    # A correctly failed-closed run still has a valid certificate of
    # what happened (no batches were ever admitted).
    assert cvab_verify_certificate(ids, cert, MASK)["valid"]


def test_preexisting_repeat_fails_closed():
    model = distinct_model(length=8)
    frozen = {0: 10, 1: 11, 2: 12, 3: 13, 4: 10, 5: 11, 6: 12, 7: 13}
    ids, trace, _ = cvab_sample(
        model, 8, MASK, config=CVABConfig(), policy="cvab",
        frozen=frozen)
    assert not trace.completed
    assert trace.preexisting_repeat
    assert trace.forward_passes == 0
    assert trace.commits == 0


def test_determinism():
    model = distinct_model()
    ids1, _, cert1 = cvab_sample(
        model, 16, MASK, config=CVABConfig(max_width=4), policy="cvab")
    ids2, _, cert2 = cvab_sample(
        model, 16, MASK, config=CVABConfig(max_width=4), policy="cvab")
    assert ids1.tolist() == ids2.tolist()
    assert cert1 == cert2


def test_policies_and_geometry_helpers():
    assert set(POLICIES) == {"vanilla", "cvab"}
    assert interaction_radius(4) == 3
    assert min_separation(4) == 4
    assert min_separation(1) == 1
    model = distinct_model()
    with pytest.raises(ValueError):
        cvab_sample(model, 16, MASK, policy="nope")
    with pytest.raises(ValueError):
        cvab_sample(model, 0, MASK)
