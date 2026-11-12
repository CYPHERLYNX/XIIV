"""Tests for xiiv/cvel.py -- Certificate-Verified Expansion Length.

Falsifiers (pre-registered in the module docstring):
- F1 no-expansion equivalence: on a satisfiable-from-start contract,
  cvel produces the same canvas as the vanilla policy and records
  zero expansions.
- F2 expansion rescues the short canvas: on the span stub (required
  4-token span, longest masked run 3), the vanilla policy completes
  but violates the contract, while cvel expands by exactly the
  deficit (k=1, at the end of the longest masked run) and the final
  canvas contains the span.
- F3 every expansion justified: recomputing the oracle on the
  pre-expansion canvas gives unsatisfiable, and on the expanded
  canvas gives satisfiable, for every expansion in the certificate.
- F4 bound: a contract unsatisfiable within max_length ends with
  completed=False, length_failure=True, and final length ==
  max_length; no silent truncation.
- F5 final soundness: completed=True implies the independent
  verify_canvas(final) is clean AND the oracle finds the final canvas
  satisfiable; a tampered canvas fails cvel_verify_certificate.
- F6 determinism: identical runs -> identical canvas and certificate.
"""

import torch

from xiiv.cvel import (
    CVELConfig, cvel_sample, cvel_certificate, cvel_verify_certificate,
    length_oracle, POLICIES,
)
from xiiv.scv import verify_canvas

MASK = 0
SPAN = [40, 41, 42, 43]


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


class SpanModel:
    """Canvas-aware stub for the F2 contract.

    Frozen prompt tokens sit at positions 0, 4, 5 (10, 99, 98), so the
    initial masked run is positions 1-3 (length 3) -- one short of the
    4-token required span. The span is anchored at absolute positions
    1-4: a masked position p proposes token 39 + p (40, 41, 42, 43),
    so the proposal is stable across rounds -- a run-index mapping
    would re-anchor after every commit and never place the span. After
    CVEL expands the run to length 4, positions 1-4 fill with the
    span; before expansion only positions 1-3 exist, so the vanilla
    run can place at most 40, 41, 42.
    """

    def __init__(self, vocab=128):
        self.vocab = vocab

    def __call__(self, x):
        canvas = x[0].tolist()
        length = len(canvas)
        logits = torch.full((1, length, self.vocab), -1e9)
        for p, tok in enumerate(canvas):
            if tok != MASK:
                continue
            span_tok = 39 + p  # positions 1..4 -> 40..43
            if 40 <= span_tok <= 43:
                logits[0, p, span_tok] = 0.0
                logits[0, p, span_tok + 50] = -3.0
            else:
                logits[0, p, 60 + p] = 0.0
                logits[0, p, 110 + p] = -3.0
        return logits


def _contains(canvas, span):
    n = len(span)
    return any(list(canvas[i:i + n]) == span
               for i in range(len(canvas) - n + 1))


def _f2_setup():
    frozen = {0: 10, 4: 99, 5: 98}
    contract = {"required_spans": [SPAN], "min_length": 6}
    cfg = CVELConfig()
    return SpanModel(), frozen, contract, cfg


def test_f1_no_expansion_equivalence():
    table = {p: [(10 + p, 0.0), (100 + p, -3.0)] for p in range(6)}
    model = ScriptedModel(6, 128, table)
    contract = {"required_spans": [[11, 12]], "min_length": 6}
    cfg = CVELConfig()
    ids_cvel, trace_cvel, _ = cvel_sample(
        model, 6, MASK, config=cfg, policy="cvel", contract=contract)
    ids_vanilla, trace_vanilla, _ = cvel_sample(
        model, 6, MASK, config=CVELConfig(), policy="vanilla",
        contract=contract)
    assert trace_cvel.completed
    assert trace_cvel.expansions == 0
    assert trace_cvel.expansions_log == []
    assert ids_cvel.tolist() == ids_vanilla.tolist()
    assert ids_cvel.tolist() == [10, 11, 12, 13, 14, 15]


def test_f2_expansion_rescues_short_canvas():
    model, frozen, contract, cfg = _f2_setup()
    ids_v, trace_v, _ = cvel_sample(
        model, 6, MASK, config=CVELConfig(), policy="vanilla",
        frozen=frozen, contract=contract)
    # Vanilla completes (no masks) but violates the contract: the
    # 4-token span never fit in the 3-long masked run.
    assert trace_v.completed
    assert not _contains(ids_v.tolist(), SPAN)
    assert not length_oracle(ids_v.tolist(), MASK, contract)["satisfiable"]

    ids, trace, cert = cvel_sample(
        model, 6, MASK, config=cfg, policy="cvel",
        frozen=frozen, contract=contract)
    assert trace.completed
    assert trace.expansions == 1
    assert len(trace.expansions_log) == 1
    entry = trace.expansions_log[0]
    assert entry["k"] == 1
    assert entry["position"] == 4  # end of the longest masked run
    assert entry["min_extra"] == 1
    assert entry["satisfiable_before"] is False
    assert entry["satisfiable_after"] is True
    assert cert["initial_length"] == 6
    assert cert["final_length"] == 7
    final = ids.tolist()
    assert _contains(final, SPAN)
    # Committed prompt tokens keep their relative order.
    assert final[0] == 10 and final[-2:] == [99, 98]
    assert trace.contract_satisfied is True
    assert cvel_verify_certificate(ids, cert, MASK, contract)["valid"]


def test_f3_every_expansion_justified():
    model, frozen, contract, cfg = _f2_setup()
    _, _, cert = cvel_sample(
        model, 6, MASK, config=cfg, policy="cvel",
        frozen=frozen, contract=contract)
    assert len(cert["expansions"]) == 1
    for entry in cert["expansions"]:
        before = entry["canvas_before"]
        assert not length_oracle(before, MASK, contract)["satisfiable"]
        expanded = (before[:entry["position"]]
                    + [MASK] * entry["k"]
                    + before[entry["position"]:])
        assert length_oracle(expanded, MASK, contract)["satisfiable"]
        assert entry["satisfiable_before"] is False
        assert entry["satisfiable_after"] is True


def test_f4_bound_fails_closed():
    model, frozen, _, _ = _f2_setup()
    contract = {"required_spans": [[70] * 20], "min_length": 6}
    cfg = CVELConfig(max_length=8)
    ids, trace, cert = cvel_sample(
        model, 6, MASK, config=cfg, policy="cvel",
        frozen=frozen, contract=contract)
    assert not trace.completed
    assert trace.length_failure is True
    assert trace.final_length == 8
    assert cert["length_failure"] is True
    assert MASK in ids.tolist()  # masks remain; no silent truncation


def test_f5_final_soundness_and_tamper():
    model, frozen, contract, cfg = _f2_setup()
    ids, trace, cert = cvel_sample(
        model, 6, MASK, config=cfg, policy="cvel",
        frozen=frozen, contract=contract)
    assert trace.completed
    final = ids.tolist()
    assert verify_canvas(final, MASK, 4)["final_no_repeat"]
    assert length_oracle(final, MASK, contract)["satisfiable"]
    assert cvel_verify_certificate(ids, cert, MASK, contract)["valid"]
    # Tamper: break the span; the certificate must fail.
    tampered = list(final)
    tampered[final.index(43)] = 99
    assert not cvel_verify_certificate(
        tampered, cert, MASK, contract)["valid"]


def test_f6_determinism():
    model, frozen, contract, _ = _f2_setup()
    ids1, _, cert1 = cvel_sample(
        model, 6, MASK, config=CVELConfig(), policy="cvel",
        frozen=frozen, contract=contract)
    ids2, _, cert2 = cvel_sample(
        model, 6, MASK, config=CVELConfig(), policy="cvel",
        frozen=frozen, contract=contract)
    assert ids1.tolist() == ids2.tolist()
    assert cert1 == cert2


def test_policies_and_oracle_edges():
    assert set(POLICIES) == {"vanilla", "cvel"}
    # Empty / missing contract is trivially satisfiable.
    assert length_oracle([MASK, 5], MASK, None)["satisfiable"]
    assert length_oracle([MASK, 5], MASK, {})["satisfiable"]
    # min_length deficit inserts at the end of the canvas.
    oracle = length_oracle([1, 2, MASK], MASK, {"min_length": 5})
    assert not oracle["satisfiable"]
    assert oracle["min_extra"] == 2
    assert oracle["insert_pos"] == 3
    # A span already present is satisfiable with no expansion.
    oracle = length_oracle([10, 40, 41, 5], MASK,
                           {"required_spans": [[40, 41]]})
    assert oracle["satisfiable"] and oracle["min_extra"] == 0
