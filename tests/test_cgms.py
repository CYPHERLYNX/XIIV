"""Tests for CGMS (Certificate-Gated Mutual-Stability unmasking) and
the partial-certificate gate — XIIV port of LESS-2026's 3-gate stability
sampler plus XIIV's fourth gate (partial-canvas certifiability).

Design: research/daily_research_2026-10-20.md. The partial check must
be SOUND in the fail direction (never reject a completable canvas);
the sampler must preserve every structural guarantee of the standard
path (exact length, completion, required/forbidden/at_most), and its
final step always runs the standard path so completion is inherited.
"""
import math

import torch

from xiiv.config import ModelConfig
from xiiv.contracts import Contract
from xiiv.diffusion import (DenoisingState, _js_divergence_topk,
                            check_constraints, partial_certificate_check)
from xiiv.model import XIIVModel

M = 99  # mask id for the pure partial-check cases


# --- partial_certificate_check: unit cases ----------------------------

def test_partial_clean_canvas_certifiable():
    ok, _ = partial_certificate_check([5, M, M], {"required_counts": {7: 1}}, M)
    assert ok


def test_partial_forbidden_filled_fails():
    ok, reason = partial_certificate_check([5, 3, M], {"forbidden_ids": [3]}, M)
    assert not ok and reason.startswith("forbidden_present")


def test_partial_cap_exceeded_fails():
    ok, reason = partial_certificate_check([4, 4, M], {"at_most": {4: 1}}, M)
    assert not ok and reason.startswith("cap_exceeded")


def test_partial_required_infeasible_fails():
    ok, reason = partial_certificate_check(
        [5, M], {"required_counts": {7: 2}}, M)
    assert not ok and reason.startswith("required_infeasible")


def test_partial_anchor_conflict_fails():
    ok, reason = partial_certificate_check([5, M], {"anchors": {0: 6}}, M)
    assert not ok and reason.startswith("anchor_conflict")


def test_partial_forbidden_phrase_completed_fails():
    ok, reason = partial_certificate_check(
        [1, 2, M], {"forbidden_phrases": [([1, 2], 0)]}, M)
    assert not ok and reason == "forbidden_phrase_completed"


def test_partial_phrase_spanning_mask_passes():
    # A forbidden phrase split by a mask is not yet a violation; the
    # sampler's preventive suppression owns that risk.
    ok, _ = partial_certificate_check(
        [1, M, 2], {"forbidden_phrases": [([1, 2], 0)]}, M)
    assert ok


def test_partial_order_violated_fails():
    ok, reason = partial_certificate_check(
        [8, 9, 1, 2], {"order_pairs": [([1, 2], [8, 9])]}, M)
    assert not ok and reason == "order_violated"


def test_partial_order_infeasible_fails():
    # b already occupies the only room before it; a can never precede.
    ok, reason = partial_certificate_check(
        [8, 9, M], {"order_pairs": [([1, 2], [8, 9])]}, M)
    assert not ok and reason.startswith("order_infeasible")


def test_partial_order_feasible_passes():
    ok, _ = partial_certificate_check(
        [1, 2, M, M], {"order_pairs": [([1, 2], [8, 9])]}, M)
    assert ok


def test_partial_frozen_masks_are_not_slots():
    frozen = torch.tensor([False, True])  # position 1 frozen as mask
    ok, reason = partial_certificate_check(
        [5, M], {"required_counts": {7: 1}}, M, frozen=frozen)
    assert not ok and reason.startswith("required_infeasible")


def test_contract_wrapper_matches():
    ok, _ = Contract.partial_certificate_check([5, 3, M],
                                               {"forbidden_ids": [3]}, M)
    assert not ok


# --- JS divergence helper ----------------------------------------------

def test_js_identical_is_zero():
    ids = [1, 2, 3]
    p = [0.5, 0.3, 0.2]
    assert abs(_js_divergence_topk(ids, p, ids, p)) < 1e-9


def test_js_disjoint_is_ln2():
    js = _js_divergence_topk([1], [1.0], [2], [1.0])
    assert abs(js - math.log(2)) < 1e-9


# --- sampler integration -------------------------------------------------

def tiny_cfg(**kw):
    d = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
             ffn_dim=128, mask_id=254, eos_id=255, seq_len=32)
    d.update(kw)
    return ModelConfig(**d)


def _tiny_model(seed=0):
    cfg = tiny_cfg()
    torch.manual_seed(seed)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    return m, cfg


CONSTRAINTS = {"required_ids": [10, 20, 30], "forbidden_ids": [200],
               "at_most": {50: 2}}


def _run_cgms(seed, **kw):
    m, cfg = _tiny_model(42)
    st = DenoisingState(
        length=24, mask_id=cfg.mask_id, eos_id=cfg.eos_id,
        steps=8, seed=seed, vocab_size=cfg.vocab_size,
        constraints=dict(CONSTRAINTS), cgms=True, **kw)
    while st.step(m):
        pass
    return st, cfg


def test_cgms_guarantees_hold():
    for seed in range(6):
        st, cfg = _run_cgms(seed)
        out = st.x.cpu()
        assert len(out) == 24
        assert not (out == cfg.mask_id).any(), f"seed {seed}: unfinished"
        rep = check_constraints(out, CONSTRAINTS, cfg.mask_id)
        assert all(rep.values()), f"seed {seed}: {rep}"


def test_cgms_stats_recorded():
    st, _ = _run_cgms(0)
    stats = st.cgms_stats
    assert stats["steps"] > 0
    # the run must have exercised the CGMS machinery, one way or another
    assert stats["commits"] + stats["fallbacks"] + stats["defers"] > 0


def test_cgms_deterministic_given_seed():
    a, _ = _run_cgms(3)
    b, _ = _run_cgms(3)
    assert torch.equal(a.x.cpu(), b.x.cpu())


class _StubModel(torch.nn.Module):
    """Deterministic stub: every position's top-1 is `fav` with high
    confidence, stable across steps — the stable case CGMS exists for.
    (The untrained tiny model's top-1 is the mask token itself, which
    CGMS correctly refuses to commit.)"""

    def __init__(self, length=24, vocab=256, fav=7):
        super().__init__()
        logits = torch.full((length, vocab), -5.0)
        logits[:, fav] = 5.0
        self.register_buffer("logits", logits)

    def forward(self, x, **kw):
        return self.logits[:x.shape[1]].unsqueeze(0)


def test_cgms_state_snapshot_resume_does_not_crash():
    # Regression (caught by the full suite the day CGMS landed):
    # from_snapshot must initialize the CGMS runtime-policy fields to
    # safe defaults, or step() on a resumed state AttributeErrors.
    m, cfg = _tiny_model(42)
    st, _ = _run_cgms(3)
    snap = st.to_snapshot()
    st2 = DenoisingState.from_snapshot(snap)
    assert st2.cgms is False  # runtime policy: re-attach after resume
    assert st2.cgms_stats["commits"] == 0
    while st2.step(m):
        pass
    assert not (st2.x.cpu() == cfg.mask_id).any()


def test_cgms_commits_with_stub_model():
    cfg = tiny_cfg()
    st = DenoisingState(
        length=24, mask_id=cfg.mask_id, eos_id=cfg.eos_id,
        steps=8, seed=1, vocab_size=cfg.vocab_size,
        constraints=dict(CONSTRAINTS), cgms=True)
    model = _StubModel()
    while st.step(model):
        pass
    out = st.x.cpu()
    assert st.cgms_stats["commits"] > 0, st.cgms_stats
    assert not (out == cfg.mask_id).any()
    rep = check_constraints(out, CONSTRAINTS, cfg.mask_id)
    assert all(rep.values()), rep
