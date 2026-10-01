"""Fast unit tests for XIIV core (no training data needed)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel, build_rope_cache, apply_rope
from xiiv.diffusion import (mask_batch, diffusion_loss, sample_trajectory,
                              sample_trajectory_stateful, DenoisingState,
                              check_constraints)


def tiny_cfg(**kw):
    d = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
             ffn_dim=128, mask_id=254, eos_id=255, seq_len=32)
    d.update(kw)
    return ModelConfig(**d)


def test_param_count():
    cfg = tiny_cfg()
    m = XIIVModel(cfg)
    assert m.count_parameters() == cfg.param_count_estimate(), \
        (m.count_parameters(), cfg.param_count_estimate())
    print("PASS test_param_count", m.count_parameters())


def test_forward_shape_and_bidirectional():
    cfg = tiny_cfg()
    m = XIIVModel(cfg)
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    with torch.no_grad():
        out = m(ids)
    assert out.shape == (2, 32, cfg.vocab_size), out.shape
    # bidirectionality: changing a LATER token must change an EARLIER logit
    ids2 = ids.clone()
    ids2[0, 20] = (ids2[0, 20] + 1) % cfg.vocab_size
    with torch.no_grad():
        out2 = m(ids2)
    assert not torch.equal(out[0, 5], out2[0, 5]), \
        "model looks causal — later tokens don't affect earlier positions!"
    print("PASS test_forward_shape_and_bidirectional")


def test_rope_consistency():
    cos, sin = build_rope_cache(16, 16, 10000.0, torch.device("cpu"))
    x = torch.randn(1, 2, 16, 16)
    y = apply_rope(x, cos, sin)
    # rotation preserves norms
    assert torch.allclose(x.pow(2).sum(-1), y.pow(2).sum(-1), atol=1e-5)
    print("PASS test_rope_consistency")


def test_mask_batch_ratios():
    cfg = tiny_cfg()
    ids = torch.randint(0, cfg.vocab_size, (64, 32))
    g = torch.Generator().manual_seed(0)
    noisy, targets, mask = mask_batch(ids, cfg.mask_id, 0.15, 1.0, g)
    assert torch.equal(targets, ids)
    assert (noisy[mask] == cfg.mask_id).all()
    assert (noisy[~mask] == ids[~mask]).all()
    r = mask.float().mean().item()
    assert 0.4 < r < 0.7, r  # E[t] = 0.575
    print("PASS test_mask_batch_ratios", round(r, 3))


def test_loss_finite_and_decreases():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    m = XIIVModel(cfg)
    m.init_weights()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ids = torch.randint(0, cfg.vocab_size, (4, 32))
    g = torch.Generator().manual_seed(0)
    l0, _ = diffusion_loss(m, ids, cfg.mask_id, generator=g)
    for _ in range(30):
        opt.zero_grad()
        l, _ = diffusion_loss(m, ids, cfg.mask_id, generator=g)
        assert torch.isfinite(l), "non-finite loss"
        l.backward()
        opt.step()
    assert l.item() < l0.item(), (l0.item(), l.item())
    print("PASS test_loss_finite_and_decreases",
          round(l0.item(), 3), "->", round(l.item(), 3))


def test_sampler_deterministic_and_terminates():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    a, _ = sample_trajectory(m, 24, cfg.mask_id, cfg.eos_id,
                             steps=6, seed=42)
    b, _ = sample_trajectory(m, 24, cfg.mask_id, cfg.eos_id,
                             steps=6, seed=42)
    assert torch.equal(a, b), "sampler not deterministic given seed"
    assert not (a == cfg.mask_id).any(), "sampler left masks behind"
    print("PASS test_sampler_deterministic_and_terminates")


def test_infill_preserves_context():
    from xiiv.tokenizer import XIIVTokenizer  # noqa
    cfg = tiny_cfg()
    torch.manual_seed(0)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    # fake a prompt: ids with a masked span, frozen context
    ids = torch.randint(0, cfg.vocab_size - 2, (40,))
    frozen = torch.ones(40, dtype=torch.bool)
    ids[10:16] = cfg.mask_id
    frozen[10:16] = False
    out, _ = sample_trajectory(m, 40, cfg.mask_id, cfg.eos_id, steps=6,
                               seed=1, prompt_ids=ids, frozen=frozen)
    assert torch.equal(out[frozen], ids[frozen]), "context was modified!"
    assert not (out[10:16] == cfg.mask_id).any(), "span not filled"
    print("PASS test_infill_preserves_context")


# ---------------------------------------------------------------------------
# Constraint-sampler hardening tests (AUDIT_FIXES.md)
# ---------------------------------------------------------------------------

def _tiny_model(seed=0):
    cfg = tiny_cfg()
    torch.manual_seed(seed)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    return m, cfg


def test_required_token_guarantee_50_seeds():
    # adversarial: 20 required ids in a length-24 canvas (near-saturated),
    # ids the random model essentially never predicts on its own.
    m, cfg = _tiny_model(0)
    required = list(range(230, 250))  # avoids mask_id=254, eos_id=255
    assert cfg.mask_id not in required and cfg.eos_id not in required
    for seed in range(50):
        out, _ = sample_trajectory(
            m, 24, cfg.mask_id, cfg.eos_id, steps=8, seed=seed,
            constraints={"required_ids": required})
        assert len(out) == 24, "exact token length violated"
        assert not (out == cfg.mask_id).any(), f"seed {seed}: masks left"
        for t in required:
            assert bool((out == t).any()), \
                f"seed {seed}: required token {t} missing"
        rep = check_constraints(out, {"required_ids": required}, cfg.mask_id)
        assert rep["required_present"] and rep["complete"]
    print("PASS test_required_token_guarantee_50_seeds")


def test_required_in_prompt_counts():
    # a required id already present in prompt/frozen context counts —
    # no force-placement needed.
    m, cfg = _tiny_model(1)
    prompt = torch.full((16,), cfg.mask_id, dtype=torch.long)
    prompt[0] = 77
    prompt[1] = 78
    out, _ = sample_trajectory(
        m, 16, cfg.mask_id, cfg.eos_id, steps=6, seed=11,
        prompt_ids=prompt, constraints={"required_ids": [77, 78, 79]})
    for t in (77, 78, 79):
        assert bool((out == t).any()), f"required token {t} missing"
    print("PASS test_required_in_prompt_counts")


def test_forbidden_never_appear_50_seeds():
    m, cfg = _tiny_model(2)
    forbidden = list(range(100, 110))
    for seed in range(50):
        out, _ = sample_trajectory(
            m, 24, cfg.mask_id, cfg.eos_id, steps=8, seed=seed,
            constraints={"forbidden_ids": forbidden})
        for t in forbidden:
            assert not bool((out == t).any()), \
                f"seed {seed}: forbidden token {t} appeared"
        assert not (out == cfg.mask_id).any()
    print("PASS test_forbidden_never_appear_50_seeds")


def test_locked_spans_bit_for_bit():
    m, cfg = _tiny_model(3)
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size - 2, (40,))
    frozen = torch.ones(40, dtype=torch.bool)
    ids[10:16] = cfg.mask_id
    frozen[10:16] = False
    anchors = {20: 33, 21: 34}
    ids[20], ids[21] = 33, 34  # anchor values live in the context too
    out, _ = sample_trajectory(m, 40, cfg.mask_id, cfg.eos_id, steps=6,
                               seed=1, prompt_ids=ids, frozen=frozen,
                               constraints={"anchors": anchors})
    assert torch.equal(out[frozen], ids[frozen]), "locked span modified!"
    assert int(out[20]) == 33 and int(out[21]) == 34, "anchor moved!"
    assert not (out[10:16] == cfg.mask_id).any(), "span not filled"
    rep = check_constraints(out, {"anchors": anchors}, cfg.mask_id)
    assert rep["anchors_intact"]
    print("PASS test_locked_spans_bit_for_bit")


def test_impossible_constraints_raise():
    m, cfg = _tiny_model(4)

    def expect_value_error(fn, label):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"{label}: expected ValueError")

    # forbidden covers the entire vocabulary except [MASK] -> no valid token
    expect_value_error(
        lambda: sample_trajectory(
            m, 16, cfg.mask_id, cfg.eos_id, steps=4, seed=0,
            constraints={"forbidden_ids": [i for i in range(cfg.vocab_size)
                                           if i != cfg.mask_id]}),
        "full-vocab forbidden")
    # required ∩ forbidden
    expect_value_error(
        lambda: sample_trajectory(
            m, 16, cfg.mask_id, cfg.eos_id, steps=4, seed=0,
            constraints={"required_ids": [10], "forbidden_ids": [10]}),
        "required/forbidden overlap")
    # more required tokens than maskable positions
    expect_value_error(
        lambda: sample_trajectory(
            m, 8, cfg.mask_id, cfg.eos_id, steps=4, seed=0,
            constraints={"required_ids": list(range(10))}),
        "required > maskable")
    # mask_id as forbidden / required
    expect_value_error(
        lambda: sample_trajectory(
            m, 16, cfg.mask_id, cfg.eos_id, steps=4, seed=0,
            constraints={"forbidden_ids": [cfg.mask_id]}),
        "mask forbidden")
    expect_value_error(
        lambda: sample_trajectory(
            m, 16, cfg.mask_id, cfg.eos_id, steps=4, seed=0,
            constraints={"required_ids": [cfg.mask_id]}),
        "mask required")
    print("PASS test_impossible_constraints_raise")


def test_determinism_all_order_presets():
    m, cfg = _tiny_model(5)
    orders = ["confidence", "left-to-right", "right-to-left",
              "ends-first", "middle-last", "random"]
    for order in orders:
        a, _ = sample_trajectory(m, 24, cfg.mask_id, cfg.eos_id,
                                 steps=6, seed=123, order=order)
        b, _ = sample_trajectory(m, 24, cfg.mask_id, cfg.eos_id,
                                 steps=6, seed=123, order=order)
        assert torch.equal(a, b), \
            f"order {order!r}: not deterministic given seed"
    print("PASS test_determinism_all_order_presets")


def test_empty_canvas_conf_initialized():
    # nothing masked -> loop never runs; confidences must still be defined
    m, cfg = _tiny_model(6)
    torch.manual_seed(6)
    ids = torch.randint(0, cfg.vocab_size - 2, (16,))
    frozen = torch.ones(16, dtype=torch.bool)
    out, conf = sample_trajectory(m, 16, cfg.mask_id, cfg.eos_id, steps=4,
                                  seed=0, prompt_ids=ids, frozen=frozen)
    assert torch.equal(out, ids)
    assert conf.shape == (16,) and bool((conf == 0).all())
    print("PASS test_empty_canvas_conf_initialized")


def _stateful_kwargs(cfg, **over):
    kw = dict(length=24, mask_id=cfg.mask_id, eos_id=cfg.eos_id, steps=8,
              seed=5, order="confidence", vocab_size=cfg.vocab_size,
              constraints={"required_ids": [10, 20],
                           "forbidden_ids": [30]})
    kw.update(over)
    return kw


def test_stateful_pause_resume_identical():
    m, cfg = _tiny_model(7)
    kw = _stateful_kwargs(cfg)
    ref_kw = {k: v for k, v in kw.items() if k != "vocab_size"}
    ref_ids, ref_conf = sample_trajectory(m, **ref_kw)
    # manual run, pause after 3 steps
    st = DenoisingState(**kw)
    for _ in range(3):
        assert st.step(m) is True
    snap = st.to_snapshot()
    # cross-process resume: torch.save / torch.load round-trip
    import io
    buf = io.BytesIO()
    torch.save(snap, buf)
    buf.seek(0)
    snap2 = torch.load(buf, weights_only=False)
    st2 = DenoisingState.from_snapshot(snap2)
    while st2.step(m):
        pass
    assert torch.equal(st2.x.cpu(), ref_ids), \
        "resumed run diverged from uninterrupted run"
    assert torch.equal(st2.conf.cpu(), ref_conf)
    print("PASS test_stateful_pause_resume_identical")


def test_stateful_pin():
    m, cfg = _tiny_model(8)
    st = DenoisingState(length=24, mask_id=cfg.mask_id, eos_id=cfg.eos_id,
                        steps=6, seed=3, vocab_size=cfg.vocab_size)
    assert st.step(m) is True
    pos = int(torch.where(st.masked)[0][0].item())
    st.pin(pos, 42)
    assert bool(st.frozen[pos]) and int(st.x[pos]) == 42
    while st.step(m):
        pass
    assert int(st.x.cpu()[pos]) == 42, "pinned token did not survive"
    print("PASS test_stateful_pin")


def test_stateful_pin_invariant_guard():
    # 6 slots, 2 required: pinning 5 non-required tokens must raise,
    # because the 2nd required id would have no slot left.
    m, cfg = _tiny_model(9)
    st = DenoisingState(length=6, mask_id=cfg.mask_id, eos_id=cfg.eos_id,
                        steps=4, seed=0, vocab_size=cfg.vocab_size,
                        constraints={"required_ids": [10, 11]})
    masked_pos = torch.where(st.masked)[0].tolist()
    for p in masked_pos[:4]:
        st.pin(p, 5)  # 4 pins ok: 2 slots remain for 2 required
    try:
        st.pin(masked_pos[4], 5)
    except ValueError:
        print("PASS test_stateful_pin_invariant_guard")
        return
    raise AssertionError("pin() did not guard the slot invariant")


def test_stateful_api_equivalence():
    # sample_trajectory_stateful runs the same algorithm and additionally
    # exposes the finished state object.
    from xiiv.diffusion import sample_trajectory_stateful
    m, cfg = _tiny_model(10)
    kw = dict(length=24, mask_id=cfg.mask_id, eos_id=cfg.eos_id, steps=8,
              seed=21, order="ends-first",
              constraints={"required_ids": [10, 20],
                           "forbidden_ids": [30]})
    a, conf_a = sample_trajectory(m, **kw)
    b, conf_b, st = sample_trajectory_stateful(m, **kw)
    assert torch.equal(a, b) and torch.equal(conf_a, conf_b)
    assert torch.equal(st.x.cpu(), a) and st.finished
    # with trajectory
    c, _, traj, st2 = sample_trajectory_stateful(m, return_trajectory=True,
                                                 **kw)
    assert torch.equal(c, a) and len(traj) == 9  # initial + 8 steps
    print("PASS test_stateful_api_equivalence")


def test_span_masks():
    from xiiv.tokenizer import XIIVTokenizer
    tok = XIIVTokenizer.__new__(XIIVTokenizer)
    tok.mask_id = 99
    tok.encode = lambda s: [ord(ch) % 60 for ch in s]  # deterministic stub
    ids, frozen = tok.encode_with_masks("ab<mask>cd<mask:5>e")
    assert len(ids) == 2 + 1 + 2 + 5 + 1, len(ids)
    assert int((ids == 99).sum()) == 6, "span token counts wrong"
    assert frozen.tolist() == ([True] * 2 + [False] + [True] * 2
                               + [False] * 5 + [True]), frozen.tolist()
    # plain <mask> stays exactly one token
    ids1, fr1 = tok.encode_with_masks("x<mask>y")
    assert int((ids1 == 99).sum()) == 1 and len(ids1) == 3
    assert fr1.tolist() == [True, False, True]
    # invalid span raises
    try:
        tok.encode_with_masks("x<mask:0>y")
    except ValueError:
        pass
    else:
        raise AssertionError("<mask:0> did not raise")
    print("PASS test_span_masks")


if __name__ == "__main__":
    test_param_count()
    test_forward_shape_and_bidirectional()
    test_rope_consistency()
    test_mask_batch_ratios()
    test_loss_finite_and_decreases()
    test_sampler_deterministic_and_terminates()
    test_infill_preserves_context()
    test_required_token_guarantee_50_seeds()
    test_required_in_prompt_counts()
    test_forbidden_never_appear_50_seeds()
    test_locked_spans_bit_for_bit()
    test_impossible_constraints_raise()
    test_determinism_all_order_presets()
    test_empty_canvas_conf_initialized()
    test_stateful_pause_resume_identical()
    test_stateful_pin()
    test_stateful_pin_invariant_guard()
    test_stateful_api_equivalence()
    test_span_masks()
    print("ALL TESTS PASSED")

