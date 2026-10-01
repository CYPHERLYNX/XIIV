"""Focused tests for XIIV's reasoning + multimodal APIs.

Covers: scratchpad determinism and region accounting, self-refine
guarantee preservation (frozen/required/forbidden/length), long-form
chunked generation, vision forward shapes/errors, multimodal frozen
prefixes, and snapshot round-trips with and without image_embeds
(including OLD snapshots that lack the image_embeds key).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.diffusion import (DenoisingState, sample_trajectory,
                              generate_with_scratchpad, self_refine,
                              generate_long)
from xiiv.vision import encode_image, make_multimodal_canvas


def tiny_cfg(**kw):
    d = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
             ffn_dim=128, mask_id=254, eos_id=255, seq_len=32)
    d.update(kw)
    return ModelConfig(**d)


def _tiny(seed=0):
    torch.manual_seed(seed)
    return XIIVModel(tiny_cfg())


def _vision_model(seed=1):
    torch.manual_seed(seed)
    cfg = tiny_cfg(image_enabled=True, image_size=64, image_patch_size=16)
    m = XIIVModel(cfg)
    m.eval()
    return m, cfg


def test_text_only_model_unchanged():
    # vision scaffold must not alter the text-only parameter count
    m = _tiny()
    assert m.count_parameters() == 98624, m.count_parameters()
    m.eval()
    with torch.no_grad():
        out = m(torch.randint(0, 256, (2, 32)))
    assert out.shape == (2, 32, 256)
    print("PASS test_text_only_model_unchanged")


def test_vision_forward_shape_and_errors():
    m, cfg = _vision_model()
    n = m.n_image_tokens
    assert n == 16, n  # (64/16)^2 patches
    px = torch.rand(3, 64, 64)
    ie = encode_image(m, px)                      # (N, d)
    assert ie.shape == (n, 64), ie.shape
    ids = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        out = m(ids, ie)                          # model also accepts (N, d)
    assert out.shape == (1, 32, 256), out.shape
    # wrong image-token count must raise, not silently misalign
    try:
        m(ids, torch.randn(n + 1, 64))
        raise SystemExit("no error for wrong image-token count")
    except ValueError:
        pass
    # image_embeds on a text-only model must raise
    mt = _tiny()
    mt.eval()
    try:
        with torch.no_grad():
            mt(ids, ie)
        raise SystemExit("no error for image_embeds on text-only model")
    except ValueError:
        pass
    print("PASS test_vision_forward_shape_and_errors")


def test_multimodal_frozen_prefix():
    m, cfg = _vision_model()
    n = m.n_image_tokens
    px = torch.rand(3, 64, 64)
    ie = encode_image(m, px)                      # (N, d)
    text = torch.full((8,), cfg.mask_id, dtype=torch.long)
    prompt_ids, frozen = make_multimodal_canvas(n, text, cfg.mask_id)
    assert len(prompt_ids) == n + 8
    assert bool(frozen[:n].all()) and not bool(frozen[n:].any())
    ids, _ = sample_trajectory(m, n + 8, cfg.mask_id, cfg.eos_id,
                               steps=8, seed=11, prompt_ids=prompt_ids,
                               frozen=frozen, image_embeds=ie)
    # frozen image-prefix placeholders are never denoised
    assert torch.equal(ids[:n], prompt_ids[:n]), "image prefix was modified!"
    # text region got filled with real tokens
    assert not bool((ids[n:] == cfg.mask_id).any())
    print("PASS test_multimodal_frozen_prefix")


def test_scratchpad_determinism_and_accounting():
    m = _tiny()
    m.eval()
    prompt = torch.tensor([10, 20, 30], dtype=torch.long)
    kw = dict(length=24, mask_id=254, eos_id=255, scratch_len=8, steps=8,
              seed=42, prompt_ids=prompt)
    a1, c1, s1 = generate_with_scratchpad(m, **kw)
    a2, c2, s2 = generate_with_scratchpad(m, **kw)
    assert torch.equal(a1, a2) and torch.equal(c1, c2) and torch.equal(s1, s2)
    # layout: [prompt(3)][scratchpad(8)][answer(13)] -> total 27
    assert len(a1) == 27 and len(s1) == 8
    assert torch.equal(a1[:3], prompt), "prompt was not preserved"
    assert torch.equal(a1[3:11], s1), "scratchpad slice mismatch"
    assert not bool((a1 == 254).any()), "masks remain in final output"
    # invalid scratch_len raises
    for bad in (0, 24, 25):
        try:
            generate_with_scratchpad(m, length=24, mask_id=254, eos_id=255,
                                     scratch_len=bad, steps=4)
            raise SystemExit(f"no error for scratch_len={bad}")
        except ValueError:
            pass
    print("PASS test_scratchpad_determinism_and_accounting")


def test_self_refine_preserves_guarantees():
    m = _tiny()
    m.eval()
    torch.manual_seed(0)
    prompt = torch.randint(0, 200, (6,))
    frozen = torch.zeros(20, dtype=torch.bool)
    frozen[10] = True
    frozen[11] = True
    prompt_ids = torch.full((20,), 254, dtype=torch.long)
    prompt_ids[:6] = prompt
    prompt_ids[10] = 50
    prompt_ids[11] = 51   # frozen positions must hold real context ids
    cons = {"required_ids": [201, 202], "forbidden_ids": [210, 211, 212]}
    kw = dict(length=20, mask_id=254, eos_id=255, steps=8, refine_rounds=2,
              remask_frac=0.3, refine_steps=6, seed=7, prompt_ids=prompt_ids,
              frozen=frozen, constraints=cons)
    ids1, cf1 = self_refine(m, **kw)
    ids2, cf2 = self_refine(m, **kw)
    assert torch.equal(ids1, ids2), "self_refine not deterministic"
    assert len(ids1) == 20, "length changed during refinement"
    assert torch.equal(ids1[:6], prompt), "prompt prefix was modified"
    assert torch.equal(ids1[10:12], prompt_ids[10:12]), "frozen touched"
    assert 201 in ids1.tolist() and 202 in ids1.tolist(), "required lost"
    assert not any(t in ids1.tolist() for t in (210, 211, 212)), \
        "forbidden token sampled during refinement"
    print("PASS test_self_refine_preserves_guarantees")


def test_generate_long_exact_and_deterministic():
    m = _tiny()
    m.eval()
    total = 40
    kw = dict(total_length=total, chunk_length=10, mask_id=254, eos_id=255,
              seq_len=32, steps=6, seed=99)
    ids1, cf1 = generate_long(m, **kw)
    ids2, cf2 = generate_long(m, **kw)
    assert len(ids1) == total and len(cf1) == total, "wrong total length"
    assert torch.equal(ids1, ids2), "generate_long not deterministic"
    assert not bool((ids1 == 254).any()), "masks remain"
    print("PASS test_generate_long_exact_and_deterministic")


def test_snapshot_roundtrip_with_image_embeds():
    m, cfg = _vision_model()
    n = m.n_image_tokens
    ie = encode_image(m, torch.rand(3, 64, 64))   # (N, d)
    st = DenoisingState(length=n + 8, mask_id=cfg.mask_id, eos_id=cfg.eos_id,
                        steps=8, seed=5, image_embeds=ie)
    assert st.image_embeds.shape == (1, n, 64)
    for _ in range(3):
        assert st.step(m) is True
    snap = st.to_snapshot()
    assert snap["image_embeds"] is not None
    import io
    buf = io.BytesIO()
    torch.save(snap, buf)
    buf.seek(0)
    st2 = DenoisingState.from_snapshot(torch.load(buf, weights_only=False))
    assert torch.equal(st2.image_embeds, st.image_embeds)
    # resume to completion without errors
    while st2.step(m):
        pass
    assert not bool(st2.masked.any())
    print("PASS test_snapshot_roundtrip_with_image_embeds")


def test_old_snapshot_without_image_key_loads():
    # backward compat: snapshots taken before the multimodal edit
    m = _tiny()
    m.eval()
    st = DenoisingState(length=16, mask_id=254, eos_id=255, steps=6, seed=3)
    for _ in range(2):
        assert st.step(m) is True
    snap = st.to_snapshot()
    del snap["image_embeds"]          # simulate a pre-multimodal snapshot
    st2 = DenoisingState.from_snapshot(snap)
    assert st2.image_embeds is None
    while st2.step(m):
        pass
    assert not bool(st2.masked.any())
    print("PASS test_old_snapshot_without_image_key_loads")


if __name__ == "__main__":
    test_text_only_model_unchanged()
    test_vision_forward_shape_and_errors()
    test_multimodal_frozen_prefix()
    test_scratchpad_determinism_and_accounting()
    test_self_refine_preserves_guarantees()
    test_generate_long_exact_and_deterministic()
    test_snapshot_roundtrip_with_image_embeds()
    test_old_snapshot_without_image_key_loads()
    print("ALL REASONING TESTS PASSED")
