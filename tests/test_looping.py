"""Tests for selective looping in XIIVModel.forward (LoopMDM /
Looped-Uniform-Diffusion lineage): re-applying a contiguous layer
group extra times must be exactly the manual repeated application,
the default path must be bit-identical, and gradients must flow.
"""
import pytest
import torch

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel


def _model(seed=0):
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                      ffn_dim=128, mask_id=254, eos_id=255, seq_len=32)
    torch.manual_seed(seed)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    return m


def test_default_path_unchanged():
    m = _model()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        a = m(ids)
        b = m(ids, loop_group=None, loop_count=0)
        c = m(ids, loop_group=(0, 2), loop_count=0)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_looping_equals_manual_reapplication():
    m = _model()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        looped = m(ids, loop_group=(0, 2), loop_count=1)
        # manual: embed -> L0 L1 L0 L1 L2 L3 -> norm -> head
        x = m.embed(ids)
        cos, sin = m._get_rope_cache(16, m.cfg.d_model // m.cfg.n_heads,
                                     ids.device, x.dtype)
        for i in (0, 1, 0, 1, 2, 3):
            x = m.layers[i](x, cos, sin, None)
        manual = m.lm_head(m.final_norm(x))
    assert torch.allclose(looped, manual, atol=1e-6)


def test_looping_changes_output():
    m = _model()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        a = m(ids)
        b = m(ids, loop_group=(0, 2), loop_count=2)
    assert not torch.allclose(a, b)


def test_invalid_group_raises():
    m = _model()
    ids = torch.randint(0, 200, (1, 8))
    with pytest.raises(ValueError):
        m(ids, loop_group=(2, 2), loop_count=1)
    with pytest.raises(ValueError):
        m(ids, loop_group=(0, 9), loop_count=1)


def test_flop_ratio():
    r = XIIVModel.selective_loop_flop_ratio
    assert r(4, None, 0) == 1.0
    assert r(4, (0, 2), 1) == 1.5
    assert r(36, (0, 6), 2) == pytest.approx((36 + 12) / 36)


def test_gradients_flow_through_loop():
    m = _model()
    m.train()
    ids = torch.randint(0, 200, (2, 16))
    logits = m(ids, loop_group=(0, 2), loop_count=1)
    loss = logits.float().pow(2).mean()
    loss.backward()
    g0 = m.layers[0].attn.qkv.weight.grad
    assert g0 is not None and float(g0.abs().sum()) > 0
