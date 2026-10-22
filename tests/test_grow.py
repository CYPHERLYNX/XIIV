"""Tests for scripts/grow_checkpoint.py: single-lineage model growth.

Growth must be approximately function-preserving (Net2Net-style widening
with symmetry-breaking noise; exact through RMSNorm is impossible), new
depth blocks must be exact identity, and genuinely new capacity must not
stay redundant (replicas get noise so gradients diverge).
"""
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from grow_checkpoint import grow_model, measure_drift


def _tiny_cfg(**kw):
    base = dict(vocab_size=256, d_model=128, n_layers=2, n_heads=2,
                ffn_dim=256, tie_embeddings=True,
                contract_enabled=True, contract_d_model=32,
                contract_n_layers=1, contract_n_heads=2,
                contract_ffn_dim=64, contract_max_tokens=16)
    base.update(kw)
    return ModelConfig(**base)


def _vocab_map(v_old=256, v_new=384):
    m = {}
    for nid in range(v_new):
        if nid < v_old:
            m[str(nid)] = {"old_id": nid}
        else:
            m[str(nid)] = {"pieces": [nid % v_old, (nid + 7) % v_old]}
    return m


@pytest.fixture
def grown_pair():
    torch.manual_seed(0)
    small = XIIVModel(_tiny_cfg())
    small.init_weights()
    # exercise the contract path: gates are zero at init
    with torch.no_grad():
        for blk in small.layers:
            blk.contract_attn.gate.normal_(0, 0.5)
    big_cfg = _tiny_cfg(vocab_size=384, d_model=192, n_layers=3, n_heads=3)
    gen = torch.Generator().manual_seed(0)
    big = grow_model(small, big_cfg, vocab_map=_vocab_map(),
                     noise_scale=0.01, gen=gen)
    return small, big


def test_param_count_matches_estimate(grown_pair):
    _, big = grown_pair
    assert big.count_parameters() == big.cfg.param_count_estimate()


def test_new_depth_blocks_are_exact_identity(grown_pair):
    small, big = grown_pair
    blk = big.layers[2]
    x = torch.randn(2, 16, 192)
    cos = torch.randn(16, 64)
    sin = torch.randn(16, 64)
    with torch.no_grad():
        y = blk(x, cos, sin, contract_embeds=torch.randn(2, 8, 192))
    assert torch.equal(y, x), "grown depth block must be exact identity"


def test_growth_is_approximately_function_preserving(grown_pair):
    small, big = grown_pair
    max_d, mean_d = measure_drift(small, big,
                                  small.cfg.vocab_size, seq_len=16,
                                  batches=4, contract_tokens=8, seed=1)
    with torch.no_grad():
        probe = small(torch.randint(0, small.cfg.vocab_size, (2, 16)),
                      contract_ids=torch.randint(
                          0, small.cfg.vocab_size, (2, 8))).float()
    rel = mean_d / (probe.std().item() + 1e-6)
    print(f"\ndrift: max|d|={max_d:.4f} mean|d|={mean_d:.4f} rel={rel:.4f}")
    # Width growth through RMSNorm cannot be bit-exact (the rms statistic
    # changes when dims are added); depth growth is exact (see identity
    # test). Bound the approximation instead of demanding exactness.
    assert mean_d < 0.20, f"width drift too large: mean|d|={mean_d:.3f}"


def test_grown_beats_random_init(grown_pair):
    """The point of growth: start much closer to the parent than scratch."""
    small, big = grown_pair
    _, mean_grown = measure_drift(
        small, big, small.cfg.vocab_size, seq_len=16, batches=4,
        contract_tokens=8, seed=1)
    torch.manual_seed(999)
    rnd = XIIVModel(big.cfg)
    rnd.init_weights()
    _, mean_rnd = measure_drift(
        small, rnd, small.cfg.vocab_size, seq_len=16, batches=4,
        contract_tokens=8, seed=1)
    print(f"\ngrown mean|d|={mean_grown:.4f} vs random {mean_rnd:.4f}")
    assert mean_grown < 0.6 * mean_rnd


def test_new_heads_diverge_from_replicas(grown_pair):
    """Replicas get noise: new head rows must NOT equal head-0 rows."""
    _, big = grown_pair
    qkv = big.layers[0].attn.qkv.weight  # (3*192, 192)
    q = qkv[:192]
    assert not torch.equal(q[128:], q[:64]), \
        "new head must differ from its replica (symmetry broken)"


def test_old_dims_untouched_by_noise(grown_pair):
    """Noise applies to NEW rows/cols only; old weights replicate exactly."""
    torch.manual_seed(0)
    small = XIIVModel(_tiny_cfg())
    small.init_weights()
    big_cfg = _tiny_cfg(vocab_size=256, d_model=192, n_layers=2, n_heads=3)
    gen = torch.Generator().manual_seed(0)
    big = grow_model(small, big_cfg, vocab_map=None,
                     noise_scale=0.01, gen=gen)
    # SwiGLU down: output widened; old rows must match exactly
    assert torch.equal(big.layers[0].mlp.down.weight[:128],
                       small.layers[0].mlp.down.weight)
    # norms: old dims replicate exactly
    assert torch.equal(big.layers[0].attn_norm.weight[:128],
                       small.layers[0].attn_norm.weight)


def test_vocab_map_embedding_init(grown_pair):
    small, big = grown_pair
    d_old = small.cfg.d_model
    E_old = small.embed.weight.float()
    E_new = big.embed.weight.float()
    # surviving surface forms: old dims exact copy, new dims noised replica
    assert torch.equal(E_new[10][:d_old], E_old[10].to(E_new.dtype))
    assert not torch.equal(E_new[10][d_old:], E_old[10][:E_new.shape[1]-d_old].to(E_new.dtype))
    # new tokens: mean of pieces (+ noise); check closeness not equality
    # (compare old-dim prefix: new dims hold the widened replica + noise)
    pieces = [300 % 256, (300 + 7) % 256]
    expected = E_old[pieces].mean(dim=0)
    assert torch.allclose(E_new[300][:d_old].float(), expected.float(),
                          atol=0.2)
    # contract encoder token embedding follows the same map
    CE_old = small.contract_encoder.tok_embed.weight.float()
    CE_new = big.contract_encoder.tok_embed.weight.float()
    assert torch.equal(CE_new[10], CE_old[10].to(CE_new.dtype))


# --- Regression: streaming-growth KV truncation (repaired 2026-10-22) ---
# grow_1b_streaming.py applied its d-length row map directly to the
# contract-attention kv.weight [2d, d], silently dropping the V half and
# shipping a 1B checkpoint whose kv was [1280, 1280] instead of
# [2560, 1280] (research/scale_plan_5b.md §1). The split-half operator
# must preserve BOTH halves, and must refuse non-[2d, d] input loudly.
from grow_1b_streaming import _dim_map, _widen_kv_streaming


def test_streaming_kv_widening_preserves_both_halves():
    d_old, hd, d_new = 16, 4, 24
    g, count = _dim_map(d_old, d_new, hd)
    gen = torch.Generator().manual_seed(0)
    k_half = torch.ones(d_old, d_old)
    v_half = torch.full((d_old, d_old), 3.0)  # distinct V marker
    w = torch.cat([k_half, v_half], dim=0)
    out = _widen_kv_streaming(w, g, count, 0.0, gen)
    assert out.shape == (2 * d_new, d_new)
    # Column widening scales both halves by the same per-column counts,
    # so the V/K ratio survives exactly: proof the bottom half came from
    # the parent's V rows, not from re-replicated K rows.
    assert torch.allclose(out[d_new:], 3.0 * out[:d_new])
    assert (out[:d_new] > 0).all() and (out[d_new:] > out[:d_new]).all()


def test_streaming_kv_widening_rejects_wrong_shape():
    g, count = _dim_map(16, 24, 4)
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(AssertionError):
        _widen_kv_streaming(torch.ones(16, 16), g, count, 0.0, gen)
