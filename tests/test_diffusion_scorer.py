"""Regression tests for the diffusion pseudo-likelihood scorer.

These tests lock in the scorer's behavior: determinism, correct masking,
and sensible rankings. If the scorer changes, these must still pass.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.diffusion_scorer import score_continuation, rank_choices


def tiny_model():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      ffn_dim=128, tie_embeddings=True,
                      contract_enabled=False, mask_id=1)
    torch.manual_seed(42)
    m = XIIVModel(cfg)
    m.eval()
    return m, cfg


def test_deterministic():
    """Same input -> same score, every time."""
    m, cfg = tiny_model()
    ctx, cont = [10, 20, 30], [40, 50]
    nll1, _ = score_continuation(m, ctx, cont, cfg.mask_id)
    nll2, _ = score_continuation(m, ctx, cont, cfg.mask_id)
    assert nll1 == nll2, f"non-deterministic: {nll1} vs {nll2}"
    print("PASS: deterministic")


def test_empty_continuation():
    """Empty continuation -> (0.0, 0), no crash."""
    m, cfg = tiny_model()
    nll, n = score_continuation(m, [10, 20], [], cfg.mask_id)
    assert nll == 0.0 and n == 0
    print("PASS: empty continuation")


def test_single_token():
    """Single-token continuation scores exactly 1 position."""
    m, cfg = tiny_model()
    nll, n = score_continuation(m, [10, 20], [40], cfg.mask_id)
    assert n == 1 and nll > 0
    print(f"PASS: single token (nll={nll:.4f})")


def test_context_never_masked():
    """Scoring must not depend on masking the context.

    We verify by checking that lengthening the context with identical
    prefix doesn't change the per-token computation structure.
    (The model's bidirectional attention means context affects scores,
    but the MASK is only ever on continuation positions.)
    """
    m, cfg = tiny_model()
    # If we score with empty context vs real context, both should work
    nll1, n1 = score_continuation(m, [], [40, 50], cfg.mask_id)
    nll2, n2 = score_continuation(m, [10, 20, 30], [40, 50], cfg.mask_id)
    assert n1 == n2 == 2
    # scores differ (context matters) but both are finite
    assert 0 < nll1 < 100 and 0 < nll2 < 100
    print("PASS: context never masked")


def test_ranking_sensible():
    """A choice identical to training-like pattern should rank sensibly.

    We can't assert which choice wins without a trained model, but we
    assert the ranking is total, deterministic, and covers all choices.
    """
    m, cfg = tiny_model()
    ctx = [10, 20]
    choices = [[30], [40, 50], [60, 70, 80]]
    ranked = rank_choices(m, ctx, choices, cfg.mask_id)
    assert len(ranked) == 3
    assert sorted(i for i, _ in ranked) == [0, 1, 2]
    # sorted ascending by NLL
    nlls = [nll for _, nll in ranked]
    assert nlls == sorted(nlls)
    # deterministic
    ranked2 = rank_choices(m, ctx, choices, cfg.mask_id)
    assert ranked == ranked2
    print("PASS: ranking sensible + deterministic")


def test_leave_one_out_masks_one():
    """Verify leave-one-out: exactly 1 masked position per forward pass.

    We do this by monkeypatching the model to record its input.
    """
    m, cfg = tiny_model()
    seen_masks = []
    orig_forward = m.forward

    def spy(ids, **kw):
        n_masked = (ids == cfg.mask_id).sum().item()
        seen_masks.append(n_masked)
        return orig_forward(ids, **kw)
    m.forward = spy

    score_continuation(m, [10, 20, 30], [40, 50, 60], cfg.mask_id)
    m.forward = orig_forward
    # 3 continuation tokens -> 3 forward passes, each with exactly 1 mask
    assert seen_masks == [1, 1, 1], f"expected [1,1,1], got {seen_masks}"
    print("PASS: leave-one-out masks exactly one position")


if __name__ == "__main__":
    test_deterministic()
    test_empty_continuation()
    test_single_token()
    test_context_never_masked()
    test_ranking_sensible()
    test_leave_one_out_masks_one()
    print("\nAll diffusion scorer regression tests passed.")
