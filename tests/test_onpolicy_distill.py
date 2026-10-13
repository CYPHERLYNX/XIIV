"""Tests for On-Policy Self-Corrective Distillation (OSCD).

Mechanism checks on a tiny toy block, CPU-only. These validate the
implementation — they do NOT measure gains on the real 916M model (that
needs SGBD Phase 2 with real teacher pairs; block CE before/after is the
metric to watch there).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from xiiv.onpolicy_distill import (
    on_policy_corrupt,
    self_corrective_distill_loss,
)
from xiiv.sgbd import BlockTrainer


class ToyBlock(nn.Module):
    """Tiny ids->logits block standing in for the SGBD output block."""

    def __init__(self, vocab=64, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return self.head(self.emb(ids))


def _batch(vocab=64, b=4, L=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab - 1, (b, L), generator=g), g


def test_on_policy_corrupt_fills_masks_with_own_predictions():
    block = ToyBlock()
    x0, g = _batch()
    mask_id = 63
    noisy = x0.clone()
    m = torch.rand(x0.shape, generator=g) < 0.5
    noisy[m] = mask_id
    with torch.no_grad():
        logits = block(noisy)
    x_p = on_policy_corrupt(noisy, logits, m)
    # clean positions untouched
    assert torch.equal(x_p[~m], noisy[~m])
    # masked positions hold the block's own argmax predictions
    assert torch.equal(x_p[m], logits.argmax(-1)[m])
    # predictions are detached — no grad path through x_p
    assert not x_p.requires_grad
    # no mask tokens remain at filled positions
    assert (x_p[m] != mask_id).all()


def test_on_policy_corrupt_sample_mode():
    block = ToyBlock()
    x0, g = _batch()
    mask_id = 63
    noisy = x0.clone()
    m = torch.rand(x0.shape, generator=g) < 0.5
    noisy[m] = mask_id
    with torch.no_grad():
        logits = block(noisy)
    g2 = torch.Generator().manual_seed(1)
    x_p = on_policy_corrupt(noisy, logits, m, generator=g2,
                            sample_predictions=True)
    assert x_p.shape == noisy.shape
    assert torch.equal(x_p[~m], noisy[~m])
    assert ((x_p[m] >= 0) & (x_p[m] < 64)).all()


def test_loss_returns_tuple_and_backprops():
    block = ToyBlock()
    x0, g = _batch()
    loss, info = self_corrective_distill_loss(
        block, x0, mask_id=63, generator=g)
    assert loss.dim() == 0 and loss.requires_grad
    assert info["n_masked"] > 0
    assert info["l_mask"] > 0 and info["l_self"] > 0
    loss.backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(gr).all() for gr in grads)


def test_lambda_self_zero_recovers_standard_loss():
    block = ToyBlock()
    x0, g = _batch()
    loss_full, info_full = self_corrective_distill_loss(
        block, x0, mask_id=63, lambda_self=1.0,
        generator=torch.Generator().manual_seed(7))
    loss_std, _ = self_corrective_distill_loss(
        block, x0, mask_id=63, lambda_self=0.0,
        generator=torch.Generator().manual_seed(7))
    # same corruption (same seed) -> full = l_mask + l_self, std = l_mask
    assert abs(info_full["l_mask"] - loss_std.item()) < 1e-5
    assert loss_full.item() > loss_std.item()


def test_degenerate_no_masks_returns_zero_tuple():
    block = ToyBlock()
    x0, g = _batch()
    # t_max tiny but mask_batch draws t ~ U(t_min, t_max); force the
    # degenerate path via t_min=t_max=0
    loss, info = self_corrective_distill_loss(
        block, x0, mask_id=63, t_min=0.0, t_max=0.0, generator=g)
    assert info["n_masked"] == 0
    assert loss.item() == 0.0
    loss.backward()  # graph must stay alive


def test_block_trainer_token_step_trains():
    # Learnable toy task: every masked position targets the SAME token, so
    # the block can genuinely improve (a random-target toy sits at the
    # uniform optimum and cannot decrease — that would be a bad test, not
    # a bad loss).
    torch.manual_seed(0)
    block = ToyBlock()
    trainer = BlockTrainer(block, block.parameters(), lr=3e-3)
    x0 = torch.full((4, 16), 7, dtype=torch.long)  # constant target
    before, _ = self_corrective_distill_loss(
        block, x0, mask_id=63,
        generator=torch.Generator().manual_seed(11))
    before = before.item()
    for s in range(50):
        gs = torch.Generator().manual_seed(100 + s)
        trainer.train_step_tokens(x0, mask_id=63, generator=gs)
    after, info = self_corrective_distill_loss(
        block, x0, mask_id=63,
        generator=torch.Generator().manual_seed(11))
    after = after.item()
    assert after < before * 0.95, f"toy loss did not learn: {before} -> {after}"
    assert info["n_masked"] > 0
