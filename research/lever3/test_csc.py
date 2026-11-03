"""CPU smoke test for research/lever3/csc.py.

Tiny real XIIVModel (d=64, 2 layers) on synthetic data with a planted
bigram pattern. Trains with cold_start_loss (p_cold=0.5 so the cold
objective is exercised heavily) and asserts:
  - loss is a finite scalar requiring grad every step
  - cold loss decreases over the run (the joint prior trains)
  - no NaN/Inf in loss or in any model parameter at the end
  - info dict carries the CSC contract (cold flag + l_cold/l_denoise)
  - MECHANISM CHECK: single-shot generation from a 100%-masked canvas
    recovers the planted pattern better AFTER training than at init
    (the step-0 void is being filled with a trained prior)

Run:  python test_csc.py   (from this dir; needs the repo on sys.path)
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from csc import cold_start_loss


def planted_batch(gen, n, L, vocab):
    """Synthetic language: every sequence is repetitions of the bigram
    (10, 20) — a joint pattern only learnable from full sequences."""
    x = torch.empty(n, L, dtype=torch.long)
    for i in range(n):
        for j in range(0, L, 2):
            x[i, j] = 10
            if j + 1 < L:
                x[i, j + 1] = 20
    return x


@torch.no_grad()
def cold_shot_accuracy(model, mask_id, L):
    """Single forward pass from 100%-masked canvas; fraction of positions
    matching the planted (10, 20) pattern."""
    x_cold = torch.full((8, L), mask_id, dtype=torch.long)
    pred = model(x_cold).argmax(-1)
    target = planted_batch(None, 8, L, None)
    return (pred == target).float().mean().item()


def main():
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      ffn_dim=128, seq_len=32, mask_id=254, eos_id=255,
                      tie_embeddings=True)
    model = XIIVModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] tiny XIIVModel params={n_params}")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(1234)

    x0 = planted_batch(gen, 4, 32, 256)

    acc_before = cold_shot_accuracy(model, cfg.mask_id, 32)
    print(f"[smoke] cold-shot pattern accuracy BEFORE: {acc_before:.3f}")

    cold_losses, denoise_losses = [], []
    for step in range(40):
        opt.zero_grad()
        loss, info = cold_start_loss(model, x0, cfg.mask_id,
                                     p_cold=0.5, generator=gen)
        assert torch.is_tensor(loss) and loss.dim() == 0, "loss not scalar"
        assert loss.requires_grad, "loss carries no grad"
        assert torch.isfinite(loss).item(), f"non-finite loss at {step}"
        assert "cold" in info, "info missing 'cold' flag"
        if info["cold"]:
            assert "l_cold" in info
            cold_losses.append(info["l_cold"].item())
        else:
            assert "l_denoise" in info
            denoise_losses.append(info["l_denoise"].item())
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "non-finite grad"
        opt.step()

    for p in model.parameters():
        assert torch.isfinite(p).all(), "non-finite param"

    assert len(cold_losses) > 5, "cold objective never sampled"
    assert len(denoise_losses) > 5, "denoise objective never sampled"
    print(f"[smoke] cold batches={len(cold_losses)} "
          f"l_cold {cold_losses[0]:.3f} -> {cold_losses[-1]:.3f}")
    print(f"[smoke] denoise batches={len(denoise_losses)} "
          f"l_denoise {denoise_losses[0]:.3f} -> {denoise_losses[-1]:.3f}")
    assert cold_losses[-1] < cold_losses[0], "cold loss did not decrease"

    acc_after = cold_shot_accuracy(model, cfg.mask_id, 32)
    print(f"[smoke] cold-shot pattern accuracy AFTER:  {acc_after:.3f}")
    assert acc_after > acc_before, \
        "cold-shot generation did not improve"
    print("[smoke] PASS: dual objective trains; step-0 prior improves")


if __name__ == "__main__":
    main()
