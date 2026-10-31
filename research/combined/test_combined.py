"""CPU smoke test for research/combined/combined.py.

Tiny real XIIVModel (d=64, 2 layers) on synthetic data with a planted
bigram pattern. Trains with combined_loss (p_cold=0.5 so both branches are
exercised heavily) and asserts:
  - loss is a finite scalar requiring grad every step
  - BOTH branches are sampled (cold and BASC-denoise)
  - cold loss decreases over the run (the joint prior trains)
  - denoise l_mask decreases over the run (the conditional skill trains)
  - no NaN/Inf in loss or in any model parameter at the end
  - info dict carries the combined contract on each branch:
      cold   -> {"cold": True,  "l_cold"}
      denoise-> {"cold": False, "mask_ratio", "n_masked",
                             "l_mask", "l_self", "l_anchor"}
  - KL anchor is non-negative (within 1e-6 float noise)
  - degenerate guard: t_min=t_max=0 denoise step returns a finite
    zero-graph loss (BASC's guard flows through)
  - MECHANISM CHECK: single-shot generation from a 100%-masked canvas
    recovers the planted pattern better AFTER training than at init

Run:  python test_combined.py   (from this dir; needs the repo on sys.path)
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
from combined import combined_loss


def planted_batch(n, L):
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
    target = planted_batch(8, L)
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

    x0 = planted_batch(4, 32)

    acc_before = cold_shot_accuracy(model, cfg.mask_id, 32)
    print(f"[smoke] cold-shot pattern accuracy BEFORE: {acc_before:.3f}")

    cold_losses, denoise_losses = [], []
    for step in range(40):
        opt.zero_grad()
        loss, info = combined_loss(model, x0, cfg.mask_id,
                                   p_cold=0.5, generator=gen)
        assert torch.is_tensor(loss) and loss.dim() == 0, "loss not scalar"
        assert loss.requires_grad, "loss carries no grad"
        assert torch.isfinite(loss).item(), f"non-finite loss at {step}"
        assert "cold" in info, "info missing 'cold' flag"
        if info["cold"]:
            assert "l_cold" in info, "cold branch missing l_cold"
            cold_losses.append(info["l_cold"].item())
        else:
            for k in ("mask_ratio", "n_masked", "l_mask",
                      "l_self", "l_anchor"):
                assert k in info, f"denoise branch info missing {k}"
            # KL is mathematically non-negative; allow 1e-6 float noise.
            assert info["l_anchor"] >= -1e-6, \
                f"KL anchor unexpectedly negative: {info['l_anchor']}"
            denoise_losses.append(info["l_mask"])
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "non-finite grad"
        opt.step()

    for p in model.parameters():
        assert torch.isfinite(p).all(), "non-finite param"

    assert len(cold_losses) > 5, "cold branch never sampled"
    assert len(denoise_losses) > 5, "denoise branch never sampled"
    print(f"[smoke] cold steps={len(cold_losses)} "
          f"l_cold {cold_losses[0]:.3f} -> {cold_losses[-1]:.3f}")
    print(f"[smoke] denoise steps={len(denoise_losses)} "
          f"l_mask {denoise_losses[0]:.3f} -> {denoise_losses[-1]:.3f}")
    assert cold_losses[-1] < cold_losses[0], "cold loss did not decrease"
    assert denoise_losses[-1] < denoise_losses[0], \
        "denoise l_mask did not decrease"

    acc_after = cold_shot_accuracy(model, cfg.mask_id, 32)
    print(f"[smoke] cold-shot pattern accuracy AFTER:  {acc_after:.3f}")
    assert acc_after > acc_before, \
        "cold-shot generation did not improve"

    # Degenerate guard: force zero masked positions via t_min=t_max=0.
    # p_cold=0.0 keeps us on the denoise branch so BASC's guard fires.
    loss0, info0 = combined_loss(model, x0, cfg.mask_id, p_cold=0.0,
                                 t_min=0.0, t_max=0.0, generator=gen)
    assert info0["cold"] is False and info0["n_masked"] == 0 \
        and torch.isfinite(loss0).item()
    print("[smoke] degenerate guard OK (n_masked=0 -> finite zero-graph)")

    print("[smoke] PASS: combined step trains both branches; "
          "step-0 prior improves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
