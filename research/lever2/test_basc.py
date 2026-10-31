"""CPU smoke test for research/lever2/basc.py.

Tiny real XIIVModel (d=64, 2 layers) on random token data, a few Adam
steps with belief_anchored_loss. Asserts:
  - loss is a finite scalar requiring grad every step
  - loss decreases over the run (mechanism trains)
  - no NaN/Inf in loss or in any model parameter at the end
  - info dict carries the full BASC contract (incl. l_anchor)
  - degenerate guard: an all-clean batch still returns (loss, info)

Run:  python test_basc.py   (from this dir; needs the repo on sys.path)
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
from basc import belief_anchored_loss


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

    # Fixed synthetic "language": random tokens, batch 4 x 32.
    x0 = torch.randint(0, 253, (4, 32))

    losses = []
    for step in range(20):
        opt.zero_grad()
        loss, info = belief_anchored_loss(model, x0, cfg.mask_id,
                                          generator=gen)
        assert torch.is_tensor(loss) and loss.dim() == 0, "loss not scalar"
        assert loss.requires_grad, "loss carries no grad"
        assert torch.isfinite(loss).item(), f"non-finite loss at step {step}"
        for k in ("mask_ratio", "n_masked", "l_mask", "l_self", "l_anchor"):
            assert k in info, f"info missing key {k}"
        # KL is mathematically non-negative; allow 1e-6 of float noise
        # (F.kl_div can return tiny negatives when the two distributions
        # are near-identical, e.g. a random-init model stuck echoing the
        # mask token so the sampled canvas equals the noisy canvas).
        assert info["l_anchor"] >= -1e-6, \
            f"KL anchor unexpectedly negative: {info['l_anchor']}"
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all().item(), \
                    f"non-finite grad at step {step}"
        opt.step()
        losses.append(float(loss.item()))
        if step % 5 == 0 or step == 19:
            print(f"[smoke] step={step:2d} loss={losses[-1]:.4f} "
                  f"l_mask={info['l_mask']:.4f} l_self={info['l_self']:.4f} "
                  f"l_anchor={info['l_anchor']:.4f} n_masked={info['n_masked']}")

    for p in model.parameters():
        assert torch.isfinite(p).all().item(), "non-finite param after run"
    assert losses[-1] < losses[0], \
        f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    # Degenerate guard: force zero masked positions via t_min=t_max=0.
    loss0, info0 = belief_anchored_loss(model, x0, cfg.mask_id,
                                        t_min=0.0, t_max=0.0, generator=gen)
    assert info0["n_masked"] == 0 and torch.isfinite(loss0).item()
    print("[smoke] degenerate guard OK (n_masked=0 -> finite zero-graph)")

    # Anchor-term unit check with a synthetic model whose beliefs provably
    # swing between the clean and corrupted canvases. mask_id=7 here.
    def fake_model(ids):
        base = torch.zeros(ids.shape[0], ids.shape[1], 8)
        base[..., 0] = (ids == 7).float() * 5.0   # mask token in -> peak at 0
        base[..., 1] = (ids != 7).float() * 5.0   # filled token in -> peak at 1
        return base

    x0b = torch.randint(0, 7, (2, 16))
    gen2 = torch.Generator().manual_seed(99)
    lossb, infob = belief_anchored_loss(fake_model, x0b, 7,
                                        t_min=0.5, t_max=0.5,
                                        lambda_self=1.0, lambda_anchor=0.5,
                                        generator=gen2)
    assert torch.isfinite(lossb).item(), "synthetic loss non-finite"
    assert infob["n_masked"] > 0, "synthetic batch masked nothing"
    # Clean canvas peaks at token 0, corrupted canvas peaks at token 1:
    # the anchor must register a large positive KL.
    assert infob["l_anchor"] > 1.0, \
        f"anchor term missed the belief swing: {infob['l_anchor']}"
    print(f"[smoke] anchor unit check OK (l_anchor={infob['l_anchor']:.3f} "
          "on synthetic belief swing)")

    print(f"[smoke] PASS: loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
          "no NaNs, full info contract, degenerate guard OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
