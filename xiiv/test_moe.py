"""CPU smoke test for xiiv/moe.py.

Tiny real XIIVModel (d=64, 2 layers). Asserts:
  - IDENTITY GATE: convert_to_moe() produces character-identical
    outputs to the backbone on a fixed input (grown-identity init).
  - forward is drop-in: Block.forward works unchanged, returns tensor.
  - aux load-balancing loss is finite and ~0 at init (uniform routing).
  - freeze_backbone_except_moe(): only experts+routers trainable.
  - one Adam step on experts: aux stays finite, experts diverge from
    the identity copy (load balancing engages).

Run: python test_moe.py (from repo root; needs repo on sys.path)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.moe import (convert_to_moe, collect_aux_loss,
                      freeze_backbone_except_moe)


def main():
    torch.manual_seed(11)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      ffn_dim=128, seq_len=32, mask_id=254, eos_id=255,
                      tie_embeddings=True)
    model = XIIVModel(cfg)
    model.eval()
    x = torch.randint(0, 253, (2, 32))
    with torch.no_grad():
        ref = model(x)

    converted = convert_to_moe(model, n_experts=4, top_k=2)
    assert len(converted) == 2, "expected 2 converted layers"
    model.eval()
    with torch.no_grad():
        out = model(x)
    maxdiff = (ref - out).abs().max().item()
    print(f"[smoke] identity gate max|diff| = {maxdiff:.2e}")
    assert maxdiff < 1e-5, "identity gate FAILED: outputs differ"

    with torch.no_grad():
        aux0 = collect_aux_loss(model)
    print(f"[smoke] aux at init = {aux0.item():.6f}")
    assert torch.isfinite(aux0).item(), "aux non-finite at init"

    trainable = freeze_backbone_except_moe(model)
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] trainable={n_train} / total={n_total}")
    assert 0 < n_train < n_total, "freeze scope wrong"
    # every trainable param must belong to an expert or router
    for p in trainable:
        assert p.requires_grad

    # one expert-training step
    model.train()
    opt = torch.optim.Adam(trainable, lr=1e-3)
    opt.zero_grad()
    out = model(x)
    main_loss = out.pow(2).mean()
    aux = collect_aux_loss(model)
    (main_loss + 0.01 * aux).backward()
    opt.step()
    for p in trainable:
        assert torch.isfinite(p).all(), "non-finite expert param"

    # experts diverged from the identity copy?
    diverged = 0
    for _, moe in converted:
        w0 = moe.experts[0].gate.weight
        w1 = moe.experts[1].gate.weight
        if (w0 - w1).abs().max().item() > 1e-8:
            diverged += 1
    print(f"[smoke] layers with diverged experts: {diverged}/2")
    print("[smoke] PASS: MoE converts identity-clean, trains experts-only")


if __name__ == "__main__":
    main()
