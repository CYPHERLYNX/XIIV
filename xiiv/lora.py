"""LoRA (Low-Rank Adaptation) for XIIV — surgical training on limited hardware.

Instead of training all 916M params (needs 10GB+ RAM), we:
1. FREEZE the entire base model (no gradients, no optimizer state)
2. Add tiny low-rank adapter matrices to each attention layer
3. Train ONLY the adapters (~3-5M params instead of 916M)

Memory: ~4GB total (model weights fp32 frozen + adapters + grads + Adam for adapters only)
This fits in 7GB RAM and runs on CPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps nn.Linear with a trainable low-rank bypass. Base weights frozen."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        d_in, d_out = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def trainable_params(self):
        return sum(p.numel() for p in [self.lora_A, self.lora_B])


def inject_lora(model, rank=8, alpha=16.0, target_modules=("qkv", "out")):
    """Replace target Linears in all attention layers with LoRA wrappers.
    Returns count of trainable params."""
    total = 0
    for block in model.layers:
        for name in target_modules:
            orig = getattr(block.attn, name)
            if not isinstance(orig, LoRALinear):
                lora = LoRALinear(orig, rank=rank, alpha=alpha)
                setattr(block.attn, name, lora)
                total += lora.trainable_params()
    # Also ensure everything else is frozen
    for name, p in model.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False
    return total


def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(model):
    """Fold LoRA weights back into base for deployment (optional)."""
    for block in model.layers:
        for name in ("qkv", "out"):
            mod = getattr(block.attn, name)
            if isinstance(mod, LoRALinear):
                with torch.no_grad():
                    delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                    mod.base.weight += delta.T if delta.shape == mod.base.weight.shape else delta
                setattr(block.attn, name, mod.base)
    return model
