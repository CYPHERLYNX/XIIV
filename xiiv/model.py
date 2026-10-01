"""XIIV: a bidirectional transformer for masked-diffusion language modeling.

Unlike autoregressive (GPT-style) models, there is NO causal mask anywhere:
every position attends to every other position. The model is trained to
reconstruct randomly-masked tokens (the "diffusion" forward process), and at
inference time it generates by iterative parallel de-masking.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device, dtype=torch.float32):
    """Precompute RoPE cos/sin tables of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                        dtype=dtype) / head_dim))
    ang = torch.outer(pos, freq)                       # (T, head_dim/2)
    emb = torch.cat([ang, ang], dim=-1)                # (T, head_dim)
    return torch.cos(emb), torch.sin(emb)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (T, D). Rotary position embedding."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    # interleave-style rotation matching the cache layout above
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        assert d % h == 0
        self.n_heads = h
        self.head_dim = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)               # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # NOTE: deliberately bidirectional — no causal mask. This is what
        # makes parallel de-masking (diffusion) possible at inference.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class PatchEmbed(nn.Module):
    """Tiny vision stem for the multimodal scaffold.

    Patchifies an (B, 3, S, S) image into (B, N, d) prefix embeddings,
    N = (image_size // patch_size)^2. Created ONLY when
    cfg.image_enabled=True, so text-only checkpoints are bit-unaffected.
    The embeddings SUBSTITUTE the first N token positions (see
    XIIVModel.forward); the sampler freezes those positions, which makes
    multimodal generation ordinary infilling conditioned on a frozen
    image prefix — no sampler fork required.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        p = cfg.image_patch_size
        self.n_tokens = (cfg.image_size // p) ** 2
        self.proj = nn.Conv2d(3, cfg.d_model, kernel_size=p, stride=p)
        self.pos = nn.Embedding(self.n_tokens, cfg.d_model)
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, S, S) float in [0, 1] -> (B, N, d)
        x = self.proj(images).flatten(2).transpose(1, 2)
        x = x + self.pos.weight.unsqueeze(0)
        return self.norm(x)


class XIIVModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # set True to trade ~30% speed for much lower activation memory
        self.grad_checkpoint = False
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight  # tied: no extra params
        # Multimodal scaffold: created only when enabled; text-only
        # checkpoints never see these parameters.
        if cfg.image_enabled:
            self.patch_embed = PatchEmbed(cfg)
            self.n_image_tokens = self.patch_embed.n_tokens
        else:
            self.patch_embed = None
            self.n_image_tokens = 0

    def forward(self, ids: torch.Tensor,
                image_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """ids: (B, T) -> logits (B, T, V).

        image_embeds: optional (B, N, d) visual prefix from PatchEmbed
        (or xiiv.vision.encode_image). When given, it SUBSTITUTES the
        first N token embeddings in place, so sequence length and the
        sampler's position accounting are unchanged. Requires
        cfg.image_enabled=True.
        """
        B, T = ids.shape
        x = self.embed(ids)
        if image_embeds is not None:
            if self.patch_embed is None:
                raise ValueError(
                    "image_embeds given but cfg.image_enabled=False")
            ie = torch.as_tensor(image_embeds)
            if ie.dim() == 2:
                ie = ie.unsqueeze(0)          # (N, d) -> (1, N, d)
            n = self.n_image_tokens
            if ie.shape[1] != n:
                raise ValueError(
                    f"expected {n} image tokens, got {ie.shape[1]}")
            if T < n:
                raise ValueError(
                    f"sequence length {T} shorter than image prefix {n}")
            x[:, :n] = ie.to(dtype=x.dtype)
        cos, sin = build_rope_cache(T, self.cfg.d_model // self.cfg.n_heads,
                                    self.cfg.rope_theta, ids.device,
                                    dtype=torch.float32)
        if self.grad_checkpoint and x.requires_grad:
            from torch.utils.checkpoint import checkpoint
            for layer in self.layers:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
        else:
            for layer in self.layers:
                x = layer(x, cos, sin)
        x = self.final_norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_weights(self, std: float = 0.02):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)
