"""AR baseline for XIIV: a same-size causal (GPT-style) transformer.

This exists for exactly one reason: a fair fight. Same architecture
dimensions as XIIVModel (d=320, L=6, h=5, SwiGLU-864, RoPE, RMSNorm, tied
embeddings -> identical parameter count), trained on the SAME tokenized
TinyStories data with the SAME optimizer/schedule/budget — the only
difference is the training objective (next-token prediction with a causal
mask) instead of masked diffusion.

The point it proves: XIIV isn't competing on fluency. It's competing on
*infilling* — a task the AR baseline cannot do natively, and a decoding
regime (parallel) the AR baseline cannot use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from xiiv.config import ModelConfig
from xiiv.model import RMSNorm, SwiGLU, build_rope_cache, apply_rope


class CausalAttention(nn.Module):
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
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # THE difference from XIIVModel: causal mask. Left-to-right only.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out(y)


class CausalBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class ARModel(nn.Module):
    """Same-size autoregressive baseline. Identical param count to XIIVModel."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([CausalBlock(cfg)
                                     for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.embed(ids)
        cos, sin = build_rope_cache(T, self.cfg.d_model // self.cfg.n_heads,
                                    self.cfg.rope_theta, ids.device,
                                    dtype=torch.float32)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_weights(self, std: float = 0.02):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)


@torch.no_grad()
def generate_left_to_right(model, prefix_ids: torch.Tensor, n_tokens: int,
                           temperature: float = 0.0, seed: int | None = None,
                           device: str = "cpu",
                           pass_counter: list | None = None) -> torch.Tensor:
    """Greedy (temp=0) left-to-right generation, no KV cache — same
    no-cache setup as XIIV, so the forward-pass comparison is apples
    to apples. Returns the full sequence (prefix + generated)."""
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    model.eval()
    ids = prefix_ids.clone().to(device)
    for _ in range(n_tokens):
        logits = model(ids.unsqueeze(0)).squeeze(0)  # (T, V)
        if pass_counter is not None:
            pass_counter[0] += 1
        nxt = logits[-1]
        if temperature > 0:
            probs = torch.softmax(nxt / temperature, dim=-1)
            tok = torch.multinomial(probs, 1, generator=g).item()
        else:
            tok = int(nxt.argmax().item())
        ids = torch.cat([ids, torch.tensor([tok], device=device)])
    return ids
