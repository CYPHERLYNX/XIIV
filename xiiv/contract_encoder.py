"""Contract encoder: the contract as a first-class model input.

A Contract is serialized to its canonical text form (Contract.to_text()),
tokenized with the main tokenizer, and encoded by a small transformer into
a sequence of contract embeddings. The main diffusion transformer attends
to these embeddings via cross-attention in every block.

This is what makes the model contract-*native* rather than
contract-*instructed*: the guarantee shapes the computation itself instead
of arriving as prompt text the model may or may not heed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RMSNorm


class _EncoderLayer(nn.Module):
    """Vanilla pre-norm transformer layer (learned positions, no RoPE:
    contract sequences are short)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = RMSNorm(d_model, eps)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.mlp_norm = RMSNorm(d_model, eps)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x, key_padding_mask=None):
        # key_padding_mask accepted for API compatibility; this torch
        # build's SDPA has no such kwarg and the encoder never pads.
        B, T, D = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        x = x + self.attn_out(y)
        h = self.mlp_norm(x)
        x = x + self.fc2(F.silu(self.fc1(h)))
        return x


class ContractEncoder(nn.Module):
    """Contract token ids -> contract embeddings (B, C, d_model)."""

    def __init__(self, cfg):
        super().__init__()
        d_enc = cfg.contract_d_model
        self.tok_embed = nn.Embedding(cfg.vocab_size, d_enc)
        self.pos_embed = nn.Embedding(cfg.contract_max_tokens, d_enc)
        self.layers = nn.ModuleList([
            _EncoderLayer(d_enc, cfg.contract_n_heads,
                          cfg.contract_ffn_dim, cfg.norm_eps)
            for _ in range(cfg.contract_n_layers)
        ])
        self.norm = RMSNorm(d_enc, cfg.norm_eps)
        self.proj = nn.Linear(d_enc, cfg.d_model, bias=False)

    def forward(self, contract_ids: torch.Tensor) -> torch.Tensor:
        # (B, C) -> (B, C, d_model)
        B, C = contract_ids.shape
        pos = torch.arange(C, device=contract_ids.device).unsqueeze(0)
        x = self.tok_embed(contract_ids) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        return self.proj(self.norm(x))


def encode_contract(contract, tokenizer, max_tokens: int) -> torch.Tensor:
    """Contract -> (1, C) token-id tensor, truncated/padded to max_tokens.

    Padding uses the tokenizer's mask id (never attended as content is
    not needed: the encoder has no padding mask, so callers should keep
    contracts short; overlong contracts raise instead of silently
    truncating meaning away).
    """
    text = contract.to_text()
    ids = [t for t in tokenizer.encode(text)
           if t not in (tokenizer.mask_id, tokenizer.eos_id)]
    if len(ids) > max_tokens:
        raise ValueError(
            f"contract text too long ({len(ids)} > {max_tokens} tokens); "
            "simplify the contract")
    return torch.tensor([ids], dtype=torch.long)
