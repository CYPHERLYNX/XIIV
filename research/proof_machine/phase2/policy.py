"""
Phase 2, step 1 — learned composition policy (supervised warmup).

A tiny encoder-decoder transformer (d=128, 2+2 layers, ~550K params — the
TIPS scale, arXiv:2610.22099) mapping a spec string to a primitive-op
sequence. Trained by supervised imitation of the deterministic VRC planner
(trajectory-contraction supervision, Sakurai et al. arXiv:2610.31576).

Input:  spec string, character-level (canon form, spaces stripped).
Output: op sequence over the 8-op vocabulary (+BOS/EOS/PAD).

No GPU. CPU-only, pure torch.
"""
import math

import torch
import torch.nn as nn

OPS = ["ADD_1", "ADD_3", "ADD_7", "MUL_2", "MUL_5", "MOD_7", "MOD_13", "NEG"]
SPEC_CHARS = list("0123456789x+-*()mod")

SRC_PAD, SRC_BOS, SRC_EOS = 0, 1, 2
TGT_PAD, TGT_BOS, TGT_EOS = 0, 1, 2

SRC_VOCAB = ["<pad>", "<bos>", "<eos>"] + SPEC_CHARS
TGT_VOCAB = ["<pad>", "<bos>", "<eos>"] + OPS
SRC2ID = {c: i for i, c in enumerate(SRC_VOCAB)}
TGT2ID = {t: i for i, t in enumerate(TGT_VOCAB)}


def encode_spec(spec):
    """Spec string -> token ids (BOS + chars + EOS). Raises KeyError on
    characters outside the vocabulary (loud, never silently dropped)."""
    s = spec.replace(" ", "")
    return [SRC_BOS] + [SRC2ID[c] for c in s] + [SRC_EOS]


def encode_ops(ops):
    """Op list -> token ids (BOS + ops + EOS)."""
    return [TGT_BOS] + [TGT2ID[o] for o in ops] + [TGT_EOS]


def decode_ops(ids):
    """Token ids -> op list. Stops at EOS; drops BOS/PAD."""
    out = []
    for i in ids:
        if i == TGT_EOS:
            break
        if i in (TGT_PAD, TGT_BOS):
            continue
        out.append(TGT_VOCAB[i])
    return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Policy(nn.Module):
    """Tiny encoder-decoder transformer: spec chars -> op sequence."""

    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_ff=256,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(len(SRC_VOCAB), d_model,
                                    padding_idx=SRC_PAD)
        self.tgt_emb = nn.Embedding(len(TGT_VOCAB), d_model,
                                    padding_idx=TGT_PAD)
        self.pos = PositionalEncoding(d_model)
        self.tf = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.out = nn.Linear(d_model, len(TGT_VOCAB))
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt_in, src_pad_mask=None, tgt_pad_mask=None):
        src_e = self.pos(self.src_emb(src))
        tgt_e = self.pos(self.tgt_emb(tgt_in))
        tgt_len = tgt_in.size(1)
        causal = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool),
                            diagonal=1)
        out = self.tf(src_e, tgt_e, tgt_mask=causal,
                      src_key_padding_mask=src_pad_mask,
                      tgt_key_padding_mask=tgt_pad_mask,
                      memory_key_padding_mask=src_pad_mask)
        return self.out(out)

    @torch.no_grad()
    def greedy(self, spec_ids, max_len=24):
        """Greedy decode for one spec (list of ids). Returns op list."""
        self.eval()
        src = torch.tensor([spec_ids], dtype=torch.long)
        src_pad = (src == SRC_PAD)
        src_e = self.pos(self.src_emb(src))
        memory = self.tf.encoder(src_e, src_key_padding_mask=src_pad)
        tgt = torch.tensor([[TGT_BOS]], dtype=torch.long)
        for _ in range(max_len):
            tgt_e = self.pos(self.tgt_emb(tgt))
            causal = torch.triu(
                torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool),
                diagonal=1)
            dec = self.tf.decoder(tgt_e, memory, tgt_mask=causal)
            logits = self.out(dec[:, -1, :])
            nxt = int(torch.argmax(logits, dim=-1).item())
            if nxt == TGT_EOS:
                break
            tgt = torch.cat([tgt, torch.tensor([[nxt]])], dim=1)
        return decode_ops(tgt[0].tolist()[1:])


def count_params(model):
    return sum(p.numel() for p in model.parameters())
