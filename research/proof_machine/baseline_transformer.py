"""
Transformer baseline for the kill gate (spec section 5.3).

Standard decoder-only transformer, trained end-to-end on (x0, S) -> x_L pairs
with the explicit op sequence in the input -- it must learn composition
IMPLICITLY through its layers. Digit-level I/O with exact-integer-match scoring.

Size: ~350K params (d=96, 3 layers, 3 heads, ffn=384). This is ~5x the proof
machine's total neural params (~70K), deliberately erring AGAINST the proof
machine: the pre-registered concern is an under-parameterized baseline, so the
baseline is given the parameter advantage. If it still degrades with depth
while the proof machine holds flat, the difference is architectural.

Training distribution: depths 1..16 (spec), same rejection-sampled generator
as the proof machine (taskgen), infinite fresh data. Test depths 1..100 probe
out-of-distribution depth generalization -- the compositionality wall.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskgen import gen_instance
from certificates import OP_NAMES

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
PAD, BOS, EOS, SEP = 0, 1, 2, 3
DIG0 = 4            # '0'..'9' -> 4..13
MINUS = 14
OP_BASE = 15        # 8 ops -> 15..22
VOCAB = 23
MAX_LEN = 128

OP_ID = {op: OP_BASE + i for i, op in enumerate(OP_NAMES)}


def enc_number(n):
    toks = []
    if n < 0:
        toks.append(MINUS)
        n = -n
    toks += [DIG0 + int(d) for d in str(n)]
    return toks


def enc_input(x0, ops):
    return [BOS] + enc_number(x0) + [SEP] + [OP_ID[o] for o in ops] + [SEP]


def enc_target(xL):
    return enc_number(xL) + [EOS]


def dec_number(toks):
    """Token list (no EOS) -> int. Returns None if malformed."""
    try:
        neg = False
        digits = []
        for t in toks:
            if t == MINUS and not digits and not neg:
                neg = True
            elif DIG0 <= t <= DIG0 + 9:
                digits.append(str(t - DIG0))
            else:
                return None
        if not digits:
            return None
        v = int("".join(digits))
        return -v if neg else v
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalMHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.dh).transpose(1, 2)
        k = k.view(B, T, self.heads, self.dh).transpose(1, 2)
        v = v.view(B, T, self.heads, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.att = CausalMHA(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, d=96, layers=3, heads=3, ffn=384):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.blocks = nn.ModuleList([Block(d, heads, ffn) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, ids):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_batch(batch_size, rng, lo_depth=1, hi_depth=16):
    seqs, masks, Ls = [], [], []
    while len(seqs) < batch_size:
        L = rng.randint(lo_depth, hi_depth)
        inst = gen_instance(L, rng)
        if inst is None:
            continue
        x0, ops, xL, _ = inst
        inp = enc_input(x0, ops)
        tgt = enc_target(xL)
        full = inp + tgt
        if len(full) > MAX_LEN:
            continue
        mask = [0] * len(inp) + [1] * len(tgt)
        seqs.append(full)
        masks.append(mask)
        Ls.append(len(full))
    T = max(Ls)
    ids = torch.tensor([s + [PAD] * (T - len(s)) for s in seqs], dtype=torch.long)
    # labels: next-token prediction; loss only on target positions
    labels = torch.full_like(ids, -100)
    attn = (ids != PAD).long()
    for i, (s, m) in enumerate(zip(seqs, masks)):
        for j in range(len(s) - 1):
            if m[j + 1]:
                labels[i, j] = s[j + 1]
    return ids, labels, attn


@torch.no_grad()
def greedy_decode(model, x0, ops, device, max_new=12):
    ids = torch.tensor([enc_input(x0, ops)], dtype=torch.long, device=device)
    out = []
    for _ in range(max_new):
        logits = model(ids)
        nxt = int(logits[0, -1].argmax().item())
        if nxt == EOS or nxt == PAD:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return dec_number(out)


@torch.no_grad()
def eval_exact_match(model, depths, n_per_depth, seed, device, batch_note=""):
    model.eval()
    rng = random.Random(seed)
    res = {}
    for L in depths:
        good = total = 0
        while total < n_per_depth:
            inst = gen_instance(L, rng)
            if inst is None:
                continue
            x0, ops, xL, _ = inst
            pred = greedy_decode(model, x0, ops, device)
            if pred == xL:
                good += 1
            total += 1
        res[L] = good / total
        print(f"  L={L:3d} A_TF={res[L]*100:6.2f}% {batch_note}", flush=True)
    model.train()
    return res


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(max_steps=10000, batch_size=64, lr=3e-4, val_every=400,
          patience=5, seed=7, device="cpu"):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = DecoderTransformer().to(device)
    print(f"baseline params: {model.n_params()}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    best_val = -1.0
    bad = 0
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "checkpoints", "baseline_tf.pt")
    t_start = time.time()
    for step in range(1, max_steps + 1):
        ids, labels, _ = make_batch(batch_size, rng)
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        # labels[i,j] already holds the next-token target for position j
        # (see make_batch); -100 elsewhere (input/pad positions).
        loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                               labels.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            el = time.time() - t_start
            print(f"step {step:5d} loss {loss.item():.4f} ({el/60:.1f} min)",
                  flush=True)
        if step % val_every == 0:
            val = eval_exact_match(model, [1, 2, 4, 8, 16], 400, seed=1000 + step,
                                   device=device, batch_note=f"[val step {step}]")
            vmean = sum(val.values()) / len(val)
            print(f"step {step}: val exact-match (depths 1-16) = {vmean*100:.2f}%",
                  flush=True)
            if vmean > best_val + 1e-4:
                best_val = vmean
                bad = 0
                torch.save(model.state_dict(), ckpt)
                print(f"  saved best ({best_val*100:.2f}%) -> {ckpt}", flush=True)
            else:
                bad += 1
                if bad >= patience:
                    print(f"early stop at step {step} (best val {best_val*100:.2f}%)",
                          flush=True)
                    break
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    ckpt = os.path.join(base, "checkpoints", "baseline_tf.pt")

    if args.train:
        train(max_steps=args.max_steps, device=args.device)
    if args.eval:
        torch.set_num_threads(4)
        model = DecoderTransformer().to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=args.device,
                                         weights_only=True))
        model.eval()
        depths = [1, 2, 4, 8, 16, 32, 64, 100]
        res = eval_exact_match(model, depths, 1000, seed=777, device=args.device)
        out = {"depths": depths, "n_per_depth": 1000,
               "A_TF": {str(k): v for k, v in res.items()}}
        path = os.path.join(base, "results",
                            f"baseline_{time.strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"A_TF(100) = {res[100]*100:.2f}%", flush=True)
        print(f"results -> {path}", flush=True)


if __name__ == "__main__":
    main()
