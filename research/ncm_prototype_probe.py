"""NCM prototype probe — coarse-to-fine vs flat masked diffusion.

Question (from research scout 2026-10-20): does generating text
coarse-to-fine (predict block-level "concept" codes first, then tokens
conditioned on them) reach a given quality with fewer denoising steps /
less training than flat masked diffusion?

Design (prototype-grade, synthetic):
  - Data: sequences of length 32 over vocab 64, built from 4 blocks of 8.
    Each block is drawn from one of K=8 block templates (a fixed random
    token pattern) + per-position noise (15% resample). So sequences have
    real global structure a coarse code can capture.
  - FLAT: tiny transformer, masked-diffusion objective (mask 50% of
    tokens, predict them). Inference: iterative unmasking by confidence.
  - C2F (NCM-style): same backbone + a coarse head. Training: backbone
    sees the TRUE block-template id as an extra embedding per block
    (teacher-forced concepts). Inference stage 1: predict each block's
    template id with the coarse head (from a fully-masked canvas +
    partial unmasking); stage 2: denoise tokens conditioned on the
    predicted codes.
  - Metric: token accuracy of the final canvas vs denoising steps, and
    train loss at matched optimizer steps.

Honest scope: tiny synthetic data, one seed family, CPU. This tests the
MECHANISM (does concept conditioning help denoising converge faster),
not NCM's published benchmark claims.
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
VOCAB = 64
MASK = 63
LEN = 32
BLOCK = 8
NBLOCK = LEN // BLOCK
K_TEMPLATES = 8
D = 128
DEVICE = "cpu"

# ---------------- data ----------------
TEMPLATES = torch.randint(0, 60, (K_TEMPLATES, BLOCK))


def make_batch(n, gen):
    block_ids = torch.randint(0, K_TEMPLATES, (n, NBLOCK), generator=gen)
    x = TEMPLATES[block_ids].reshape(n, LEN).clone()
    noise = torch.rand(n, LEN, generator=gen) < 0.15
    repl = torch.randint(0, 60, (n, LEN), generator=gen)
    x = torch.where(noise, repl, x)
    return x, block_ids


# ---------------- model ----------------
class TinyDenoiser(nn.Module):
    def __init__(self, use_concepts: bool):
        super().__init__()
        self.use_concepts = use_concepts
        self.emb = nn.Embedding(VOCAB + 1, D)
        self.pos = nn.Embedding(LEN, D)
        if use_concepts:
            self.concept_emb = nn.Embedding(K_TEMPLATES + 1, D)
        layer = nn.TransformerEncoderLayer(
            D, 4, 256, batch_first=True, dropout=0.0)
        self.trunk = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(D, VOCAB)
        if use_concepts:
            self.coarse_head = nn.Linear(D, K_TEMPLATES)

    def forward(self, x, concepts=None):
        h = self.emb(x) + self.pos.weight[: x.size(1)].unsqueeze(0)
        if self.use_concepts and concepts is not None:
            # concepts: (B, NBLOCK) ids, K_TEMPLATES = unknown
            ce = self.concept_emb(concepts)  # (B, NBLOCK, D)
            ce = ce.repeat_interleave(BLOCK, dim=1)
            h = h + ce
        h = self.trunk(h)
        return self.head(h), h


def train_model(use_concepts, steps=500, bs=64, lr=3e-3):
    gen = torch.Generator().manual_seed(1234)
    m = TinyDenoiser(use_concepts).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        x, block_ids = make_batch(bs, gen)
        mask = torch.rand(bs, LEN) < 0.5
        xm = torch.where(mask, torch.full_like(x, MASK), x)
        concepts = None
        if use_concepts:
            # teacher-force true concepts, but hide 25% of blocks
            # (concept dropout so inference-time predicted concepts work)
            concepts = block_ids.clone()
            drop = torch.rand(bs, NBLOCK) < 0.25
            concepts = torch.where(
                drop, torch.full_like(concepts, K_TEMPLATES), concepts)
        logits, h = m(xm, concepts)
        loss = F.cross_entropy(logits[mask], x[mask])
        if use_concepts:
            # train the coarse head: predict each block's true template id
            # from mean-pooled block hidden states (full canvas visible).
            hb = h.reshape(bs, NBLOCK, BLOCK, D).mean(dim=2)
            closs = F.cross_entropy(
                m.coarse_head(hb).reshape(-1, K_TEMPLATES),
                block_ids.reshape(-1))
            loss = loss + 0.5 * closs
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return m, losses


@torch.no_grad()
def denoise_accuracy(m, use_concepts, n_steps, n=256, seed=99,
                     visible_frac=0.25):
    """Inpainting eval: visible_frac of tokens stay visible; the model
    must fill the rest. C2F first infers block concepts from the visible
    evidence, then denoises conditioned on them."""
    gen = torch.Generator().manual_seed(seed)
    x_true, block_ids = make_batch(n, gen)
    visible = torch.rand(n, LEN, generator=gen) < visible_frac
    x = torch.where(visible, x_true, torch.full_like(x_true, MASK))
    remaining = ~visible
    concepts = None
    if use_concepts == 2:  # oracle: true concepts (upper bound)
        concepts = block_ids.clone()
        concept_acc = 1.0
    elif use_concepts:
        _, h = m(x, torch.full((n, NBLOCK), K_TEMPLATES, dtype=torch.long))
        hb = h.reshape(n, NBLOCK, BLOCK, D).mean(dim=2)
        concepts = m.coarse_head(hb).argmax(-1)
        concept_acc = (concepts == block_ids).float().mean().item()
    else:
        concept_acc = None
    per_step = max(1, int(remaining.sum(dim=1).float().mean().item())
                   // max(1, n_steps))
    for s in range(n_steps):
        logits, _ = m(x, concepts)
        probs = logits.softmax(-1)
        conf, pred = probs.max(-1)
        conf = torch.where(remaining, conf, torch.full_like(conf, -1.0))
        k = min(per_step, int(remaining.sum(dim=1).max().item()))
        if k <= 0:
            break
        idx = conf.topk(k, dim=1).indices
        x.scatter_(1, idx, pred.gather(1, idx))
        remaining.scatter_(1, idx, False)
        if not remaining.any():
            break
    if remaining.any():  # fill the rest in one shot
        logits, _ = m(x, concepts)
        x = torch.where(remaining, logits.argmax(-1), x)
    acc = (x[~visible] == x_true[~visible]).float().mean().item()
    return acc, concept_acc



def main():
    t0 = time.time()
    print("training FLAT ...")
    flat, flat_losses = train_model(False)
    print("training C2F ...")
    c2f, c2f_losses = train_model(True)
    print(f"train loss (last 50 avg): flat={sum(flat_losses[-50:]) / 50:.3f} "
          f"c2f={sum(c2f_losses[-50:]) / 50:.3f}")
    print(f"{'steps':>6} {'flat_acc':>9} {'c2f_acc':>9}")
    for steps in (2, 4, 8, 16):
        fa, _ = denoise_accuracy(flat, False, steps)
        ca, cacc = denoise_accuracy(c2f, True, steps)
        oa, _ = denoise_accuracy(c2f, 2, steps)
        print(f"{steps:>6} {fa:>9.3f} {ca:>9.3f}   (concept acc {cacc:.3f}, oracle {oa:.3f})")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
