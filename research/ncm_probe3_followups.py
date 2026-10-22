"""NCM Probe 3 — the documented fair follow-ups (2026-10-22).

Probe 2 (research/ncm_probe2_results.md) closed with four documented
next attacks. This probe runs #1, #2 and #4; #3 is deferred with its
reason recorded in the results file.

  #1 MATCHED-CONDITION HEAD TRAINING. Probes 1-2 trained the coarse
     head on hidden states from a 50%-masked canvas WITH teacher
     concepts present; at inference the head sees a 25%-visible canvas
     with concepts UNKNOWN. Variant A computes the head loss from a
     dedicated pass under exact inference conditions (25% visible,
     concepts unknown), token loss unchanged.
  #2 END-TO-END SOFT CONDITIONING. Probe 2's soft variant blended
     embeddings at inference into a hard-trained model (OOD inputs —
     an unfair test, recorded at the time). Variant B trains with soft
     conditioning: pass 1 infers a concept DISTRIBUTION from the
     visible view, pass 2 denoises conditioned on the probability-
     weighted concept embedding, gradients flowing through both.
  #4 VISIBLE-FRACTION SWEEP. Concept accuracy and token accuracy at
     visible fractions {0.125, 0.25, 0.5, 0.75} for flat, probe-1 C2F,
     A and B — pricing the evidence limit directly.

Setup identical to probes 1-2 by construction (imports probe 1's
data/model/training; same seeds; 500 steps; bs 64; eval n=256,
16 denoise steps headline). Baselines re-run in-process.
"""
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
REPO = os.path.expanduser("~/workspace/ai-tools/corsac")
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location(
    "ncm1", os.path.join(REPO, "research/ncm_prototype_probe.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

VOCAB, MASK, LEN, BLOCK, NBLOCK = P.VOCAB, P.MASK, P.LEN, P.BLOCK, P.NBLOCK
K, D = P.K_TEMPLATES, P.D
UNKNOWN = K  # concept id for "unknown"


def visible_view(x, gen, frac=0.25):
    vis = torch.rand(x.shape, generator=gen) < frac
    return torch.where(vis, x, torch.full_like(x, MASK))


def pooled(h, bs):
    return h.reshape(bs, NBLOCK, BLOCK, D).mean(dim=2)


# ---------------- Variant A: matched-condition head ----------------
def train_variant_a(steps=500, bs=64, lr=3e-3):
    gen = torch.Generator().manual_seed(1234)
    m = P.TinyDenoiser(True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        x, block_ids = P.make_batch(bs, gen)
        mask = torch.rand(bs, LEN) < 0.5
        xm = torch.where(mask, torch.full_like(x, MASK), x)
        concepts = block_ids.clone()
        drop = torch.rand(bs, NBLOCK) < 0.25
        concepts = torch.where(drop, torch.full_like(concepts, UNKNOWN),
                               concepts)
        logits, _ = m(xm, concepts)
        loss = F.cross_entropy(logits[mask], x[mask])
        # head pass under EXACT inference conditions
        xv = visible_view(x, gen)
        unk = torch.full((bs, NBLOCK), UNKNOWN, dtype=torch.long)
        _, hv = m(xv, unk)
        closs = F.cross_entropy(
            m.coarse_head(pooled(hv, bs)).reshape(-1, K),
            block_ids.reshape(-1))
        loss = loss + 0.5 * closs
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


# ---------------- Variant B: end-to-end soft ----------------
class SoftDenoiser(P.TinyDenoiser):
    def forward_soft(self, x, soft_probs):
        """soft_probs: (B, NBLOCK, K) distribution over concepts."""
        h = self.emb(x) + self.pos.weight[:x.size(1)].unsqueeze(0)
        w = self.concept_emb.weight[:K]              # (K, D)
        ce = soft_probs @ w                           # (B, NBLOCK, D)
        h = h + ce.repeat_interleave(BLOCK, dim=1)
        h = self.trunk(h)
        return self.head(h), h

    def infer_soft(self, x_visible):
        bs = x_visible.size(0)
        unk = torch.full((bs, NBLOCK), UNKNOWN, dtype=torch.long)
        _, h = self(x_visible, unk)
        logits = self.coarse_head(pooled(h, bs))
        return logits.softmax(-1), logits


def train_variant_b(steps=500, bs=64, lr=3e-3):
    gen = torch.Generator().manual_seed(1234)
    m = SoftDenoiser(True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(steps):
        x, block_ids = P.make_batch(bs, gen)
        xv = visible_view(x, gen)
        probs, clogits = m.infer_soft(xv)
        mask = torch.rand(bs, LEN) < 0.5
        xm = torch.where(mask, torch.full_like(x, MASK), x)
        logits, _ = m.forward_soft(xm, probs.detach())
        loss = F.cross_entropy(logits[mask], x[mask])
        closs = F.cross_entropy(clogits.reshape(-1, K),
                                block_ids.reshape(-1))
        loss = loss + 0.5 * closs
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def denoise_accuracy_soft(m, n_steps, n=256, seed=99, visible_frac=0.25):
    gen = torch.Generator().manual_seed(seed)
    x_true, block_ids = P.make_batch(n, gen)
    visible = torch.rand(n, LEN, generator=gen) < visible_frac
    x = torch.where(visible, x_true, torch.full_like(x_true, MASK))
    remaining = ~visible
    probs, _ = m.infer_soft(x)
    concept_acc = (probs.argmax(-1) == block_ids).float().mean().item()
    per_step = max(1, int(remaining.sum(dim=1).float().mean().item())
                   // max(1, n_steps))
    for _ in range(n_steps):
        logits, _ = m.forward_soft(x, probs)
        p = logits.softmax(-1)
        conf, pred = p.max(-1)
        conf = torch.where(remaining, conf, torch.full_like(conf, -1.0))
        k = min(per_step, int(remaining.sum(dim=1).max().item()))
        if k <= 0:
            break
        idx = conf.topk(k, dim=1).indices
        x.scatter_(1, idx, pred.gather(1, idx))
        remaining.scatter_(1, idx, False)
        if not remaining.any():
            break
    if remaining.any():
        logits, _ = m.forward_soft(x, probs)
        x = torch.where(remaining, logits.argmax(-1), x)
    acc = (x[~visible] == x_true[~visible]).float().mean().item()
    return acc, concept_acc


def main():
    t0 = time.time()
    print("training FLAT ...", flush=True)
    flat, _ = P.train_model(False)
    print("training C2F (probe-1 recipe) ...", flush=True)
    c2f, _ = P.train_model(True)
    print("training Variant A (matched-condition head) ...", flush=True)
    var_a = train_variant_a()
    print("training Variant B (end-to-end soft) ...", flush=True)
    var_b = train_variant_b()

    print("\n== headline (visible 0.25, 16 steps) ==", flush=True)
    fa, _ = P.denoise_accuracy(flat, False, 16)
    ca, cacc = P.denoise_accuracy(c2f, True, 16)
    aa, aacc = P.denoise_accuracy(var_a, True, 16)
    ba, bacc = denoise_accuracy_soft(var_b, 16)
    oa, _ = P.denoise_accuracy(c2f, 2, 16)
    print(f"flat={fa:.3f}  c2f-hard={ca:.3f} (concept {cacc:.3f})  "
          f"A={aa:.3f} (concept {aacc:.3f})  B={ba:.3f} "
          f"(concept {bacc:.3f})  oracle={oa:.3f}", flush=True)

    print("\n== visible-fraction sweep (16 steps) ==", flush=True)
    print(f"{'frac':>6} {'flat':>6} {'c2f':>6} {'cAcc':>6} {'A':>6} "
          f"{'aAcc':>6} {'B':>6} {'bAcc':>6} {'oracle':>7}", flush=True)
    for frac in (0.125, 0.25, 0.5, 0.75):
        fa, _ = P.denoise_accuracy(flat, False, 16, visible_frac=frac)
        ca, cc = P.denoise_accuracy(c2f, True, 16, visible_frac=frac)
        aa, ac = P.denoise_accuracy(var_a, True, 16, visible_frac=frac)
        ba, bc = denoise_accuracy_soft(var_b, 16, visible_frac=frac)
        oa, _ = P.denoise_accuracy(c2f, 2, 16, visible_frac=frac)
        print(f"{frac:>6.3f} {fa:>6.3f} {ca:>6.3f} {cc:>6.3f} {aa:>6.3f} "
              f"{ac:>6.3f} {ba:>6.3f} {bc:>6.3f} {oa:>7.3f}", flush=True)
    print(f"elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
