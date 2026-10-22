#!/usr/bin/env python3
"""Selective-looping training-efficiency probe (XIIV stack).

Question (from Looped Uniform Diffusion, arXiv:2610.30066, and LoopMDM,
arXiv:2605.26106 — authors' claims, unverified): does re-applying a few
early layers within each training forward reach a target validation
NLL with fewer total training FLOPs than the non-looped model?

Arms (identical init seed, data order, optimizer, step budget):
  base     — no looping (layer-pass ratio 1.0)
  fixed    — loop_group=(0,2), loop_count=1 every step (ratio 1.5)
  sampled  — loop_count drawn per step from {0:.25, 1:.5, 2:.25}
             (expected ratio 1.5): the papers' count-sampling recipe

Data: structured synthetic corpus — sequences of 4 blocks drawn from
16 templates of 8 tokens + 10% token noise (same family as the NCM
probes, so results are comparable in kind). Metric: masked-token CE
on a fixed validation set with a fixed eval mask seed, every 50 steps.
FLOP accounting is analytic layer-pass counting (uniform layer cost),
labeled as such — not wall-clock.
"""
import json
import time

import torch
import torch.nn.functional as F

from xiiv.config import ModelConfig
from xiiv.diffusion import mask_batch
from xiiv.model import XIIVModel

VOCAB_TOKENS = 64
SEQ = 32
STEPS = 300
BATCH = 32
EVAL_EVERY = 50


def make_corpus(n, gen):
    templates = torch.randint(0, VOCAB_TOKENS, (16, 8), generator=gen)
    seqs = []
    for _ in range(n):
        blocks = templates[torch.randint(0, 16, (4,), generator=gen)]
        s = blocks.reshape(-1).clone()
        noise = torch.rand(SEQ, generator=gen) < 0.10
        s[noise] = torch.randint(0, VOCAB_TOKENS, (int(noise.sum()),),
                                 generator=gen)
        seqs.append(s)
    return torch.stack(seqs)


def eval_nll(model, val, mask_id):
    model.eval()
    g = torch.Generator().manual_seed(1234)
    losses = []
    with torch.no_grad():
        for i in range(0, len(val), BATCH):
            batch = val[i:i + BATCH]
            noisy, targets, mask = mask_batch(batch, mask_id, generator=g)
            logits = model(noisy)
            losses.append(F.cross_entropy(logits[mask], targets[mask]).item())
    model.train()
    return sum(losses) / len(losses)


def train_arm(name, loop_mode, train, val, cfg):
    torch.manual_seed(7)  # identical init across arms
    model = XIIVModel(cfg)
    model.init_weights()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    gen = torch.Generator().manual_seed(99)  # identical data order
    ratio = XIIVModel.selective_loop_flop_ratio(cfg.n_layers, (0, 2), 1)
    curve = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(train), (BATCH,), generator=gen)
        batch = train[idx]
        noisy, targets, mask = mask_batch(batch, cfg.mask_id, generator=gen)
        if loop_mode == "fixed":
            lc, r = 1, ratio
        elif loop_mode == "sampled":
            u = torch.rand((), generator=gen).item()
            lc = 0 if u < 0.25 else (1 if u < 0.75 else 2)
            r = XIIVModel.selective_loop_flop_ratio(cfg.n_layers, (0, 2), lc)
        else:
            lc, r = 0, 1.0
        logits = model(noisy, loop_group=(0, 2), loop_count=lc)
        loss = F.cross_entropy(logits[mask], targets[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            curve.append({"step": step, "ratio": r,
                          "cum_passes": None,
                          "val_nll": eval_nll(model, val, cfg.mask_id)})
    wall = time.time() - t0
    # cumulative layer-passes: recompute deterministically per arm type
    passes_per_step = {"base": 1.0, "fixed": ratio, "sampled": 1.5}
    for pt in curve:
        pt["cum_passes"] = pt["step"] * passes_per_step[loop_mode] \
            * cfg.n_layers
    return {"arm": name, "wall_s": round(wall, 1), "curve": curve}


def main():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                      ffn_dim=128, mask_id=254, eos_id=255, seq_len=SEQ)
    gen = torch.Generator().manual_seed(2026)
    train = make_corpus(1024, gen)
    val = make_corpus(256, gen)
    results = [train_arm(n, n, train, val, cfg)
               for n in ("base", "fixed", "sampled")]
    base_final = results[0]["curve"][-1]["val_nll"]
    print(f"baseline final val NLL: {base_final:.4f}")
    for res in results:
        c = res["curve"]
        hit = next((p for p in c if p["val_nll"] <= base_final), None)
        print(f"{res['arm']:8s} final {c[-1]['val_nll']:.4f} "
              f"wall {res['wall_s']}s | first step reaching baseline "
              f"final: {hit['step'] if hit else 'never'} "
              f"(cum layer-passes {hit['cum_passes'] if hit else '-'})")
    out = {"baseline_final_val_nll": base_final, "arms": results,
           "steps": STEPS, "batch": BATCH,
           "note": "FLOPs = analytic layer-pass counts, not wall-clock; "
                   "toy corpus, tiny model — directional only."}
    with open("research/selective_looping_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote research/selective_looping_results.json")


if __name__ == "__main__":
    main()
