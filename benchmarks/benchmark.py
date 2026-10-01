"""Honest efficiency benchmark: XIIV vs same-size AR baseline.

Measures (all wall-clock on CPU, 2 threads, near-greedy decoding):
  A. raw generation: 128 tokens — tok/s, wall time, forward passes, peak RSS
  B. infill task: 200 held-out 128-token blocks, 32-token masked span —
     masked-span recovery accuracy (exact token match), wall time,
     forward passes per block

Usage:
    python benchmarks/benchmark.py --model xiiv   # -> results_xiiv.json
    python benchmarks/benchmark.py --model ar       # -> results_ar.json
Each model runs in its own process so peak-RSS attribution is clean.

XIIV weights: checkpoints/best.pt (or --xiiv-ckpt). AR weights:
checkpoints_ar/best_ar.pt. Run AFTER both trainings finish.
"""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv import load_ckpt_trusted
from xiiv.diffusion import sample_trajectory
from benchmarks.ar_model import ARModel, generate_left_to_right

N_BLOCKS = 200        # validation blocks used for the infill task
SPAN = (48, 80)       # masked span inside each 128-token block (32 tokens)
GEN_LEN = 128
XIIV_STEPS = 16


class CountWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.cfg = getattr(m, "cfg", None)  # keep constraint-id validation live
        self.n = 0

    def forward(self, x):
        self.n += 1
        return self.m(x)


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xiiv", "ar"], required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--xiiv-ckpt", default="checkpoints/best.pt")
    ap.add_argument("--ar-ckpt", default="checkpoints_ar/best_ar.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = "cpu"

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    cfg = ModelConfig(vocab_size=meta["vocab_size"],
                      mask_id=meta["mask_id"], eos_id=meta["eos_id"],
                      seq_len=128)
    valid_mm = np.memmap(os.path.join(args.data, "valid.bin"), dtype=np.uint16,
                         mode="r", shape=(meta["valid_blocks"], 128))

    if args.model == "xiiv":
        base = XIIVModel(cfg)
        ck = load_ckpt_trusted(args.xiiv_ckpt)
        base.load_state_dict(ck["model"])
        sampler = "parallel de-masking"
    else:
        base = ARModel(cfg)
        ck = load_ckpt_trusted(args.ar_ckpt)
        base.load_state_dict(ck["model"])
        sampler = "left-to-right"
    base.eval()
    model = CountWrapper(base)
    n_params = base.count_parameters()
    print(f"[{args.model}] params={n_params} sampler={sampler} "
          f"rss_after_load={peak_rss_mb():.1f} MB", flush=True)

    res = {"model": args.model, "params": n_params, "threads": args.threads,
           "sampler": sampler}

    # ---- A. raw generation: 128 tokens ----
    model.n = 0
    t0 = time.perf_counter()
    if args.model == "xiiv":
        ids, _ = sample_trajectory(model, GEN_LEN, cfg.mask_id, cfg.eos_id,
                                   steps=XIIV_STEPS, temperature=1e-3,
                                   top_p=1.0, seed=7, device=device)
    else:
        prefix = torch.from_numpy(
            np.asarray(valid_mm[0][:8])).long()  # fixed 8-token prompt
        ids = generate_left_to_right(model, prefix, GEN_LEN - 8,
                                     temperature=0.0, seed=7, device=device)
    wall = time.perf_counter() - t0
    res["gen_128"] = {
        "wall_s": round(wall, 3),
        "tok_s": round(GEN_LEN / wall, 1),
        "forward_passes": model.n,
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    print(f"[{args.model}] gen128: {GEN_LEN/wall:.1f} tok/s, "
          f"{model.n} passes, {wall:.2f}s", flush=True)

    # ---- B. infill task: 200 blocks, 32-token span ----
    s0, s1 = SPAN
    span_len = s1 - s0
    correct, total = 0, 0
    model.n = 0
    t0 = time.perf_counter()
    for b in range(N_BLOCKS):
        truth = torch.from_numpy(np.asarray(valid_mm[b])).long()
        if args.model == "xiiv":
            noisy = truth.clone()
            noisy[s0:s1] = cfg.mask_id
            frozen = noisy != cfg.mask_id
            out, _ = sample_trajectory(
                model, 128, cfg.mask_id, cfg.eos_id, steps=XIIV_STEPS,
                temperature=1e-3, top_p=1.0, seed=1000 + b,
                prompt_ids=noisy, frozen=frozen, device=device)
            pred = out[s0:s1]
        else:
            # "prompted to infill": the AR model only ever sees the prefix
            # and generates the span left-to-right. It cannot see the
            # suffix — that asymmetry IS the point being measured.
            full = generate_left_to_right(model, truth[:s0], span_len,
                                          temperature=0.0, seed=1000 + b,
                                          device=device)
            pred = full[s0:s0 + span_len]
        correct += int((pred == truth[s0:s1]).sum())
        total += span_len
    wall = time.perf_counter() - t0
    res["infill"] = {
        "blocks": N_BLOCKS,
        "span_tokens": span_len,
        "recovery_accuracy": round(correct / total, 4),
        "correct_tokens": correct,
        "total_tokens": total,
        "wall_s_total": round(wall, 2),
        "wall_s_per_block": round(wall / N_BLOCKS, 3),
        "forward_passes_per_block": model.n // N_BLOCKS,
        "forward_passes_total": model.n,
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }
    print(f"[{args.model}] infill acc={correct/total:.4f} "
          f"({correct}/{total}), {wall/N_BLOCKS:.3f}s/block, "
          f"{model.n//N_BLOCKS} passes/block", flush=True)

    res["peak_rss_mb"] = round(peak_rss_mb(), 1)
    out = args.out or f"benchmarks/results_{args.model}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[{args.model}] wrote {out}")


if __name__ == "__main__":
    main()
