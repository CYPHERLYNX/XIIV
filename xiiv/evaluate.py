"""Honest evaluation harness for XIIV.

Reports ONLY measured numbers:
  * held-out masked-LM NLL upper bound (nats/token) on a frozen valid split
  * generation throughput (tok/s) + peak RSS on this machine's CPU
  * parameter count, weight file size

Usage: python -m xiiv.evaluate --weights weights/xiiv.safetensors ...
"""
import argparse
import json
import os
import resource
import time

import numpy as np
import torch

from .config import ModelConfig
from .sample import load_model
from .tokenizer import XIIVTokenizer
from .diffusion import evaluate_nll_bound, sample_trajectory


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cpu_name():
    try:
        info = open("/proc/cpuinfo").read()
        return info.split("model name")[1].split(":")[1].split("\n")[0].strip()
    except Exception:
        import platform
        return f"{platform.processor() or platform.machine()} ({platform.system()})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/xiiv.safetensors")
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--val-batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--gen-steps", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    rss0 = peak_rss_mb()
    tok = XIIVTokenizer(args.tokenizer)
    model, cfg = load_model(args.weights)
    n_params = model.count_parameters()
    w_mb = os.path.getsize(args.weights) / 1e6

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    valid_mm = np.memmap(os.path.join(args.data, "valid.bin"), dtype=np.uint16,
                         mode="r",
                         shape=(meta["valid_blocks"], meta["seq_len"]))
    g = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(args.val_batches):
        idx = torch.randint(0, len(valid_mm), (args.batch_size,), generator=g)
        batches.append(torch.from_numpy(np.asarray(valid_mm[idx.numpy()])).long())

    t0 = time.time()
    nll = evaluate_nll_bound(model, batches, cfg.mask_id, cfg.t_min, cfg.t_max)
    nll_time = time.time() - t0

    throughput = {}
    for steps in args.gen_steps:
        t0 = time.time()
        ids, conf = sample_trajectory(model, args.gen_length, tok.mask_id,
                                      tok.eos_id, steps=steps, seed=123,
                                      temperature=1.0, top_p=0.95)
        dt = time.time() - t0
        n_tok = int((ids != tok.mask_id).sum())
        throughput[f"steps_{steps}"] = {
            "tok_s": round(n_tok / dt, 1),
            "seconds": round(dt, 2),
            "tokens": n_tok,
        }

    results = {
        "params": n_params,
        "params_M": round(n_params / 1e6, 2),
        "weights_file_MB": round(w_mb, 1),
        "weights_dtype": str(next(model.parameters()).dtype),
        "val_nll_bound_nats_per_token": round(nll, 4),
        "val_note": ("upper bound on true sequence NLL from the masked "
                     "objective; NOT comparable to autoregressive perplexity"),
        "val_batches": args.val_batches,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "gen_length": args.gen_length,
        "throughput": throughput,
        "peak_rss_MB": round(peak_rss_mb(), 1),
        "baseline_rss_MB": round(rss0, 1),
        "cpu": cpu_name(),
    }
    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
