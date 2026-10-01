"""Train the AR baseline: same data, same budget, same size as XIIV.

Fair-fight checklist (must match xiiv/train.py):
  - same tokenized data (data/train.bin, 36.5M tokens)
  - same batch size (48 x 128) and total steps (5000)
  - same optimizer: AdamW lr 3e-4 -> 3e-5, cosine, warmup 500, wd 0.05
  - same seed (1337) -> identical batch sampling order
  - same bf16 autocast, grad clip 1.0, 2 threads
The ONLY difference: next-token cross-entropy with a causal mask
instead of the masked-diffusion objective.

Checkpoints go to checkpoints_ar/ (never touch checkpoints/).

Usage:
    python benchmarks/train_ar.py --data data --ckpt-dir checkpoints_ar --bf16
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from xiiv.config import ModelConfig
from xiiv import load_ckpt_trusted
from benchmarks.ar_model import ARModel


def get_batch(memmap, batch_size, seq_len, device, generator):
    idx = torch.randint(0, len(memmap), (batch_size,), generator=generator)
    batch = torch.from_numpy(np.asarray(memmap[idx.numpy()])).long()
    return batch.to(device)


def cosine_schedule(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    p = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))


def ar_loss(model, ids):
    """Standard next-token cross-entropy."""
    logits = model(ids)
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                           ids[:, 1:].reshape(-1))


@torch.no_grad()
def evaluate_nll(model, batches, device):
    """Held-out next-token NLL (nats/token). NOTE: not comparable to XIIV's
    masked-diffusion NLL bound — different objectives. Reported for
    completeness, never ranked against the diffusion number."""
    model.eval()
    total, n = 0.0, 0
    for ids in batches:
        ids = ids.to(device)
        logits = model(ids)
        total += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1), reduction="sum").item()
        n += ids[:, 1:].numel()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt-dir", default="checkpoints_ar")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=5000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every-min", type=float, default=30.0)
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    g = torch.Generator().manual_seed(args.seed)

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    cfg = ModelConfig(vocab_size=meta["vocab_size"],
                      mask_id=meta["mask_id"], eos_id=meta["eos_id"],
                      seq_len=args.seq_len)
    print(f"[AR baseline] ~{cfg.param_count_estimate()/1e6:.2f}M params "
          f"(d={cfg.d_model} L={cfg.n_layers} h={cfg.n_heads}), causal")

    train_mm = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16,
                         mode="r", shape=(meta["train_blocks"], args.seq_len))
    valid_mm = np.memmap(os.path.join(args.data, "valid.bin"), dtype=np.uint16,
                         mode="r", shape=(meta["valid_blocks"], args.seq_len))

    model = ARModel(cfg)
    model.init_weights()
    model.to(device)
    print(f"[AR baseline] actual params: {model.count_parameters()/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)
    amp = torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=args.bf16)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, "train_log.jsonl")

    step, best_val, tokens_seen = 0, float("inf"), 0
    ckpts = sorted(
        (f for f in os.listdir(args.ckpt_dir)
         if f.startswith("ckpt_") and f.endswith(".pt")),
        key=lambda f: int(f.split("_")[1].split(".")[0]))
    if ckpts:
        ck = load_ckpt_trusted(os.path.join(args.ckpt_dir, ckpts[-1]))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best_val = ck["step"], ck["best_val"]
        tokens_seen = ck.get("tokens_seen", 0)
        if "rng_state" in ck:
            g.set_state(ck["rng_state"])
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"])
        if "np_rng" in ck:
            np.random.set_state(ck["np_rng"])
        print(f"resumed from {ckpts[-1]} (step {step})")

    def save(name, val_nll=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "best_val": best_val,
                    "tokens_seen": tokens_seen, "config": cfg.to_json(),
                    "val_nll": val_nll,
                    "rng_state": g.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state()},
                   os.path.join(args.ckpt_dir, name))

    logf = open(log_path, "a")
    last_ckpt_time = time.time()
    model.train()
    t0 = time.time()
    nonfinite_streak = 0

    while step < args.total_steps:
        lr = cosine_schedule(step, args.warmup_steps, args.total_steps,
                             args.lr, args.lr_min)
        for pg in opt.param_groups:
            pg["lr"] = lr

        batch = get_batch(train_mm, args.batch_size, args.seq_len, device, g)
        opt.zero_grad(set_to_none=True)
        with amp:
            loss = ar_loss(model, batch)
        if not torch.isfinite(loss):
            nonfinite_streak += 1
            print(f"[step {step}] NON-FINITE LOSS (streak {nonfinite_streak})",
                  flush=True)
            if nonfinite_streak >= 20:
                logf.write(json.dumps({"aborted": "non-finite",
                                       "step": step}) + "\n")
                break
            args.lr *= 0.5
            args.lr_min *= 0.5
            continue
        nonfinite_streak = 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        step += 1
        tokens_seen += args.batch_size * args.seq_len

        if step % args.log_every == 0:
            dt = time.time() - t0
            tps = args.log_every * args.batch_size * args.seq_len / dt
            t0 = time.time()
            logf.write(json.dumps(
                {"step": step, "loss": round(loss.item(), 4), "lr": lr,
                 "tok_s": round(tps, 1),
                 "tokens_seen_M": round(tokens_seen / 1e6, 2)}) + "\n")
            logf.flush()
            print(f"[AR] step {step:6d} | loss {loss.item():.4f} | "
                  f"{tps:7.1f} tok/s | {tokens_seen/1e6:6.1f}M tok | "
                  f"lr {lr:.1e}", flush=True)

        if step % args.val_every == 0:
            vb = [get_batch(valid_mm, args.batch_size, args.seq_len,
                            device, g) for _ in range(args.val_batches)]
            val_nll = evaluate_nll(model, vb, device)
            model.train()
            improved = val_nll < best_val
            if improved:
                best_val = val_nll
                save("best_ar.pt", val_nll)
            print(f"[AR val] step {step}: NLL {val_nll:.4f} nats/tok "
                  f"{'(new best)' if improved else ''}", flush=True)
            logf.write(json.dumps({"step": step, "val_nll": round(val_nll, 4),
                                   "best": improved}) + "\n")
            logf.flush()

        if (time.time() - last_ckpt_time) / 60 >= args.ckpt_every_min:
            save(f"ckpt_{step}.pt")
            last_ckpt_time = time.time()
            olds = sorted(
                (f for f in os.listdir(args.ckpt_dir)
                 if f.startswith("ckpt_") and f.endswith(".pt")),
                key=lambda f: int(f.split("_")[1].split(".")[0]))[:-3]
            for f in olds:
                os.remove(os.path.join(args.ckpt_dir, f))

    save(f"ckpt_{step}.pt")
    print(f"[AR] done: step {step}, tokens_seen {tokens_seen/1e6:.1f}M, "
          f"best val NLL {best_val:.4f}")


if __name__ == "__main__":
    main()
