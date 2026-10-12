"""Train XIIV with the LLaDA-style masked-diffusion objective.

Usage:
    python -m xiiv.train --data data --ckpt ckpts --config-out ...

Checkpoints: ckpt_<step>.pt (full training state) + best.pt. A JSONL log
records loss/tokens/sec. Safe to interrupt and resume: the latest checkpoint
is picked up automatically.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from .config import ModelConfig
from .model import XIIVModel
from . import load_ckpt_trusted
from .diffusion import diffusion_loss, evaluate_nll_bound
from .run_config import RunConfig


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step; effective "
                         "batch = batch-size * grad-accum")
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adamw8bit"],
                    help="adamw8bit = bitsandbytes 8-bit AdamW "
                         "(~4x smaller optimizer states; for 6GB VRAM)")
    ap.add_argument("--seq-len", type=int, default=128)
    # Model size (defaults = 10M XIIV-10M config). For XIIV-50M use:
    # --d-model 576 --n-layers 12 --n-heads 9 --ffn-dim 1536 (~52.5M params).
    ap.add_argument("--d-model", type=int, default=320)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=5)
    ap.add_argument("--ffn-dim", type=int, default=864)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=20000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true",
                    help="use bfloat16 autocast on CPU (faster if AVX512-BF16)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (inductor, CPU)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every-min", type=float, default=30.0)
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-name", default=None,
                    help="run name for provenance (default: xiiv-<params>)")
    ap.add_argument("--dataset-id", default=None,
                    help="dataset snapshot id for provenance "
                         "(default: derived from --data)")
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda | auto (auto picks cuda when available)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="activation checkpointing: less VRAM, ~30% slower")
    ap.add_argument("--lock-prob", type=float, default=0.0,
                    help="constraint-aware training: fraction of sequences "
                         "with locked spans (never masked); 0.0 = classic")
    ap.add_argument("--init-ckpt", default=None,
                    help="load model weights from this checkpoint and "
                         "fine-tune (optimizer restarts fresh)")
    # --- guarantee-native pre-training ---
    ap.add_argument("--contract-enabled", action="store_true",
                    help="build the contract-native architecture (contract "
                         "encoder + per-block cross-attention)")
    ap.add_argument("--contract-prob", type=float, default=0.0,
                    help="fraction of batches trained contract-conditioned; "
                         "0.0 = classic unconditional training")
    ap.add_argument("--contract-adversarial-frac", type=float, default=0.3,
                    help="fraction of synthetic contracts that are "
                         "adversarial (rare-span, tight-count, ordering)")
    ap.add_argument("--contract-max-tokens", type=int, default=96,
                    help="max contract text tokens (longer -> ValueError "
                         "fallback to a trivial contract)")
    ap.add_argument("--tokenizer", default="weights/tokenizer.json",
                    help="tokenizer path (only loaded when "
                         "--contract-prob > 0)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: --device cuda requested but no GPU found.")
    g = torch.Generator().manual_seed(args.seed)
    # Fixed-seed validation generator: every val uses IDENTICAL corruption
    # masks, so val-to-val deltas measure the model, not mask luck.
    # (Unseeded masks made val swing ±0.4 nats between checks — larger
    # than real gains — and best.pt selection was luck-based.)
    val_g = torch.Generator().manual_seed(args.seed + 1_000_003)

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    # Special-token ids come from the trained tokenizer (recorded in
    # meta.json) — never from ModelConfig defaults.
    cfg = ModelConfig(vocab_size=meta["vocab_size"],
                      mask_id=meta["mask_id"], eos_id=meta["eos_id"],
                      seq_len=args.seq_len,
                      d_model=args.d_model, n_layers=args.n_layers,
                      n_heads=args.n_heads, ffn_dim=args.ffn_dim,
                      contract_enabled=args.contract_enabled)
    print(f"model config: ~{cfg.param_count_estimate()/1e6:.2f}M params "
          f"(d={cfg.d_model} L={cfg.n_layers} h={cfg.n_heads})"
          f"{' +contract-native' if args.contract_enabled else ''}")
    if args.contract_prob > 0 and not args.contract_enabled:
        raise SystemExit(
            "ERROR: --contract-prob > 0 needs --contract-enabled (the "
            "model has no contract encoder to condition on).")
    tokenizer = None
    if args.contract_prob > 0:
        from .tokenizer import XIIVTokenizer
        from .guarantee_train import sample_contracts, encode_contract_batch
        tokenizer = XIIVTokenizer(args.tokenizer)
        print(f"guarantee-native: contract_prob={args.contract_prob} "
              f"adversarial_frac={args.contract_adversarial_frac}")
    print(f"data: {meta['train_tokens']/1e6:.1f}M train tokens, "
          f"{meta['valid_tokens']/1e6:.1f}M valid tokens")

    train_mm = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16,
                         mode="r", shape=(meta["train_blocks"], args.seq_len))
    valid_mm = np.memmap(os.path.join(args.data, "valid.bin"), dtype=np.uint16,
                         mode="r", shape=(meta["valid_blocks"], args.seq_len))

    model = XIIVModel(cfg)
    model.init_weights()
    if args.init_ckpt:
        ck = load_ckpt_trusted(args.init_ckpt)
        model.load_state_dict(ck["model"])
        print(f"fine-tuning from {args.init_ckpt} "
              f"(step {ck.get('step', '?')})")
    model.to(device)
    model.grad_checkpoint = args.grad_checkpoint
    if args.compile:
        print("torch.compile ...")
        model = torch.compile(model)
    print(f"actual params: {model.count_parameters()/1e6:.2f}M")

    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise SystemExit(
                "adamw8bit needs bitsandbytes: pip install bitsandbytes")
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)
        print("optimizer: bitsandbytes AdamW8bit")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.95), weight_decay=args.weight_decay,
                                fused=(device.type == "cuda"))
    if device.type == "cuda":
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)
        amp = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({vram:.1f} GB), amp={amp_dtype}, "
              f"grad_checkpoint={args.grad_checkpoint}")
    else:
        amp = torch.amp.autocast("cpu", dtype=torch.bfloat16,
                                 enabled=args.bf16)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, "train_log.jsonl")

    # --- frozen run provenance (MLE workflow §3): one immutable record of
    # everything this run depends on, hashed and written before training ---
    # (run_name/dataset_id used to be hardcoded to the 10M TinyStories run —
    # every later run inherited wrong provenance until this was fixed.)
    rc = RunConfig(
        run_name=args.run_name or f"xiiv-{cfg.param_count_estimate()/1e6:.0f}m",
        dataset_snapshot_id=args.dataset_id or
        f"{os.path.basename(os.path.normpath(args.data))}-{meta.get('train_blocks', 0)}b",
        train_blocks=meta.get("train_blocks", 0),
        valid_blocks=meta.get("valid_blocks", 0),
        seq_len=args.seq_len, vocab_size=meta["vocab_size"],
        mask_id=meta["mask_id"], eos_id=meta["eos_id"],
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        optimizer=args.optimizer, total_steps=args.total_steps,
        lr=args.lr, lr_min=args.lr_min, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        bf16=args.bf16, seed=args.seed, threads=args.threads,
    ).with_provenance(
        os.path.dirname(os.path.abspath(__file__)), torch.__version__)
    rc.save(os.path.join(args.ckpt_dir, "run_config.json"))
    print(f"run_config hash={rc.config_hash} code_sha256={rc.code_sha256[:12]}…")

    # --- resume ---
    step, best_val, tokens_seen = 0, float("inf"), 0
    ckpts = sorted(
        (f for f in os.listdir(args.ckpt_dir)
         if f.startswith("ckpt_") and f.endswith(".pt")),
        key=lambda f: int(f.split("_")[1].split(".")[0]))
    if ckpts:
        ck = load_ckpt_trusted(os.path.join(args.ckpt_dir, ckpts[-1]))
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            # grown checkpoint (scripts/grow_checkpoint.py): optimizer
            # restarts fresh; step/RNG state still continue the lineage
            print("grown checkpoint: optimizer starts fresh")
        step, best_val = ck["step"], ck["best_val"]
        tokens_seen = ck.get("tokens_seen", 0)
        if "rng_state" in ck:                      # exact data-stream resume
            g.set_state(ck["rng_state"])
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"])
        if "np_rng" in ck:
            np.random.set_state(ck["np_rng"])
        print(f"resumed from {ckpts[-1]} (step {step}, best_val {best_val:.4f})")
    if os.path.exists(os.path.join(args.ckpt_dir, "best.pt")) and best_val == float("inf"):
        ck = load_ckpt_trusted(os.path.join(args.ckpt_dir, "best.pt"))
        best_val = ck.get("val_nll", best_val)

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
    lr_scale = 1.0  # halved on non-finite recovery; args.lr stays frozen

    while step < args.total_steps:
        lr = lr_scale * cosine_schedule(step, args.warmup_steps,
                                        args.total_steps,
                                        args.lr, args.lr_min)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        # gradient accumulation: grad_accum micro-batches make one
        # optimizer step; effective batch = batch-size * grad_accum.
        step_loss, step_mask, n_cond = 0.0, 0.0, 0
        bad_micro = False
        for _ in range(args.grad_accum):
            batch = get_batch(train_mm, args.batch_size, args.seq_len,
                              device, g)
            # guarantee-native: a fraction of batches is
            # contract-conditioned. Contracts are synthesized from the
            # CLEAN batch (pre-masking), so every clause is satisfiable
            # by the target by construction.
            contract_ids = None
            if tokenizer is not None and \
                    torch.rand((), generator=g).item() < args.contract_prob:
                contracts = sample_contracts(
                    batch, tokenizer, meta["vocab_size"], cfg.mask_id,
                    cfg.eos_id, g,
                    adversarial_frac=args.contract_adversarial_frac)
                contract_ids = encode_contract_batch(
                    contracts, tokenizer, args.contract_max_tokens, device)
                n_cond += 1
            with amp:
                loss, info = diffusion_loss(model, batch, cfg.mask_id,
                                            cfg.t_min, cfg.t_max,
                                            lock_prob=args.lock_prob,
                                            contract_ids=contract_ids)
            if not torch.isfinite(loss):
                bad_micro = True
                break
            (loss / args.grad_accum).backward()
            step_loss += loss.item() / args.grad_accum
            step_mask += info["mask_ratio"] / args.grad_accum
        conditioned = n_cond > 0
        if bad_micro:
            nonfinite_streak += 1
            print(f"[step {step}] NON-FINITE LOSS (streak {nonfinite_streak})"
                  f" — restoring best.pt, halving LR", flush=True)
            if nonfinite_streak >= 20:
                print("aborting: 20 consecutive non-finite losses",
                      flush=True)
                logf.write(json.dumps({"aborted": "non-finite loss",
                                       "step": step}) + "\n")
                break
            ck = load_ckpt_trusted(os.path.join(args.ckpt_dir, "best.pt")) \
                if os.path.exists(os.path.join(args.ckpt_dir, "best.pt")) \
                else None
            if ck is not None:
                model.load_state_dict(ck["model"])
            # halve the LR scale — args.lr/args.lr_min stay frozen so the
            # run config provenance never drifts from what actually ran.
            lr_scale *= 0.5
            print(f"halved LR (scale now {lr_scale})", flush=True)
            continue
        nonfinite_streak = 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        step += 1
        eff_batch = args.batch_size * args.grad_accum
        tokens_seen += eff_batch * args.seq_len

        if step % args.log_every == 0:
            dt = time.time() - t0
            tps = args.log_every * eff_batch * args.seq_len / dt
            t0 = time.time()
            rec = {"step": step, "loss": round(step_loss, 4), "lr": lr,
                   "tok_s": round(tps, 1),
                   "mask_ratio": round(step_mask, 3),
                   "tokens_seen_M": round(tokens_seen / 1e6, 2),
                   "contract": conditioned}
            vram_str = ""
            if device.type == "cuda":
                peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
                rec["peak_vram_GB"] = round(peak_vram_gb, 2)
                vram_str = f" | vram {peak_vram_gb:.2f}GB"
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:6d} | loss {step_loss:.4f} | "
                  f"{tps:7.1f} tok/s | {tokens_seen/1e6:6.1f}M tok | "
                  f"lr {lr:.1e}{vram_str}", flush=True)

        if step % args.val_every == 0:
            model.eval()
            vb = [get_batch(valid_mm, args.batch_size, args.seq_len,
                            device, g) for _ in range(args.val_batches)]
            val_nll = evaluate_nll_bound(model, vb, cfg.mask_id,
                                         cfg.t_min, cfg.t_max, device,
                                         generator=val_g)
            model.train()
            improved = val_nll < best_val
            if improved:
                best_val = val_nll
                save("best.pt", val_nll)
            print(f"[val] step {step}: NLL bound {val_nll:.4f} nats/tok "
                  f"{'(new best)' if improved else ''}", flush=True)
            logf.write(json.dumps({"step": step, "val_nll": round(val_nll, 4),
                                   "best": improved}) + "\n")
            logf.flush()

        if (time.time() - last_ckpt_time) / 60 >= args.ckpt_every_min:
            save(f"ckpt_{step}.pt")
            last_ckpt_time = time.time()
            # keep only the 3 newest rolling checkpoints
            olds = sorted(
                (f for f in os.listdir(args.ckpt_dir)
                 if f.startswith("ckpt_") and f.endswith(".pt")),
                key=lambda f: int(f.split("_")[1].split(".")[0]))[:-3]
            for f in olds:
                os.remove(os.path.join(args.ckpt_dir, f))

    save(f"ckpt_{step}.pt")
    print(f"done: step {step}, tokens_seen {tokens_seen/1e6:.1f}M, "
          f"best val NLL {best_val:.4f}")


if __name__ == "__main__":
    main()
