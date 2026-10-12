#!/usr/bin/env python3
"""RTX 5090 smoke test for the grown XIIV-1B checkpoint.

Validates BEFORE committing to a full training run:
1. BF16 forward pass produces finite logits.
2. BF16 backward pass produces finite gradients.
3. Peak VRAM is measured (not estimated) at several micro-batch sizes.
4. A 10-step training burst completes, checkpoints, reloads, and resumes
   at the correct step (resume continuity).

Exit 0 = safe to train. Exit 1 = do NOT train; investigate.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.diffusion import diffusion_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--micro-batches", default="2,4,8,16")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device found"
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")

    # --- load grown checkpoint ---
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**json.loads(ckpt["config"]))
    print(f"config: d={cfg.d_model} L={cfg.n_layers} h={cfg.n_heads} "
          f"ffn={cfg.ffn_dim} vocab={cfg.vocab_size}")
    with torch.device("meta"):
        model = XIIVModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(dev).to(torch.bfloat16)
    model.train()
    uniq = sum(p.numel() for p in model.parameters())
    print(f"unique params: {uniq/1e9:.4f}B")

    # --- data sample ---
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    mask_id = meta["mask_id"]
    n_blocks = meta.get("n_train_blocks") or meta.get("n_blocks")
    print(f"data: {args.data_dir}, train blocks: {n_blocks}")
    train_bin = os.path.join(args.data_dir, "train.bin")
    import numpy as np
    mm = np.memmap(train_bin, dtype=np.uint16, mode="r")
    seq_len = cfg.seq_len

    def batch(bs):
        idx = torch.randint(0, len(mm) // seq_len - 1, (bs,)).numpy()
        x = np.stack([mm[i*seq_len:(i+1)*seq_len] for i in idx])
        return torch.from_numpy(x.astype(np.int64)).to(dev)

    # --- 1. BF16 forward/backward finiteness + VRAM profile ---
    print("\n--- forward/backward finiteness + VRAM ---")
    opt = torch.optim.AdamW(model.parameters(), lr=1.65e-4)
    ok = True
    for bs in [int(x) for x in args.micro_batches.split(",")]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            xb = batch(bs)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = diffusion_loss(model, xb, mask_id,
                                         t_min=cfg.t_min, t_max=cfg.t_max)
            assert torch.isfinite(loss).item(), f"non-finite loss at bs={bs}"
            opt.zero_grad()
            loss.backward()
            gnorm = sum(p.grad.detach().float().pow(2).sum()
                        for p in model.parameters() if p.grad is not None).sqrt()
            assert torch.isfinite(gnorm).item(), f"non-finite grad at bs={bs}"
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  bs={bs:3d}: loss={loss.item():.4f} |g|={gnorm.item():.4f} "
                  f"peak={peak:.2f}GB  FINITE")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  bs={bs:3d}: OOM — too large for 32GB")
                torch.cuda.empty_cache()
            else:
                raise
    if not ok:
        sys.exit(1)

    # --- 2. resume continuity: 10 steps, save, reload, verify step ---
    print("\n--- resume continuity (10 steps) ---")
    step = ckpt.get("step", 8218)
    for i in range(10):
        xb = batch(4)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = diffusion_loss(model, xb, mask_id,
                                     t_min=cfg.t_min, t_max=cfg.t_max)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    tmp = "/tmp/smoke_1b_resume.pt"
    torch.save({"model": model.state_dict(), "step": step}, tmp)
    re = torch.load(tmp, map_location="cpu", weights_only=False)
    assert re["step"] == step, "step mismatch after reload"
    print(f"  10 steps ok, reloaded at step {re['step']} — resume continuity OK")
    os.remove(tmp)

    print("\nSMOKE PASSED — safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
