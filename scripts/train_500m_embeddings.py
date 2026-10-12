#!/usr/bin/env python3
"""Train only the 500M's untrained embeddings (vocab 8192-32767).

The 500M was grown from the 133M (vocab 8k -> 32k). The 24,934 new token
embeddings are random — full-vocab output is garbled. This script freezes
the entire backbone and trains ONLY those embeddings on real data.

Memory: ~2.5GB (fp16 model + fp32 Adam on embeddings only).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.diffusion import diffusion_loss

PARENT_VOCAB = 8192  # embeddings [8192:] are untrained


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_500m_gn/ckpt_8218.pt")
    ap.add_argument("--data-dir", default="data_fineweb")
    ap.add_argument("--out-dir", default="checkpoints_500m_emb")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=32,
                    help="shorter seq = faster CPU backward (4x speedup vs 128)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading 500M checkpoint...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**json.loads(ckpt["config"]))
    # Meta-device init to avoid 2x memory (random init + checkpoint)
    with torch.device("meta"):
        model = XIIVModel(cfg)
    model.to_empty(device="cpu")
    model.load_state_dict(ckpt["model"], strict=True, assign=True)
    # Stash metadata needed for the final checkpoint BEFORE freeing the dict
    ckpt_config = ckpt["config"]
    ckpt_step = ckpt.get("step", 8218)
    ckpt_grown_from = ckpt.get("grown_from")
    del ckpt  # free checkpoint dict memory
    model.to(torch.float16)
    model.eval()

    # Freeze everything except the embedding matrix
    for p in model.parameters():
        p.requires_grad_(False)
    model.embed.weight.requires_grad_(True)

    n_trainable = model.embed.weight[PARENT_VOCAB:].numel()
    print(f"Trainable: {n_trainable/1e6:.1f}M params "
          f"(embeddings [{PARENT_VOCAB}:])")

    opt = torch.optim.Adam([model.embed.weight], lr=args.lr)

    # Data
    train_bin = os.path.join(args.data_dir, "train.bin")
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    seq_len = args.seq_len
    # Data is stored as 128-length blocks; slice to seq_len for faster CPU
    store_seq = 128
    n_blocks = meta.get("train_blocks",
                        os.path.getsize(train_bin) // (2 * store_seq))
    mm = np.memmap(train_bin, dtype=np.uint16, mode="r",
                   shape=(n_blocks, store_seq))
    mask_id = meta["mask_id"]
    print(f"Data: {n_blocks} blocks")

    g = torch.Generator().manual_seed(args.seed)
    step = 0
    t0 = time.time()

    resume_ckpt = os.path.join(args.out_dir, "emb_latest.pt")
    if os.path.exists(resume_ckpt):
        r = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.embed.weight.data.copy_(r["embed_weight"])
        opt.load_state_dict(r["opt"])
        step = r["step"]
        print(f"Resumed from step {step}")

    model.zero_grad()
    while step < args.steps:
        idx = torch.randint(0, n_blocks, (args.batch_size,), generator=g)
        batch = torch.from_numpy(np.asarray(mm[idx.numpy()])).long()
        batch = batch[:, :args.seq_len]  # slice to target length

        loss, _info = diffusion_loss(model, batch, mask_id, generator=g)

        opt.zero_grad()
        loss.backward()

        # Zero grads for parent embeddings — only train the new ones
        with torch.no_grad():
            if model.embed.weight.grad is not None:
                model.embed.weight.grad[:PARENT_VOCAB].zero_()

        torch.nn.utils.clip_grad_norm_([model.embed.weight], 1.0)
        opt.step()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step}/{args.steps} loss={loss.item():.4f} "
                  f"({dt:.0f}s)", flush=True)

        if step % args.save_every == 0:
            torch.save({
                "embed_weight": model.embed.weight.detach().cpu(),
                "opt": opt.state_dict(),
                "step": step,
            }, resume_ckpt)

    # Final checkpoint with trained embeddings (lm_head tied — stays in sync)
    print("Writing final checkpoint...")
    final_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model": final_sd,
        "config": ckpt_config,
        "step": ckpt_step,
        "emb_trained_steps": step,
        "grown_from": ckpt_grown_from,
    }, os.path.join(args.out_dir, "ckpt_500m_emb_trained.pt"))
    print("DONE")


if __name__ == "__main__":
    main()
