"""SGBD Phase 2 — independent block training with full gradients.

Trains ONE student block (a slice of the 1B model's layers) against cached
teacher activations from Phase 1 (xiiv/sgbd_capture.py).

Architecture mapping (documented design decision):
  - Student: 36 layers, d=1280 -> 8 blocks: [5,5,5,5,4,4,4,4] layers
  - Teacher: 15 layers, d=704  -> 8 groups: [2,2,2,2,2,2,2,1] layers
  - Block b input  = in_proj(teacher[t_start].input)    # 704 -> 1280
  - Block b target = out_proj(teacher[t_end-1].output)  # 704 -> 1280
  - Loss = MSE(block_out, target). Projections train jointly (FitNets-style
    scaffolding); they are discarded when blocks are composed in Phase 3.

Usage:
    .venv/bin/python xiiv/sgbd_train_block.py --block 0 --steps 200
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xiiv.config import ModelConfig
from xiiv.model import Block, build_rope_cache

REPO = Path(__file__).resolve().parent.parent
STUDENT_CKPT = REPO / "checkpoints_1b_gn" / "ckpt_8218.pt"

# student layer slices per block (36 layers -> 8 blocks)
BLOCKS = [(0, 5), (5, 10), (10, 15), (15, 20),
          (20, 24), (24, 28), (28, 32), (32, 36)]
# teacher layer groups per block (15 layers -> 8 groups)
T_GROUPS = [(0, 2), (2, 4), (4, 6), (6, 8),
            (8, 10), (10, 12), (12, 14), (14, 15)]

TEACHER_DIM = 704
STUDENT_DIM = 1280
SEQ_LEN = 128


def build_student_block(s_start, s_end, ckpt_sd):
    # NOTE: contract cross-attention is EXCLUDED from block distillation.
    # The 1B ckpt has contract_attn only on layers 0-26, with an older kv
    # shape (d->d) than current code (d->2d), and max |gate| ~0.03 — a
    # near-zero residual. Teacher capture ran with contract_embeds=None,
    # so targets are main-path only. Consistent: distill the main path.
    cfg = ModelConfig(
        vocab_size=32768, d_model=STUDENT_DIM, n_layers=36,
        n_heads=20, ffn_dim=3584, rope_theta=10000.0, norm_eps=1e-6,
        contract_enabled=False, tie_embeddings=True,
    )
    layers = nn.ModuleList()
    for i in range(s_start, s_end):
        blk = Block(cfg)
        prefix = f"layers.{i}."
        sub = {k[len(prefix):]: v for k, v in ckpt_sd.items()
               if k.startswith(prefix) and "contract_attn" not in k
               and "contract_norm" not in k}
        blk.load_state_dict(sub, strict=True)
        layers.append(blk)
    n_params = sum(p.numel() for p in layers.parameters())
    print(f"[train] student block layers [{s_start},{s_end}): "
          f"{n_params/1e6:.1f}M params")
    return layers, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cache-dir", type=str, default="sgbd_cache/iter0")
    ap.add_argument("--out-dir", type=str, default="sgbd_blocks")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--resume-step", type=int, default=0,
                    help="resume from out_dir/block{b}_step{resume_step}.pt "
                         "(restores layer weights, projections, optimizer, step)")
    args = ap.parse_args()

    b = args.block
    s_start, s_end = BLOCKS[b]
    t_start, t_end = T_GROUPS[b]
    cache_dir = REPO / args.cache_dir
    out_dir = REPO / args.out_dir / f"block{b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(STUDENT_CKPT, map_location="cpu", weights_only=False)
    layers, cfg = build_student_block(s_start, s_end, ckpt["model"])
    del ckpt
    gc.collect()

    # dimension scaffolding (trained jointly, discarded at composition)
    in_proj = nn.Linear(TEACHER_DIM, STUDENT_DIM)
    out_proj = nn.Linear(TEACHER_DIM, STUDENT_DIM)

    params = list(layers.parameters()) + list(in_proj.parameters()) \
        + list(out_proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"[train] trainable: {sum(p.numel() for p in params)/1e6:.1f}M")

    # RoPE tables for the student block
    head_dim = STUDENT_DIM // cfg.n_heads
    cos, sin = build_rope_cache(SEQ_LEN, head_dim, cfg.rope_theta,
                                "cpu", torch.float32)

    # chunk files for this block's teacher layers
    chunks = sorted((cache_dir).glob(f"layer{t_start:02d}_c*.pt"))
    n_chunks = len(chunks)
    assert n_chunks > 0, f"no cache chunks for teacher layer {t_start}"
    print(f"[train] {n_chunks} cache chunks; "
          f"teacher layers [{t_start},{t_end}) -> student [{s_start},{s_end})")

    def block_forward(x):
        for layer in layers:
            x = layer(x, cos, sin, None)
        return x

    layers.train()
    in_proj.train()
    out_proj.train()

    t0 = time.time()
    step = 0
    loss_hist = []

    # Resume from a step checkpoint (e.g. after OOM kill or reboot):
    # restores layer weights, scaffolding projections, optimizer state,
    # and the step counter so the deterministic chunk schedule continues.
    if args.resume_step > 0:
        resume_path = out_dir / f"block{b}_step{args.resume_step}.pt"
        assert resume_path.exists(), f"resume checkpoint missing: {resume_path}"
        r = torch.load(resume_path, map_location="cpu", weights_only=False)
        assert r["block"] == b, f"checkpoint block {r['block']} != {b}"
        for g_idx, sd in r["layers"]:
            layers[g_idx - s_start].load_state_dict(sd)
        in_proj.load_state_dict(r["in_proj"])
        out_proj.load_state_dict(r["out_proj"])
        opt.load_state_dict(r["opt"])
        step = int(r["step"])
        loss_hist = r.get("loss_hist", [])
        print(f"[train] resumed from {resume_path} at step {step}", flush=True)
        del r
        gc.collect()

    chunk_order = list(range(n_chunks))
    while step < args.steps:
        #reshuffle chunk order each epoch
        import random
        random.seed(1234 + step)
        random.shuffle(chunk_order)
        for c in chunk_order:
            if step >= args.steps:
                break
            din = torch.load(cache_dir / f"layer{t_start:02d}_c{c:03d}.pt",
                             map_location="cpu", weights_only=False)
            dout = torch.load(
                cache_dir / f"layer{t_end-1:02d}_c{c:03d}.pt",
                map_location="cpu", weights_only=False)
            t_in, t_out = din["input"], dout["output"]
            del din, dout
            n_seq = t_in.shape[0]
            perm = torch.randperm(n_seq)
            for s in range(0, n_seq, args.batch_size):
                if step >= args.steps:
                    break
                idx = perm[s:s + args.batch_size]
                opt.zero_grad()
                x = in_proj(t_in[idx])
                with torch.no_grad():
                    tgt = out_proj(t_out[idx])
                pred = block_forward(x)
                loss = torch.nn.functional.mse_loss(pred, tgt)
                loss.backward()
                opt.step()
                step += 1
                if step % args.log_every == 0:
                    el = time.time() - t0
                    print(f"[train] block {b} step {step}/{args.steps} "
                          f"loss={loss.item():.4f} ({step/el:.2f} st/s)",
                          flush=True)
                if step % args.save_every == 0:
                    torch.save({
                        "block": b,
                        "layers": [(s_start + i, layers[i].state_dict())
                                   for i in range(len(layers))],
                        "in_proj": in_proj.state_dict(),
                        "out_proj": out_proj.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                    }, out_dir / f"block{b}_step{step}.pt")
            del t_in, t_out
            gc.collect()
            loss_hist.append(loss.item())

    torch.save({
        "block": b,
        "layers": [(s_start + i, layers[i].state_dict())
                   for i in range(len(layers))],
        "in_proj": in_proj.state_dict(),
        "out_proj": out_proj.state_dict(),
        "step": step,
        "loss_hist": loss_hist,
    }, out_dir / f"block{b}_final.pt")
    print(f"[train] DONE block {b}: {step} steps in {time.time()-t0:.0f}s -> "
          f"{out_dir}")


if __name__ == "__main__":
    main()
