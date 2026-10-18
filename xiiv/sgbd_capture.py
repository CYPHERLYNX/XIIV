"""SGBD Phase 1 — real teacher activation capture.

Loads the 133M teacher (checkpoints_100m_gn/best.pt), registers forward
hooks on all 15 transformer blocks, streams token blocks from data_all
(8k vocab, matches the teacher), and saves per-layer (input, output)
activation pairs to disk for block-distillation training.

Usage:
    .venv/bin/python xiiv/sgbd_capture.py --n-sequences 128 --batch-size 16
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xiiv.config import ModelConfig
from xiiv.model import XIIVModel

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints_100m_gn" / "best.pt"
DATA = REPO / "data_all" / "train.bin"
SEQ_LEN = 128


def load_teacher():
    cfg = ModelConfig(
        vocab_size=8192, mask_id=2, eos_id=3,
        d_model=704, n_layers=15, n_heads=11, ffn_dim=2048,
        rope_theta=10000.0, norm_eps=1e-6,
        contract_enabled=True, tie_embeddings=True,
    )
    model = XIIVModel(cfg)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"[capture] teacher loaded: {model.count_parameters()/1e6:.1f}M params, "
          f"{len(model.layers)} layers, ckpt step {ckpt['step']}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sequences", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-dir", type=str, default="sgbd_cache/iter0")
    ap.add_argument("--data", type=str, default=str(DATA))
    args = ap.parse_args()

    cache_dir = REPO / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = load_teacher()
    n_layers = len(model.layers)

    # --- register hooks: capture (input, output) of every block ---
    acts = {}

    def make_hook(idx):
        def hook(module, inp, out):
            # inp[0] is x (B,T,d); out is x (B,T,d)
            acts[idx] = (inp[0].detach().cpu(), out.detach().cpu())
        return hook

    handles = [layer.register_forward_hook(make_hook(i))
               for i, layer in enumerate(model.layers)]

    # --- stream data, flushing each chunk to disk (bounded RAM) ---
    CHUNK_SEQS = 64  # sequences per chunk file per layer
    raw = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_blocks_avail = len(raw) // SEQ_LEN
    assert args.n_sequences <= n_blocks_avail, \
        f"asked {args.n_sequences}, only {n_blocks_avail} available"
    print(f"[capture] streaming {args.n_sequences} sequences "
          f"(batch {args.batch_size}, chunk {CHUNK_SEQS}) from {args.data}",
          flush=True)

    n_chunks = (args.n_sequences + CHUNK_SEQS - 1) // CHUNK_SEQS
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for c in range(n_chunks):
            c_start = c * CHUNK_SEQS
            c_end = min(c_start + CHUNK_SEQS, args.n_sequences)
            # per-layer accumulators for this chunk only
            layer_ins = [[] for _ in range(n_layers)]
            layer_outs = [[] for _ in range(n_layers)]
            for start in range(c_start, c_end, args.batch_size):
                bs = min(args.batch_size, c_end - start)
                blk = raw[start * SEQ_LEN:(start + bs) * SEQ_LEN]
                ids = torch.from_numpy(np.array(blk, dtype=np.int64)
                                       .reshape(bs, SEQ_LEN))
                acts.clear()
                model(ids)  # forward populates acts via hooks
                assert len(acts) == n_layers, \
                    f"hooks missed: {len(acts)}/{n_layers}"
                for i in range(n_layers):
                    inp, out = acts[i]
                    layer_ins[i].append(inp)
                    layer_outs[i].append(out)
                done += bs
                del ids
            # flush chunk to disk, one file per layer
            for i in range(n_layers):
                ins = torch.cat(layer_ins[i], dim=0)
                outs = torch.cat(layer_outs[i], dim=0)
                torch.save({"input": ins, "output": outs},
                           cache_dir / f"layer{i:02d}_c{c:03d}.pt")
                del ins, outs
            del layer_ins, layer_outs
            gc.collect()
            el = time.time() - t0
            print(f"[capture] chunk {c+1}/{n_chunks} done "
                  f"({done}/{args.n_sequences} seqs, {done/el:.1f} seq/s)",
                  flush=True)

    for h in handles:
        h.remove()

    meta = {
        "teacher_ckpt": str(CKPT),
        "teacher_params_M": round(model.count_parameters() / 1e6, 1),
        "n_layers": n_layers,
        "n_sequences": args.n_sequences,
        "batch_size": args.batch_size,
        "seq_len": SEQ_LEN,
        "data": args.data,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[capture] DONE in {time.time()-t0:.0f}s -> {cache_dir}")


if __name__ == "__main__":
    main()
