#!/usr/bin/env python3
"""Initialize 500M's untrained embeddings from parent 8k subwords.

Each new 32k token (ids 8192-32767) is a BPE merge of 8k subwords.
Its embedding = mean of the parent subword embeddings.
This is instant and far better than random init (standard technique).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizers import Tokenizer

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel

PARENT_VOCAB = 8192


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_500m_gn/ckpt_8218.pt")
    ap.add_argument("--tok8k", default="weights/tokenizer.json")
    ap.add_argument("--tok32k", default="weights/tokenizer_32k.json")
    ap.add_argument("--out", default="checkpoints_500m_emb/ckpt_500m_emb_init.pt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading tokenizers...")
    tok8k = Tokenizer.from_file(args.tok8k)
    tok32k = Tokenizer.from_file(args.tok32k)

    print("Loading 500M checkpoint...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**json.loads(ckpt["config"]))
    with torch.device("meta"):
        model = XIIVModel(cfg)
    model.to_empty(device="cpu")
    model.load_state_dict(ckpt["model"], strict=True, assign=True)
    emb = model.embed.weight.float()
    print(f"Embedding shape: {emb.shape}")

    print("Initializing new embeddings from parent subwords...")
    n_init = 0
    for tid in range(PARENT_VOCAB, 32768):
        text = tok32k.decode([tid])
        sub_ids = tok8k.encode(text).ids
        # Filter to valid parent ids (exclude new tokens to avoid recursion)
        sub_ids = [s for s in sub_ids if 0 <= s < PARENT_VOCAB]
        if sub_ids:
            with torch.no_grad():
                emb[tid] = emb[sub_ids].mean(dim=0)
            n_init += 1
        if (tid - PARENT_VOCAB) % 5000 == 0:
            print(f"  {tid - PARENT_VOCAB}/24576", flush=True)

    print(f"Initialized {n_init}/24576 embeddings")
    # lm_head is tied — shares storage, auto-updated

    torch.save({
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": ckpt["config"],
        "step": ckpt.get("step", 8218),
        "emb_init": "parent_subword_mean",
        "grown_from": ckpt.get("grown_from"),
    }, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
