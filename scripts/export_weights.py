"""Export the best training checkpoint to weights/ as safetensors + config."""
import argparse
import json
import os
import sys

# allow running as `python scripts/export_weights.py` from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import torch
from safetensors.torch import save_model

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv import load_ckpt_trusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--fp16", action="store_true",
                    help="store weights in float16 (halves file size)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ck = load_ckpt_trusted(args.ckpt)
    cfg = ModelConfig.from_json(ck["config"])
    model = XIIVModel(cfg)
    # checkpoints may come from torch.compile (keys prefixed "_orig_mod.")
    state = {k.replace("_orig_mod.", ""): v
             for k, v in ck["model"].items()}
    model.load_state_dict(state, strict=True)
    if args.fp16:
        model = model.half()
    out = os.path.join(args.out_dir, "xiiv.safetensors")
    # save_model (not save_file): tied embeddings share storage between
    # embed.weight and lm_head.weight, and save_file refuses shared tensors.
    save_model(model, out)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())
    # copy the tokenizer next to the weights: inference needs both, offline
    import shutil
    tok_src = "data/tokenizer.json"
    if os.path.exists(tok_src):
        shutil.copy(tok_src, os.path.join(args.out_dir, "tokenizer.json"))
    print(f"exported {out} ({os.path.getsize(out)/1e6:.1f} MB), "
          f"val NLL {ck.get('val_nll', float('nan')):.4f}, "
          f"tokens seen {ck.get('tokens_seen', 0)/1e6:.1f}M")


if __name__ == "__main__":
    main()
