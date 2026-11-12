"""CPU baseline for the assembled XIIV 1B — same spec as the GPU v1 eval.

Purpose: give the verdict a CPU arm that is directly comparable to the
GPU run when it lands. Identical spec to
``xiiv-kaggle-gpu/multi-platform/xiiv_v1_eval_only.py``:
prompt "What is quantum computing vs quantum mechanics.",
length 64, steps 16, seed 7, temperature 0.8, top_p 0.9.

What this is NOT: a benchmark score. It records raw generated text,
generation seconds, and tokens/second for one fixed prompt, plus the
weights file's size and SHA-256 so the run is auditable against the
assembled file (3,093,698,487 bytes, verified 2026-10-27). No quality
claim is made from this output; the verdict is Fahad's, side-by-side
with the GPU timing.

Usage (from the repo root):
    python3 eval/assembled_cpu_eval.py \
        --weights checkpoints_1b_sgbd/xiiv_1b_sgbd.safetensors \
        --tokenizer weights/tokenizer_32k.json
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from xiiv.diffusion import sample_trajectory
from xiiv.sample import load_model
from xiiv.tokenizer import XIIVTokenizer

# Fixed eval spec — do not change without also changing the GPU job,
# or the CPU/GPU comparison stops being apples-to-apples.
PROMPT = "What is quantum computing vs quantum mechanics."
LENGTH = 64
STEPS = 16
SEED = 7
TEMPERATURE = 0.8
TOP_P = 0.9


def sha256_file(path, chunk=1 << 20):
    """Streaming SHA-256 (no full-file read into RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def tokens_per_second(n_tokens, seconds):
    return n_tokens / seconds if seconds > 0 else 0.0


def build_record(weights_path, text, ids, seconds, config, mask_id=2):
    """Pure record builder — unit-testable without loading the model."""
    n_tok = int((ids != mask_id).sum()) if hasattr(ids, "sum") else len(ids)
    return {
        "spec": {
            "prompt": PROMPT,
            "length": LENGTH,
            "steps": STEPS,
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "device": "cpu",
        },
        "weights": {
            "path": weights_path,
            "bytes": os.path.getsize(weights_path),
            "sha256": sha256_file(weights_path),
        },
        "config": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "vocab_size": config.vocab_size,
            "contract_enabled": config.contract_enabled,
            "tie_embeddings": config.tie_embeddings,
        },
        "output": {
            "text": text,
            "token_ids": [int(t) for t in ids.tolist()],
            "n_tokens_non_mask": n_tok,
        },
        "timing": {
            "generation_seconds": round(seconds, 3),
            "tokens_per_second": round(tokens_per_second(n_tok, seconds), 3),
        },
        "disclaimer": (
            "Raw generation only. Not a benchmark score, not a quality "
            "verdict, not omega-level evidence on its own."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights",
                    default="checkpoints_1b_sgbd/xiiv_1b_sgbd.safetensors")
    ap.add_argument("--tokenizer", default="weights/tokenizer_32k.json")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: eval/results/"
                         "assembled_1b_cpu_eval_<date>.json)")
    args = ap.parse_args()

    tok = XIIVTokenizer(args.tokenizer)
    model, cfg = load_model(args.weights, device="cpu")
    assert tok.mask_id == cfg.mask_id and tok.eos_id == cfg.eos_id, \
        "tokenizer/weights special-token mismatch"
    assert tok.vocab_size == cfg.vocab_size, \
        f"tokenizer vocab ({tok.vocab_size}) != model vocab ({cfg.vocab_size})"

    prompt_ids = torch.tensor(tok.encode(PROMPT), dtype=torch.long)
    assert len(prompt_ids) < LENGTH, \
        f"prompt is {len(prompt_ids)} tokens but length is only {LENGTH}"

    t0 = time.time()
    ids, _conf = sample_trajectory(
        model, LENGTH, tok.mask_id, tok.eos_id, steps=STEPS,
        temperature=TEMPERATURE, top_p=TOP_P, seed=SEED,
        prompt_ids=prompt_ids, device="cpu")
    seconds = time.time() - t0

    text = tok.decode(ids.tolist())
    record = build_record(args.weights, text, ids, seconds, cfg,
                          mask_id=tok.mask_id)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        f"assembled_1b_cpu_eval_{time.strftime('%Y-%m-%d')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)

    print(text)
    print(f"\n[xiiv] cpu baseline: {record['output']['n_tokens_non_mask']} "
          f"tokens, {seconds:.2f}s "
          f"({record['timing']['tokens_per_second']:.2f} tok/s), seed={SEED}")
    print(f"Record: {out}")


if __name__ == "__main__":
    main()
