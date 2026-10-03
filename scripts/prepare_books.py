"""Build a book-prose fine-tune dataset using the EXISTING tokenizer.

Strips Project Gutenberg headers/footers, tokenizes data_books/raw/*.txt
with weights_52m/tokenizer.json (never retrains it), writes packed
uint16 memmap blocks (seq_len=128, EOS-separated) + meta.json.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
import numpy as np

from xiiv.tokenizer import XIIVTokenizer


def clean(text):
    # cut gutenberg header/footer
    m = re.search(r"\*\*\* START OF .*?\*\*\*", text, re.S)
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* END OF .*?\*\*\*", text, re.S)
    if m:
        text = text[:m.start()]
    # normalize whitespace, drop very short lines (toc junk)
    lines = [ln.strip() for ln in text.splitlines()
             if len(ln.strip()) > 40]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data_books/raw")
    ap.add_argument("--out-dir", default="data_books")
    ap.add_argument("--tokenizer", default="weights_52m/tokenizer.json")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tok = XIIVTokenizer(args.tokenizer)
    ids_all = []
    for path in sorted(glob.glob(os.path.join(args.raw_dir, "*.txt"))):
        raw = open(path, encoding="utf-8", errors="ignore").read()
        text = clean(raw)
        if len(text) < 50000:
            print("skip (short):", os.path.basename(path))
            continue
        ids = [i for i in tok.encode(text) if i < 65536]
        ids_all.extend(ids)
        ids_all.append(tok.eos_id)
        print(f"{os.path.basename(path)}: {len(ids)/1e3:.0f}k tokens")
    print(f"total {len(ids_all)/1e6:.2f}M tokens")

    rng = np.random.RandomState(args.seed)
    arr = np.array(ids_all, dtype=np.uint16)
    n_blocks = len(arr) // args.seq_len
    arr = arr[:n_blocks * args.seq_len].reshape(n_blocks, args.seq_len)
    perm = rng.permutation(n_blocks)
    n_valid = max(16, int(n_blocks * args.valid_frac))

    os.makedirs(args.out_dir, exist_ok=True)
    arr[perm[n_valid:]].tofile(os.path.join(args.out_dir, "train.bin"))
    arr[perm[:n_valid]].tofile(os.path.join(args.out_dir, "valid.bin"))
    meta = {"vocab_size": 8192, "mask_id": tok.mask_id, "eos_id": tok.eos_id,
            "seq_len": args.seq_len,
            "train_blocks": int(n_blocks - n_valid),
            "valid_blocks": int(n_valid),
            "train_tokens": int((n_blocks - n_valid) * args.seq_len),
            "valid_tokens": int(n_valid * args.seq_len)}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"),
              indent=2)
    print("wrote", args.out_dir, json.dumps(meta))


if __name__ == "__main__":
    main()
