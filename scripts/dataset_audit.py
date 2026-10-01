"""Dataset audit: snapshot hashes + leakage check (ecc_mle_workflow §2).

- SHA256 of train.bin / valid.bin / tokenizer.json -> data/dataset_snapshot.json
- Leakage check: exact 128-token block overlap between train and valid
  (must be 0 — upstream files are disjoint story sets).

Usage:
    python scripts/dataset_audit.py --data data
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def block_hashes(memmap):
    """SHA256 (truncated) per 128-token block. Returns a set of digests."""
    out = set()
    n = len(memmap)
    for i in range(n):
        out.add(hashlib.sha256(
            np.ascontiguousarray(memmap[i]).tobytes()).hexdigest()[:16])
        if (i + 1) % 50000 == 0:
            print(f"  hashed {i + 1}/{n} blocks...", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    seq_len = meta["seq_len"]
    snap = {
        "snapshot_id": "tinystories-v2-gpt4-20261001",
        "sha256": {},
        "blocks": {"train": meta["train_blocks"],
                   "valid": meta["valid_blocks"]},
    }
    for name in ("train.bin", "valid.bin", "tokenizer.json"):
        p = os.path.join(args.data, name)
        print(f"hashing {name}...", flush=True)
        snap["sha256"][name] = sha256_file(p)

    print("leakage check: hashing train blocks...", flush=True)
    train_mm = np.memmap(os.path.join(args.data, "train.bin"),
                         dtype=np.uint16, mode="r",
                         shape=(meta["train_blocks"], seq_len))
    train_hashes = block_hashes(train_mm)
    print("leakage check: hashing valid blocks...", flush=True)
    valid_mm = np.memmap(os.path.join(args.data, "valid.bin"),
                         dtype=np.uint16, mode="r",
                         shape=(meta["valid_blocks"], seq_len))
    overlap = sum(1 for i in range(len(valid_mm))
                  if hashlib.sha256(
                      np.ascontiguousarray(valid_mm[i]).tobytes()
                  ).hexdigest()[:16] in train_hashes)
    snap["leakage_check"] = {
        "method": "exact 128-token block hash overlap, valid vs train",
        "overlapping_blocks": overlap,
        "pass": overlap == 0,
    }
    print(f"overlapping blocks: {overlap} -> "
          f"{'PASS' if overlap == 0 else 'FAIL'}", flush=True)

    out = os.path.join(args.data, "dataset_snapshot.json")
    json.dump(snap, open(out, "w"), indent=2)
    print(f"wrote {out}")
    if overlap != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
