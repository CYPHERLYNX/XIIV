#!/usr/bin/env python3
"""mix_corpus.py — concatenate prepared corpus dirs into one mixed training dir.

Each input dir must contain train.bin / valid.bin (raw uint16 LE token ids,
packed in seq_len blocks, EOS-separated — the format scripts/prepare_*.py
writes) plus meta.json. Output dir gets concatenated train.bin / valid.bin
and a fresh meta.json with full provenance (sources, token counts, mix id).

Mixing is plain concatenation in the order given (FineWeb first, then code,
then books) — the runbook's documented mix. Block alignment is preserved
because every source file's byte length is a multiple of seq_len*2.

Usage:
  python scripts/mix_corpus.py --inputs data_fineweb,data_code,data_books \\
      --out data_heavy [--force]

Exit 0 + skip message if the out dir's meta.json already matches the inputs
(same dirs, same source token counts) and --force is not given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

DTYPE_BYTES = 2  # uint16


def read_meta(d):
    path = os.path.join(d, "meta.json")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {d} has no meta.json")
    with open(path) as f:
        return json.load(f)


def blocks_of(meta, name):
    n = meta.get(name)
    if not isinstance(n, int) or n <= 0:
        sys.exit(f"ERROR: meta.json has bad {name}={n!r}")
    return n


def concat_bin(src_dirs, fname, seq_len, out_path):
    """Concatenate one .bin across dirs; returns (blocks, tokens)."""
    block_bytes = seq_len * DTYPE_BYTES
    total_blocks = 0
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as out:
        for d in src_dirs:
            p = os.path.join(d, fname)
            if not os.path.isfile(p):
                sys.exit(f"ERROR: missing {p}")
            size = os.path.getsize(p)
            if size % block_bytes != 0:
                sys.exit(
                    f"ERROR: {p} is {size} bytes — not a multiple of "
                    f"seq_len*2 ({block_bytes}); refusing to misalign blocks"
                )
            with open(p, "rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            total_blocks += size // block_bytes
    os.replace(tmp, out_path)
    return total_blocks, total_blocks * seq_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="comma-separated corpus dirs, in mix order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_dirs = [d.strip().rstrip("/") for d in args.inputs.split(",") if d.strip()]
    if len(src_dirs) < 1:
        sys.exit("ERROR: --inputs needs at least one dir")
    for d in src_dirs:
        if not os.path.isdir(d):
            sys.exit(f"ERROR: input dir not found: {d}")

    metas = [read_meta(d) for d in src_dirs]
    seq_len = metas[0].get("seq_len")
    vocab = metas[0].get("vocab_size")
    for d, m in zip(src_dirs, metas):
        if m.get("seq_len") != seq_len:
            sys.exit(f"ERROR: seq_len mismatch: {src_dirs[0]}={seq_len}, {d}={m.get('seq_len')}")
        if m.get("vocab_size") != vocab:
            sys.exit(f"ERROR: vocab_size mismatch: {src_dirs[0]}={vocab}, {d}={m.get('vocab_size')}")

    sources = []
    for d, m in zip(src_dirs, metas):
        sources.append({
            "dir": d,
            "source": m.get("source", "?"),
            "seq_len": m.get("seq_len"),
            "train_blocks": blocks_of(m, "train_blocks"),
            "train_tokens": m.get("train_tokens"),
            "valid_blocks": blocks_of(m, "valid_blocks"),
            "valid_tokens": m.get("valid_tokens"),
            "docs_seen": m.get("docs_seen"),
            "docs_dropped": m.get("docs_dropped"),
        })

    mix_id_src = "|".join(
        f"{s['dir']}:{s['source']}:{s['train_blocks']}" for s in sources
    )
    mix_id = hashlib.sha256(mix_id_src.encode()).hexdigest()[:12]

    meta_path = os.path.join(args.out, "meta.json")
    if os.path.isfile(meta_path) and not args.force:
        try:
            old = json.load(open(meta_path))
            if old.get("mix_id") == mix_id and old.get("out_dir") == args.out:
                print(f"mix up to date (mix_id={mix_id}); --force to rebuild")
                return
        except (json.JSONDecodeError, OSError):
            pass  # fall through and rebuild

    os.makedirs(args.out, exist_ok=True)
    train_blocks, train_tok = concat_bin(
        src_dirs, "train.bin", seq_len, os.path.join(args.out, "train.bin"))
    valid_blocks, valid_tok = concat_bin(
        src_dirs, "valid.bin", seq_len, os.path.join(args.out, "valid.bin"))

    meta = {
        "vocab_size": vocab,
        "mask_id": metas[0].get("mask_id"),
        "eos_id": metas[0].get("eos_id"),
        "seq_len": seq_len,
        "train_blocks": train_blocks,
        "train_tokens": train_tok,
        "valid_blocks": valid_blocks,
        "valid_tokens": valid_tok,
        "mix_id": mix_id,
        "out_dir": args.out,
        "mix_order": src_dirs,
        "sources": sources,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "concatenated mix (no stories); order = mix priority",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)

    print(f"mixed {len(src_dirs)} corpora -> {args.out} (mix_id={mix_id})")
    for s in sources:
        frac = 100.0 * s["train_blocks"] / train_blocks
        print(f"  {s['dir']:<16} {s['train_tokens']/1e6:8.2f}M train tok ({frac:5.1f}%)  src={s['source']}")
    print(f"  total: {train_tok/1e6:.2f}M train / {valid_tok/1e6:.2f}M valid tokens, "
          f"{train_blocks} train blocks x {seq_len}")


if __name__ == "__main__":
    main()
