#!/usr/bin/env python3
"""
Upload a tokenized XIIV dataset (.bin + meta) to HuggingFace Datasets Hub.

Usage:
    export HF_TOKEN=hf_...                # write token (one-time setup)
    python3 hf_upload_dataset.py \
        --bin data_frontier/fineweb_edu/train.bin \
        --repo CYPHERLYNX/xiiv-frontier-fineweb-edu \
        --seq-len 128 --private False

Converts uint16 token .bin -> sharded parquet -> pushes to Hub.
Public repos = free best-effort unlimited storage.
"""
import argparse
import os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to train.bin (uint16 token ids)")
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. CYPHERLYNX/xiiv-frontier-fineweb-edu")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--shard-tokens", type=int, default=50_000_000,
                    help="tokens per parquet shard (~100MB)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN env var (huggingface.co/settings/tokens, Write role)")

    from huggingface_hub import HfApi, create_repo
    import pyarrow as pa
    import pyarrow.parquet as pq

    api = HfApi(token=token)
    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"[ok] repo ready: https://huggingface.co/datasets/{args.repo}")

    ids = np.memmap(args.bin, dtype=np.uint16, mode="r")
    total = len(ids)
    print(f"[ok] loaded {total:,} tokens from {args.bin}")

    shard_len = (args.shard_tokens // args.seq_len) * args.seq_len
    tmpdir = "/tmp/xiiv_hf_upload"
    os.makedirs(tmpdir, exist_ok=True)

    shard = 0
    for start in range(0, total, shard_len):
        chunk = np.asarray(ids[start:start + shard_len])
        n_blocks = len(chunk) // args.seq_len
        chunk = chunk[: n_blocks * args.seq_len].reshape(n_blocks, args.seq_len)
        table = pa.table({"input_ids": chunk.tolist()})
        path = os.path.join(tmpdir, f"train-{shard:05d}.parquet")
        pq.write_table(table, path)
        shard += 1
        print(f"  shard {shard}: {n_blocks:,} blocks -> {path}")

    print(f"[ok] {shard} shards written, uploading...")
    api.upload_folder(
        folder_path=tmpdir,
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"[done] https://huggingface.co/datasets/{args.repo}")

    # Dataset card
    card = f"""---
license: odc-by
task_categories: [text-generation]
---

# {args.repo}

Tokenized training data for XIIV (first open-source ASI model).
- Tokens: {total:,}
- Seq len: {args.seq_len}
- Format: parquet shards of `input_ids` (uint16-range int64)
- Produced by `scripts/build_frontier_data.py`
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print("[done] README uploaded")


if __name__ == "__main__":
    main()
