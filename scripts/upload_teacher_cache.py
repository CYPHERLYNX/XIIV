#!/usr/bin/env python3
"""Upload the SGBD teacher activation cache to Hugging Face Hub (private).

Why: the 16GB teacher cache (133M-teacher activations, sgbd_cache/iter0/)
must be reachable from Kaggle GPU kernels for block training. HF is the
established data layer (30GB of tokenized data already there as elova89/*).

Usage: HF_TOKEN=<token> python3 upload_teacher_cache.py [--dry-run]

- Creates elova89/xiiv-sgbd-teacher-cache (private) if missing.
- upload_folder is idempotent: files already on the Hub are skipped.
- NEVER deletes local files (CPU fallback still needs them).
- Token ONLY from HF_TOKEN env var; never logged.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "sgbd_cache" / "iter0"
REPO_ID = "elova89/xiiv-sgbd-teacher-cache"


def log(msg: str) -> None:
    print(f"[teacher-cache] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        log("HF_TOKEN env var not set")
        return 2
    if not CACHE_DIR.is_dir():
        log(f"cache dir missing: {CACHE_DIR}")
        return 2

    files = sorted(p for p in CACHE_DIR.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    log(f"{len(files)} files, {total/1e9:.1f} GB in {CACHE_DIR}")
    if args.dry_run:
        for p in files[:5]:
            log(f"  would upload: {p.name} ({p.stat().st_size/1e6:.0f} MB)")
        log(f"  ... and {len(files)-5} more")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    # create repo if missing (private)
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)
        log(f"repo ready: {REPO_ID}")
    except Exception as e:
        log(f"create_repo note: {type(e).__name__}: {str(e)[:120]}")

    log("uploading (idempotent, resumable)...")
    api.upload_folder(
        folder_path=str(CACHE_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo="iter0",
    )
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
