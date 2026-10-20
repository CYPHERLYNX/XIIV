#!/usr/bin/env python3
"""Priority upload for Kaggle block-2: layer04+layer05 cache chunks (2.2GB)
+ student ckpt (1.83GB) to elova89/xiiv-sgbd-teacher-cache.

The background 16GB full-cache upload is idempotent and will skip these.
Token ONLY from HF_TOKEN env var; never logged.
"""
import os, sys
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from pathlib import Path
from huggingface_hub import HfApi

REPO = "elova89/xiiv-sgbd-teacher-cache"
CACHE = Path("/home/hatch/workspace/ai-tools/corsac/sgbd_cache/iter0")
CKPT = Path("/home/hatch/workspace/ai-tools/corsac/checkpoints_1b_gn/ckpt_8218.pt")

token = os.environ.get("HF_TOKEN", "")
assert token.startswith("hsurr:"), "HF_TOKEN not set"

api = HfApi(token=token)
# ckpt FIRST (1.8GB — the long pole; smoke test needs it before any chunks)
print("[prio] uploading student ckpt (1.8GB) first...", flush=True)
api.upload_file(path_or_fileobj=str(CKPT), path_in_repo="ckpt_8218.pt",
                repo_id=REPO, repo_type="dataset")
print("[prio] ckpt done; uploading cache chunks...", flush=True)
files = []
for lyr in ("layer04", "layer05"):
    files += sorted(CACHE.glob(f"{lyr}_c*.pt"))
print(f"[prio] {len(files)} cache chunks to upload", flush=True)
for i, f in enumerate(files):
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                    repo_id=REPO, repo_type="dataset")
    if (i + 1) % 12 == 0:
        print(f"[prio] {i+1}/{len(files)} chunks done", flush=True)
print("[prio] ALL DONE", flush=True)
