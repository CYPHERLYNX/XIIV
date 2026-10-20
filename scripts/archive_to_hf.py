#!/usr/bin/env python3
"""Archive finished XIIV tokenized datasets to Hugging Face Hub, then free local disk.

Why: the 100G box cannot hold the full frontier corpus. Finished datasets
(train.bin/valid.bin) are archived to HF Hub dataset repos, verified by size,
and only then deleted locally. The download queue (stackexchange, pg19) can
then proceed without stalling on disk.

Usage:
  HF_TOKEN=<write token> python3 scripts/archive_to_hf.py --datasets fineweb_edu papers reasoning math
  HF_TOKEN=<write token> python3 scripts/archive_to_hf.py --datasets dclm --public --delete-local

Flags:
  --datasets  space-separated dataset dir names under data_frontier/ (default: all finished)
  --public    create public repos (default: private; the public/private call stays with Fahad)
  --delete-local  remove local train.bin/valid.bin ONLY after Hub verification (default: keep)
  --namespace HF username/org for the repos (default: CYPHERLYNX)
  --list      dry run: show what would be archived and how much disk it frees

Safety:
  - Token is read ONLY from the HF_TOKEN env var; never logged, never written to disk.
  - Uploads are idempotent: files already on the Hub with matching size are skipped.
  - --delete-local refuses to run unless every file for that dataset verified OK.
  - meta.json / resume_state.json are always kept locally (tiny, needed for provenance).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data_frontier"
FILES = ("train.bin", "valid.bin", "meta.json", "README.md")

# Datasets whose builds are finished and safe to archive. dclm is finishing tonight.
FINISHED = ("fineweb_edu", "papers", "reasoning", "math")


def log(msg: str) -> None:
    print(f"[archive] {msg}", flush=True)


def head_content_length(url: str, token: str | None = None) -> int | None:
    # NOTE: private repos need the Authorization header or HEAD returns 401.
    req = urllib.request.Request(url, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive finished XIIV datasets to HF Hub.")
    ap.add_argument("--datasets", nargs="*", default=list(FINISHED))
    ap.add_argument("--public", action="store_true", help="create public repos (default: private)")
    ap.add_argument("--delete-local", action="store_true")
    ap.add_argument("--namespace", default="CYPHERLYNX")
    ap.add_argument("--list", action="store_true", help="dry run: report only")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        log("huggingface_hub not installed")
        return 2

    plan = []
    for name in args.datasets:
        d = DATA_DIR / name
        if not d.is_dir():
            log(f"skip {name}: not a directory")
            continue
        sizes = {}
        ok = True
        for f in FILES:
            p = d / f
            if not p.exists():
                log(f"skip {name}: missing {f}")
                ok = False
                break
            sizes[f] = p.stat().st_size
        if not ok:
            continue
        meta = json.loads((d / "meta.json").read_text())
        plan.append((name, d, sizes, meta))

    if not plan:
        log("nothing to archive")
        return 0

    total = sum(s for _, _, sizes, _ in plan for s in sizes.values())
    log(f"{len(plan)} dataset(s), {total / 1e9:.1f} GB total:")
    for name, _, sizes, meta in plan:
        toks = meta.get("train_tokens", 0)
        log(f"  {name}: train={sizes['train.bin'] / 1e9:.1f}G valid={sizes['valid.bin'] / 1e6:.0f}M "
            f"tokens={toks / 1e9:.2f}B src={meta.get('source', '?')}")

    if args.list:
        log(f"dry run: would free {total / 1e9:.1f} GB with --delete-local")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        log("HF_TOKEN env var not set; refusing to run. Nothing uploaded, nothing deleted.")
        return 2

    api = HfApi(token=token)
    visibility = "public" if args.public else "private"
    log(f"repos will be {visibility} under namespace {args.namespace}")

    all_verified: dict[str, bool] = {}
    for name, d, sizes, meta in plan:
        repo_id = f"{args.namespace}/xiiv-{name}-tok32k"
        log(f"--- {name} -> {repo_id} ---")
        api.create_repo(repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
        try:
            remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
        except Exception as e:
            log(f"  cannot list remote files: {e}")
            remote = set()

        verified = True
        for f in FILES:
            local_size = sizes[f]
            if f in remote:
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}"
                remote_size = head_content_length(url, token)
                if remote_size == local_size:
                    log(f"  {f}: already on Hub, size matches ({local_size / 1e9:.2f}G) - skip")
                    continue
                log(f"  {f}: remote size mismatch/unknown ({remote_size} vs {local_size}) - re-upload")
            log(f"  uploading {f} ({local_size / 1e9:.2f}G)...")
            try:
                api.upload_file(
                    path_or_fileobj=str(d / f),
                    path_in_repo=f,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
            except Exception as e:
                log(f"  UPLOAD FAILED for {f}: {e}")
                verified = False
                continue
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}"
            remote_size = head_content_length(url, token)
            if remote_size == local_size:
                log(f"  {f}: verified on Hub ({remote_size} bytes)")
            else:
                log(f"  {f}: VERIFICATION FAILED (hub={remote_size}, local={local_size})")
                verified = False
        all_verified[name] = verified

    if args.delete_local:
        for name, d, sizes, _ in plan:
            if not all_verified.get(name):
                log(f"{name}: NOT verified - keeping local files")
                continue
            for f in ("train.bin", "valid.bin"):
                (d / f).unlink()
                log(f"{name}: deleted local {f} ({sizes[f] / 1e9:.2f}G freed)")
        log("local provenance (meta.json, README.md, resume_state.json) kept in all cases")
    else:
        log("uploads done; local files kept (pass --delete-local to free disk after verification)")

    failed = [n for n, v in all_verified.items() if not v]
    if failed:
        log(f"DONE WITH FAILURES: {failed}")
        return 1
    log("all datasets archived and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
