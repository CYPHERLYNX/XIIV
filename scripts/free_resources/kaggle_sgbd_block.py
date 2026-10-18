#!/usr/bin/env python3
"""
Push one XIIV SGBD block training job to Kaggle free GPU.

Usage:
    python3 kaggle_sgbd_block.py --block 0 --dry-run   # preview only
    python3 kaggle_sgbd_block.py --block 0             # launch on Kaggle

Prereq: ~/.kaggle/kaggle.json (see kaggle_setup.sh)
Each block trains ~115M params on T4x2 — fits in a 9h Kaggle session.
30 GPU-hrs/week = ~2-3 blocks/week = full 1B in ~3-4 weeks, free.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

# --- The training payload that runs ON Kaggle ---
KAGGLE_TRAIN_SCRIPT = textwrap.dedent('''
    import os, sys, json, math, time
    print("=== XIIV SGBD Block {block} on Kaggle ===", flush=True)

    # Install deps (cached on warm workers, fast on reruns)
    os.system("pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1")
    os.system("pip install -q safetensors numpy 2>&1 | tail -1")

    import torch
    print(f"CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # --- Clone XIIV repo (public after push; until then this is a placeholder) ---
    # os.system("git clone --depth 1 https://github.com/CYPHERLYNX/XIIV.git /kaggle/working/xiiv")

    # --- Placeholder: replace with actual xiiv.sgbd block training ---
    # from xiiv.sgbd import train_block
    # train_block(block_id={block}, teacher_cache="/kaggle/input/xiiv-teacher/iter0",
    #             out_dir="/kaggle/working/block{block}", device="cuda")

    print("PAYLOAD PLACEHOLDER — wire to xiiv.sgbd.train_block when repo is public", flush=True)

    # Kaggle auto-saves /kaggle/working as versioned output on completion.
    print("=== done ===", flush=True)
''')

KERNEL_METADATA_TEMPLATE = {
    "id": "cypherlynx/xiiv-sgbd-block{block}",
    "title": "XIIV SGBD Block {block} Training",
    "code_file": "kaggle_sgbd_block.py",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}


def build_notebook(block: int) -> str:
    """Build a .ipynb that runs the training payload."""
    script = KAGGLE_TRAIN_SCRIPT.replace("{block}", str(block))
    # Escape for embedding in a notebook JSON cell
    lines = script.split("\n")
    cell_source = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_source,
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return json.dumps(nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True, help="SGBD block id 0-7")
    ap.add_argument("--dry-run", action="store_true", help="build files but do not push")
    args = ap.parse_args()

    if not (0 <= args.block <= 7):
        sys.exit("block must be 0-7")

    tmpdir = tempfile.mkdtemp(prefix=f"xiiv_block{args.block}_")
    nb_path = os.path.join(tmpdir, "kaggle_sgbd_block.py")
    # kaggle CLI expects code_file to be the notebook; we write .ipynb content
    # into a .py-named file? No — write proper .ipynb and point metadata at it.
    nb_path = os.path.join(tmpdir, "notebook.ipynb")
    with open(nb_path, "w") as f:
        f.write(build_notebook(args.block))

    meta = dict(KERNEL_METADATA_TEMPLATE)
    meta["id"] = meta["id"].format(block=args.block)
    meta["title"] = meta["title"].format(block=args.block)
    meta["code_file"] = "notebook.ipynb"
    meta_path = os.path.join(tmpdir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Built Kaggle kernel in {tmpdir}")
    print(f"  id: {meta['id']}")
    print(f"  gpu: {meta['enable_gpu']}, internet: {meta['enable_internet']}")

    if args.dry_run:
        print("[dry-run] not pushing. Files ready for `kaggle kernels push -p <dir>`")
        return

    # Push = upload AND trigger run
    r = subprocess.run(
        ["kaggle", "kernels", "push", "-p", tmpdir],
        capture_output=True, text=True,
    )
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    print(f"\nLaunched. Monitor: kaggle kernels status {meta['id']}")
    print(f"Outputs: kaggle kernels output {meta['id']} -p ./out_block{args.block}")


if __name__ == "__main__":
    main()
