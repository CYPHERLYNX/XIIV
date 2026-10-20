#!/usr/bin/env python3
"""
Push one XIIV SGBD block training job to Kaggle free GPU (REAL payload).

Usage:
    python3 kaggle_sgbd_block.py --block 2 --dry-run   # build + validate only
    python3 kaggle_sgbd_block.py --block 2 --smoke      # push PRIVATE smoke kernel (20 steps)
    python3 kaggle_sgbd_block.py --block 2              # push PRIVATE full kernel (1000 steps)

Prereq: ~/.kaggle/access_token (KGAT_ token; see KAGGLE_WIRING_NOTES.md).
Data: private Kaggle dataset x0d4n7/xiiv-block2-inputs (attached to kernel).
Code: embedded minimal xiiv package in the notebook (repo is private, no clone).
No HF tokens anywhere (short-lived surrogates expired before kernels ran).
Kernels are PRIVATE (owner-only).

What the kernel does:
  1. pip install torch cu121 + safetensors + numpy
  2. Write embedded training code to /kaggle/working/pkg/
  3. Read block inputs from /kaggle/input/xiiv-block2-inputs/:
     layer04_c*.pt + layer05_c*.pt (teacher cache) + ckpt_8218.pt (student init)
  4. Run sgbd_train_block.py --block 2 --steps N --device cuda
     (resume-capable via --resume-step; step ckpts every 200 steps)
  5. Checkpoints land in /kaggle/working/sgbd_blocks/block2/ (auto-versioned)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

# Block -> teacher layer map (must match xiiv/sgbd_train_block.py T_GROUPS)
BLOCK_TLAYERS = {0: (0, 2), 1: (2, 4), 2: (4, 6), 3: (6, 8),
                 4: (8, 10), 5: (10, 12), 6: (12, 14), 7: (14, 15)}

KAGGLE_TRAIN_SCRIPT = textwrap.dedent('''
    import os, sys, time, glob
    t0 = time.time()
    print("=== XIIV SGBD Block {block} on Kaggle GPU ===", flush=True)

    # ---- 0. deps ----
    os.system("pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1")
    os.system("pip install -q safetensors huggingface_hub numpy 2>&1 | tail -1")

    import torch
    assert torch.cuda.is_available(), "NO CUDA — aborting (need GPU)"
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

    # ---- 1. training code (embedded; repo is private so no git clone) ----
    # Files: xiiv/__init__.py (stub), xiiv/config.py, xiiv/model.py,
    #        sgbd_train_block.py — bundled at build time from the local repo.
    import json as _json
    _bundle = _json.loads('{CODE_BUNDLE}')
    for _rel, _src in _bundle.items():
        _p = "/kaggle/working/pkg/" + _rel
        os.makedirs(os.path.dirname(_p), exist_ok=True)
        open(_p, "w").write(_src)
    print(f"[code] wrote {len(_bundle)} files", flush=True)
    del _bundle

    # ---- 2. data from attached Kaggle dataset (no HF auth needed) ----
    # Dataset x0d4n7/xiiv-block2-inputs mounted at /kaggle/input/xiiv-block2-inputs/
    ddir = "/kaggle/input/xiiv-block2-inputs"
    n_chunks = len(glob.glob(ddir + "/layer04_c*.pt"))
    assert n_chunks >= 1, f"expected layer04 chunks, got {n_chunks}"
    if not {smoke_chunks}:
        assert n_chunks == 24, f"expected 24 layer04 chunks, got {n_chunks}"
    assert os.path.exists(ddir + "/ckpt_8218.pt"), "student ckpt missing"
    print(f"[data] {n_chunks} chunks/layer + student ckpt ready "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- 3. train ----
    os.chdir("/kaggle/working/pkg")
    env = "PYTHONPATH=/kaggle/working/pkg"
    cmd = (f"{env} python sgbd_train_block.py --block {block} --steps {steps} "
           f"--device cuda --batch-size {batch} --cache-dir /kaggle/input/xiiv-block2-inputs "
           f"--out-dir /kaggle/working/sgbd_blocks "
           f"--student-ckpt /kaggle/input/xiiv-block2-inputs/ckpt_8218.pt "
           f"--save-every 200 --log-every 25 {resume}")
    print("[train]", cmd, flush=True)
    rc = os.system(cmd)
    assert rc == 0, f"training exited rc={rc}"
    print(f"=== DONE block {block}: {steps} steps in {(time.time()-t0)/60:.1f} min ===",
          flush=True)
''')

KERNEL_METADATA_TEMPLATE = {
    "id": "x0d4n7/xiiv-sgbd-block{block}",
    "title": "XIIV SGBD Block {block} Training",
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": ["x0d4n7/xiiv-block2-inputs"],
    "competition_sources": [],
    "kernel_sources": [],
}


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # .../corsac


def get_code_bundle():
    """Bundle the minimal training code: stub xiiv/__init__ + config + model
    + sgbd_train_block.py. Avoids git clone (repo is private)."""
    xiiv_dir = os.path.join(REPO_ROOT, "xiiv")
    bundle = {"xiiv/__init__.py": '"""xiiv stub for Kaggle kernels."""\n'}
    for name in ("config.py", "model.py"):
        with open(os.path.join(xiiv_dir, name)) as f:
            bundle[f"xiiv/{name}"] = f.read()
    with open(os.path.join(xiiv_dir, "sgbd_train_block.py")) as f:
        bundle["sgbd_train_block.py"] = f.read()
    total = sum(len(v) for v in bundle.values())
    print(f"[build] code bundle: {len(bundle)} files, {total//1024} KB")
    return json.dumps(bundle)


def build_notebook(block, steps, batch_size, resume_step, smoke=False):
    t_start, t_end = BLOCK_TLAYERS[block]
    resume = f"--resume-step {resume_step}" if resume_step > 0 else ""
    code_bundle = get_code_bundle().replace("\\", "\\\\").replace("'", "\\'")
    script = KAGGLE_TRAIN_SCRIPT.replace("{block}", str(block)) \
        .replace("{steps}", str(steps)) \
        .replace("{batch}", str(batch_size)) \
        .replace("{resume}", resume) \
        .replace("{smoke_chunks}", "True" if smoke else "False") \
        .replace("{CODE_BUNDLE}", code_bundle)
    assert "{CODE_BUNDLE}" not in script and "{block}" not in script
    lines = script.split("\n")
    nb = {
        "cells": [{
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }
    return json.dumps(nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--resume-step", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="20-step smoke test kernel (separate slug)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (0 <= args.block <= 7):
        sys.exit("block must be 0-7")

    steps = 20 if args.smoke else args.steps
    # NOTE: Kaggle slugifies the title for the URL; keep title slug-safe so the
    # id is predictable. Verified 2026-10-18: mismatched title/id causes
    # kernels.get permission errors on status checks.
    slug = f"x0d4n7/xiiv-sgbd-block{args.block}-smoke" if args.smoke \
        else f"x0d4n7/xiiv-sgbd-block{args.block}"
    title = slug.split("/", 1)[1]  # slug-safe title

    # data comes from attached Kaggle dataset x0d4n7/xiiv-block2-inputs — no HF token needed
    nb_json = build_notebook(args.block, steps, args.batch_size,
                             args.resume_step, smoke=args.smoke)
    # validate notebook JSON parses and has one code cell
    nb = json.loads(nb_json)
    assert nb["cells"][0]["cell_type"] == "code"
    print(f"[build] notebook OK ({len(nb_json)//1024} KB, 1 code cell)")

    tmpdir = tempfile.mkdtemp(prefix=f"xiiv_kaggle_b{args.block}_")
    with open(os.path.join(tmpdir, "notebook.ipynb"), "w") as f:
        f.write(nb_json)
    meta = dict(KERNEL_METADATA_TEMPLATE)
    meta.update({"id": slug, "title": title})
    with open(os.path.join(tmpdir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build] kernel dir: {tmpdir}")
    print(f"  id: {slug} | steps: {steps} | gpu: True | private: True")

    if args.dry_run:
        print("[dry-run] not pushing.")
        return

    r = subprocess.run(["kaggle", "kernels", "push", "-p", tmpdir],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        sys.exit(1)
    print(f"\nLaunched. Monitor: kaggle kernels status {slug}")
    print(f"Output DL: kaggle kernels output {slug} -p ./out_b{args.block}")


if __name__ == "__main__":
    main()
