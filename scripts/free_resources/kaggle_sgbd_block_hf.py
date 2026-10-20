#!/usr/bin/env python3
"""Push one XIIV SGBD block training job to Kaggle GPU, data from HuggingFace.

The token is read ONLY from the HF_TOKEN env var (never logged, never written
to disk except embedded in the PRIVATE kernel notebook). The kernel is PRIVATE
(owner-only).

Usage:
    HF_TOKEN=<token> python3 kaggle_sgbd_block_hf.py --block 3 --dry-run
    HF_TOKEN=<token> python3 kaggle_sgbd_block_hf.py --block 3

What the kernel does:
  1. pip install torch cu121 + safetensors + huggingface_hub + numpy
  2. Write embedded training code to /kaggle/working/pkg/
     (REPO resolves to /kaggle/working, so cache/out dirs live there)
  3. snapshot_download block inputs from elova89/xiiv-sgbd-teacher-cache:
     iter0/layer{TT}_c*.pt (teacher in/out) + ckpt_8218.pt -> /kaggle/working/sgbd_cache/
  4. Run sgbd_train_block.py --block N --steps 1000 --device cuda
  5. Checkpoints -> /kaggle/working/sgbd_blocks/blockN/ (auto-versioned)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

BLOCK_TLAYERS = {0: (0, 2), 1: (2, 4), 2: (4, 6), 3: (6, 8),
                 4: (8, 10), 5: (10, 12), 6: (12, 14), 7: (14, 15)}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # .../corsac

KERNEL_SCRIPT = textwrap.dedent('''\
    import os, sys, time, glob
    t0 = time.time()
    print("=== XIIV SGBD Block {block} on Kaggle GPU (HF data) ===", flush=True)

    # ---- 0. deps ----
    os.system("pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1")
    os.system("pip install -q safetensors huggingface_hub numpy 2>&1 | tail -1")

    import torch
    assert torch.cuda.is_available(), "NO CUDA - aborting"
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

    # ---- 1. training code (embedded; repo is private so no git clone) ----
    import json as _json
    _bundle = _json.loads('{CODE_BUNDLE}')
    for _rel, _src in _bundle.items():
        _p = "/kaggle/working/pkg/" + _rel
        os.makedirs(os.path.dirname(_p), exist_ok=True)
        open(_p, "w").write(_src)
    print(f"[code] wrote {len(_bundle)} files", flush=True)
    del _bundle

    # ---- 2. data from HuggingFace (long-lived token, embedded at build) ----
    from huggingface_hub import snapshot_download
    _tok = "{HF_TOKEN}"
    _t0, _t1 = {t_start}, {t_end}
    _pats = [f"iter0/layer{_t0:02d}_c*.pt", f"iter0/layer{_t1-1:02d}_c*.pt",
             "ckpt_8218.pt"]
    _ddir = snapshot_download(
        repo_id="elova89/xiiv-sgbd-teacher-cache", repo_type="dataset",
        allow_patterns=_pats, token=_tok, local_dir="/kaggle/working/sgbd_cache")
    del _tok
    import glob as _g
    _n0 = len(_g.glob(f"/kaggle/working/sgbd_cache/iter0/layer{_t0:02d}_c*.pt"))
    _n1 = len(_g.glob(f"/kaggle/working/sgbd_cache/iter0/layer{_t1-1:02d}_c*.pt"))
    assert _n0 == 24 and _n1 == 24, f"chunks: {_n0}/{_n1}"
    assert os.path.exists("/kaggle/working/sgbd_cache/ckpt_8218.pt"), "ckpt missing"
    print(f"[data] {_n0}+{_n1} chunks + ckpt ready ({(time.time()-t0)/60:.1f} min)",
          flush=True)

    # ---- 3. train ----
    # REPO resolves to /kaggle/working (script is at /kaggle/working/pkg/),
    # so --cache-dir/--out-dir are relative to /kaggle/working.
    cmd = ("PYTHONPATH=/kaggle/working/pkg python /kaggle/working/pkg/sgbd_train_block.py "
           f"--block {block} --steps {steps} --device cuda --batch-size {batch} "
           f"--cache-dir sgbd_cache/iter0 --out-dir sgbd_blocks "
           f"--student-ckpt /kaggle/working/sgbd_cache/ckpt_8218.pt "
           f"--save-every 200 --log-every 25 {resume}")
    print("[train]", cmd[:120] + "...", flush=True)
    rc = os.system(cmd)
    assert rc == 0, f"training exited rc={rc}"
    print(f"=== DONE block {block}: {steps} steps in {(time.time()-t0)/60:.1f} min ===",
          flush=True)
''')


def get_code_bundle():
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


def build_notebook(block, steps, batch_size, resume_step, hf_token):
    t_start, t_end = BLOCK_TLAYERS[block]
    resume = f"--resume-step {resume_step}" if resume_step > 0 else ""
    code_bundle = get_code_bundle().replace("\\", "\\\\").replace("'", "\\'")
    script = KERNEL_SCRIPT.replace("{block}", str(block)) \
        .replace("{steps}", str(steps)) \
        .replace("{batch}", str(batch_size)) \
        .replace("{t_start}", str(t_start)) \
        .replace("{t_end}", str(t_end)) \
        .replace("{resume}", resume) \
        .replace("{CODE_BUNDLE}", code_bundle) \
        .replace("{HF_TOKEN}", hf_token)
    assert "{CODE_BUNDLE}" not in script and "{HF_TOKEN}" not in script
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (0 <= args.block <= 7):
        sys.exit("block must be 0-7")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("HF_TOKEN env var not set")
    if len(hf_token) < 20 or not hf_token.startswith("hf_"):
        sys.exit("HF_TOKEN looks invalid")

    slug = f"x0d4n7/xiiv-sgbd-block{args.block}-hf"
    title = slug.split("/", 1)[1]

    nb_json = build_notebook(args.block, args.steps, args.batch_size,
                             args.resume_step, hf_token)
    nb = json.loads(nb_json)
    assert nb["cells"][0]["cell_type"] == "code"
    # sanity: token appears exactly once in the notebook source
    src = "".join(nb["cells"][0]["source"])
    assert src.count(hf_token) == 1, "token must appear exactly once"
    print(f"[build] notebook OK ({len(nb_json)//1024} KB, 1 code cell)")

    tmpdir = tempfile.mkdtemp(prefix=f"xiiv_kaggle_b{args.block}_hf_")
    with open(os.path.join(tmpdir, "notebook.ipynb"), "w") as f:
        f.write(nb_json)
    meta = {
        "id": slug, "title": title, "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(tmpdir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[build] kernel dir: {tmpdir}")
    print(f"  id: {slug} | steps: {args.steps} | gpu: True | private: True")

    if args.dry_run:
        print("[dry-run] not pushing.")
        return

    r = subprocess.run(["kaggle", "kernels", "push", "-p", tmpdir],
                       capture_output=True, text=True, timeout=300)
    # never print the token; show only the tail
    out = r.stdout[-2000:].replace(hf_token, "[TOKEN]")
    print(out)
    if r.returncode != 0:
        err = r.stderr[-2000:].replace(hf_token, "[TOKEN]")
        print(err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
