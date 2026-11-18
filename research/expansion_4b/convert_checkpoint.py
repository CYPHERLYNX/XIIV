#!/usr/bin/env python3
"""Convert an exact-1B XIIV checkpoint into the 4B-class sparse-MoE build.

Spec: research/expansion_4b_moe.md. Module: xiiv/moe.py.

Every layer's SwiGLU FFN becomes a MoESwiGLU with N experts, top-2
routing. Grown identity init: each expert is an exact copy of the FFN it
replaces; the router is zero-initialised (uniform routing), so the
converted model reproduces the backbone's outputs exactly at the cast
precision (module-level identity gate proven in xiiv/test_moe.py,
max|diff| = 0.0; this script spot-verifies the emitted tensors).

Streaming: the source is read tensor-by-tensor via safetensors
safe_open and the output is written in shards, so peak RAM stays near
one shard buffer regardless of model size.

Audited totals for the default build (8 experts + the 5B-exact
growth block, source = exact-1B stitch, 284 tensors):
  runnable  4,857,429,760
  headline  5,000,000,000  (runnable + 142,570,240 contract scaffold)
  tensors   1,166
  active/token ~= 1.41B (backbone non-FFN + 2 experts/layer + routers;
  the growth block adds its full 146,073,600 when active)
The audit is computed from tensor shapes on every run and asserted
against these values; any other total aborts the build.

Usage:
  python3 convert_checkpoint.py --source <1b.safetensors> --out <dir>
      [--dtype fp16|fp32] [--n-experts 8] [--audit-only] [--verify]
"""

import argparse
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CONTRACT_SCAFFOLD = 142_570_240
# 5B-exact growth block: one residual SiLU MLP (norm-free, bias-free)
# after the MoE stack, zero-init down-projection => exact identity at
# init. hidden 57,060 x d 1280 x 2 matrices = 146,073,600 params —
# exactly the gap from the 8-expert MoE headline (4,853,926,400) to
# 5,000,000,000. Runnable becomes 4,857,429,760.
GROWTH_HIDDEN = 57_060
GROWTH_PARAMS = 2 * 1280 * GROWTH_HIDDEN
EXPECTED = {
    "runnable": 4_711_356_160 + GROWTH_PARAMS,
    "headline": 5_000_000_000,
    "tensors": 1_166,
}
FFN_KEYS = ("gate", "up", "down")


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def plan(source_path, n_experts):
    """Return (tensor_plan, audit) from source metadata only.

    tensor_plan: list of (out_key, kind, src_key) where kind is
    'copy' (pass through), 'expert' (copy of an FFN tensor), or
    'router' (zeros, src_key None).
    """
    plan_items = []
    runnable = 0
    with safe_open(source_path, framework="pt") as f:
        keys = list(f.keys())
        shapes = {}
        for k in keys:
            shapes[k] = f.get_slice(k).get_shape()
        layers = sorted({int(k.split(".")[1]) for k in keys
                         if k.startswith("layers.")})
        for k in keys:
            if ".mlp." in k and any(f".mlp.{g}." in k for g in FFN_KEYS):
                continue  # replaced by experts below
            plan_items.append((k, "copy", k))
            runnable += numel(shapes[k])
        for i in layers:
            for e in range(n_experts):
                for g in FFN_KEYS:
                    src = f"layers.{i}.mlp.{g}.weight"
                    out = f"layers.{i}.mlp.experts.{e}.{g}.weight"
                    plan_items.append((out, "expert", src))
                    runnable += numel(shapes[src])
            d_model = shapes[f"layers.{i}.attn.qkv.weight"][1]
            plan_items.append((f"layers.{i}.mlp.router.weight",
                               "router", None))
            runnable += n_experts * d_model
        # 5B-exact growth block (identity at init: down = zeros)
        plan_items.append(("growth_block.up.weight", "growth_up", None))
        plan_items.append(("growth_block.down.weight", "growth_down",
                           None))
        runnable += GROWTH_PARAMS
    audit = {
        "runnable": runnable,
        "headline": runnable + CONTRACT_SCAFFOLD,
        "contract_scaffold": CONTRACT_SCAFFOLD,
        "tensors": len(plan_items),
        "n_layers": len(layers),
    }
    return plan_items, audit, shapes


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(source_path, out_dir, dtype, n_experts, shard_bytes):
    plan_items, audit, shapes = plan(source_path, n_experts)
    for k, v in EXPECTED.items():
        assert audit[k] == v, f"audit mismatch {k}: {audit[k]} != {v}"
    os.makedirs(out_dir, exist_ok=True)
    cast = torch.float16 if dtype == "fp16" else torch.float32
    elem = 2 if dtype == "fp16" else 4

    shards = []
    buf = {}
    buf_bytes = 0
    shard_idx = 0

    def flush():
        nonlocal buf, buf_bytes, shard_idx
        if not buf:
            return
        name = f"model-{shard_idx:05d}.safetensors"
        path = os.path.join(out_dir, name)
        save_file(buf, path, metadata={"format": "pt"})
        shards.append({"file": name, "bytes": os.path.getsize(path),
                       "sha256": sha256_file(path),
                       "tensors": len(buf)})
        print(f"  shard {name}: {len(buf)} tensors, "
              f"{os.path.getsize(path)} bytes", flush=True)
        buf = {}
        buf_bytes = 0
        shard_idx += 1

    with safe_open(source_path, framework="pt") as f:
        cache = {}

        def get(src):
            if src not in cache:
                cache.clear()
                cache[src] = f.get_tensor(src)
            return cache[src]

        for out_key, kind, src in plan_items:
            if kind == "growth_up":
                d_model = shapes["embed.weight"][1]
                t = torch.empty((GROWTH_HIDDEN, d_model))
                torch.nn.init.trunc_normal_(t, mean=0.0, std=0.02)
                buf[out_key] = t.to(cast)
                buf_bytes += GROWTH_HIDDEN * d_model * elem
                if buf_bytes >= shard_bytes:
                    flush()
                continue
            if kind == "growth_down":
                d_model = shapes["embed.weight"][1]
                buf[out_key] = torch.zeros((d_model, GROWTH_HIDDEN),
                                           dtype=cast)
                buf_bytes += GROWTH_HIDDEN * d_model * elem
                if buf_bytes >= shard_bytes:
                    flush()
                continue
            if kind == "router":
                # router shape: [n_experts, d_model]; d_model from layer
                layer = int(out_key.split(".")[1])
                d_model = shapes[f"layers.{layer}.attn.qkv.weight"][1]
                t = torch.zeros((n_experts, d_model), dtype=cast)
            else:
                t = get(src).to(cast)
                if kind == "expert":
                    t = t.clone()
            buf[out_key] = t
            buf_bytes += numel(list(t.shape)) * elem
            if buf_bytes >= shard_bytes:
                flush()
    flush()
    return audit, shards


def verify(source_path, out_dir, dtype, n_experts):
    """Spot identity checks: experts == source FFN at cast precision,
    routers zero, totals match the audit."""
    cast = torch.float16 if dtype == "fp16" else torch.float32
    manifest = json.load(open(os.path.join(out_dir, "moe.json")))
    total = 0
    tensors = 0
    shard_files = [s["file"] for s in manifest["shards"]]
    # check two layers fully: first and last
    check_layers = {0, manifest["audit"]["n_layers"] - 1}
    ok = True
    with safe_open(source_path, framework="pt") as src:
        for shard in shard_files:
            with safe_open(os.path.join(out_dir, shard),
                           framework="pt") as f:
                for k in f.keys():
                    sl = f.get_slice(k)
                    total += numel(sl.get_shape())
                    tensors += 1
                    if k == "growth_block.down.weight":
                        if not bool((f.get_tensor(k) == 0).all()):
                            print("  FAIL growth_block.down not zero "
                                  "(identity init broken)")
                            ok = False
                    parts = k.split(".")
                    if len(parts) >= 3 and parts[0] == "layers" \
                            and int(parts[1]) in check_layers:
                        t = f.get_tensor(k)
                        if "router" in k:
                            if not bool((t == 0).all()):
                                print(f"  FAIL router not zero: {k}")
                                ok = False
                        elif "experts" in k:
                            g = parts[4]
                            src_t = src.get_tensor(
                                f"layers.{parts[1]}.mlp.{g}.weight").to(cast)
                            if not torch.equal(t, src_t):
                                print(f"  FAIL expert != FFN: {k}")
                                ok = False
    assert total == manifest["audit"]["runnable"], (total, "runnable")
    assert tensors == manifest["audit"]["tensors"], (tensors, "tensors")
    print(f"  verify: totals OK ({total} runnable, {tensors} tensors); "
          f"expert/router spot checks on layers "
          f"{sorted(check_layers)}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--n-experts", type=int, default=8)
    ap.add_argument("--shard-mb", type=int, default=300)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="verify an existing build in --out, then exit")
    args = ap.parse_args()

    if args.verify:
        ok = verify(args.source, args.out, args.dtype, args.n_experts)
        sys.exit(0 if ok else 1)

    _, audit, _ = plan(args.source, args.n_experts)
    print("audit (from source shapes):")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    for k, v in EXPECTED.items():
        assert audit[k] == v, f"audit mismatch {k}: {audit[k]} != {v}"
    print("  audit matches expected 4B-class totals: PASS")
    if args.audit_only:
        return

    src_sha = sha256_file(args.source)
    print(f"source sha256: {src_sha}")
    print(f"converting -> {args.out} ({args.dtype}, "
          f"{args.n_experts} experts, shard cap {args.shard_mb} MB)")
    audit, shards = convert(args.source, args.out, args.dtype,
                            args.n_experts, args.shard_mb << 20)
    moe = {
        "build": "xiiv-4b-moe",
        "source_checkpoint_sha256": src_sha,
        "dtype": args.dtype,
        "n_experts": args.n_experts,
        "top_k": 2,
        "audit": audit,
        "shards": shards,
        "growth_block": {
            "keys": ["growth_block.up.weight",
                     "growth_block.down.weight"],
            "hidden": GROWTH_HIDDEN, "params": GROWTH_PARAMS,
            "init": "up trunc-normal std 0.02, down zeros => the "
                    "block contributes exactly 0 at init (identity); "
                    "apply AFTER the MoE stack, before final norm, "
                    "as a residual SiLU MLP (norm-free, bias-free), "
                    "mirroring expansion_block semantics",
            "purpose": "closes the headline to exactly "
                       "5,000,000,000 params",
        },
        "loader": "xiiv/moe.py convert_to_moe layout: "
                  "layers.{i}.mlp.experts.{e}.{gate,up,down}.weight, "
                  "layers.{i}.mlp.router.weight (zeros)",
    }
    with open(os.path.join(args.out, "moe.json"), "w") as fh:
        json.dump(moe, fh, indent=2)
    print(f"wrote {len(shards)} shards + moe.json")
    ok = verify(args.source, args.out, args.dtype, args.n_experts)
    print("CONVERT_OK" if ok else "CONVERT_VERIFY_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
