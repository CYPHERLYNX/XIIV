#!/usr/bin/env python3
"""Assemble the trained XIIV-1B from its 8 SGBD block checkpoints.

Stitch procedure (derived from xiiv/sgbd_train_block.py, xiiv/model.py,
docs/MODEL_SPEC_1B.md, and the on-disk parent
checkpoints_1b_gn/ckpt_8218.safetensors):

  1. The 1B model has 36 layers (d=1280, 20 heads, ffn=3584, vocab=32768).
     SGBD trained it as 8 independent layer slices ("blocks"):
     block 0 -> layers 0-4, block 1 -> 5-9, block 2 -> 10-14,
     block 3 -> 15-19, block 4 -> 20-23, block 5 -> 24-27,
     block 6 -> 28-31, block 7 -> 32-35.
  2. Each block's ``block{b}_final.pt`` holds, per layer, exactly the 7
     main-path tensors (attn_norm, attn.qkv, attn.out, mlp_norm, mlp.gate,
     mlp.up, mlp.down) plus FitNets scaffolding (in_proj/out_proj) that
     was trained jointly and is DISCARDED at composition (it is not part
     of the model). Step checkpoints (``block{b}_step{step}.pt``) additionally
     carry optimizer state and are NOT used: the ``*_final.pt`` files are
     the authoritative trained weights (all verified step=1000).
  3. Embeddings and the final norm were never trained by SGBD (block files
     contain no embed/final_norm keys). They are carried over verbatim
     from the grown-init parent (F16 -> cast to fp32 here).
  4. Contract machinery (per-layer contract_attn on layers 0-26 and the
     contract_encoder) was EXCLUDED from SGBD distillation by design
     (targets were main-path only) and is NOT in the runnable output.
     Reasons, all recorded in the manifest:
       - SGBD never trained it, so including it would present grown-init
         cross-attention as "trained" weights;
       - the parent's contract_attn.kv is [1280, 1280] (a truncated
         artifact of the old growth bug: the K/V halves were never
         restacked; current code expects [2560, 1280]), so a
         contract-enabled load fails on shape;
       - xiiv/sample.py rejects unexpected keys, so a file carrying
         contract keys cannot load into a contract-free model either.
     The parent file is left untouched, so contract parts can be
     re-attached later by a separate, explicit repair decision.
  5. Output is a single fp32 safetensors file (no pickle), tied weights
     stored ONCE (embed.weight covers lm_head.weight, matching
     xiiv/sample.py's loader), plus a sibling config.json (the loader
     requires config.json next to the weights) and a manifest.json with
     full provenance.

Parameter-count reconciliation (read this before quoting a number):
  - Runnable output (trained layers + parent embed/final_norm):
    773,418,240 unique parameters.
  - Contract machinery excluded from the runnable file:
    142,570,240 parameters (contract_attn 132,779,520 + encoder 9,790,720).
  - Headline figure from docs/MODEL_SPEC_1B.md:
    773,418,240 + 142,570,240 = 915,988,480 unique parameters.
  Quoting 915,988,480 for the runnable file alone is WRONG. The manifest
  records all three numbers with their provenance.

Memory discipline: the script is streaming and two-pass. The plan phase
reads only the parent safetensors header (no tensor data). The write
phase holds at most ONE block file (~0.4 GB fp32) plus ONE output tensor
buffer (largest: embed, ~168 MB fp32) at a time, plus interpreter/torch
overhead. Expected peak RSS ~1.2-1.5 GB. It never loads all 8 blocks at
once and never loads the full parent. Sources are opened read-only and
never modified; output is written to a temp file and atomically renamed
on success (refuses to overwrite existing outputs unless --overwrite).

Usage (daytime, supervised, box quiet):
    .venv/bin/python scripts/stitch_1b_sgbd.py \
        --blocks-dir sgbd_blocks \
        --parent checkpoints_1b_gn/ckpt_8218.safetensors \
        --out-dir checkpoints_1b_sgbd

Preflight (cheap, header-only; writes nothing, imports no torch):
    .venv/bin/python scripts/stitch_1b_sgbd.py --plan-only
"""
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Layer slices per block, from xiiv/sgbd_train_block.py BLOCKS.
DEFAULT_BLOCK_MAP = [(0, 5), (5, 10), (10, 15), (15, 20),
                     (20, 24), (24, 28), (28, 32), (32, 36)]

# Main-path tensor suffixes per layer, in canonical output order.
MAIN_KEYS = ["attn_norm.weight", "attn.qkv.weight", "attn.out.weight",
             "mlp_norm.weight", "mlp.gate.weight", "mlp.up.weight",
             "mlp.down.weight"]
# Contract suffixes (present on parent layers 0-26 only; excluded here).
CONTRACT_KEYS = ["contract_attn.gate", "contract_attn.kv.weight",
                 "contract_attn.out.weight", "contract_attn.q.weight",
                 "contract_norm.weight"]
# FitNets scaffolding keys in block files (trained jointly, discarded).
SCAFFOLD_KEYS = {"in_proj", "out_proj"}

EXPECTED_STEP = 1000  # all 8 block finals verified step=1000 (2026-10-26)


class StitchError(Exception):
    pass


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_plan(parent_path, block_map):
    """Read ONLY the parent header; derive the full output key plan."""
    from safetensors import safe_open
    with safe_open(str(parent_path), framework="np") as f:
        parent_keys = list(f.keys())
        parent_meta = {k: (f.get_slice(k).get_shape(),
                           f.get_slice(k).get_dtype()) for k in parent_keys}
    parent_set = set(parent_keys)
    # Contract layers are derived from the header, never hardcoded.
    contract_layers = sorted({
        int(k.split(".")[1]) for k in parent_keys
        if k.startswith("layers.") and ".contract_attn." in k})
    # Validate the block map covers every layer exactly once, contiguously.
    covered = []
    for (s, e) in block_map:
        if not (0 <= s < e):
            raise StitchError(f"invalid block slice {(s, e)}")
        covered.extend(range(s, e))
    n_layers = max(covered) + 1 if covered else 0
    if sorted(covered) != list(range(n_layers)) or len(set(covered)) != len(covered):
        raise StitchError(f"block map does not cover layers 0..{n_layers - 1} "
                          f"exactly once: {block_map}")
    if "embed.weight" not in parent_set or "final_norm.weight" not in parent_set:
        raise StitchError("parent header lacks embed.weight/final_norm.weight")
    # Output key list (canonical order): embed, per-layer main keys,
    # final_norm. Tied lm_head stored once via embed (never duplicated).
    out_keys = ["embed.weight"]
    for i in range(n_layers):
        for suffix in MAIN_KEYS:
            key = f"layers.{i}.{suffix}"
            if key not in parent_set:
                raise StitchError(f"parent header lacks main-path key {key}")
            out_keys.append(key)
    out_keys.append("final_norm.weight")
    shapes = {}
    for key in out_keys:
        shape, _dtype = parent_meta[key]
        shapes[key] = tuple(int(s) for s in shape)
    n_params = sum(int(np.prod(shapes[k])) for k in out_keys)
    # Contract totals, for the manifest reconciliation (not in output).
    contract_keys = [k for k in parent_keys
                     if ".contract_attn." in k or ".contract_norm." in k
                     or k.startswith("contract_encoder.")]
    contract_params = sum(int(np.prod(parent_meta[k][0])) for k in contract_keys)
    return {
        "parent_keys": parent_keys,
        "parent_meta": {k: (list(v[0]), str(v[1])) for k, v in parent_meta.items()},
        "contract_layers": contract_layers,
        "contract_keys": sorted(contract_keys),
        "contract_params": contract_params,
        "n_layers": n_layers,
        "out_keys": out_keys,
        "shapes": {k: list(v) for k, v in shapes.items()},
        "out_params": n_params,
    }


def safetensors_header_bytes(entries):
    """entries: list of (key, dtype_str, shape, (start, end)) in order."""
    hdr = {}
    for key, dtype_str, shape, (start, end) in entries:
        hdr[key] = {"dtype": dtype_str, "shape": list(shape),
                    "data_offsets": [int(start), int(end)]}
    raw = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    return raw


def compute_offsets(out_keys, shapes):
    """Fixpoint: header length depends on offsets, offsets on header len."""
    offsets = {}
    pos = 0
    header_len = 0
    for _ in range(10):
        entries = []
        pos = 0
        for key in out_keys:
            nbytes = int(np.prod(shapes[key])) * 4  # fp32
            entries.append((key, "F32", shapes[key], (pos, pos + nbytes)))
            pos += nbytes
        raw = safetensors_header_bytes(entries)
        new_len = len(raw)
        if new_len == header_len:
            return entries, raw
        header_len = new_len
    raise StitchError("safetensors header length did not converge")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks-dir", default="sgbd_blocks")
    ap.add_argument("--parent",
                    default="checkpoints_1b_gn/ckpt_8218.safetensors")
    ap.add_argument("--out-dir", default="checkpoints_1b_sgbd")
    ap.add_argument("--block-map", default=None,
                    help="JSON list of [start, end) layer slices per block; "
                         "default is the real 8-block SGBD map")
    ap.add_argument("--expect-step", type=int, default=EXPECTED_STEP)
    ap.add_argument("--plan-only", action="store_true",
                    help="validate the plan (parent header + block files) "
                         "and exit without importing torch or writing "
                         "any output")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    blocks_dir = (Path(args.blocks_dir) if Path(args.blocks_dir).is_absolute()
                  else repo / args.blocks_dir)
    parent_path = (Path(args.parent) if Path(args.parent).is_absolute()
                   else repo / args.parent)
    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute()
               else repo / args.out_dir)
    block_map = ([tuple(x) for x in json.loads(args.block_map)]
                 if args.block_map else DEFAULT_BLOCK_MAP)
    if not parent_path.exists():
        raise StitchError(f"parent not found: {parent_path}")
    if not blocks_dir.is_dir():
        raise StitchError(f"blocks dir not found: {blocks_dir}")

    out_weights = out_dir / "xiiv_1b_sgbd.safetensors"
    out_config = out_dir / "config.json"
    out_manifest = out_dir / "manifest.json"

    print(f"[stitch] parent: {parent_path}", flush=True)
    plan = build_plan(parent_path, block_map)
    print(f"[stitch] layers: {plan['n_layers']}, output tensors: "
          f"{len(plan['out_keys'])}, output params: {plan['out_params']:,}",
          flush=True)
    print(f"[stitch] contract layers in parent (excluded): "
          f"{plan['contract_layers']}", flush=True)
    print(f"[stitch] contract params excluded: {plan['contract_params']:,}",
          flush=True)
    entries, header_raw = compute_offsets(plan["out_keys"], plan["shapes"])
    total_bytes = 8 + len(header_raw) + entries[-1][3][1]
    print(f"[stitch] output size: {total_bytes:,} bytes "
          f"({total_bytes / 1e9:.2f} GB)", flush=True)

    if args.plan_only:
        # Header-only preflight: block finals must exist (stat only, no
        # reads, no hashes, no torch import, no output directory created).
        for b, (s, e) in enumerate(block_map):
            p = blocks_dir / f"block{b}" / f"block{b}_final.pt"
            if not p.exists():
                raise StitchError(f"missing block file: {p}")
            print(f"[stitch] plan-only: block {b} layers {s}-{e - 1}: "
                  f"{p} ({p.stat().st_size:,} bytes)", flush=True)
        print("[stitch] plan-only: OK — no outputs written", flush=True)
        return 0

    if not args.overwrite:
        for p in (out_weights, out_config, out_manifest):
            if p.exists():
                raise StitchError(f"refusing to overwrite existing {p} "
                                  f"(pass --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors import safe_open

    # Source file hashes (chunked reads; sources never opened for writing).
    block_sources = []
    for b, (s, e) in enumerate(block_map):
        p = blocks_dir / f"block{b}" / f"block{b}_final.pt"
        if not p.exists():
            raise StitchError(f"missing block file: {p}")
        block_sources.append({
            "block": b, "layers": [s, e], "path": str(p),
            "size_bytes": p.stat().st_size, "sha256": sha256_file(p),
        })
    parent_sha = sha256_file(parent_path)
    parent_size = parent_path.stat().st_size

    # ---- streaming write (temp file + atomic rename) ----
    tmp_path = out_dir / ".xiiv_1b_sgbd.safetensors.tmp"
    tensor_sha = {}
    written_keys = []
    try:
        with safe_open(str(parent_path), framework="np") as parent_f:
            with open(tmp_path, "wb") as out:
                out.write(len(header_raw).to_bytes(8, "little"))
                out.write(header_raw)
                # embed.weight + final_norm.weight come from the parent;
                # layer keys come from block files, one block at a time.
                parent_wanted = {"embed.weight", "final_norm.weight"}
                # Map output key -> (block index, short key) for layer keys.
                layer_src = {}
                for b, (s, e) in enumerate(block_map):
                    for i in range(s, e):
                        for suffix in MAIN_KEYS:
                            layer_src[f"layers.{i}.{suffix}"] = (b, suffix)
                loaded_block = None
                loaded_ckpt = None
                for key, _dtype, shape, (start, end) in entries:
                    if key in parent_wanted:
                        arr = parent_f.get_slice(key)[:]
                        arr = np.ascontiguousarray(arr.astype(np.float32))
                    else:
                        b, suffix = layer_src[key]
                        if loaded_block != b:
                            if loaded_ckpt is not None:
                                del loaded_ckpt
                                gc.collect()
                            bp = blocks_dir / f"block{b}" / f"block{b}_final.pt"
                            ckpt = torch.load(str(bp), map_location="cpu",
                                              weights_only=True)
                            # ---- fail-loud gates on every block file ----
                            if ckpt.get("block") != b:
                                raise StitchError(
                                    f"{bp}: block field {ckpt.get('block')} != {b}")
                            if ckpt.get("step") != args.expect_step:
                                raise StitchError(
                                    f"{bp}: step {ckpt.get('step')} != "
                                    f"{args.expect_step}")
                            got_layers = [li for (li, _sd) in ckpt["layers"]]
                            if got_layers != list(range(block_map[b][0],
                                                       block_map[b][1])):
                                raise StitchError(
                                    f"{bp}: layer indices {got_layers} != "
                                    f"expected {list(range(block_map[b][0], block_map[b][1]))}")
                            extra = set(ckpt.keys()) - {
                                "block", "layers", "step", "loss_hist"} - SCAFFOLD_KEYS
                            if extra:
                                raise StitchError(
                                    f"{bp}: unexpected top-level keys {extra}")
                            loaded_block, loaded_ckpt = b, ckpt
                            print(f"[stitch] block {b}: layers "
                                  f"{block_map[b][0]}-{block_map[b][1] - 1} "
                                  f"loaded, step {ckpt['step']}", flush=True)
                        layer_idx = int(key.split(".")[1])
                        # layers list is [(global_idx, state_dict), ...]
                        sd = None
                        for li, sdi in loaded_ckpt["layers"]:
                            if li == layer_idx:
                                sd = sdi
                                break
                        if sd is None:
                            raise StitchError(f"block {b} lacks layer {layer_idx}")
                        if set(sd.keys()) != set(MAIN_KEYS):
                            raise StitchError(
                                f"block {b} layer {layer_idx} keys "
                                f"{sorted(sd.keys())} != main-path keys")
                        t = sd[suffix]
                        if tuple(t.shape) != tuple(shape):
                            raise StitchError(
                                f"{key}: block tensor shape {tuple(t.shape)} "
                                f"!= parent shape {tuple(shape)}")
                        arr = np.ascontiguousarray(
                            t.detach().cpu().numpy().astype(np.float32))
                    if arr.shape != tuple(shape):
                        raise StitchError(f"{key}: shape {arr.shape} != {shape}")
                    if not np.isfinite(arr).all():
                        raise StitchError(f"{key}: non-finite values")
                    raw = arr.tobytes(order="C")
                    if len(raw) != end - start:
                        raise StitchError(f"{key}: byte length {len(raw)} "
                                          f"!= {end - start}")
                    tensor_sha[key] = hashlib.sha256(raw).hexdigest()
                    out.write(raw)
                    written_keys.append(key)
                    del arr, raw
                if loaded_ckpt is not None:
                    del loaded_ckpt
                    gc.collect()
        os.replace(tmp_path, out_weights)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise

    # ---- sibling config.json (required by xiiv/sample.py's loader) ----
    # Valid ModelConfig fields only (from_json passes kwargs to the
    # dataclass); mask/eos 2/3 agree with data_32k/meta.json and
    # weights/tokenizer_32k.json ([MASK]=2, [EOS]=3).
    # All dims are derived from the parent header, never hardcoded:
    # head_dim 64 is the documented invariant (docs/MODEL_SPEC_1B.md:
    # "Head dim preserved at 64").
    d_model = plan["shapes"]["embed.weight"][1]
    vocab = plan["shapes"]["embed.weight"][0]
    ffn_dim = plan["shapes"]["layers.0.mlp.gate.weight"][0]
    n_heads = d_model // 64
    config = {
        "vocab_size": vocab, "mask_id": 2, "eos_id": 3,
        "d_model": d_model, "n_layers": plan["n_layers"],
        "n_heads": n_heads, "ffn_dim": ffn_dim, "rope_theta": 10000.0,
        "norm_eps": 1e-6, "dropout": 0.0, "tie_embeddings": True,
        "seq_len": 128, "t_min": 0.15, "t_max": 1.0,
        "image_enabled": False, "image_size": 64, "image_patch_size": 16,
        "contract_enabled": False, "contract_d_model": 256,
        "contract_n_layers": 2, "contract_n_heads": 4,
        "contract_ffn_dim": 512, "contract_max_tokens": 96,
    }
    # n_heads/ffn_dim are cross-checked, not assumed, for the real map:
    qkv_shape = plan["shapes"]["layers.0.attn.qkv.weight"]
    if qkv_shape[0] != 3 * d_model:
        raise StitchError(f"unexpected qkv shape {qkv_shape}")
    out_config.write_text(json.dumps(config, indent=2) + "\n")

    manifest = {
        "model": "XIIV-1B (SGBD-assembled, runnable main-path model)",
        "method": "Seed-Grown Block Distillation Phase 2 composition "
                  "(xiiv/sgbd_train_block.py BLOCKS map); scaffolding discarded",
        "lineage": {
            "parent": str(parent_path),
            "parent_size_bytes": parent_size,
            "parent_sha256": parent_sha,
            "parent_format": "safetensors F16, 406 tensors (incl. tied lm_head dup)",
            "growth_lineage": "XIIV-133M @ step 8218 -> grown -> XIIV-500M -> "
                              "grown -> XIIV-1B @ step 8218 (docs/MODEL_SPEC_1B.md)",
            "note": "parent file untouched; read-only throughout",
        },
        "blocks": [
            {**src, "step": args.expect_step,
             "params_trained": sum(
                 int(np.prod(plan["shapes"][f"layers.{i}.{s}"]))
                 for i in range(src["layers"][0], src["layers"][1])
                 for s in MAIN_KEYS)}
            for src in block_sources
        ],
        "discarded_scaffolding": {
            "keys_per_block": sorted(SCAFFOLD_KEYS),
            "reason": "FitNets dimension scaffolding trained jointly with each "
                      "block (teacher 704 -> student 1280); not part of the "
                      "model; discarded at composition per sgbd_train_block.py",
        },
        "excluded_contract": {
            "keys": plan["contract_keys"],
            "params": plan["contract_params"],
            "contract_layers_in_parent": plan["contract_layers"],
            "reasons": [
                "SGBD never trained contract parts (targets were main-path only)",
                "parent contract_attn.kv is [1280, 1280] (truncated growth "
                "artifact; current xiiv/model.py expects [2560, 1280]) — a "
                "contract-enabled load fails on shape",
                "xiiv/sample.py rejects unexpected keys, so contract keys "
                "cannot ride along in a contract-free runnable file",
            ],
            "status": "NOT in runnable output; parent preserved untouched for "
                      "a separate, explicit repair decision",
        },
        "embed_final_norm_provenance": "carried over verbatim from the "
            "grown-init parent (F16 -> fp32); never trained by SGBD",
        "tie_convention": "lm_head.weight stored once via embed.weight "
            "(tie_embeddings=True); loader reports lm_head.weight as the "
            "only missing key, which xiiv/sample.py explicitly allows",
        "counts": {
            "output_tensors": len(written_keys),
            "output_params_unique": plan["out_params"],
            "contract_params_excluded": plan["contract_params"],
            "headline_reconciliation": plan["out_params"]
                + plan["contract_params"],
            "headline_reference": 915988480,
            "headline_matches": (plan["out_params"]
                                 + plan["contract_params"]) == 915988480,
            "serialized_with_tied_dup": plan["out_params"]
                + plan["shapes"]["embed.weight"][0]
                * plan["shapes"]["embed.weight"][1],
        },
        "output": {
            "weights": str(out_weights),
            "size_bytes": out_weights.stat().st_size,
            "sha256": sha256_file(out_weights),
            "dtype": "F32",
            "tensor_sha256": tensor_sha,
        },
        "config": config,
        "validation_performed_by_script": [
            "block index field == directory block index, per file",
            "step field == expected step, per file",
            "layer indices == block map slice, exactly, per file",
            "per-layer state-dict keys == exactly the 7 main-path keys",
            "no unexpected top-level keys in block files (only scaffolding)",
            "every output tensor shape == parent header shape",
            "every output tensor finite (full scan at write time)",
            "output key set == parent main-path key set (embed + layers + "
            "final_norm), tied lm_head stored once",
        ],
        "command": sys.argv,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[stitch] DONE: {out_weights} ({out_weights.stat().st_size:,} bytes)",
          flush=True)
    print(f"[stitch] output params: {plan['out_params']:,} | "
          f"headline reconciliation: "
          f"{plan['out_params'] + plan['contract_params']:,} "
          f"(reference 915,988,480)", flush=True)
    print(f"[stitch] manifest: {out_manifest}", flush=True)


if __name__ == "__main__":
    main()
