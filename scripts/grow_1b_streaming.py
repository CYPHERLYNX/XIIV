#!/usr/bin/env python3
"""Memory-efficient 500M -> 1B growth via streaming state-dict surgery.

Never instantiates the 1B model. Grows each tensor from the 500M state dict
directly, building the 1B state dict incrementally. Peak RAM ~1.2GB.
"""
import argparse
import copy
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from xiiv.config import ModelConfig


def _dim_map(d_old, d_new, hd):
    """Map new dims to old dims (replicate whole heads)."""
    g = torch.arange(d_new) % d_old
    # Ensure whole-head replication for clean attention growth
    n_old_heads = d_old // hd
    n_new_heads = d_new // hd
    g = torch.cat([
        torch.arange(h * hd, (h + 1) * hd)
        for h in range(n_new_heads)
    ]) % d_old
    # Count replications per old dim (for column-split scaling)
    count = torch.zeros(d_old)
    for i in range(d_new):
        count[g[i]] += 1
    return g, count


def _row_map(f_old, f_new):
    gf = torch.arange(f_new) % f_old
    return gf, None


def _widen_norm(w, g):
    return w[g]


def _widen_rows(w, gf, noise_scale, gen):
    out = w[gf].clone()
    return out


def _widen_cols(w, g, count, noise_scale, gen):
    out = w[:, g].clone()
    # Split columns: divide replicated cols by replication count
    # to preserve function
    for j in range(w.shape[1]):
        mask = (g == j)
        if mask.sum() > 1:
            out[:, mask] /= mask.sum()
    return out


def _noise_like(shape, std, noise_scale, gen):
    if noise_scale <= 0:
        return None
    return torch.randn(shape, generator=gen) * std * noise_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ckpt", required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--d-new", type=int, required=True)
    ap.add_argument("--layers-new", type=int, required=True)
    ap.add_argument("--heads-new", type=int, required=True)
    ap.add_argument("--ffn-new", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-scale", type=float, default=0.01)
    ap.add_argument("--phase", type=int, default=0,
                    help="0=all, 1=layers 0-17 only, 2=layers 18-35 (loads phase1)")
    ap.add_argument("--phase1-ckpt", default=None,
                    help="phase 1 output (for phase 2)")
    args = ap.parse_args()

    gen = torch.Generator().manual_seed(args.seed)
    dt = torch.float16

    # Phase 2: load only the needed parent tensors (saved by phase 1)
    # to avoid OOM from loading the full 500M
    if args.phase == 2:
        print("Loading phase 1 config...")
        p1_cfg = torch.load(args.phase1_ckpt, map_location="cpu", weights_only=False)
        cfg_big = ModelConfig.from_json(p1_cfg["config"])
        del p1_cfg
        # Reconstruct parent cfg (500M dims)
        cfg_small = copy.deepcopy(cfg_big)
        cfg_small.d_model = 1024
        cfg_small.n_layers = 27
        cfg_small.n_heads = 16
        cfg_small.ffn_dim = 2816
        cfg_small.vocab_size = 32768
        parent_needed_path = os.path.join(
            os.path.dirname(os.path.abspath(args.phase1_ckpt)),
            "ckpt_1b_parent_needed.pt")
        print(f"Loading needed parent tensors from {parent_needed_path}...")
        pn = torch.load(parent_needed_path, map_location="cpu", weights_only=False)
        sd_small = {k: v.to(dt) if torch.is_tensor(v) else v
                    for k, v in pn["model"].items()}
        del pn
        gc.collect()
        # Dummy ck for step info
        ck = {"step": 8218}
        # Skip the cfg_big deepcopy below (already have it)
        _cfg_big_ready = True
    else:
        print("Loading 500M checkpoint...")
        ck = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)
        cfg_small = ModelConfig.from_json(ck["config"])
        sd_small = {k: v.to(dt) if torch.is_tensor(v) else v
                    for k, v in ck["model"].items()}
        _cfg_big_ready = False
    print(f"Parent: {sum(v.numel() for v in sd_small.values() if torch.is_tensor(v))/1e6:.1f}M params")

    if not _cfg_big_ready:
        cfg_big = copy.deepcopy(cfg_small)
        cfg_big.d_model = args.d_new
        cfg_big.n_layers = args.layers_new
        cfg_big.n_heads = args.heads_new
        cfg_big.ffn_dim = args.ffn_new

    d_old, d_new = cfg_small.d_model, cfg_big.d_model
    L_old, L_new = cfg_small.n_layers, cfg_big.n_layers
    h_old, h_new = cfg_small.n_heads, cfg_big.n_heads
    hd = d_old // h_old
    assert d_new // h_new == hd
    f_old, f_new = cfg_small.ffn_dim, cfg_big.ffn_dim

    g, count = _dim_map(d_old, d_new, hd)
    gf, _ = _row_map(f_old, f_new)

    sd_big = {}
    n_new = 0

    def grow_tensor(name, w, fn, *a, **kw):
        out = fn(w.float(), *a, **kw).to(dt)
        sd_big[name] = out
        return out.numel()

    # Phase 2 loads phase 1 output first
    if args.phase == 2:
        print(f"Loading phase 1 from {args.phase1_ckpt}...")
        p1 = torch.load(args.phase1_ckpt, map_location="cpu", weights_only=False)
        sd_big.update(p1["model"])
        cfg_big = ModelConfig.from_json(p1["config"])
        del p1
        gc.collect()
        print(f"Phase 1 loaded: {len(sd_big)} tensors")
        # Free parent layers 0-17 (already grown in phase 1)
        for i in range(18):
            p = f"layers.{i}."
            for k in list(sd_small.keys()):
                if k.startswith(p):
                    del sd_small[k]
        gc.collect()
        print("Freed parent layers 0-17")

    if args.phase != 2:
        print("Growing embeddings...")
        E = sd_small.pop("embed.weight").float()
        Enew = E[:, g].clone()
        # Add noise to new dims only
        std = E.std().item()
        n = _noise_like(Enew.shape, std, args.noise_scale, gen)
        if n is not None:
            nm = torch.zeros(d_new, dtype=torch.bool)
            nm[d_old:] = True
            Enew[:, nm] += n[:, nm]
        sd_big["embed.weight"] = Enew.to(dt)
        n_new += Enew.numel()
        del E, Enew
        gc.collect()

    print("Growing transformer blocks...")
    # Phase support: split layer range to bound RAM
    layer_start = 0
    layer_end = L_new
    if args.phase == 1:
        layer_end = 18
        print("PHASE 1: layers 0-17 only")
    elif args.phase == 2:
        layer_start = 18
        print("PHASE 2: layers 18-35")
        # Load phase 1 output
        p1 = torch.load(args.phase1_ckpt, map_location="cpu", weights_only=False)
        sd_big.update(p1["model"])
        del p1
        gc.collect()
    for i in range(layer_start, layer_end):
        p = f"layers.{i}."
        if i >= L_old:
            # New blocks as exact identity: zero output projections
            # We need the shapes from cfg_big; construct zeros
            # attn.out: (d_new, d_new), mlp.down: (d_new, f_new)
            sd_big[p + "attn_norm.weight"] = torch.ones(d_new, dtype=dt)
            sd_big[p + "mlp_norm.weight"] = torch.ones(d_new, dtype=dt)
            # QKV: (3*d_new, d_new) - zeros for identity
            sd_big[p + "attn.qkv.weight"] = torch.zeros(3*d_new, d_new, dtype=dt)
            sd_big[p + "attn.out.weight"] = torch.zeros(d_new, d_new, dtype=dt)
            sd_big[p + "mlp.gate.weight"] = torch.zeros(f_new, d_new, dtype=dt)
            sd_big[p + "mlp.up.weight"] = torch.zeros(f_new, d_new, dtype=dt)
            sd_big[p + "mlp.down.weight"] = torch.zeros(d_new, f_new, dtype=dt)
            # Contract attention (if present in parent)
            if f"{p}contract_attn.q.weight" in sd_small or True:
                # Check parent structure
                has_ca = any(k.startswith(f"layers.0.contract_attn") for k in sd_small)
                if has_ca:
                    sd_big[p + "contract_norm.weight"] = torch.ones(d_new, dtype=dt)
                    sd_big[p + "contract_attn.q.weight"] = torch.zeros(d_new, d_new, dtype=dt)
                    sd_big[p + "contract_attn.kv.weight"] = torch.zeros(2*d_new, d_new, dtype=dt)
                    sd_big[p + "contract_attn.out.weight"] = torch.zeros(d_new, d_new, dtype=dt)
                    sd_big[p + "contract_attn.gate"] = torch.zeros(d_new, dtype=dt)
            n_new += 1  # placeholder
        else:
            # Grow from parent layer
            # Norms: replicate
            sd_big[p + "attn_norm.weight"] = _widen_norm(
                sd_small.pop(p + "attn_norm.weight").float(), g).to(dt)
            sd_big[p + "mlp_norm.weight"] = _widen_norm(
                sd_small.pop(p + "mlp_norm.weight").float(), g).to(dt)
            # QKV: special handling for head replication
            qkv = sd_small.pop(p + "attn.qkv.weight").float()  # (3*d_old, d_old)
            # Reshape to (3, h_old, hd, d_old), replicate heads, reshape back
            qkv_r = qkv.reshape(3, h_old, hd, d_old)
            # New heads replicate head 0
            h_idx = torch.arange(h_new) % h_old
            qkv_new = qkv_r[:, h_idx].reshape(3*d_new, d_old)
            # Widen input dim (columns)
            qkv_new = _widen_cols(qkv_new, g, count, args.noise_scale, gen)
            sd_big[p + "attn.qkv.weight"] = qkv_new.to(dt)
            del qkv, qkv_r, qkv_new
            # Out proj: widen rows then cols (with split)
            out_w = sd_small.pop(p + "attn.out.weight").float()
            out_w = _widen_rows(out_w, g, args.noise_scale, gen)
            out_w = _widen_cols(out_w, g, count, args.noise_scale, gen)
            sd_big[p + "attn.out.weight"] = out_w.to(dt)
            del out_w
            # MLP: gate, up, down
            for name_in, name_out in [("mlp.gate.weight", None),
                                      ("mlp.up.weight", None),
                                      ("mlp.down.weight", None)]:
                w = sd_small.pop(p + name_in).float()
                if "down" in name_in:
                    # (d_old, f_old) -> (d_new, f_new)
                    w = _widen_rows(w, g, args.noise_scale, gen)
                    w = _widen_cols(w, gf, None, args.noise_scale, gen)
                else:
                    # (f_old, d_old) -> (f_new, d_new)
                    w = _widen_cols(w, g, count, args.noise_scale, gen)
                    w = _widen_rows(w, gf, args.noise_scale, gen)
                sd_big[p + name_in] = w.to(dt)
                del w
            # Contract attention
            ca_keys = [k for k in sd_small if k.startswith(p + "contract_attn")]
            if ca_keys:
                sd_big[p + "contract_norm.weight"] = _widen_norm(
                    sd_small.pop(p + "contract_norm.weight").float(), g).to(dt)
                # Q, KV, out, gate
                for cn in ["q.weight", "kv.weight", "out.weight"]:
                    w = sd_small.pop(p + "contract_attn." + cn).float()
                    w = _widen_rows(w, g, args.noise_scale, gen)
                    w = _widen_cols(w, g, count, args.noise_scale, gen)
                    sd_big[p + "contract_attn." + cn] = w.to(dt)
                    del w
                gate = torch.zeros(d_new, dtype=dt)
                gate[:d_old] = sd_small.pop(p + "contract_attn.gate").to(dt)
                sd_big[p + "contract_attn.gate"] = gate
        if (i + 1) % 3 == 0:
            gc.collect()
            print(f"  {i+1}/{L_new} layers", flush=True)

    # Phase 1 skips final norm, encoder, lm_head (done in phase 2)
    if args.phase != 1:
        print("Growing final norm...")
        sd_big["final_norm.weight"] = _widen_norm(
            sd_small.pop("final_norm.weight").float(), g).to(dt)

        # Contract encoder (if present)
        enc_keys = [k for k in sd_small if k.startswith("contract_encoder.")]
        if enc_keys:
            print("Growing contract encoder...")
            for k in enc_keys:
                if "tok_embed" in k or "proj" in k:
                    # These have d_model dim; widen
                    w = sd_small.pop(k).float()
                    if w.shape[-1] == d_old:
                        # Last dim is d_model
                        w = _widen_cols(w, g, count, args.noise_scale, gen)
                    elif w.shape[0] == d_old:
                        w = _widen_rows(w, g, args.noise_scale, gen)
                    sd_big[k] = w.to(dt)
                else:
                    sd_big[k] = sd_small.pop(k).to(dt)

        # Tie embeddings: lm_head shares storage with embed
        sd_big["lm_head.weight"] = sd_big["embed.weight"]

    total = sum(v.numel() for v in sd_big.values() if torch.is_tensor(v))
    # Subtract double-counted tied embeddings (if lm_head present)
    if "lm_head.weight" in sd_big:
        total_unique = total - sd_big["embed.weight"].numel()
    else:
        total_unique = total
    print(f"\nGrown: {total_unique/1e6:.1f}M unique params")

    out = {
        "model": sd_big,
        "step": ck["step"],
        "config": cfg_big.to_json(),
        "grown_from": os.path.basename(args.src_ckpt),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ckpt)), exist_ok=True)
    torch.save(out, args.out_ckpt)
    print(f"Wrote {args.out_ckpt}")

    # Phase 1: also save parent layers 18-26 + final norm + encoder for phase 2
    # (so phase 2 doesn't need to load the full 500M)
    if args.phase == 1:
        parent_needed = {}
        for i in range(18, 27):  # layers 18-26 (grown from parent)
            p = f"layers.{i}."
            for k in list(sd_small.keys()):
                if k.startswith(p):
                    parent_needed[k] = sd_small[k]
        # final norm and encoder
        for k in list(sd_small.keys()):
            if k == "final_norm.weight" or k.startswith("contract_encoder."):
                parent_needed[k] = sd_small[k]
        parent_path = os.path.join(
            os.path.dirname(os.path.abspath(args.out_ckpt)),
            "ckpt_1b_parent_needed.pt")
        torch.save({"model": parent_needed}, parent_path)
        print(f"Wrote {parent_path} "
              f"({sum(v.numel() for v in parent_needed.values())/1e6:.1f}M params)")


if __name__ == "__main__":
    main()
