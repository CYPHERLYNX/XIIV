#!/usr/bin/env python3
"""Grow a trained XIIV checkpoint to a larger architecture.

Approximately function-preserving growth (Net2Net-style):
- Width d_old -> d_new: new dims replicate the first (d_new-d_old) dims
  (whole heads). Linear layers with a widened INPUT get column-split
  weights (replicated columns divided by the replication count); layers
  with a widened OUTPUT get row-replicated weights. New attention heads
  replicate head 0 (Q/K/V rows copied, output-projection columns split).
- Symmetry breaking: small Gaussian noise (default 1% of the source
  tensor's std) is added to every NEW row/column. Without it, replicas
  receive identical gradients forever and the new capacity stays
  permanently redundant.
- Depth L_old -> L_new: new blocks appended as EXACT identity (all output
  projections zeroed; pre-norm residuals pass x through unchanged).
- Contract encoder internals are unchanged; only its output projection
  widens. Its token embedding follows --vocab-map like the main embedding.
- Optional --vocab-map (from scripts/expand_vocab.py): embedding rows for
  surviving surface forms are copied; genuinely new tokens are
  initialized from the mean of their old-token pieces (+ noise).

Exact preservation is impossible through RMSNorm (the rms statistic
changes when dims are added), so growth is approximate: verification
measures parent-vs-grown drift on random inputs and aborts past tolerance.

Output checkpoint: grown model state + source step/best_val/tokens_seen and
RNG state, so the data stream and LR schedule continue exactly where the
parent left off. Optimizer state is NOT carried (train.py starts it fresh
when "opt" is absent).

Example:
  .venv/bin/python scripts/grow_checkpoint.py \
      --src-ckpt checkpoints_100m_gn/ckpt_10000.pt \
      --out-ckpt checkpoints_200m_gn/ckpt_10000.pt \
      --d-new 768 --layers-new 18 --heads-new 12 \
      --vocab-map weights/vocab_map_8k_32k.json
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel


# ---------------------------------------------------------------- helpers

def _row_map(n_old, n_new):
    """Replication map for widening rows: new rows copy rows [0, extra)."""
    extra = n_new - n_old
    assert extra >= 0, "only growth is supported"
    g = torch.arange(n_new)
    count = torch.ones(n_old)
    if extra:
        g[n_old:] = torch.arange(extra)
        count[:extra] += 1.0
    return g, count


def _dim_map(d_old, d_new, head_dim):
    """Replication map for widening: new dims copy dims [0, extra)."""
    extra = d_new - d_old
    assert extra % head_dim == 0, \
        f"new dims ({extra}) must be whole heads of dim {head_dim}"
    return _row_map(d_old, d_new)


def _noise_like(shape, std, scale, gen):
    if scale <= 0 or std == 0:
        return None
    return torch.randn(shape, generator=gen) * (std * scale)


def _widen_rows(W, g, noise_scale, gen):
    """(d_old, in) -> (d_new, in): replicate rows; noise on new rows only."""
    W32 = W.float()
    out = W32[g].clone()
    n = _noise_like(out.shape, W32.std().item(), noise_scale, gen)
    if n is not None:
        new_mask = torch.zeros(out.shape[0], dtype=torch.bool)
        new_mask[W.shape[0]:] = True
        out[new_mask] += n[new_mask]
    return out.to(W.dtype)


def _widen_cols(W, g, count, noise_scale, gen):
    """(out, d_old) -> (out, d_new): split columns; noise on new cols only."""
    W32 = W.float()
    out = (W32[:, g] / count[g]).clone()
    n = _noise_like(out.shape, W32.std().item(), noise_scale, gen)
    if n is not None:
        new_mask = torch.zeros(out.shape[1], dtype=torch.bool)
        new_mask[W.shape[1]:] = True
        out[:, new_mask] += n[:, new_mask]
    return out.to(W.dtype)


def _widen_norm(w, g):
    return w[g].to(w.dtype)


def _widen_qkv(W, d_old, d_new, heads_old, head_dim, g, count,
               noise_scale, gen):
    """Fused qkv (3*d_old, d_old) -> (3*d_new, d_new), head-aware.

    New heads replicate head 0's rows; input columns are split."""
    W32 = W.float()
    std = W32.std().item()
    parts = []
    for k in range(3):
        src = W32[k * d_old:(k + 1) * d_old, :]
        src = src[:, g] / count[g]                      # widen input
        dst = torch.empty(d_new, src.shape[1])
        dst[:d_old] = src
        if d_new > d_old:
            # replicate head 0 once per new head
            n_new_heads = (d_new - d_old) // head_dim
            dst[d_old:] = src[:head_dim].repeat(n_new_heads, 1)
            n = _noise_like(dst.shape, std, noise_scale, gen)
            if n is not None:
                nm = torch.zeros(d_new, dtype=torch.bool)
                nm[d_old:] = True
                dst[nm] += n[nm]
        parts.append(dst)
    return torch.cat(parts, dim=0).to(W.dtype)


def _widen_kv(W, d_old, d_new, head_dim, g, count, noise_scale, gen):
    """Cross-attn kv (2*d_old, d_old) -> (2*d_new, d_new).

    Input dim is the (widened) contract-embedding dim -> column split."""
    W32 = W.float()
    std = W32.std().item()
    parts = []
    for k in range(2):
        src = W32[k * d_old:(k + 1) * d_old, :]
        src = src[:, g] / count[g]
        dst = torch.empty(d_new, src.shape[1])
        dst[:d_old] = src
        if d_new > d_old:
            n_new_heads = (d_new - d_old) // head_dim
            dst[d_old:] = src[:head_dim].repeat(n_new_heads, 1)
            n = _noise_like(dst.shape, std, noise_scale, gen)
            if n is not None:
                nm = torch.zeros(d_new, dtype=torch.bool)
                nm[d_old:] = True
                dst[nm] += n[nm]
        parts.append(dst)
    return torch.cat(parts, dim=0).to(W.dtype)


def _zero_block(block):
    """Make a transformer block an exact identity: x -> x."""
    with torch.no_grad():
        block.attn.qkv.weight.zero_()
        block.attn.out.weight.zero_()
        block.mlp.gate.weight.zero_()
        block.mlp.up.weight.zero_()
        block.mlp.down.weight.zero_()
        if block.contract_attn is not None:
            block.contract_attn.q.weight.zero_()
            block.contract_attn.kv.weight.zero_()
            block.contract_attn.out.weight.zero_()
            block.contract_attn.gate.zero_()
    return block


# ---------------------------------------------------------------- growth

def grow_model(small, cfg_big, vocab_map=None, noise_scale=0.01, gen=None):
    """Return a new XIIVModel at cfg_big, grown from `small`.

    vocab_map: {new_id: {"old_id": i} | {"pieces": [old_ids]}} or None
    (None = pure arch growth, vocab unchanged).
    """
    cfg_small = small.cfg
    d_old, d_new = cfg_small.d_model, cfg_big.d_model
    L_old, L_new = cfg_small.n_layers, cfg_big.n_layers
    h_old, h_new = cfg_small.n_heads, cfg_big.n_heads
    hd = d_old // h_old
    assert d_new // h_new == hd, "head_dim must be preserved by growth"
    assert h_new == h_old + (d_new - d_old) // hd
    assert L_new >= L_old
    v_old = cfg_small.vocab_size
    v_new = cfg_big.vocab_size

    g, count = _dim_map(d_old, d_new, hd)
    f_old, f_new = cfg_small.ffn_dim, cfg_big.ffn_dim
    gf, countf = _row_map(f_old, f_new)
    # match source dtype (fp16 low-mem mode): allocate directly, no fp32 transient
    try:
        src_dt = next(small.parameters()).dtype
    except StopIteration:
        src_dt = torch.float32
    if src_dt != torch.float32:
        with torch.device("meta"):
            big = XIIVModel(cfg_big)
        big.to_empty(device="cpu").to(src_dt)
        # to_empty breaks tied-embedding sharing (allocates separate
        # tensors); re-tie explicitly so lm_head follows embed.
        if cfg_big.tie_embeddings:
            big.lm_head.weight = big.embed.weight
    else:
        big = XIIVModel(cfg_big)
    sp = dict(small.named_parameters())

    def S(name):
        return sp[name]

    with torch.no_grad():
        # --- embeddings (tied head follows automatically) ---
        if vocab_map is None:
            assert v_new == v_old
            E = S("embed.weight").float()
            Enew = E[:, g].clone()          # replicate cols, no split
            n = _noise_like(Enew.shape, E.std().item(), noise_scale, gen)
            if n is not None:
                nm = torch.zeros(d_new, dtype=torch.bool)
                nm[d_old:] = True
                Enew[:, nm] += n[:, nm]
            Enew = Enew.to(S("embed.weight").dtype)
        else:
            E = S("embed.weight").float()
            Ew = E[:, g]  # width-widened old rows (v_old, d_new)
            Enew = torch.empty(v_new, d_new)
            new_col = torch.zeros(d_new, dtype=torch.bool)
            new_col[d_old:] = True
            n = _noise_like((v_new, d_new), E.std().item(),
                            noise_scale, gen)
            for nid_s, spec in vocab_map.items():
                nid = int(nid_s)
                if "old_id" in spec:
                    row = Ew[spec["old_id"]].clone()
                else:
                    row = Ew[spec["pieces"]].mean(dim=0)
                if n is not None:
                    row[new_col] += n[nid][new_col]
                    if "pieces" in spec:      # genuinely new token
                        row[~new_col] += n[nid][~new_col]
                Enew[nid] = row
            Enew = Enew.to(S("embed.weight").dtype)
        big.embed.weight.copy_(Enew)

        # --- transformer blocks ---
        for i in range(L_new):
            b_big, b_small = big.layers[i], small.layers[i] \
                if i < L_old else (None, None)
            if i >= L_old:
                _zero_block(b_big)      # exact identity
                continue
            bs = small.layers[i]
            # norms: replicate
            b_big.attn_norm.weight.copy_(_widen_norm(
                bs.attn_norm.weight, g))
            b_big.mlp_norm.weight.copy_(_widen_norm(
                bs.mlp_norm.weight, g))
            # self-attention
            b_big.attn.qkv.weight.copy_(_widen_qkv(
                bs.attn.qkv.weight, d_old, d_new, h_old, hd, g, count,
                noise_scale, gen))
            b_big.attn.out.weight.copy_(_widen_cols(
                _widen_rows(bs.attn.out.weight, g, 0.0, gen),
                g, count, noise_scale, gen))
            # SwiGLU (gate/up/down): widen d_model AND ffn_dim.
            # gate/up (f,d): replicate new ffn rows, split new d cols.
            # down (d,f): replicate new d rows, split new ffn cols
            # (division by countf keeps the block function-preserving).
            b_big.mlp.gate.weight.copy_(_widen_rows(
                _widen_cols(bs.mlp.gate.weight, g, count,
                            noise_scale, gen),
                gf, noise_scale, gen))
            b_big.mlp.up.weight.copy_(_widen_rows(
                _widen_cols(bs.mlp.up.weight, g, count,
                            noise_scale, gen),
                gf, noise_scale, gen))
            b_big.mlp.down.weight.copy_(_widen_cols(
                _widen_rows(bs.mlp.down.weight, g, noise_scale, gen),
                gf, countf, noise_scale, gen))
            # contract cross-attention
            if bs.contract_attn is not None:
                b_big.contract_norm.weight.copy_(_widen_norm(
                    bs.contract_norm.weight, g))
                ca_s, ca_b = bs.contract_attn, b_big.contract_attn
                ca_b.q.weight.copy_(_widen_cols(
                    _widen_rows(ca_s.q.weight, g, 0.0, gen),
                    g, count, noise_scale, gen))
                ca_b.kv.weight.copy_(_widen_kv(
                    ca_s.kv.weight, d_old, d_new, hd, g, count,
                    noise_scale, gen))
                ca_b.out.weight.copy_(_widen_cols(
                    _widen_rows(ca_s.out.weight, g, 0.0, gen),
                    g, count, noise_scale, gen))
                gate = torch.zeros(d_new, dtype=ca_s.gate.dtype)
                gate[:d_old] = ca_s.gate
                ca_b.gate.copy_(gate)

        big.final_norm.weight.copy_(_widen_norm(
            small.final_norm.weight, g))

        # --- contract encoder: internals copied; proj + tok_embed widen ---
        if small.contract_encoder is not None:
            es, eb = small.contract_encoder, big.contract_encoder
            eb.tok_embed.weight.copy_(
                _grow_encoder_embed(es.tok_embed.weight, v_old, v_new,
                                    vocab_map, noise_scale, gen))
            eb.pos_embed.weight.copy_(es.pos_embed.weight)
            for ls, lb in zip(es.layers, eb.layers):
                lb.load_state_dict(ls.state_dict())
            eb.norm.weight.copy_(es.norm.weight)
            eb.proj.weight.copy_(_widen_rows(
                es.proj.weight, g, noise_scale, gen))

    return big


def _grow_encoder_embed(W, v_old, v_new, vocab_map, noise_scale, gen):
    """Contract encoder token embedding (v, d_enc): vocab may expand,
    width never does."""
    if vocab_map is None:
        assert v_new == v_old
        return W.to(W.dtype)
    W32 = W.float()
    out = torch.empty(v_new, W.shape[1])
    for nid_s, spec in vocab_map.items():
        nid = int(nid_s)
        if "old_id" in spec:
            out[nid] = W32[spec["old_id"]]
        else:
            out[nid] = W32[spec["pieces"]].mean(dim=0)
    n = _noise_like(out.shape, W32.std().item(), noise_scale, gen)
    if n is not None:
        for nid_s, spec in vocab_map.items():
            if "pieces" in spec:
                out[int(nid_s)] += n[int(nid_s)]
    return out.to(W.dtype)


@torch.no_grad()
def measure_drift(small, big, vocab_size, seq_len=128, batches=4,
                  contract_tokens=32, seed=0):
    """Max/mean abs logit drift between parent and grown model.

    Runs the forward passes in float32 for numerical stability (fp16
    logits can overflow to inf, producing NaN drift). Models are cast
    sequentially to bound peak memory.
    """
    gen = torch.Generator().manual_seed(seed)
    small.eval()
    big.eval()
    # remember dtypes to restore (Module.float() is in-place)
    dt_small = next(small.parameters()).dtype
    dt_big = next(big.parameters()).dtype
    max_d, sum_d, n = 0.0, 0.0, 0
    for _ in range(batches):
        ids = torch.randint(0, vocab_size, (2, seq_len), generator=gen)
        cids = torch.randint(0, vocab_size, (2, contract_tokens),
                             generator=gen)
        # contract ids must fit the encoder's own (possibly smaller) vocab
        cids = torch.remainder(cids, small.cfg.vocab_size)
        # sequential fp32 with cast-back: peak is one fp32 model at a time
        if dt_small != torch.float32:
            small.float()
        l0 = small(ids, contract_ids=cids).float()
        if dt_small != torch.float32:
            small.to(dt_small)
        if dt_big != torch.float32:
            big.float()
        l1 = big(ids, contract_ids=cids).float()
        if dt_big != torch.float32:
            big.to(dt_big)
        # vocabs may differ after expansion: compare the shared prefix
        d = (l0.float() - l1.float()[..., :l0.shape[-1]]).abs()
        # FAIL CLOSED on non-finite drift: zeroing it would mask broken growth
        if not torch.isfinite(d).all():
            n_bad = (~torch.isfinite(d)).sum().item()
            raise RuntimeError(
                f"NON-FINITE DRIFT: {n_bad}/{d.numel()} values are NaN/inf. "
                f"Growth is broken — refusing to certify. Do not use this checkpoint."
            )
        max_d = max(max_d, d.max().item())
        sum_d += d.sum().item()
        n += d.numel()
    return max_d, sum_d / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ckpt", required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--d-new", type=int, required=True)
    ap.add_argument("--layers-new", type=int, required=True)
    ap.add_argument("--heads-new", type=int, required=True)
    ap.add_argument("--ffn-new", type=int, default=None,
                    help="new SwiGLU dim (default: keep parent ffn_dim)")
    ap.add_argument("--vocab-size-new", type=int, default=None)
    ap.add_argument("--vocab-map", default=None)
    ap.add_argument("--noise-scale", type=float, default=0.01)
    ap.add_argument("--drift-ratio-tol", type=float, default=0.6,
                    help="max allowed (grown drift)/(random-init drift); "
                         "the scale-invariant gate pre-registered in "
                         "tests/test_grow.py (absolute drift does not "
                         "generalize from toy to real growth sizes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="compute dtype for growth (float16 halves RAM; "
                         "safe for Net2Net-style copying)")
    ap.add_argument("--skip-drift", action="store_true",
                    help="skip the drift-gate measurement (for memory-"
                         "constrained boxes; run verify_drift.py separately)")
    args = ap.parse_args()

    dt = torch.float16 if args.dtype == "float16" else torch.float32
    gen = torch.Generator().manual_seed(args.seed)
    ck = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)
    cfg_small = ModelConfig.from_json(ck["config"])
    # memory-efficient: convert state dict to target dtype first, then
    # allocate the model directly in that dtype (no fp32 transient)
    sd_small = {k: v.to(dt) if torch.is_tensor(v) else v
                for k, v in ck["model"].items()}
    del ck["model"]
    with torch.device("meta"):
        small = XIIVModel(cfg_small)
    small.to_empty(device="cpu").to(dt)
    small.load_state_dict(sd_small, strict=True)
    del sd_small
    print(f"parent: {small.count_parameters()/1e6:.2f}M params, "
          f"step {ck['step']}")

    import copy as _copy
    cfg_big = _copy.deepcopy(cfg_small)
    cfg_big.d_model = args.d_new
    cfg_big.n_layers = args.layers_new
    cfg_big.n_heads = args.heads_new
    if args.ffn_new:
        cfg_big.ffn_dim = args.ffn_new
    if args.vocab_size_new:
        cfg_big.vocab_size = args.vocab_size_new

    # --- drift gate: random-init baseline FIRST (then freed), so the peak
    # is always two models (small+rnd, then small+big), never three. ---
    mean_rnd = None
    if not args.skip_drift:
        torch.manual_seed(args.seed + 1)
        with torch.device("meta"):
            rnd = XIIVModel(cfg_big)
        rnd.to_empty(device="cpu").to(dt)
        _, mean_rnd = measure_drift(small, rnd, cfg_small.vocab_size)
        del rnd

    vocab_map = None
    if args.vocab_map:
        with open(args.vocab_map) as f:
            vocab_map = json.load(f)
        assert cfg_big.vocab_size == len(vocab_map), \
            "vocab map size must match new vocab size"

    big = grow_model(small, cfg_big, vocab_map=vocab_map,
                     noise_scale=args.noise_scale, gen=gen)
    print(f"grown:  {big.count_parameters()/1e6:.2f}M params "
          f"(estimate {cfg_big.param_count_estimate()/1e6:.2f}M)")

    # --- drift gate (relative, scale-invariant; pre-registered in
    # tests/test_grow.py): the grown init must be far closer to the
    # parent than a random init of the same architecture. ---
    if not args.skip_drift:
        max_d, mean_d = measure_drift(small, big, cfg_small.vocab_size)
        ratio = mean_d / (mean_rnd + 1e-9)
        print(f"drift: grown max|d|={max_d:.4f} mean|d|={mean_d:.4f} nats; "
              f"random-init mean|d|={mean_rnd:.4f} nats; "
              f"ratio={ratio:.3f} (gate < {args.drift_ratio_tol})")
        assert ratio < args.drift_ratio_tol, \
            (f"grown init not closer to parent than random "
             f"({ratio:.3f} >= {args.drift_ratio_tol})")
    else:
        print("drift gate SKIPPED (--skip-drift); run verify_drift.py separately")

    out = {"model": big.state_dict(),
           "step": ck["step"], "best_val": ck["best_val"],
           "tokens_seen": ck.get("tokens_seen", 0),
           "config": cfg_big.to_json(),
           "grown_from": os.path.basename(args.src_ckpt)}
    for k in ("rng_state", "torch_rng", "np_rng"):
        if k in ck:
            out[k] = ck[k]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ckpt)),
                exist_ok=True)
    torch.save(out, args.out_ckpt)
    print(f"wrote {args.out_ckpt} (no optimizer state: starts fresh)")


if __name__ == "__main__":
    main()
