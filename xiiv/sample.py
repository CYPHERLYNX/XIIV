"""XIIV command-line interface: generate text or fill in masked spans.

Examples:
    python -m xiiv.sample generate --length 128 --steps 16 --seed 7
    python -m xiiv.sample infill --text "Lily found a <mask> in the garden."
    python -m xiiv.sample generate --prompt "Once upon a time" --length 160

Zero network calls. Deterministic given --seed.
"""
import argparse
import json
import os
import sys
import time

import torch

from .config import ModelConfig
from .model import XIIVModel
from .tokenizer import XIIVTokenizer
from .diffusion import sample_trajectory


def load_model(weights_path: str, device="cpu"):
    """Load safetensors weights + sibling config.json. Zero network calls."""
    from safetensors.torch import load_file
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(weights_path)),
                            "config.json")
    cfg = ModelConfig.from_json(open(cfg_path).read())
    state = load_file(weights_path, device=device)
    model = XIIVModel(cfg)
    # strict=False: safetensors stores tied weights once (embed.weight covers
    # lm_head.weight — same tensor object by construction in XIIVModel).
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys in {weights_path}: {unexpected}"
    assert set(missing) <= {"lm_head.weight"}, \
        f"missing keys in {weights_path}: {missing}"
    model.to(device).eval()
    return model, cfg


def cmd_generate(args, model, cfg, tok):
    from xiiv.diffusion import generate_with_scratchpad, self_refine
    prompt_ids = None
    if args.prompt:
        prompt_ids = torch.tensor(tok.encode(args.prompt), dtype=torch.long)
    t0 = time.time()
    scratch_txt, traj = None, None
    if args.scratchpad > 0:
        # Two-phase reasoning: scratchpad first, answer conditioned on it.
        ids, conf, scratch = generate_with_scratchpad(
            model, args.length, tok.mask_id, tok.eos_id,
            scratch_len=args.scratchpad, steps=args.steps,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            prompt_ids=prompt_ids, device="cpu")
        scratch_txt = tok.decode(scratch.tolist())
    elif args.refine_rounds > 0:
        # Draft + self-correction: remask lowest-confidence spans,
        # re-denoise in place (diffusion-native revision).
        ids, conf = self_refine(
            model, args.length, tok.mask_id, tok.eos_id, steps=args.steps,
            refine_rounds=args.refine_rounds, remask_frac=args.refine_frac,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            prompt_ids=prompt_ids, device="cpu")
    else:
        out = sample_trajectory(
            model, args.length, tok.mask_id, tok.eos_id, steps=args.steps,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            prompt_ids=prompt_ids, remask_ratio=args.remask,
            return_trajectory=args.trajectory, device="cpu")
        if args.trajectory:
            ids, conf, traj = out
        else:
            ids, conf = out
    dt = time.time() - t0
    if scratch_txt is not None:
        print(f"[scratchpad] {scratch_txt}\n")
    text = tok.decode(ids.tolist())
    # trim at EOS / trailing masks (already skipped by decode)
    print(text)
    n_tok = int((ids != tok.mask_id).sum())
    mode = ("scratchpad+answer" if scratch_txt is not None
            else f"refine x{args.refine_rounds}" if args.refine_rounds > 0
            else "generate")
    print(f"\n[xiiv] {mode}: {n_tok} tokens, {args.steps} denoise steps, "
          f"{dt:.2f}s ({n_tok / max(dt, 1e-6):.1f} tok/s), seed={args.seed}",
          file=sys.stderr)
    if args.trajectory:
        return text, traj, tok
    return text, None, tok


def cmd_infill(args, model, cfg, tok):
    ids, frozen = tok.encode_with_masks(args.text)
    length = len(ids)
    if args.length and args.length > length:
        pad = args.length - length
        import torch as _t
        ids = _t.cat([ids, _t.full((pad,), tok.mask_id, dtype=_t.long)])
        frozen = _t.cat([frozen, _t.zeros(pad, dtype=_t.bool)])
        length = args.length
    t0 = time.time()
    out_ids, conf = sample_trajectory(
        model, length, tok.mask_id, tok.eos_id, steps=args.steps,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        prompt_ids=ids, frozen=frozen, remask_ratio=args.remask,
        device="cpu")
    # re-apply frozen context exactly (safety: sampler never touches it,
    # but belt-and-braces for the demo)
    out_ids[frozen] = ids[frozen]
    dt = time.time() - t0
    text = tok.decode(out_ids.tolist())
    print(text)
    print(f"\n[xiiv] infilled {int((~frozen).sum())} masked tokens in "
          f"{dt:.2f}s, seed={args.seed}", file=sys.stderr)
    return text


def main():
    ap = argparse.ArgumentParser(prog="xiiv",
                                 description="XIIV masked-diffusion LM")
    ap.add_argument("--weights", default="weights/xiiv.safetensors")
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate from noise (or a prompt)")
    g.add_argument("--prompt", default="")
    g.add_argument("--length", type=int, default=128)
    g.add_argument("--steps", type=int, default=16)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--remask", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--trajectory", action="store_true")
    g.add_argument("--scratchpad", type=int, default=0,
                   help="reasoning-first decoding: denoise SCRATCHPAD tokens "
                        "as a scratchpad, freeze it, then denoise the answer "
                        "on the full trace (0 = off)")
    g.add_argument("--refine-rounds", type=int, default=0,
                   help="self-correction passes: remask lowest-confidence "
                        "spans and re-denoise in place (0 = off)")
    g.add_argument("--refine-frac", type=float, default=0.15,
                   help="fraction of open positions remasked per refine round")

    f = sub.add_parser("infill",
                       help="fill <mask> / <mask:N> spans, e.g. --text 'a <mask:5> sat'")
    f.add_argument("--text", required=True)
    f.add_argument("--length", type=int, default=0,
                   help="pad total length with extra masks (0 = exact)")
    f.add_argument("--steps", type=int, default=16)
    f.add_argument("--temperature", type=float, default=1.0)
    f.add_argument("--top-p", type=float, default=0.95)
    f.add_argument("--remask", type=float, default=0.0)
    f.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()
    # generate needs a positive canvas; infill allows --length 0 (= exact,
    # no padding — the canvas comes from the masked text itself).
    if args.cmd == "generate" and args.length < 1:
        ap.error("--length must be >= 1")
    if args.cmd == "infill" and args.length < 0:
        ap.error("--length must be >= 0")
    if args.cmd == "generate":
        if args.scratchpad < 0:
            ap.error("--scratchpad must be >= 0")
        if args.scratchpad >= args.length:
            ap.error("--scratchpad must be < --length")
        if args.refine_rounds < 0:
            ap.error("--refine-rounds must be >= 0")
        if not 0.0 <= args.refine_frac <= 1.0:
            ap.error("--refine-frac must be in [0, 1]")
    if args.steps < 1:
        ap.error("--steps must be >= 1")
    tok = XIIVTokenizer(args.tokenizer)
    model, cfg = load_model(args.weights)
    assert tok.mask_id == cfg.mask_id and tok.eos_id == cfg.eos_id, \
        "tokenizer/weights special-token mismatch"

    if args.cmd == "generate":
        cmd_generate(args, model, cfg, tok)
    else:
        cmd_infill(args, model, cfg, tok)


if __name__ == "__main__":
    main()
