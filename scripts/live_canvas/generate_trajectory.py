"""Generate a Live Canvas trajectory: real XIIV diffusion, step-by-step capture.

Runs DenoisingState with return_trajectory=True and saves:
  - trajectory: list of canvas states (token ids) per step
  - confidences: per-position confidence per step
  - priorities: the fill priority used (for GLOD visualization)
  - interventions: the guarantee intervention trail
  - final_text: decoded output
  - meta: model, steps, length, seed, order

Usage:
  PYTHONPATH=. .venv/bin/python scripts/live_canvas/generate_trajectory.py \
      --model tiny --length 48 --steps 12 --seed 0 --order guarantee_load \
      --out live_canvas_traj.json
  PYTHONPATH=. .venv/bin/python scripts/live_canvas/generate_trajectory.py \
      --model ckpt --ckpt checkpoints_1b_gn/ckpt_8218.pt \
      --tokenizer <tokenizer.json> --length 64 --steps 16 --seed 1 \
      --out live_canvas_traj_133m.json
"""
import argparse
import json
import sys

import torch


def build_tiny():
    from xiiv.config import ModelConfig
    from xiiv.model import XIIVModel
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      ffn_dim=128, mask_id=254, eos_id=255, seq_len=64)
    torch.manual_seed(0)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    return m, cfg, None


def build_ckpt(ckpt_path, tokenizer_path):
    from xiiv.config import ModelConfig
    from xiiv.model import XIIVModel
    from xiiv.tokenizer import XIIVTokenizer
    tok = XIIVTokenizer(tokenizer_path)
    # ModelConfig must match the checkpoint; read from ckpt if available.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("checkpoint has no config; cannot build model")
    m = XIIVModel(cfg)
    m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    m.eval()
    return m, cfg, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tiny", "ckpt"], default="tiny")
    ap.add_argument("--ckpt", default="checkpoints_1b_gn/ckpt_8218.pt")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", default="guarantee_load")
    ap.add_argument("--constraints", default=None,
                    help="JSON dict of constraints")
    ap.add_argument("--out", default="live_canvas_traj.json")
    args = ap.parse_args()

    if args.model == "tiny":
        m, cfg, tok = build_tiny()
    else:
        if not args.tokenizer:
            sys.exit("--tokenizer required for --model ckpt")
        m, cfg, tok = build_ckpt(args.ckpt, args.tokenizer)

    from xiiv.diffusion import DenoisingState
    constraints = json.loads(args.constraints) if args.constraints else None

    st = DenoisingState(
        length=args.length, mask_id=cfg.mask_id, eos_id=cfg.eos_id,
        steps=args.steps, seed=args.seed, vocab_size=cfg.vocab_size,
        order=args.order, constraints=constraints,
        return_trajectory=True, device="cpu")

    # Capture per-step: canvas + confidence snapshot.
    frames = []
    frames.append({"x": st.x.cpu().tolist(),
                   "conf": st.conf.cpu().tolist(),
                   "masked": st.masked.cpu().tolist(),
                   "step": args.steps})
    while st.step(m):
        frames.append({"x": st.x.cpu().tolist(),
                       "conf": st.conf.cpu().tolist(),
                       "masked": st.masked.cpu().tolist(),
                       "step": st.steps_remaining})

    final_ids = st.x.cpu().tolist()
    if tok is not None:
        final_text = tok.decode(final_ids)
    else:
        final_text = None

    # Which positions resolved on which step (for the parallel-reveal viz).
    resolved_step = [-1] * args.length
    for fi in range(1, len(frames)):
        prev = frames[fi - 1]["masked"]
        cur = frames[fi]["masked"]
        for p in range(args.length):
            if prev[p] and not cur[p] and resolved_step[p] == -1:
                resolved_step[p] = fi

    out = {
        "meta": {"model": args.model, "length": args.length,
                 "steps": args.steps, "seed": args.seed,
                 "order": args.order, "constraints": constraints,
                 "n_frames": len(frames)},
        "frames": frames,
        "resolved_step": resolved_step,
        "priority": (st.priority.cpu().tolist()
                     if st.priority is not None else None),
        "interventions": st.interventions,
        "final_ids": final_ids,
        "final_text": final_text,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"trajectory: {len(frames)} frames, {args.length} positions, "
          f"{len(st.interventions)} interventions -> {args.out}")
    if final_text:
        print(f"final: {final_text[:200]}")


if __name__ == "__main__":
    main()
