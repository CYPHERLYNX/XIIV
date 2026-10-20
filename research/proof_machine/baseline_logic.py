"""
Transformer baseline for logic chains (spec section 5.6, task 1).

Same decoder-only architecture as baseline_transformer.py (d=96, 3 layers,
3 heads, ffn=384 -- ~370K params; the larger vocab adds ~24K embedding
params vs Phase 1, still erring AGAINST the proof machine whose 8 boolean
nets total ~13K params). Byte-level I/O: one token per byte value, op
tokens for the 8 boolean primitives. Trained end-to-end on (x0, S) -> x_L
with the explicit op sequence in the input; it must learn composition
IMPLICITLY. Training depths 1..16 (spec 5.3); test depths 1..100 probe
out-of-distribution depth generalization.

Reuses baseline_transformer's model + training/eval machinery by
overriding its module-level encoding hooks (same code path, different
tokenization) -- the architecture and training loop are untouched.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_transformer as bt  # noqa: E402
from taskgen_logic import gen_instance as gen_logic  # noqa: E402
from logic import BOOL_OP_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Byte-level tokenization
# ---------------------------------------------------------------------------
PAD, BOS, EOS, SEP = 0, 1, 2, 3
BYTE0 = 4                    # bytes 0..255 -> 4..259
OP_BASE = 260                # 8 boolean ops -> 260..267
VOCAB = 268

OP_ID = {op: OP_BASE + i for i, op in enumerate(BOOL_OP_NAMES)}


def enc_input(x0, ops):
    return [BOS, BYTE0 + int(x0), SEP] + [OP_ID[o] for o in ops] + [SEP]


def enc_target(xL):
    return [BYTE0 + int(xL), EOS]


def dec_number(toks):
    """Token list (no EOS) -> int. Single byte token expected."""
    if len(toks) != 1:
        return None
    t = toks[0]
    if BYTE0 <= t < BYTE0 + 256:
        return t - BYTE0
    return None


# Override baseline_transformer's module hooks (architecture untouched).
# NOTE: arithmetic gen_instance returns (x0, ops, xL, _) -- the shim keeps
# the 4-tuple shape make_batch/eval_exact_match unpack.
def _gen_instance4(L, rng):
    x0, ops, xL = gen_logic(L, rng)
    return x0, ops, xL, None


bt.VOCAB = VOCAB
bt.gen_instance = _gen_instance4
bt.enc_input = enc_input
bt.enc_target = enc_target
bt.dec_number = dec_number

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints_logic", "baseline_tf_logic.pt")


def train(max_steps=10000, batch_size=64, lr=3e-4, device="cpu"):
    torch.manual_seed(7)
    torch.set_num_threads(4)
    model = bt.DecoderTransformer().to(device)
    print(f"baseline params: {model.n_params()}", flush=True)
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(7)
    best_val = -1.0
    bad = 0
    t_start = time.time()
    for step in range(1, max_steps + 1):
        ids, labels, _ = bt.make_batch(batch_size, rng)
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                               labels.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            el = time.time() - t_start
            print(f"step {step:5d} loss {loss.item():.4f} ({el/60:.1f} min)",
                  flush=True)
        if step % 400 == 0:
            val = bt.eval_exact_match(model, [1, 2, 4, 8, 16], 400,
                                      seed=1000 + step, device=device,
                                      batch_note=f"[val step {step}]")
            vmean = sum(val.values()) / len(val)
            print(f"step {step}: val exact-match (depths 1-16) = "
                  f"{vmean*100:.2f}%", flush=True)
            if vmean > best_val + 1e-4:
                best_val = vmean
                bad = 0
                torch.save(model.state_dict(), CKPT)
                print(f"  saved best ({best_val*100:.2f}%) -> {CKPT}",
                      flush=True)
            else:
                bad += 1
                if bad >= 5:
                    print(f"early stop at step {step} "
                          f"(best val {best_val*100:.2f}%)", flush=True)
                    break
    return CKPT


@torch.no_grad()
def evaluate(device="cpu"):
    torch.set_num_threads(4)
    model = bt.DecoderTransformer().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device,
                                     weights_only=True))
    model.eval()
    depths = [1, 2, 4, 8, 16, 32, 64, 100]
    res = bt.eval_exact_match(model, depths, 1000, seed=777, device=device)
    out = {"depths": depths, "n_per_depth": 1000,
           "A_TF": {str(k): v for k, v in res.items()},
           "vocab": VOCAB, "arch": "d96-L3-H3-ffn384 (same as Phase 1)"}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results",
                        f"baseline_logic_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"A_TF(100) = {res[100]*100:.2f}%", flush=True)
    print(f"results -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.train:
        train(max_steps=args.max_steps, device=args.device)
    if args.eval:
        evaluate(device=args.device)


if __name__ == "__main__":
    main()
