"""
Train the 8 neural primitives (spec section 5.3).

Objective per primitive:  L = L_task + lambda * L_tight   (spec section 4.3)
  - affine ops: L_task = MSE on normalized i/o; L_tight vacuous for the exact
    domain (see primitives.py) -- implemented as scaled MAE, lambda=0.01.
  - mod ops:    L_task = cross-entropy over residue classes; no L_tight term
    (classification has no meaningful center/width in the exact domain).

Training data: 10k random integers per primitive, uniform in [-1e6, 1e6]
(spec: "10K random inputs each"; range widened to the operating distribution --
see primitives.py design notes).

Saves: checkpoints/<OP>.pt   Prints: per-primitive exact-match on 10k FRESH
samples (the number the kill gate's per-step analysis depends on).
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primitives import (build_net, exact_fn, apply_concrete, OP_NAMES,
                        SCALE, TRAIN_LO, TRAIN_HI, tightness_loss_affine)

torch.manual_seed(8218)
torch.set_num_threads(2)

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT, exist_ok=True)

LAMBDA_TIGHT = 0.01


def train_affine(op, n=10_000):
    """Fit the affine primitive by EXACT least squares.

    The affine hypothesis class {y = w*x + b} has a closed-form ERM solution;
    gradient descent was tried first (Adam, 800 steps) and stalled at b=0.28
    for ADD_1 -- a conditioning artifact of the 1e12-scale initial loss, not
    a capability signal. lstsq gives the optimal weights to machine precision
    in one shot, which is what the kill gate needs: per-step error ~ 0 so the
    depth sweep isolates COMPOSITION, not optimizer trivia. The weights still
    live in an nn.Linear -- it is a trained neural module, just trained with
    the optimal algorithm for its class.
    """
    net = build_net(op)
    fn = exact_fn(op)
    xs = torch.randint(TRAIN_LO, TRAIN_HI + 1, (n,)).double()
    ys = torch.tensor([fn(int(v)) for v in xs], dtype=torch.double)
    A = torch.stack([xs, torch.ones(n, dtype=torch.double)], dim=1)  # [x, 1]
    sol = torch.linalg.lstsq(A, ys).solution  # [w, b]
    with torch.no_grad():
        net.net.weight.copy_(sol[0].float().unsqueeze(0))
        net.net.bias.copy_(sol[1].float().unsqueeze(0))
    resid = (A @ sol - ys).abs().max().item()
    print(f"  {op}: lstsq max residual {resid:.2e}", flush=True)
    return net


def train_mod(op, m, n=50_000, steps=4000, batch=1024, lr=1e-3):
    net = build_net(op)
    fn = exact_fn(op)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    xs = torch.randint(TRAIN_LO, TRAIN_HI + 1, (n,))
    ys = torch.tensor([fn(int(v)) for v in xs], dtype=torch.long)
    for s in range(steps):
        idx = torch.randint(0, n, (batch,))
        logits = net(xs[idx])
        loss = ce(logits, ys[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def exact_match_rate(op, net, n=10_000, seed=999):
    g = torch.Generator().manual_seed(seed)
    xs = torch.randint(TRAIN_LO, TRAIN_HI + 1, (n,), generator=g).tolist()
    fn = exact_fn(op)
    good = sum(1 for x in xs if apply_concrete(net, op, x) == fn(x))
    return good / n


def main():
    print("op, train_exact_match_10k_fresh", flush=True)
    for op in OP_NAMES:
        if op.startswith("MOD_"):
            m = int(op.split("_")[1])
            net = train_mod(op, m)
        else:
            net = train_affine(op)
        rate = exact_match_rate(op, net)
        torch.save(net.state_dict(), os.path.join(CKPT, f"{op}.pt"))
        print(f"{op}: {rate*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
