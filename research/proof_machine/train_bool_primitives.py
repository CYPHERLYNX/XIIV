"""
Train the 8 neural boolean primitives (spec section 5.6, task 1).

Objective: per-bit binary cross-entropy against the exact function.
Training data: the FULL domain -- all 256 byte values (documented choice;
see bool_primitives.py design notes). Per-step exactness ~100% is what the
kill gate needs: the depth sweep must isolate COMPOSITION.

Saves: checkpoints_logic/<OP>.pt
Prints: per-primitive exact-match on all 256 values.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bool_primitives import build_net, exact_fn, apply_concrete, bits_tensor
from logic import BOOL_OP_NAMES, W

torch.manual_seed(8218)
torch.set_num_threads(2)

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "checkpoints_logic")
os.makedirs(CKPT, exist_ok=True)

STEPS = 3000
LR = 1e-3


def train_op(op):
    net = build_net(op)
    fn = exact_fn(op)
    xs = torch.arange(256)
    X = torch.stack([bits_tensor(int(x))[0] for x in xs])          # (256, 8)
    Y = torch.stack([bits_tensor(fn(int(x)))[0] for x in xs])      # (256, 8)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()
    for s in range(STEPS):
        logits = net(X)
        loss = bce(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def exact_match_rate(op, net):
    fn = exact_fn(op)
    good = sum(1 for x in range(256) if apply_concrete(net, op, x) == fn(x))
    return good / 256


def main():
    print("op, exact_match_all_256", flush=True)
    for op in BOOL_OP_NAMES:
        net = train_op(op)
        rate = exact_match_rate(op, net)
        torch.save(net.state_dict(), os.path.join(CKPT, f"{op}.pt"))
        print(f"{op}: {rate * 100:.2f}%", flush=True)
        assert rate == 1.0, f"{op} did not reach exactness -- investigate"


if __name__ == "__main__":
    main()
