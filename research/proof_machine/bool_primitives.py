"""
Neural primitives for logic chains (spec section 5.6, task 1).

Each primitive is a pair (M_i, M_i#):
  - M_i:  a small neural module (torch MLP) implementing the bit-vector op.
  - M_i#: a FIXED symbolic abstract transformer (from logic.RULES_BOOL).
          Not learned. Exact for this op set.

Primitive set (8, mirroring the 8-primitive Phase-1 library):
    NOT            x -> ~x        (bitwise, 8-bit)
    XOR_A5, XOR_3C x -> x ^ c    (fixed constant operand, like ADD_k's k)
    AND_F0         x -> x & 0xF0
    OR_0F          x -> x | 0x0F
    SHL_1          x -> (x << 1) & 0xFF
    SHR_1          x -> x >> 1
    ROL_1          x -> rotate-left by 1

Design notes (documented choices, not hidden):
  - The net sees the 8-bit vector (not the scalar): 8 inputs -> 8 sigmoid
    outputs, one per bit. Bitwise ops are linear-ish in bit space; an
    8->32->32->8 ReLU MLP learns them to exactness. A scalar-input MLP
    would merely memorize 256 scalar mappings -- the vector form is the
    honest "learning the boolean function" setup.
  - The domain is FINITE (256 values) and training enumerates the full
    domain (see train_bool_primitives.py). This is deliberate, mirroring
    the Phase-1 lstsq note in train_primitives.py: per-step error must be
    ~0 so the depth sweep isolates COMPOSITION, not optimizer trivia.
    The kill gate measures depth scaling of composition, not
    out-of-domain generalization of the primitive nets.
"""

import torch
import torch.nn as nn

from logic import W, int_to_bits, bits_to_int


class BitNet(nn.Module):
    """8-bit in, 8-bit out (logits). Learns the boolean function."""

    def __init__(self, h1=32, h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(W, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, W),
        )

    def forward(self, bits):
        """bits: (B, 8) float. Returns (B, 8) logits."""
        return self.net(bits)


def build_net(op):
    if op in ("NOT", "XOR_A5", "XOR_3C", "AND_F0", "OR_0F",
              "SHL_1", "SHR_1", "ROL_1"):
        return BitNet()
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Exact reference functions (ground truth for training / eval)
# ---------------------------------------------------------------------------

def exact_fn(op):
    if op == "NOT":
        return lambda x: (~x) & 0xFF
    if op == "XOR_A5":
        return lambda x: x ^ 0xA5
    if op == "XOR_3C":
        return lambda x: x ^ 0x3C
    if op == "AND_F0":
        return lambda x: x & 0xF0
    if op == "OR_0F":
        return lambda x: x | 0x0F
    if op == "SHL_1":
        return lambda x: (x << 1) & 0xFF
    if op == "SHR_1":
        return lambda x: x >> 1
    if op == "ROL_1":
        return lambda x: ((x << 1) | (x >> 7)) & 0xFF
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Concrete execution of one primitive (neural forward -> int)
# ---------------------------------------------------------------------------

def bits_tensor(x):
    """int -> (1, 8) float tensor, LSB-first."""
    return torch.tensor([list(int_to_bits(x))], dtype=torch.float32)


@torch.no_grad()
def apply_concrete(net, op, x):
    """Apply the neural primitive to byte x, returning an int in [0,255]."""
    logits = net(bits_tensor(x))
    bits = (torch.sigmoid(logits[0]) > 0.5).int().tolist()
    return bits_to_int(bits)
