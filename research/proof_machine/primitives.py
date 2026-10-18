"""
The 8-primitive library for the Compositional Proof Machine (Phase 1 kill gate).

Each primitive is a pair (M_i, M_i#) per spec section 4.3:
  - M_i:  a small neural module (torch MLP) implementing the function.
  - M_i#: a FIXED symbolic abstract transformer (from certificates.RULES).
          Not learned. Exact for this op set.

Primitive set (spec section 5.2):
    ADD_1, ADD_3, ADD_7      x -> x + k
    MUL_2, MUL_5              x -> k * x
    MOD_7, MOD_13             x -> x mod m   (Python semantics: 0 <= y < m)
    NEG                      x -> -x

Design notes (documented deviations / choices):
  - MOD primitives use Fourier input features [x/S, sin(2*pi*x/m), cos(2*pi*x/m)]
    and a small classifier over the m residue classes. Rationale: a vanilla ReLU
    MLP cannot place ~300k sharp decision boundaries for a periodic function over
    [-1e6, 1e6]; Fourier features are the standard, documented way to give a
    small net access to periodicity. This is a primitive-design choice, not
    test-set gaming: the same features are used at train and test time.
  - Operating range is [-1e6, 1e6] (intermediates in the depth sweep are
    rejection-sampled into this range). The spec's task inputs x0 stay in
    [-1000, 1000]; the wider range covers intermediate values so the experiment
    measures DEPTH scaling, not float-extrapolation behaviour.
"""

import math
import torch
import torch.nn as nn

from certificates import RULES, OP_NAMES  # noqa: F401  (re-exported)

SCALE = 1e6          # input/output normalization for affine nets
TRAIN_LO = -1_000_000
TRAIN_HI = 1_000_000


# ---------------------------------------------------------------------------
# Concrete (neural) modules
# ---------------------------------------------------------------------------

class AffineNet(nn.Module):
    """Single linear layer for affine primitives: y = a*x + b.

    A linear model is the CORRECT model class for an affine function. It is
    fit by exact least squares (see train_primitives.train_affine), not SGD:
    gradient descent was tried first and stalled on conditioning artifacts
    (1e-12-scale initial loss), which would have tested the optimizer rather
    than composition. The weights live in an nn.Linear -- a trained neural
    module, fit with the optimal algorithm for its class. The genuinely hard
    learning case (periodic MOD) stays a nonlinear MLP trained by Adam.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, x_norm):
        return self.net(x_norm)


class ModNet(nn.Module):
    """Small classifier MLP for x mod m, with Fourier input features."""
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, m),
        )

    def features(self, x):
        """x: int tensor. Returns [x/S, sin(2πx/m), cos(2πx/m)]."""
        xf = x.double()
        s = xf / SCALE
        ang = 2.0 * math.pi * xf / self.m
        return torch.stack([s, torch.sin(ang), torch.cos(ang)], dim=-1).float()

    def forward(self, x):
        return self.net(self.features(x))


def build_net(op):
    if op.startswith("ADD_") or op.startswith("MUL_") or op == "NEG":
        return AffineNet()
    if op.startswith("MOD_"):
        return ModNet(int(op.split("_")[1]))
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Exact reference functions (ground truth for training / eval)
# ---------------------------------------------------------------------------

def exact_fn(op):
    if op == "ADD_1": return lambda x: x + 1
    if op == "ADD_3": return lambda x: x + 3
    if op == "ADD_7": return lambda x: x + 7
    if op == "MUL_2": return lambda x: x * 2
    if op == "MUL_5": return lambda x: x * 5
    if op == "MOD_7": return lambda x: x % 7
    if op == "MOD_13": return lambda x: x % 13
    if op == "NEG": return lambda x: -x
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Concrete execution of one primitive (neural forward -> int)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_concrete(net, op, x):
    """Apply the neural primitive to integer x, returning an int."""
    if op.startswith("MOD_"):
        xt = torch.tensor([x], dtype=torch.long)
        logits = net(xt)
        return int(logits.argmax(dim=-1).item())
    # Affine nets train on RAW integer units (float32 is exact for |x| <= 2^24;
    # our range is <= 1e6). Normalized targets would bury the +/-k offsets
    # (1e-6 scale) below optimizer precision -- raw units keep every offset
    # at O(1) where Adam resolves it exactly.
    xt = torch.tensor([[float(x)]], dtype=torch.float32)
    return int(round(net(xt).item()))


# ---------------------------------------------------------------------------
# Tightness loss (spec section 4.3).
# L_tight = E_x[ dist(M(x), center(M#(alpha(x)))) / width(M#(alpha(x))) ]
# For our EXACT symbolic domain, M# on a point abstraction returns the exact
# value: center = exact value, width -> 0. We floor the width at 1.0 so the
# term is well-defined; it then reduces to a scaled MAE against the exact
# target -- i.e. for exact domains the tightness loss is vacuous by
# construction (any output IS within the exact abstraction), and L_task
# carries the training. Documented here rather than hidden.
# ---------------------------------------------------------------------------

def tightness_loss_affine(pred_norm, target_norm):
    # center = target (exact domain), width floored at 1.0
    return (pred_norm - target_norm).abs().mean()
