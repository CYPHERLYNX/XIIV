"""
Kill-gate task generator (spec section 5.2 / 5.4).

Task instance (depth L): input x0 in [-1000, 1000] (spec) and an explicit op
sequence S = (op_1, ..., op_L) sampled uniformly from the 8 primitives.
Expected answer x_L computed by exact integer arithmetic.

Intermediate bound: sequences are rejection-sampled so every intermediate
value stays within [-1e6, 1e6]. Rationale (documented, not hidden): the claim
under test is accuracy-vs-DEPTH. Unbounded intermediates would conflate depth
scaling with numerical-extrapolation behaviour (float64 is exact only to 2^53,
and the neural primitives train on [-1e6, 1e6]). Both the proof machine and the
transformer baseline train and test on this same distribution, so the comparison
is apples-to-apples; the survivor bias (fewer long MUL runs at high depth) is
reported in the results.
"""

import random

from certificates import RULES, OP_NAMES, var, eval_expr

X0_LO, X0_HI = -1000, 1000
INTERMEDIATE_BOUND = 1_000_000
MAX_TRIALS = 200_000


def gen_instance(L, rng):
    """
    Returns (x0, ops, expected, intermediates) or None if no valid instance
    found within MAX_TRIALS (survivor-bias guard for deep L).
    """
    for _ in range(MAX_TRIALS):
        x0 = rng.randint(X0_LO, X0_HI)
        ops = [rng.choice(OP_NAMES) for _ in range(L)]
        e = var()
        ok = True
        intermediates = []
        for op in ops:
            e = RULES[op](e)
            v = eval_expr(e, x0)
            if abs(v) > INTERMEDIATE_BOUND:
                ok = False
                break
            intermediates.append(v)
        if ok:
            return x0, ops, eval_expr(e, x0), intermediates
    return None


def gen_batch(L, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        inst = gen_instance(L, rng)
        if inst is None:
            raise RuntimeError(f"gen_instance failed too often at L={L}")
        out.append(inst)
    return out
