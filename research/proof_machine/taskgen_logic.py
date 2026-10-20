"""
Task generator for logic chains (spec section 5.6, task 1).

Instance: (x0, ops, expected) where x0 in [0,255], ops is a flat list of
L boolean op names, and expected is the exact integer result.

Generator design (documented choices):
  - No rejection sampling is needed: every op is total on bytes, so every
    (x0, ops) pair is a valid instance. The intermediate-bound discipline
    of taskgen.py is vacuous here by construction (the domain is closed).
  - x0 uniform in [0,255]; ops uniform over the 8 primitives.
  - expected computed by pure symbolic execution (Python ints), the same
    reference the certificate verifier independently re-derives.
"""

import random

from logic import RULES_BOOL, START_VEC, eval_bvec, BOOL_OP_NAMES


def gen_instance(L, rng):
    """Returns (x0, ops, expected). Never fails (domain is closed)."""
    x0 = rng.randint(0, 255)
    ops = [rng.choice(BOOL_OP_NAMES) for _ in range(L)]
    v = START_VEC
    for op in ops:
        v = RULES_BOOL[op](v)
    return x0, ops, eval_bvec(v, x0)


def gen_batch(L, n, seed):
    rng = random.Random(seed)
    return [gen_instance(L, rng) for _ in range(n)]
