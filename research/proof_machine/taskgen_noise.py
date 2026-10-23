"""
Noisy-input task generators (spec 5.6, task 3).

Flat instances: reuse taskgen.gen_instance (same distribution discipline:
x0 in [-1000, 1000], rejection-sampled intermediates in [-1e6, 1e6]), then
NOISE-VALIDATE: every concrete x in [x0 - r, x0 + r] must keep every
intermediate in bound. Instances failing this are rejected (not clipped) --
the claim under test is depth scaling under noise, not extrapolation.

Conditional instances: reuse taskgen_conditional.gen_cinstance, then
noise-validate with per-x branch semantics: for each x in the input interval,
the guard, BOTH branch texts, and the taken path must stay in bound.
"""

import random

from certificates import RULES, var, eval_expr
from conditionals import is_ifz
from taskgen import gen_instance, INTERMEDIATE_BOUND
from taskgen_conditional import gen_cinstance


def _flat_ok_interval(x0, r, ops):
    """True iff every x in [x0-r, x0+r] keeps all intermediates in bound."""
    for x in range(x0 - r, x0 + r + 1):
        e = var()
        for op in ops:
            e = RULES[op](e)
            if abs(eval_expr(e, x)) > INTERMEDIATE_BOUND:
                return False
    return True


def gen_noisy_instance(L, r, rng):
    """
    Returns (x0, r, ops) or None. r: noise radius (int >= 0).
    Distribution: identical to taskgen.gen_instance at the nominal point,
    plus the noise-validity condition above.
    """
    for _ in range(2000):  # bounded extra trials for the noise condition
        inst = gen_instance(L, rng)
        if inst is None:
            return None
        x0, ops, _, _ = inst
        if _flat_ok_interval(x0, r, ops):
            return x0, r, ops
    return None


def gen_noisy_batch(L, n, r, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        inst = gen_noisy_instance(L, r, rng)
        if inst is None:
            raise RuntimeError(f"gen_noisy_instance failed at L={L}, r={r}")
        out.append(inst)
    return out


def _prog_ok_interval(x0, r, prog):
    """Per-x bound check over the whole input interval (branch-per-x)."""
    for x in range(x0 - r, x0 + r + 1):
        if _walk_bound(var(), prog, x) is None:
            return False
    return True


def _walk_bound(e, prog, x):
    for step in prog:
        if is_ifz(step):
            _, cond, then_p, else_p = step
            ec = e
            for op in cond:
                ec = RULES[op](ec)
                if abs(eval_expr(ec, x)) > INTERMEDIATE_BOUND:
                    return None
            cv = eval_expr(ec, x)
            for branch in (then_p, else_p):
                eb = e
                for op in branch:
                    eb = RULES[op](eb)
                    if abs(eval_expr(eb, x)) > INTERMEDIATE_BOUND:
                        return None
            e = _walk_bound(e, then_p if cv == 0 else else_p, x)
            if e is None:
                return None
        else:
            e = RULES[step](e)
            if abs(eval_expr(e, x)) > INTERMEDIATE_BOUND:
                return None
    return e


def gen_noisy_cinstance(L, r, rng):
    """Returns (x0, r, prog) or None."""
    for _ in range(2000):
        inst = gen_cinstance(L, rng)
        if inst is None:
            return None
        x0, prog, _, _ = inst
        if _prog_ok_interval(x0, r, prog):
            return x0, r, prog
    return None


def gen_noisy_cbatch(L, n, r, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        inst = gen_noisy_cinstance(L, r, rng)
        if inst is None:
            raise RuntimeError(f"gen_noisy_cinstance failed at L={L}, r={r}")
        out.append(inst)
    return out
