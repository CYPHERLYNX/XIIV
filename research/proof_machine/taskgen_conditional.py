"""
Conditional task generator (spec section 5.6, task 2: IFZERO branching).

Instance: (x0, prog, expected) where prog is a program (flat ops and/or IFZ
nodes) whose max-path primitive cost == L (the depth). The expected answer is
the exact symbolic value of the TAKEN path.

Generator design (documented choices, not hidden):
  - IFZ nodes: guard = single MOD_7 / MOD_13 (cost 1); then/else branches are
    flat op chains of EQUAL length b in 1..4 (cost b each). Equal branch
    lengths make taken-path cost == max-path cost == L exactly, so depth stays
    comparable with the Phase-1 flat sweep. Asymmetric branches are future work.
  - Guard semantics: branch on guard_value == 0, i.e. then iff z % m == 0.
    Natural branch distribution (~1/7 then for MOD_7): no forced balancing;
    across 1000 instances x ~dozens of nodes both outcomes are well covered.
  - Intermediate bound [-1e6, 1e6] enforced on EVERY intermediate, including
    guard intermediates and UNTAKEN-branch intermediates (they are program
    text; out-of-range programs are rejected, not executed).
  - x0 in [-1000, 1000] (spec). Same distribution discipline as taskgen.py.
"""

import random

from certificates import RULES, OP_NAMES, var, eval_expr
from conditionals import ifz, is_ifz, max_path_cost

X0_LO, X0_HI = -1000, 1000
INTERMEDIATE_BOUND = 1_000_000
MAX_TRIALS = 200_000
BRANCH_P = 0.4
MAX_BRANCH_LEN = 4
GUARDS = ["MOD_7", "MOD_13"]


def _rand_chain(n, rng):
    return [rng.choice(OP_NAMES) for _ in range(n)]


def build_program(budget, rng):
    """
    Build a program with max_path_cost == budget (exactly).
    Branches have equal taken-cost, so taken-path == max-path.
    """
    assert budget >= 1
    steps = []
    remaining = budget
    while remaining > 0:
        if remaining >= 3 and rng.random() < BRANCH_P:
            b = rng.randint(1, min(MAX_BRANCH_LEN, remaining - 1))
            guard = rng.choice(GUARDS)
            then_ops = _rand_chain(b, rng)
            else_ops = _rand_chain(b, rng)
            steps.append(ifz([guard], then_ops, else_ops))
            remaining -= (1 + b)
        else:
            steps.append(rng.choice(OP_NAMES))
            remaining -= 1
    assert max_path_cost(steps) == budget
    return steps


def _check_chain(e, ops, x0):
    """Symbolically apply flat ops; return final expr or None if out of bound."""
    for op in ops:
        e = RULES[op](e)
        if abs(eval_expr(e, x0)) > INTERMEDIATE_BOUND:
            return None
    return e


def instantiate(x0, prog):
    """
    Walk prog symbolically at x0; bound-check every intermediate (guards and
    both branches). Returns (expected, taken_summary) or None if any
    intermediate leaves [-1e6, 1e6].
    taken_summary: list of (guard_value, taken) per IFZ node, in order.
    """
    taken_summary = []

    def walk(e, p):
        for step in p:
            if is_ifz(step):
                _, cond, then_p, else_p = step
                ec = _check_chain(e, cond, x0)
                if ec is None:
                    return None
                cv = eval_expr(ec, x0)
                taken_lbl = "then" if cv == 0 else "else"
                taken_summary.append((cv, taken_lbl))
                # bound-check BOTH branches on the pre-guard value e
                for branch in (then_p, else_p):
                    if _check_chain(e, branch, x0) is None:
                        return None
                e = walk(e, then_p if taken_lbl == "then" else else_p)
                if e is None:
                    return None
            else:
                e = RULES[step](e)
                if abs(eval_expr(e, x0)) > INTERMEDIATE_BOUND:
                    return None
        return e

    e_final = walk(var(), prog)
    if e_final is None:
        return None
    return eval_expr(e_final, x0), taken_summary


def gen_cinstance(L, rng):
    """Returns (x0, prog, expected, taken_summary) or None."""
    for _ in range(MAX_TRIALS):
        x0 = rng.randint(X0_LO, X0_HI)
        prog = build_program(L, rng)
        inst = instantiate(x0, prog)
        if inst is not None:
            expected, taken_summary = inst
            return x0, prog, expected, taken_summary
    return None


def gen_cbatch(L, n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        inst = gen_cinstance(L, rng)
        if inst is None:
            raise RuntimeError(f"gen_cinstance failed too often at L={L}")
        out.append(inst)
    return out


def branch_stats(batch):
    """Fraction of then-taken branches across a batch (distribution check)."""
    n_then = n_tot = 0
    for (_, _, _, summary) in batch:
        for (_, taken) in summary:
            n_tot += 1
            n_then += (taken == "then")
    return {"n_branches": n_tot,
            "frac_then": (n_then / n_tot) if n_tot else 0.0}
