"""
Phase 2, step 1 — supervised warmup dataset for the learned composition policy.

Generates (spec string -> op sequence) pairs by running the deterministic VRC
planner (vrc_harness.skill_genseq) over random affine specs in the 8-op closure.
The planner is the reference trajectory: training on its plans is the
trajectory-contraction supervision of Sakurai et al. (arXiv:2610.31576) — the
policy is only ever shown commitments that stay on a reference plan.

Spec generation: random op chains from x0 (ADD c in {+-1,+-3,+-7}, MUL k in
{+-2,+-5}, NEG, MOD m in {7,13}), rendered with certificates.canon as the
model input. The planner internally applies realize(), so the model must
learn the NEG-sandwich rewrites from data — that is the interesting part.

Filters (all counted, never silently dropped):
  - planner SpecRejected (outside the 8-op closure, e.g. coefficient with a
    factor 3, constant target) -> skip
  - probe bound: |eval| > 1e5 at any probe x0 in (5,-3,42,1000,-1000) -> skip
    (keeps neural primitive execution in its exact regime)
  - plan length > 16 -> skip
  - duplicate spec string -> skip
  - roundtrip: parse(canon(e)) must realize to the same expr -> else skip

Output: JSONL, one {"spec": str, "ops": [...], "plan_len": int} per line.
Deterministic given --seed. No GPU, pure Python + the VRC planner.

Usage:
    python gen_dataset.py --train 8000 --test 1500 --out data
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, PM)

from certificates import (  # noqa: E402
    add, canon, eval_expr, mod, mul, neg, var)
from vrc_harness import (  # noqa: E402
    SpecRejected, _plan, parse_target, realize)

PROBES = (5, -3, 42, 1000, -1000)
MAX_PLAN_LEN = 16
BOUND = 100000

ADD_CONSTS = (1, 3, 7, -1, -3, -7)
MUL_CONSTS = (2, 5, -2, -5)
MODS = (7, 13)


def rand_expr(rng):
    """Random affine expression over x0 as an op chain from var()."""
    e = var()
    n_ops = rng.randint(1, 6)
    for _ in range(n_ops):
        r = rng.random()
        if r < 0.35:
            e = add(e, rng.choice(ADD_CONSTS))
        elif r < 0.60:
            e = mul(e, rng.choice(MUL_CONSTS))
        elif r < 0.75:
            e = neg(e)
        elif r < 0.90:
            e = mod(e, rng.choice(MODS))
        else:
            # two-op flourish: add then mul (produces (x0+c)*k shapes)
            e = mul(add(e, rng.choice(ADD_CONSTS)), rng.choice((2, 5)))
    return e


def label_expr(e):
    """Plan one expr. Returns (spec_str, ops) or None with a skip reason."""
    try:
        realized = realize(e)
    except SpecRejected:
        return None, "realize-rejected"
    # Roundtrip: the model only ever sees the canon string; it must parse
    # back to the same realized target.
    try:
        reparsed = realize(parse_target(canon(e)))
    except SpecRejected:
        return None, "roundtrip-rejected"
    if reparsed != realized:
        return None, "roundtrip-mismatch"
    try:
        ops, _ = _plan(realized, 0, frozenset())
    except SpecRejected:
        return None, "plan-rejected"
    if len(ops) > MAX_PLAN_LEN:
        return None, "plan-too-long"
    for x0 in PROBES:
        if abs(eval_expr(realized, x0)) > BOUND:
            return None, "probe-out-of-bound"
    return {"spec": canon(e), "ops": ops, "plan_len": len(ops)}, None


def gen(n, rng, seen):
    """Generate n labeled pairs; returns (pairs, skip_counts)."""
    pairs, skips = [], {}
    while len(pairs) < n:
        item, reason = label_expr(rand_expr(rng))
        if item is None:
            skips[reason] = skips.get(reason, 0) + 1
            continue
        if item["spec"] in seen:
            skips["duplicate"] = skips.get("duplicate", 0) + 1
            continue
        seen.add(item["spec"])
        pairs.append(item)
    return pairs, skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    train, sk_train = gen(args.train, rng, seen)
    test, sk_test = gen(args.test, rng, seen)
    os.makedirs(args.out, exist_ok=True)
    for name, pairs in (("train", train), ("test", test)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
    stats = {
        "seed": args.seed,
        "train": len(train), "test": len(test),
        "skips_train": sk_train, "skips_test": sk_test,
        "plan_len_train": {str(k): sum(1 for p in train if p["plan_len"] == k)
                           for k in sorted({p["plan_len"] for p in train})},
        "plan_len_test": {str(k): sum(1 for p in test if p["plan_len"] == k)
                          for k in sorted({p["plan_len"] for p in test})},
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
