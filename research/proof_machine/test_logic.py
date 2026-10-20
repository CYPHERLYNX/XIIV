"""
Tests for logic chains (spec section 5.6, task 1).

Deterministic, CPU-only. Run:
    ~/workspace/ai-tools/corsac/.venv/bin/python test_logic.py
Needs checkpoints_logic/ (run train_bool_primitives.py first).

Conventions: plain asserts; any failed check prints and the script exits 1.
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from logic import (  # noqa: E402
    W, RULES_BOOL, BOOL_OP_NAMES, START_VEC, bconst, bvar, bnot, band, bor,
    bxor, canon, canon_bvec, eval_bexpr, eval_bvec, int_to_bits, bits_to_int,
    vec_to_json, vec_from_json, bexpr_to_json, bexpr_from_json,
    verify_bcertificate, verify_btrace)
from bool_primitives import exact_fn  # noqa: E402
from compose_logic import LogicProofMachine  # noqa: E402
from taskgen_logic import gen_batch  # noqa: E402

CKPT = os.path.join(HERE, "checkpoints_logic")

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {name} {detail}", flush=True)


# -- 1. canonical constructors ------------------------------------------------
check("not-not", bnot(bnot(bvar(3))) == bvar(3))
check("not-const", bnot(bconst(1)) == bconst(0))
check("and-annihilator", band(bvar(0), bconst(0)) == bconst(0))
check("and-identity", band(bvar(0), bconst(1)) == bvar(0))
check("and-idempotent", band(bvar(1), bvar(1)) == bvar(1))
check("and-complement", band(bvar(2), bnot(bvar(2))) == bconst(0))
check("or-annihilator", bor(bvar(0), bconst(1)) == bconst(1))
check("or-identity", bor(bvar(0), bconst(0)) == bvar(0))
check("or-complement", bor(bvar(5), bnot(bvar(5))) == bconst(1))
check("xor-self", bxor(bvar(1), bvar(1)) == bconst(0))
check("xor-const-fold", bxor(bvar(4), bconst(1)) == bnot(bvar(4)))
check("xor-identity", bxor(bvar(4), bconst(0)) == bvar(4))
# canonical ordering: prover/verifier agreement depends on determinism
check("and-order", canon(band(bvar(7), bvar(0))) == canon(band(bvar(0), bvar(7))))
check("xor-order", bxor(bvar(3), bvar(1)) == bxor(bvar(1), bvar(3)))

# -- 2. symbolic transformers are EXACT (brute force over all 256 inputs) ----
all_exact = True
for op in BOOL_OP_NAMES:
    fn = exact_fn(op)
    for x in range(256):
        v = RULES_BOOL[op](START_VEC)
        if eval_bvec(v, x) != fn(x):
            all_exact = False
            print(f"  transformer mismatch {op} x={x}", flush=True)
            break
check("RULES_BOOL exact on all 256 inputs x 8 ops", all_exact)

# -- 3. bit helpers ------------------------------------------------------------
check("bits round-trip",
      all(bits_to_int(int_to_bits(x)) == x for x in range(256)))
check("eval_bexpr",
      eval_bexpr(bxor(bvar(0), bconst(1)), 0b101) == 0 and
      eval_bexpr(band(bvar(2), bvar(0)), 0b101) == 1)

# -- 4. JSON round-trip ---------------------------------------------------------
v_mid = RULES_BOOL["XOR_A5"](RULES_BOOL["SHL_1"](START_VEC))
check("vec json round-trip",
      vec_from_json(vec_to_json(v_mid)) == v_mid)
e = bxor(band(bvar(0), bvar(7)), bnot(bvar(3)))
check("bexpr json round-trip",
      bexpr_from_json(bexpr_to_json(e)) == e)

# -- 5. end-to-end: neural == symbolic, certificates verify --------------------
pm = LogicProofMachine(CKPT)
rng = random.Random(11)
e2e_ok = True
for trial in range(60):
    L = rng.randint(1, 40)
    x0 = rng.randint(0, 255)
    ops = [rng.choice(BOOL_OP_NAMES) for _ in range(L)]
    pred, cert = pm.run(x0, ops)
    ok, info = verify_bcertificate(cert)
    if pred != pm.symbolic_result(x0, ops) or not ok:
        e2e_ok = False
        print(f"  e2e fail x0={x0} L={L} pred={pred} "
              f"sym={pm.symbolic_result(x0, ops)} vok={ok} {info}",
              flush=True)
check("60 random compositions: neural==symbolic + cert verifies", e2e_ok)

# -- 6. generator sanity ---------------------------------------------------------
batch = gen_batch(16, 50, seed=3)
gen_ok = all(pm.symbolic_result(x0, ops) == exp for (x0, ops, exp) in batch)
check("gen_batch: 50 instances match symbolic reference", gen_ok)
check("gen_batch x0 range",
      all(0 <= x0 <= 255 for (x0, _, _) in batch))

# -- 7. adversarial verifier: tampering must fail --------------------------------
x0, ops, exp = batch[0]
_, cert = pm.run(x0, ops)
import copy
bad = copy.deepcopy(cert)
bad["steps"][0]["out"] = vec_to_json(START_VEC)  # corrupt first step
ok, _ = verify_bcertificate(bad)
check("tampered step rejected", not ok)
bad2 = copy.deepcopy(cert)
bad2["result"] = (int(cert["result"]) + 1) & 0xFF  # wrong claimed result
ok, _ = verify_bcertificate(bad2)
check("wrong result rejected", not ok)
bad3 = copy.deepcopy(cert)
bad3["steps"][1]["op"] = "ADD_1"  # unknown op in this domain
ok, _ = verify_bcertificate(bad3)
check("unknown op rejected", not ok)
bad4 = copy.deepcopy(cert)
bad4["steps"] = []  # empty trace
ok, _ = verify_bcertificate(bad4)
check("empty trace rejected", not ok)

# -- 8. verify_btrace direct ------------------------------------------------------
_, cert = pm.run(0x3C, ["NOT", "XOR_A5", "ROL_1"])
ok, info = verify_btrace(cert["steps"], 0x3C, START_VEC,
                         pm.symbolic_result(0x3C, ["NOT", "XOR_A5", "ROL_1"]))
check("verify_btrace direct", ok and info["steps"] == 3, str(info))
# expected value hand-computed: NOT(0x3C)=0xC3, ^0xA5=0x66, ROL=0xCC
check("hand-computed chain", pm.symbolic_result(0x3C, ["NOT", "XOR_A5", "ROL_1"]) == 0xCC)

print(f"\n{passed} passed, {failed} failed", flush=True)
sys.exit(1 if failed else 0)
