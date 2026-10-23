"""
Tests for noisy-input interval execution (spec 5.6 task 3):
interval transformer exactness, join/guard-decision logic, the noisy executor
(flat + conditional with straddle joins), the independent interval verifier
(incl. adversarial rejections), and the loud neural-divergence halt.
"""

import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from certificates import (RULES, OP_NAMES, verify_noise_certificate,
                          verify_interval_trace)
from conditionals import ifz
from intervals import (iv, width, contains, join, guard_decision,
                       INTERVAL_RULES, iv_to_json, iv_from_json)
from noise import NoiseProofMachine, NoiseDivergence
from compose import ProofMachine

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {name} {detail}", flush=True)


def concrete_image(op, lo, hi):
    """Brute-force true image of [lo, hi] under op (exact integer semantics)."""
    from certificates import var, eval_expr
    e = var()
    e2 = RULES[op](e)
    return {eval_expr(e2, x) for x in range(lo, hi + 1)}


# -- 1. interval transformer exactness vs brute force -------------------------
rng = random.Random(0)
for op in OP_NAMES:
    for _ in range(60):
        lo = rng.randint(-5000, 5000)
        hi = lo + rng.randint(0, 40)
        got = INTERVAL_RULES[op]((lo, hi))
        img = concrete_image(op, lo, hi)
        check(f"exact {op} [{lo},{hi}]",
              got == (min(img), max(img)) and
              all(got[0] <= v <= got[1] for v in img),
              f"got {got}, image range {(min(img), max(img))}")

# -- 2. MOD wrap corners ------------------------------------------------------
check("mod full cover", INTERVAL_RULES["MOD_7"]((3, 10)) == (0, 6))
check("mod no wrap", INTERVAL_RULES["MOD_7"]((8, 12)) == (1, 5))
check("mod wrap hull", INTERVAL_RULES["MOD_7"]((6, 9)) == (0, 6))
check("mod negative", INTERVAL_RULES["MOD_7"]((-3, 2)) == (0, 6))
check("mod point", INTERVAL_RULES["MOD_13"]((26, 26)) == (0, 0))
check("mod13 exact", INTERVAL_RULES["MOD_13"]((14, 20)) == (1, 7))

# -- 3. domain ops ------------------------------------------------------------
check("join hull", join((1, 5), (3, 9)) == (1, 9))
check("join disjoint", join((-5, -1), (2, 4)) == (-5, 4))
check("join idempotent", join((2, 2), (2, 2)) == (2, 2))
check("guard then", guard_decision((0, 0)) == "then")
check("guard else-pos", guard_decision((1, 8)) == "else")
check("guard else-neg", guard_decision((-8, -1)) == "else")
check("guard straddle", guard_decision((-2, 3)) == "straddle")
check("guard straddle-hi0", guard_decision((-4, 0)) == "straddle")
check("width", width((3, 10)) == 7)
check("contains", contains((3, 10), 3) and contains((3, 10), 10)
      and not contains((3, 10), 11))
check("iv json roundtrip", iv_from_json(iv_to_json((-3, 42))) == (-3, 42))

# -- 4. noisy flat execution + verifier ---------------------------------------
pm = ProofMachine(CKPT)
npm = NoiseProofMachine(pm)

x, cert = npm.run_noisy(42, 3, ["ADD_3", "MUL_2", "MOD_7", "ADD_1"])
check("noisy flat nominal", x == ((42 + 3) * 2 % 7) + 1, f"got {x}")
ok, info = verify_noise_certificate(cert)
check("noisy flat verifies", ok, str(info.get("error", "")))
check("noisy flat width sane", cert["final_width"] == info["final_width"])

# true-output-set containment (brute force)
true_set = NoiseProofMachine.true_output_set_flat(42, 3,
                                                 ["ADD_3", "MUL_2", "MOD_7", "ADD_1"])
flo, fhi = cert["final_interval"]
check("containment flat",
      all(flo <= v <= fhi for v in true_set),
      f"set outside [{flo},{fhi}]")
# exactness: flat interval trace == true reachable range (documented property)
check("flat exactness",
      (flo, fhi) == (min(true_set), max(true_set)),
      f"[{flo},{fhi}] vs true {(min(true_set), max(true_set))}")

# r = 0 degenerates to point execution
x0, c0 = npm.run_noisy(-7, 0, ["NEG", "MUL_5", "ADD_7"])
check("r=0 point", x0 == 42 and c0["final_interval"] == [42, 42],
      f"got {x0} {c0['final_interval']}")
ok0, _ = verify_noise_certificate(c0)
check("r=0 verifies", ok0)

# -- 5. adversarial verifier rejections --------------------------------------
bad = copy.deepcopy(cert)
bad["steps"][1]["out"] = [0, 999999]  # tampered interval
ok, info = verify_noise_certificate(bad)
check("reject tampered out", not ok)

bad = copy.deepcopy(cert)
bad["steps"][2]["in"] = [123, 456]  # broken chaining
ok, info = verify_noise_certificate(bad)
check("reject broken chain", not ok)

bad = copy.deepcopy(cert)
bad["result"] = 10 ** 9  # nominal outside final interval
ok, info = verify_noise_certificate(bad)
check("reject nominal escape", not ok)

bad = copy.deepcopy(cert)
bad["steps"][0]["op"] = "NOPE"  # unknown op
ok, info = verify_noise_certificate(bad)
check("reject unknown op", not ok)

bad = copy.deepcopy(cert)
bad["steps"] = []  # empty trace
ok, info = verify_noise_certificate(bad)
check("reject empty trace", not ok)

# -- 6. conditionals under noise ----------------------------------------------
# Decisive then: guard interval is exactly {0} for every x in [-r, r]... use
# MOD_7 on multiples of 7: x0 = 14, r = 0 -> {0}. For a real interval test use
# a guard that is constant-0 on the interval: NEG/NEG? simplest: guard =
# ["MOD_7"] with x0=14, r=0 is decisive; straddle needs a varying guard.
prog_then = [ifz(["MOD_7"], ["ADD_1"], ["MUL_2"])]
x, cert = npm.run_program_noisy(14, 0, prog_then)
check("decisive then", x == 15 and cert["straddles"] == 0, f"got {x}")
ok, info = verify_noise_certificate(cert)
check("decisive then verifies", ok, str(info.get("error", "")))

prog_else = [ifz(["MOD_7"], ["ADD_1"], ["MUL_2"])]
x, cert = npm.run_program_noisy(15, 0, prog_else)
check("decisive else", x == 30 and cert["straddles"] == 0, f"got {x}")

# Straddle: guard MOD_7 over x0=13, r=2 -> inputs 11..15, residues 4,5,6,0,1
# -> guard interval [0,6] straddles 0 -> join of ADD_1 branch [12,16] and
# MUL_2 branch [22,30] -> [12,30].
prog_str = [ifz(["MOD_7"], ["ADD_1"], ["MUL_2"])]
x, cert = npm.run_program_noisy(13, 2, prog_str)
check("straddle detected", cert["straddles"] == 1, str(cert["straddles"]))
check("straddle join interval", cert["final_interval"] == [12, 30],
      str(cert["final_interval"]))
ok, info = verify_noise_certificate(cert)
check("straddle verifies", ok, str(info.get("error", "")))
true_set = NoiseProofMachine.true_output_set_program(13, 2, prog_str)
flo, fhi = cert["final_interval"]
check("containment straddle",
      all(flo <= v <= fhi for v in true_set),
      f"{sorted(true_set)} outside [{flo},{fhi}]")
# join over-approximation is real here: true set has a gap
check("join over-approx measured",
      (fhi - flo) > (max(true_set) - min(true_set)),
      f"true range {(min(true_set), max(true_set))}")

# J3: fake straddle claim rejected (guard interval decisive but node claims join)
bad = copy.deepcopy(cert)
# find the branch node and flip a decisive one... build a decisive cert and
# corrupt its node into a fake straddle
_, cert_d = npm.run_program_noisy(14, 0, prog_then)
bad = copy.deepcopy(cert_d)
node = bad["steps"][0]
node["straddle"] = True
node["then_out"] = node["out"]
node["else_out"] = node["out"]
node["then_steps"] = node["steps"]
node["else_steps"] = node["steps"]
ok, info = verify_noise_certificate(bad)
check("reject fake straddle (J3)", not ok, str(info.get("error", "")))

# J5: wrong join rejected
_, cert_s = npm.run_program_noisy(13, 2, prog_str)
bad = copy.deepcopy(cert_s)
bad["steps"][0]["out"] = [12, 29]  # off-by-one join
ok, info = verify_noise_certificate(bad)
check("reject wrong join (J5)", not ok)

# I3: taken inconsistent with guard interval rejected
_, cert_d2 = npm.run_program_noisy(15, 0, prog_else)
bad = copy.deepcopy(cert_d2)
bad["steps"][0]["taken"] = "then"  # guard says else
ok, info = verify_noise_certificate(bad)
check("reject wrong taken (I3)", not ok)

# -- 7. loud neural-divergence halt -------------------------------------------
import primitives as prim_mod
real_apply = prim_mod.apply_concrete


def evil_apply(net, op, x):
    return real_apply(net, op, x) + 10 ** 6  # shove the nominal out of range


prim_mod.apply_concrete = evil_apply
try:
    npm.run_noisy(5, 1, ["ADD_1"])
    check("divergence halts loud", False, "no exception raised")
except NoiseDivergence:
    check("divergence halts loud", True)
except Exception as ex:
    check("divergence halts loud", False, f"wrong exception {type(ex).__name__}")
finally:
    prim_mod.apply_concrete = real_apply

# -- 8. nested program under noise --------------------------------------------
nested = [ifz(["MOD_7"], [ifz(["MOD_13"], ["ADD_1"], ["ADD_3"])], ["NEG"])]
x, cert = npm.run_program_noisy(27, 2, nested)
ok, info = verify_noise_certificate(cert)
check("nested noisy verifies", ok, str(info.get("error", "")))
true_set = NoiseProofMachine.true_output_set_program(27, 2, nested)
flo, fhi = cert["final_interval"]
check("containment nested",
      all(flo <= v <= fhi for v in true_set))

# -- 9. batch generators (regression: batch wrappers pass r through) ---------
from taskgen_noise import gen_noisy_batch, gen_noisy_cbatch
b = gen_noisy_batch(4, 3, 2, seed=7)
check("gen_noisy_batch", len(b) == 3 and all(x[1] == 2 for x in b),
      str(len(b)))
bc = gen_noisy_cbatch(4, 2, 1, seed=11)
check("gen_noisy_cbatch", len(bc) == 2 and all(x[1] == 1 for x in bc),
      str(len(bc)))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
