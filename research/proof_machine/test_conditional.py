"""
Tests for conditional composition (spec 5.6 task 2): IFZ branch execution,
branch certificates, the independent verifier's B1-B4 checks, the non-vacuous
interface check, and the loud neural-divergence halt.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compose as compose_mod
from compose import ProofMachine
from certificates import verify_certificate, verify_trace
from conditionals import (ifz, is_ifz, max_path_cost, check_branch_interface,
                          decide_branch, prog_to_json, prog_from_json,
                          prog_to_str)
from taskgen_conditional import gen_cbatch, build_program, instantiate

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {name} {detail}", flush=True)


pm = ProofMachine(CKPT)

# -- 1. branch execution: then / else ---------------------------------------
p1 = [ifz(["MOD_7"], ["ADD_1"], ["MUL_2"])]
r, _ = pm.run_program(14, p1)   # 14 % 7 == 0 -> then: 14 + 1
check("then-branch taken", r == 15, f"got {r}")
r, _ = pm.run_program(15, p1)   # 15 % 7 == 1 -> else: 15 * 2
check("else-branch taken", r == 30, f"got {r}")
check("symbolic agrees (then)", pm.symbolic_result_program(14, p1) == 15)
check("symbolic agrees (else)", pm.symbolic_result_program(15, p1) == 30)

# -- 2. nesting --------------------------------------------------------------
p2 = [ifz(["MOD_7"],
          [ifz(["MOD_13"], ["ADD_3"], ["NEG"])],
          ["MUL_5"])]
# x0=14: 14%7==0 -> then; inner: 14%13=1 -> else -> NEG -> -14
r, c2 = pm.run_program(14, p2)
check("nested then/else", r == -14, f"got {r}")
# x0=15: 15%7==1 -> else -> 75
r, _ = pm.run_program(15, p2)
check("nested else", r == 75, f"got {r}")
check("symbolic nested", pm.symbolic_result_program(14, p2) == -14)

# -- 3. certificates verify (incl. nested) ------------------------------------
ok, info = verify_certificate(c2)
check("branch cert verifies", ok, str(info))
check("branch node counted (top level)", info.get("branch_nodes") == 1, str(info))
inner = c2["steps"][0]["steps"][0]
check("nested branch node present", inner.get("branch") is True and
      inner["taken"] == "else", str(inner.get("taken")))
_, c1 = pm.run_program(14, p1)
ok, _ = verify_certificate(c1)
check("simple branch cert verifies", ok)

# -- 4. verifier REJECTS forgeries ---------------------------------------------
def forged(cert, mutate):
    c = copy.deepcopy(cert)
    mutate(c)
    return c

# 4a. flipped taken label
bad = forged(c1, lambda c: c["steps"][0].update(taken="else"))
ok, info = verify_certificate(bad)
check("reject flipped taken", not ok, str(info))

# 4b. wrong cond_value
bad = forged(c1, lambda c: c["steps"][0].update(cond_value=999))
ok, info = verify_certificate(bad)
check("reject wrong cond_value", not ok, str(info))

# 4c. tampered guard sub-trace (claim MOD_7 output that re-derivation refutes)
def tamper_guard(c):
    node = c["steps"][0]
    node["cond_steps"][0]["out"] = ["const", 12345]
bad = forged(c1, tamper_guard)
ok, info = verify_certificate(bad)
check("reject tampered guard trace", not ok, str(info))

# 4d. branch sub-trace not starting at the pre-guard input
def tamper_branch_in(c):
    node = c["steps"][0]
    node["steps"][0]["in"] = ["const", 777]
bad = forged(c1, tamper_branch_in)
ok, info = verify_certificate(bad)
check("reject branch chaining break", not ok, str(info))

# 4e. tampered branch output
def tamper_branch_out(c):
    c["steps"][0]["out"] = ["const", 424242]
bad = forged(c1, tamper_branch_out)
ok, info = verify_certificate(bad)
check("reject tampered branch out", not ok, str(info))

# -- 5. interface check is REAL (adversarial) ------------------------------------
ok, reason = check_branch_interface(0, "then")
check("iface then/0 ok", ok, reason)
ok, _ = check_branch_interface(5, "then")
check("iface then/5 FAILS", not ok)
ok, _ = check_branch_interface(0, "else")
check("iface else/0 FAILS", not ok)
ok, reason = check_branch_interface(5, "else")
check("iface else/5 ok", ok, reason)
ok, _ = check_branch_interface(0, "sideways")
check("iface bad label FAILS", not ok)
check("decide_branch", decide_branch(0) == "then" and decide_branch(9) == "else")

# -- 6. neural divergence at a guard halts LOUD, names the condition ------------
_real_apply = compose_mod.apply_concrete


def _corrupt_guard(net, op, x):
    v = _real_apply(net, op, x)
    if op in ("MOD_7", "MOD_13"):
        m = int(op.split("_")[1])
        return (v + 1) % m  # guard value wrong -> cv != sv
    return v


compose_mod.apply_concrete = _corrupt_guard
try:
    pm.run_program(14, p1)
    check("divergence halts loud", False, "no exception raised")
except RuntimeError as ex:
    check("divergence halts loud", "neural-divergence" in str(ex), str(ex)[:120])
finally:
    compose_mod.apply_concrete = _real_apply

# sanity: real path still fine after restore
r, _ = pm.run_program(14, p1)
check("restore after fault injection", r == 15)

# -- 7. flat-program equivalence with run() --------------------------------------
flat = ["ADD_3", "MUL_2", "NEG", "MOD_7"]
r1, _ = pm.run(10, flat)
r2, _ = pm.run_program(10, flat)
check("run_program == run on flat", r1 == r2, f"{r1} vs {r2}")

# -- 8. serialization round-trip --------------------------------------------------
p3 = [ifz(["MOD_7"], ["ADD_1", ifz(["MOD_13"], ["NEG"], ["MUL_2"])], ["ADD_3"])]
rt = prog_from_json(prog_to_json(p3))
check("prog json round-trip", rt == p3, prog_to_str(rt))
check("max_path_cost nested", max_path_cost(p3) == 1 + max(1 + 1 + 1, 1),
      f"got {max_path_cost(p3)}")

# -- 9. generator: cost exactness + bound discipline -------------------------------
import random
rng = random.Random(0)
for L in (1, 3, 10, 50):
    prog = build_program(L, rng)
    check(f"gen cost == {L}", max_path_cost(prog) == L)
batch = gen_cbatch(8, 30, seed=7)
all_ok = True
for (x0, prog, expected, _) in batch:
    try:
        pred, cert = pm.run_program(x0, prog)
    except RuntimeError as ex:
        all_ok = False
        print(f"  batch LOUD HALT x0={x0}: {ex}", flush=True)
        continue
    ok, _ = verify_certificate(cert)
    if pred != expected or not ok:
        all_ok = False
        print(f"  batch fail x0={x0} pred={pred} exp={expected} vok={ok}",
              flush=True)
check("gen batch: 30/30 correct + certs verify", all_ok)

# -- 10. verify_trace generalized entry -------------------------------------------
_, cert = pm.run_program(14, p1)
ok, info = verify_trace(cert["steps"], 14, ("var",), 15)
check("verify_trace direct",
      ok and info["final_expr_tuple"] == ("add", ("var",), 1), str(info))

print(f"\n{passed} passed, {failed} failed", flush=True)
sys.exit(1 if failed else 0)
