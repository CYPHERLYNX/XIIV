"""
Tests for the VRC (Verity-Routed Composition) harness.

Deterministic, CPU-only, no training. Run:
    ~/workspace/ai-tools/corsac/.venv/bin/python test_vrc.py

Conventions: plain asserts; any VRCHalted where success is expected fails the test.
Fault-injection tests (corrupted checkpoint, simulated Phase-2 interface failure)
are explicitly labeled — they test the harness's LOUD failure modes, not the oracle.
"""

import os
import shutil
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from certificates import canon, var, verify_certificate  # noqa: E402
from compose import ProofMachine  # noqa: E402
from vrc_harness import (VRC, VRCHalted, SpecRejected, parse_target, realize,  # noqa: E402
                         _plan)
import vrc_harness as vh  # noqa: E402  (monkeypatch point for the interface oracle)

CKPT = os.path.join(HERE, "checkpoints")
VENV_PM = None


def get_pm():
    global VENV_PM
    if VENV_PM is None:
        VENV_PM = ProofMachine(CKPT)
    return VENV_PM


# ---------------------------------------------------------------------------
# 1. Parser + realize + constructive planner
# ---------------------------------------------------------------------------

def test_parse_and_plan():
    cases = {
        "((x0+3)*2 mod 7)+1": ["ADD_3", "MUL_2", "MOD_7", "ADD_1"],
        "5*(x0+3)": ["ADD_3", "MUL_5"],
        "(x0 mod 7)-3": ["MOD_7", "NEG", "ADD_3", "NEG"],
        "(-2*x0)+7": ["MUL_2", "NEG", "ADD_7"],
        "10*(x0+2)+1": ["MUL_2", "MUL_5", "ADD_7", "ADD_7", "ADD_7"],
        "(((x0+1)*2+3) mod 7)": ["MUL_2", "ADD_3", "ADD_1", "ADD_1", "MOD_7"],
    }
    for spec, want in cases.items():
        ops, _ = _plan(realize(parse_target(spec)), 0, frozenset())
        assert ops == want, f"{spec}: got {ops}, want {want}"
    print("ok test_parse_and_plan")


def test_realize_lemma():
    # ("add", E, c<0) from the parser is rewritten to the NEG sandwich,
    # which the trace language can produce; ("mul", E, k<0) similarly.
    t = realize(parse_target("x0-3"))
    assert t == ("neg", ("add", ("neg", ("var",)), 3)), canon(t)
    t = realize(parse_target("-2*x0"))
    assert t == ("neg", ("mul", ("var",), 2)), canon(t)
    # constants pass through realize untouched (rejected later by _plan)
    assert realize(parse_target("5")) == ("const", 5)
    print("ok test_realize_lemma")


# ---------------------------------------------------------------------------
# 2. End-to-end synthesis: verified compositions, bounded iterations
# ---------------------------------------------------------------------------

SUCCESS_TARGETS = [
    "((x0+3)*2 mod 7)+1",          # the spec's example; depth 4
    "5*(x0+3)",                    # depth 4
    "(x0 mod 7)-3",                # NEG sandwich; depth 4
    "(-2*x0)+7",                   # negative coefficient; depth 3
    "10*(x0+2)+1",                 # depth 5
    "(((x0+1)*2+3) mod 7)",        # depth 5
    "((((x0+1)*2+3)*5+7) mod 13)+8",  # depth 11
]


def test_synthesize_success():
    vrc = VRC(get_pm())
    for spec in SUCCESS_TARGETS:
        out = vrc.synthesize(spec)
        assert out["status"] == "ok", spec
        assert out["iters"] <= VRC.MAX_ITERS, (spec, out["iters"])
        assert out["iters"] == 1, (spec, out["iters"])  # constructive: first try
        # every probe: independent verifier accepts, concrete == symbolic
        for p in out["probes"]:
            result, cert = get_pm().run(p["x0"], out["ops"])
            vok, _ = verify_certificate(cert)
            assert vok, (spec, p["x0"])
            assert result == p["result"], (spec, p["x0"])
        # symbolic result equals the realized target (checked inside, re-assert)
        e = var()
        from certificates import RULES
        for op in out["ops"]:
            e = RULES[op](e)
        assert e == realize(parse_target(spec)), spec
    print(f"ok test_synthesize_success ({len(SUCCESS_TARGETS)} targets, "
          f"all 1-iteration, all certs verify)")


# ---------------------------------------------------------------------------
# 3. Spec rejection: outside the closure -> loud, categorized, with cause
# ---------------------------------------------------------------------------

def test_spec_rejections():
    vrc = VRC(get_pm())
    cases = {
        "3*x0": "spec-rejected",       # coefficient with prime factor 3
        "6*(x0+1)": "spec-rejected",   # 6 = 2*3
        "x0 mod 9": "spec-rejected",   # modulus not in {7, 13}
        "5": "spec-rejected",          # constant: x0 cannot be discarded
        "x0*x0": "spec-rejected",      # nonlinear
        "x0/2": "spec-rejected",       # unparsable
        "x0+(x0+3)": "spec-rejected",  # addition of non-constants
    }
    for spec, want_cat in cases.items():
        try:
            vrc.synthesize(spec)
        except VRCHalted as h:
            assert h.category == want_cat, (spec, h.category, h.diagnosis)
            assert h.diagnosis, spec  # a reason is always given
        else:
            raise AssertionError(f"{spec}: expected VRCHalted, got success")
    # the factor-3 diagnosis names the offending coefficient
    try:
        vrc.synthesize("3*x0")
    except VRCHalted as h:
        assert "3" in h.diagnosis and "{2, 5}" in h.diagnosis, h.diagnosis
    print(f"ok test_spec_rejections ({len(cases)} specs, all loud)")


# ---------------------------------------------------------------------------
# 4. FAULT INJECTION: corrupted primitive -> neural-divergence, loud, localized
#    (tests the harness's failure mode, not the oracle)
# ---------------------------------------------------------------------------

def test_neural_divergence_fault_injection():
    tmp = tempfile.mkdtemp()
    try:
        dst = os.path.join(tmp, "checkpoints")
        shutil.copytree(CKPT, dst)
        # Corrupt MOD_7: zero its weights -> argmax always 0.
        p = os.path.join(dst, "MOD_7.pt")
        sd = torch.load(p, map_location="cpu", weights_only=True)
        for k in sd:
            sd[k].zero_()
        torch.save(sd, p)
        vrc = VRC(ProofMachine(dst))
        try:
            vrc.synthesize("(x0 mod 7)+1", probes=[5])
        except VRCHalted as h:
            assert h.category == "neural-divergence", (h.category, h.diagnosis)
            assert "MOD_7" in h.diagnosis, h.diagnosis
            assert "no certificate shipped" in h.diagnosis
        else:
            raise AssertionError("expected VRCHalted(neural-divergence)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok test_neural_divergence_fault_injection")


# ---------------------------------------------------------------------------
# 5. SIMULATED Phase-2 interface failure -> REPAIRSTEP reroutes, then succeeds.
#    (The Phase-1 interface leg is vacuous, so the oracle leg is simulated here;
#    the ROUTER and the taboo-replan machinery under test are real.)
# ---------------------------------------------------------------------------

def test_interface_routing_simulated():
    real = vh.interface_check
    calls = {"n": 0}

    def fake(prev_out_expr, next_op):
        calls["n"] += 1
        if calls["n"] == 2:  # fail exactly once: step index 1
            return False, "simulated Phase-2 precondition failure: P_g unsatisfied"
        return True, "ok"

    vh.interface_check = fake
    try:
        vrc = VRC(get_pm())
        out = vrc.synthesize("2*x0+3", probes=[5])
    finally:
        vh.interface_check = real
    # Repair reroutes around the tabooed (1, ADD_3): +3 becomes +1+1+1.
    assert out["ops"] == ["MUL_2", "ADD_1", "ADD_1", "ADD_1"], out["ops"]
    assert out["iters"] == 2, out["iters"]
    assert out["repairs"] == 1
    h0 = out["history"][0]
    assert h0["category"] == "interface-fail" and h0["routed_to"] == "REPAIRSTEP"
    assert h0["step"] == 1 and h0["op"] == "ADD_3"
    # the repaired composition still verifies end-to-end
    result, cert = get_pm().run(5, out["ops"])
    vok, _ = verify_certificate(cert)
    assert vok and result == 2 * 5 + 3
    print("ok test_interface_routing_simulated (repair reroutes, cert verifies)")


def test_repair_exhausted_simulated():
    real = vh.interface_check

    def always_fail(prev_out_expr, next_op):
        return False, "simulated Phase-2 precondition failure"

    vh.interface_check = always_fail
    try:
        vrc = VRC(get_pm())
        try:
            vrc.synthesize("2*x0+3", probes=[5])
        except VRCHalted as h:
            assert h.category == "repair-exhausted", (h.category, h.diagnosis)
            assert "taboo" in h.diagnosis
        else:
            raise AssertionError("expected VRCHalted(repair-exhausted)")
    finally:
        vh.interface_check = real
    print("ok test_repair_exhausted_simulated")


# ---------------------------------------------------------------------------
# 6. Phase-2 hooks exist and are honest no-ops in Phase 1
# ---------------------------------------------------------------------------

def test_phase2_hooks_noop():
    vrc = VRC(get_pm())
    for skill in (vrc.skill_strengthen, vrc.skill_weaken):
        r = skill({"category": "interface-fail"})
        assert r["status"] == "noop-phase1", r
    print("ok test_phase2_hooks_noop")


if __name__ == "__main__":
    test_parse_and_plan()
    test_realize_lemma()
    test_synthesize_success()
    test_spec_rejections()
    test_neural_divergence_fault_injection()
    test_interface_routing_simulated()
    test_repair_exhausted_simulated()
    test_phase2_hooks_noop()
    print("\nALL VRC TESTS PASSED")
