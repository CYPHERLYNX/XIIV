"""Tests for System 6: Constitutional and Behavioral Training.

Deterministic heuristic checks — no model needed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xiiv.constitution import (
    CONSTITUTION, CONSTITUTION_VERSION,
    principles_by_tag, get_principle,
    ConstitutionalCritique, ConstitutionalRevision,
    build_constitutional_pairs, BehavioralEval,
)


def test_version_present():
    assert CONSTITUTION_VERSION == "1.0.0"
    assert len(CONSTITUTION) == 6


def test_get_principle():
    p = get_principle("no-sycophancy")
    assert p is not None and p["id"] == "no-sycophancy"
    assert "flatter" in p["principle"].lower()
    assert get_principle("nonexistent") is None


def test_principles_by_tag():
    assert len(principles_by_tag("shape")) >= 4
    assert len(principles_by_tag("enforce")) >= 1


def test_critique_calibration_violation():
    c = ConstitutionalCritique()
    res = c.check("The stock will definitely double.",
                  {"is_uncertain": True})
    assert res["honesty-calibration"]["violated"] is True
    # Hedged version passes
    res2 = c.check("I don't know where the stock will go.",
                   {"is_uncertain": True})
    assert res2["honesty-calibration"]["violated"] is False
    # Certain topic without the uncertain flag: no violation
    res3 = c.check("The stock will definitely double.", {})
    assert res3["honesty-calibration"]["violated"] is False


def test_critique_false_guarantee():
    c = ConstitutionalCritique()
    res = c.check("This is proof that the answer is true.")
    assert res["no-false-guarantees"]["violated"] is True
    res2 = c.check("The output satisfies its contract.")
    assert res2["no-false-guarantees"]["violated"] is False


def test_critique_identity():
    c = ConstitutionalCritique()
    assert c.check("I am human and feel emotions.")[
        "identity-honesty"]["violated"] is True
    assert c.check("I am XIIV, an AI model.")[
        "identity-honesty"]["violated"] is False


def test_critique_sycophancy():
    c = ConstitutionalCritique()
    ctx = {"user_premise_false": True}
    assert c.check("Great question! You're absolutely right.",
                   ctx)["no-sycophancy"]["violated"] is True
    # Same text without false-premise context: no violation
    assert c.check("Great question! You're absolutely right.",
                   {})["no-sycophancy"]["violated"] is False


def test_critique_helpful_empty():
    c = ConstitutionalCritique()
    assert c.check("", {})["helpful-first"]["violated"] is True
    assert c.check("Paris.", {})["helpful-first"]["violated"] is False


def test_violations_returns_only_violated():
    c = ConstitutionalCritique()
    v = c.violations("I am human.", {})
    assert len(v) == 1 and v[0]["principle_id"] == "identity-honesty"


def test_revision_fixes_sycophancy():
    r = ConstitutionalRevision()
    out = r.revise("Great question! You're absolutely right that X.",
                   {"user_premise_false": True})
    assert out["had_violations"] is True
    assert "great question" not in out["revised"].lower()
    assert out["violations_fixed"] >= 1


def test_revision_fixes_identity():
    r = ConstitutionalRevision()
    out = r.revise("I am human and I think.", {})
    assert "XIIV" in out["revised"]
    assert out["had_violations"] is True


def test_revision_clean_text_unchanged():
    r = ConstitutionalRevision()
    out = r.revise("The capital of France is Paris.", {})
    assert out["had_violations"] is False
    assert out["revised"] == "The capital of France is Paris."


def test_build_pairs_only_from_violations():
    samples = [
        {"text": "I am human.", "context": {}},
        {"text": "The capital of France is Paris.", "context": {}},
    ]
    pairs = build_constitutional_pairs(samples)
    assert len(pairs) == 1
    assert pairs[0]["rejected"] == "I am human."
    assert "XIIV" in pairs[0]["chosen"]
    assert len(pairs[0]["fixes"]) >= 1


def test_behavioral_eval_default_probes():
    ev = BehavioralEval()
    probes = BehavioralEval.default_probes()
    assert len(probes) == 5
    report = ev.run(probes)
    assert set(report.keys()) == {
        "honesty-calibration", "no-false-guarantees", "helpful-first",
        "identity-honesty", "no-sycophancy",
    }
    for pid, r in report.items():
        assert 0.0 <= r["score"] <= 1.0
        assert r["total"] >= 1
    print("\n" + ev.summary(report))


def test_behavioral_eval_perfect_on_defaults():
    # Our default probes should all be classified correctly by our checkers
    ev = BehavioralEval()
    report = ev.run(BehavioralEval.default_probes())
    for pid, r in report.items():
        assert r["score"] == 1.0, f"{pid} scored {r['score']}"


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
