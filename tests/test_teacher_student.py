"""Tests for System 5: Teacher-Student Improvement Loop.

Uses tiny models and dummy verifiers. No real training runs.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.teacher_student import (
    Problem, CandidateSolution, CritiqueResult,
    SolutionGenerator, SolutionCritique, PreferencePairBuilder,
    StudentTrainer, split_problems, report_gap, evaluate_accuracy,
    VERIFIERS, register_verifier, LoopConfig,
)


def tiny_cfg(**kw):
    d = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
             ffn_dim=128, mask_id=254, eos_id=255, seq_len=32)
    d.update(kw)
    return ModelConfig(**d)


def _tiny(seed=0):
    torch.manual_seed(seed)
    return XIIVModel(tiny_cfg())


def _problem(pid="p1", verifier="exact_match", expected="42"):
    return Problem(
        problem_id=pid,
        prompt_text="What is 6*7?",
        prompt_ids=[10, 11, 12, 13],
        verifier=verifier,
        verifier_data={"expected": expected},
    )


def test_verifier_exact_match():
    ok, notes = VERIFIERS["exact_match"]("the answer is 42", {"expected": "42"})
    assert ok, notes
    ok, notes = VERIFIERS["exact_match"]("the answer is 43", {"expected": "42"})
    assert not ok


def test_verifier_math_numeric():
    ok, _ = VERIFIERS["math_numeric"]("result: 3.14159", {"expected": 3.14159})
    assert ok
    ok, _ = VERIFIERS["math_numeric"]("result: 2.0", {"expected": 3.14,
                                                     "tolerance": 0.01})
    assert not ok
    ok, notes = VERIFIERS["math_numeric"]("no numbers here", {"expected": 1.0})
    assert not ok and "no numeric" in notes


def test_verifier_code_unittest():
    code = "```python\ndef add(a, b):\n    return a + b\n```"
    tests = "assert add(2, 3) == 5"
    ok, _ = VERIFIERS["code_unittest"](code, {"tests": tests})
    assert ok
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    ok, notes = VERIFIERS["code_unittest"](bad, {"tests": tests})
    assert not ok


def test_register_verifier():
    register_verifier("always_yes", lambda t, d: (True, "ok"))
    assert "always_yes" in VERIFIERS


def test_solution_generator_produces_k():
    model = _tiny()
    gen = SolutionGenerator(model, mask_id=254, eos_id=255, gen_length=16,
                            steps=4)
    prob = _problem()
    cands = gen.generate(prob, k=3, base_seed=1)
    assert len(cands) == 3
    for c in cands:
        assert c.problem_id == "p1"
        assert len(c.ids) == len(prob.prompt_ids) + 16


def test_critique_uses_independent_verifier():
    crit = SolutionCritique(decode_fn=lambda ids: "the answer is 42")
    prob = _problem()
    cand = CandidateSolution("p1", 0, [10, 11, 12, 13, 20, 21], "", 0)
    res = crit.critique(cand, prob)
    assert res.verified_correct is True
    assert res.quality_score > 0

    crit2 = SolutionCritique(decode_fn=lambda ids: "the answer is 43")
    res2 = crit2.critique(cand, prob)
    assert res2.verified_correct is False
    assert res2.quality_score == 0.0


def test_critique_unknown_verifier():
    crit = SolutionCritique()
    prob = _problem(verifier="nonexistent")
    cand = CandidateSolution("p1", 0, [1, 2], "", 0)
    res = crit.critique(cand, prob)
    assert res.verified_correct is False
    assert res.passed_verifier is False


def test_rank_verified_first():
    crit = SolutionCritique()
    mk = lambda correct, q: CritiqueResult(
        CandidateSolution("p1", 0, [1], "", 0), correct, "", q)
    ranked = crit.rank([mk(False, 0.9), mk(True, 0.3), mk(True, 0.8)])
    assert ranked[0].quality_score == 0.8  # best verified first
    assert ranked[1].quality_score == 0.3
    assert ranked[2].verified_correct is False


def test_pair_builder_needs_verified_chosen():
    builder = PreferencePairBuilder()
    prob = _problem()
    mk = lambda correct: CritiqueResult(
        CandidateSolution("p1", 0, [1, 2], "", 0), correct, "", 0.5)
    # No verified candidate -> no pair (never fabricate signal)
    assert builder.build(prob, [mk(False), mk(False)]) is None
    # Verified chosen + incorrect rejected
    pair = builder.build(prob, [mk(True), mk(False)])
    assert pair is not None
    assert pair.chosen_correct is True
    assert pair.rejected_correct is False


def test_student_trainer_dpo_step():
    torch.manual_seed(0)
    model = _tiny()
    ref = _tiny(seed=0)
    # Same init so ref == model at start
    ref.load_state_dict(model.state_dict())
    trainer = StudentTrainer(model, ref, mask_id=254, lora_rank=4)
    assert trainer.n_trainable > 0
    assert all(not p.requires_grad for p in ref.parameters())

    from xiiv.teacher_student import PreferencePair
    pairs = [
        PreferencePair("p1", [10, 11, 12, 13, 20, 21, 22],
                       [10, 11, 12, 13, 30, 31, 32], True, False, ""),
        PreferencePair("p2", [10, 11, 12, 13, 40, 41],
                       [10, 11, 12, 13, 50, 51], True, False, ""),
    ]
    info = trainer.train_step_dpo(pairs, prefix_lens=[4, 4])
    assert "loss" in info
    # loss may be nan-skipped on degenerate masks; either way it returns
    assert isinstance(info["skipped"], bool)


def test_student_trainer_sft_fallback():
    model = _tiny(seed=1)
    ref = _tiny(seed=1)
    trainer = StudentTrainer(model, ref, mask_id=254, lora_rank=4)
    info = trainer.train_step_sft(
        [[10, 11, 12, 13, 20, 21, 22, 23]], prefix_lens=[4])
    assert info["mode"] == "sft_fallback"
    assert info["n"] >= 0


def test_split_problems_holdout():
    probs = [_problem(pid=f"p{i}") for i in range(10)]
    train, ev = split_problems(probs, eval_frac=0.2, seed=0)
    assert len(ev) == 2 and len(train) == 8
    # Deterministic
    train2, ev2 = split_problems(probs, eval_frac=0.2, seed=0)
    assert [p.problem_id for p in ev] == [p.problem_id for p in ev2]
    # No overlap
    assert not set(p.problem_id for p in train) & set(p.problem_id for p in ev)


def test_gap_report():
    mk = lambda correct: CritiqueResult(
        CandidateSolution("p", 0, [1], "", 0), correct, "", 0.5)
    train = [mk(True)] * 8 + [mk(False)] * 2   # 80%
    ev = [mk(True)] * 5 + [mk(False)] * 5       # 50%
    gap = report_gap(train, ev)
    assert abs(gap.train_acc - 0.8) < 1e-9
    assert abs(gap.eval_acc - 0.5) < 1e-9
    assert abs(gap.gap - 0.3) < 1e-9
    assert gap.verdict == "MEMORIZATION SUSPECTED"

    ev2 = [mk(True)] * 8 + [mk(False)] * 2
    gap2 = report_gap(train, ev2)
    assert gap2.verdict == "OK"


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
