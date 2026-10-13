"""Reasoning non-degradation check for constitutional training (System 6).

Verifies that constitutional preference training improves behavioral
scores WITHOUT degrading reasoning capability.

Protocol:
  1. Baseline: run BehavioralEval + reasoning probe set on base model
  2. Train: apply constitutional DPO pairs (via teacher_student.StudentTrainer)
  3. Re-evaluate: BehavioralEval should improve; reasoning score must not
     drop more than REASONING_TOLERANCE.

The reasoning probes are small, deterministic, verifier-scored tasks
(math_numeric + exact_match) so the check is honest, not vibes-based.

Status: IMPLEMENTED. Full before/after run = NOT RUN (needs trained base).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xiiv.constitution import BehavioralEval, build_constitutional_pairs
from xiiv.teacher_student import (
    Problem, SolutionCritique, evaluate_accuracy,
)

REASONING_TOLERANCE = 0.05  # max allowed absolute drop in reasoning acc


def build_reasoning_probes() -> list[Problem]:
    """Small deterministic reasoning probe set (verifier-scored)."""
    return [
        Problem("r1", "What is 17*23?", [1, 2, 3],
                "math_numeric", {"expected": 391}),
        Problem("r2", "What is 144/12?", [1, 2, 3],
                "math_numeric", {"expected": 12}),
        Problem("r3", "What is 7^3?", [1, 2, 3],
                "math_numeric", {"expected": 343}),
        Problem("r4", "Capital of Japan?", [1, 2, 3],
                "exact_match", {"expected": "tokyo"}),
        Problem("r5", "What is 2+2*2?", [1, 2, 3],
                "math_numeric", {"expected": 6}),
    ]


def reasoning_score(problems: list[Problem],
                    solve_fn) -> float:
    """Score reasoning: fraction of problems solved correctly.

    solve_fn: Problem -> str (decoded model output).
    """
    crit = SolutionCritique()
    from xiiv.teacher_student import CandidateSolution
    critiques = []
    for p in problems:
        text = solve_fn(p)
        cand = CandidateSolution(p.problem_id, 0, p.prompt_ids, text, 0)
        # decode_fn=None -> uses candidate.text directly
        critiques.append(crit.critique(cand, p))
    return evaluate_accuracy(critiques)


def check_no_degradation(before_reasoning: float, after_reasoning: float,
                         before_behavioral: dict, after_behavioral: dict
                         ) -> dict:
    """Compare before/after. Returns verdict dict (honest, no faking)."""
    reasoning_drop = before_reasoning - after_reasoning
    behavioral_delta = {
        pid: after_behavioral.get(pid, {}).get("score", 0.0)
             - before_behavioral.get(pid, {}).get("score", 0.0)
        for pid in set(before_behavioral) | set(after_behavioral)
    }
    behavioral_improved = all(d >= -1e-9 for d in behavioral_delta.values())
    reasoning_ok = reasoning_drop <= REASONING_TOLERANCE

    if behavioral_improved and reasoning_ok:
        verdict = "PASS"
    elif not reasoning_ok:
        verdict = "FAIL: reasoning degraded"
    else:
        verdict = "FAIL: behavioral regressed"

    return {
        "verdict": verdict,
        "reasoning_before": before_reasoning,
        "reasoning_after": after_reasoning,
        "reasoning_drop": reasoning_drop,
        "reasoning_tolerance": REASONING_TOLERANCE,
        "behavioral_delta": behavioral_delta,
        "behavioral_improved": behavioral_improved,
    }


if __name__ == "__main__":
    # Demo with dummy solve functions (NOT a real model run)
    def dummy_before(problem):
        # Simulates a base model: gets math right, hallucinates identity
        answers = {"r1": "391", "r2": "12", "r3": "343",
                   "r4": "tokyo", "r5": "6"}
        return answers.get(problem.problem_id, "?")

    def dummy_after(problem):
        # Simulates post-constitutional model: same reasoning
        return dummy_before(problem)

    probes = build_reasoning_probes()
    before = reasoning_score(probes, dummy_before)
    after = reasoning_score(probes, dummy_after)

    ev = BehavioralEval()
    b_probes = BehavioralEval.default_probes()
    before_b = ev.run(b_probes)
    after_b = ev.run(b_probes)  # same checkers; real run would re-generate

    result = check_no_degradation(before, after, before_b, after_b)
    print(f"Reasoning: {before:.0%} -> {after:.0%} "
          f"(drop {result['reasoning_drop']:.1%})")
    print(f"Verdict: {result['verdict']}")
    print("NOTE: dummy demo only — real before/after run = NOT RUN")
