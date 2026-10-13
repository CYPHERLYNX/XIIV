"""Tests for System 3 (test_time_search) and System 4 (environments).

Run: .venv/bin/python -m pytest tests/test_systems_3_4.py -v
(or plain python for the smoke path at the bottom).
"""
import json
import sys
import os

sys.path.insert(0, os.path.expanduser("~/workspace/ai-tools/corsac"))

from xiiv.test_time_search import BestOfN, BranchingSearch, AdaptiveCompute
from xiiv.environments import (
    CodeEnvironment, CodeTask, MathEnvironment, MathTask,
    GameEnvironment, SudokuAdapter, TrajectoryRecorder, run_episode,
    run_untrusted_code,
)


# --- dummy generator/verifier -------------------------------------------

def make_dummy_generate(answers):
    """Cycle through canned answers; counts calls."""
    state = {"i": 0}
    def gen(*args):
        a = answers[state["i"] % len(answers)]
        state["i"] += 1
        return a
    gen.calls = lambda: state["i"]
    return gen


def keyword_verifier(keyword):
    def v(text):
        return 1.0 if keyword in text else 0.0
    return v


# --- System 3 tests ------------------------------------------------------

def test_best_of_n_finds_best():
    gen = make_dummy_generate(["wrong", "alsowrong", "RIGHT answer here"])
    res = BestOfN(n=3, verifier=keyword_verifier("RIGHT")).run("prompt", gen)
    assert res.best_score == 1.0, res
    assert "RIGHT" in res.best_text
    assert res.stopped_early  # perfect score -> early stop
    assert res.n_candidates == 3
    print("test_best_of_n_finds_best OK", res.to_dict()["strategy"])


def test_best_of_n_no_early_stop_all_scored():
    gen = make_dummy_generate(["a", "bb", "ccc"])
    # verifier prefers longer text
    res = BestOfN(n=3, verifier=lambda t: len(t) / 10.0,
                  early_stop=True).run("p", gen)
    assert res.best_text == "ccc", res.best_text
    assert not res.stopped_early
    assert res.n_candidates == 3
    print("test_best_of_n_no_early_stop_all_scored OK")


def test_best_of_n_respects_n():
    gen = make_dummy_generate(["x"])
    res = BestOfN(n=5, verifier=lambda t: 0.0).run("p", gen)
    assert res.n_candidates == 5 and gen.calls() == 5
    print("test_best_of_n_respects_n OK")


def test_branching_search_completes():
    # step generator appends a token; verifier scores full text length
    def gen_step(prompt, partial):
        return "tok"
    res = BranchingSearch(steps=3, k=2, beam_width=2,
                          verifier=lambda t: min(1.0, len(t.split()) / 3.0),
                          max_total_candidates=50).run("p", gen_step)
    assert res.n_candidates <= 50, res.n_candidates
    assert res.best_score > 0
    assert len(res.log) > 0
    print("test_branching_search_completes OK, spent", res.n_candidates)


def test_branching_search_budget_cap():
    def gen_step(prompt, partial):
        return "tok"
    res = BranchingSearch(steps=100, k=4, beam_width=4,
                          verifier=lambda t: 0.0,  # never viable
                          min_score=0.5,
                          max_total_candidates=20).run("p", gen_step)
    assert res.n_candidates <= 20, res.n_candidates
    print("test_branching_search_budget_cap OK, spent", res.n_candidates)


def test_adaptive_compute_allocates():
    gen = make_dummy_generate(["a", "bb", "ccc", "dddd"])
    ac = AdaptiveCompute(total_budget=20, probe_candidates=2)
    allocs = ac.allocate(["short", "a much longer problem statement here"],
                         gen, lambda t: 0.5)
    assert sum(allocs) <= 20, allocs
    assert all(a >= 1 for a in allocs)
    # longer prompt should get >= allocation of the short one
    assert allocs[1] >= allocs[0], allocs
    print("test_adaptive_compute_allocates OK", allocs)


def test_adaptive_compute_report():
    gen = make_dummy_generate(["RIGHT", "wrong"])
    ac = AdaptiveCompute(total_budget=12, probe_candidates=1)
    ac.run_best_of_n(["p1", "p2"], gen, keyword_verifier("RIGHT"))
    rep = ac.cost_benefit_report()
    assert rep["runs"] == 2 and rep["total_compute"] > 0
    print("test_adaptive_compute_report OK", rep)


# --- System 4 tests ------------------------------------------------------

def test_run_untrusted_code_hello():
    out = run_untrusted_code("print('hello world')")
    assert out["returncode"] == 0, out
    assert "hello world" in out["stdout"]
    assert not out["timed_out"]
    print("test_run_untrusted_code_hello OK")


def test_run_untrusted_code_timeout():
    out = run_untrusted_code("import time; time.sleep(30)", timeout=2)
    assert out["timed_out"], out
    print("test_run_untrusted_code_timeout OK")


def test_run_untrusted_code_error_captured():
    out = run_untrusted_code("raise ValueError('boom')")
    assert out["returncode"] != 0
    assert "boom" in out["stderr"]
    print("test_run_untrusted_code_error_captured OK")


def test_code_environment_hello_task():
    task = CodeTask(
        description="Print exactly: hello world",
        tests=[{"stdin": "", "expected_stdout": "hello world"}],
        task_id="hello-1",
    )
    env = CodeEnvironment(task, max_steps=3)
    obs = env.reset()
    assert "hello world" in obs.lower() or "Task" in obs
    obs2, reward, done, info = env.step("print('hello world')")
    assert reward == 1.0, (reward, obs2)
    assert done and env.is_done()
    print("test_code_environment_hello_task OK")


def test_code_environment_failure_then_retry():
    task = CodeTask(
        description="Print exactly: 42",
        tests=[{"stdin": "", "expected_stdout": "42"}],
        task_id="num-1",
    )
    env = CodeEnvironment(task, max_steps=2)
    env.reset()
    obs, reward, done, info = env.step("print('wrong')")
    assert reward == 0.0 and not done
    obs, reward, done, info = env.step("print(42)")
    assert reward == 1.0 and done
    print("test_code_environment_failure_then_retry OK")


def test_math_environment():
    env = MathEnvironment(MathTask(question="2+2?", answer="4", task_id="m1"))
    env.reset()
    obs, reward, done, info = env.step("4")
    assert reward == 1.0 and done and info["correct"]
    env2 = MathEnvironment(MathTask(question="2+2?", answer="4"))
    env2.reset()
    _, reward2, _, _ = env2.step("5")
    assert reward2 == 0.0
    print("test_math_environment OK")


def test_game_environment_sudoku_adapter():
    adapter = SudokuAdapter()
    env = GameEnvironment(adapter, max_steps=1, game_name="sudoku")
    prompt = env.reset()
    assert "Sudoku" in prompt
    # garbage action -> invalid solution -> reward 0, done
    obs, reward, done, info = env.step("not a grid")
    assert reward == 0.0 and done
    assert "valid" in info
    print("test_game_environment_sudoku_adapter OK")


def test_trajectory_recorder(tmp_path="/tmp/xiiv_test_traj"):
    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)
    rec = TrajectoryRecorder(tmp_path)
    task = CodeTask(description="Print: hi",
                    tests=[{"stdin": "", "expected_stdout": "hi"}],
                    task_id="t1")
    env = CodeEnvironment(task)
    total = run_episode(env, lambda obs: "print('hi')",
                        recorder=rec, env_name="code")
    assert total == 1.0
    manifest = [json.loads(l) for l in
                open(os.path.join(tmp_path, "manifest.jsonl"))]
    assert len(manifest) == 1 and manifest[0]["total_reward"] == 1.0
    print("test_trajectory_recorder OK")


if __name__ == "__main__":
    logging_off = __import__("logging")
    logging_off.disable(logging_off.CRITICAL)
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
