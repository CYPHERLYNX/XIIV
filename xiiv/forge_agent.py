"""XIIV Forge Agent — the Jarvis/Ultron upgrade.

The base Forge (xiiv/forge.py) is task-level: code in, tests out, fix on
failure. This module adds the AGENTIC layer Fahad asked for (2026-10-11):
the system thinks, learns from itself, and corrects itself — like the
movie AI, but real.

The agent loop:
  1. THINK   — explicit reasoning trace: understand goal, plan steps,
               anticipate failures (not hidden; logged and auditable)
  2. DECOMPOSE — break the goal into verifiable sub-tasks
  3. ACT     — solve each sub-task via the Forge loop (generate → sandbox
               → verify → rewrite until 100%)
  4. REFLECT — after each sub-task: what worked, what didn't, why
  5. CORRECT — if the plan is wrong, revise the plan (not just the code)
  6. REMEMBER — store the full episode (thinking + actions + outcomes)
               in persistent memory for future goals

What this is NOT (honest):
  - Not sentient, not conscious — it's a structured reasoning loop.
  - Thinking quality is bounded by the base model. A weak model thinks
    weak thoughts; the loop amplifies but doesn't create intelligence.
  - Movie Jarvis/Ultron are AGI fiction. This is the real, working
    version: verifiable self-improvement with explicit reasoning.

What it IS:
  - An agent that reasons explicitly, acts, verifies independently,
    corrects its own plans, and accumulates experience across goals.
  - Every thought, action, and correction is logged — auditable, unlike
    a black-box agent.
"""
import json
import time

from .forge import ForgeTask, SkillLibrary, forge_loop, ForgeResult


class ThinkTrace:
    """An explicit, auditable reasoning trace for one goal."""

    def __init__(self, goal):
        self.goal = goal
        self.steps = []  # [{"t": ..., "kind": ..., "content": ...}]
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S")

    def add(self, kind, content):
        """kind: plan | observation | reflection | correction | decision"""
        self.steps.append({
            "t": time.strftime("%H:%M:%S"),
            "kind": kind,
            "content": content,
        })

    def to_dict(self):
        return {"goal": self.goal, "started": self.started,
                "steps": self.steps}


class ForgeAgent:
    """A thinking, self-correcting agent built on the Forge loop.

    think_fn: the model's reasoning function.
        think_fn(prompt) -> text. Used for planning, reflection, correction.
        In the full system this is XIIV itself; in tests it's a stub.
    act_fn: maps a sub-task to a ForgeTask + generate_fn.
        act_fn(subtask_description) -> (ForgeTask, generate_fn).
    """

    def __init__(self, think_fn, act_fn, memory_path="forge_agent_memory.db",
                 max_plan_revisions=3, verbose=True):
        self.think = think_fn
        self.act_fn = act_fn
        self.memory = SkillLibrary(memory_path)
        self.max_plan_revisions = max_plan_revisions
        self.verbose = verbose

    def _say(self, msg):
        if self.verbose:
            print(f"  [agent] {msg}")

    def run(self, goal):
        """Pursue a goal: think → decompose → act → reflect → correct → remember."""
        trace = ThinkTrace(goal)
        self._say(f"goal: {goal}")

        # 1. THINK — understand and plan
        plan_prompt = (
            f"Goal: {goal}\n\n"
            f"Think step by step. Break this goal into concrete, verifiable "
            f"sub-tasks. For each sub-task, state exactly how you would verify "
            f"it succeeded. Output one sub-task per line, numbered.")
        plan_text = self.think(plan_prompt)
        trace.add("plan", plan_text)
        self._say(f"plan:\n{plan_text}")

        # recall relevant past episodes
        past = self.memory.recall(goal, limit=3)
        if past:
            trace.add("observation",
                      f"recalled {len(past)} relevant past episodes")
            self._say(f"recalled {len(past)} past episodes")

        # 2-5. DECOMPOSE → ACT → REFLECT → CORRECT (with plan revision)
        subtasks = self._parse_plan(plan_text)
        results = []
        for rev in range(self.max_plan_revisions + 1):
            results = []
            plan_failed = False
            for i, st in enumerate(subtasks):
                self._say(f"sub-task {i+1}/{len(subtasks)}: {st}")
                task, generate_fn = self.act_fn(st)
                r = forge_loop(task, generate_fn, self.memory,
                               verbose=self.verbose)
                results.append((st, r))
                if r.ok:
                    trace.add("observation",
                              f"sub-task {i+1} verified in "
                              f"{r.iterations} iterations")
                    # REFLECT
                    refl = self.think(
                        f"Sub-task '{st}' succeeded in {r.iterations} "
                        f"iterations. Failure history: "
                        f"{json.dumps(r.audit_log)[:500]}. "
                        f"In one sentence, what lesson should be remembered?")
                    trace.add("reflection", refl)
                else:
                    trace.add("observation",
                              f"sub-task {i+1} FAILED after "
                              f"{r.iterations} iterations")
                    plan_failed = True
                    break
            if not plan_failed:
                break
            # CORRECT — revise the plan, not just the code
            self._say("plan failed — revising")
            failed_st, failed_r = results[-1]
            correction = self.think(
                f"Goal: {goal}\nPlan so far:\n{plan_text}\n\n"
                f"Sub-task '{failed_st}' failed. Audit: "
                f"{json.dumps(failed_r.audit_log)[:800]}\n\n"
                f"The approach is wrong, not just the code. Propose a "
                f"revised plan: different sub-tasks or a different "
                f"verification strategy. Output numbered sub-tasks.")
            trace.add("correction", correction)
            plan_text = correction
            subtasks = self._parse_plan(correction)
            self._say(f"revised plan:\n{plan_text}")

        # 6. REMEMBER — store the full episode
        all_ok = all(r.ok for _, r in results)
        self.memory.store(
            task_id=f"episode:{goal[:60]}",
            contract_hash="episode",
            prompt=goal,
            solution=json.dumps(trace.to_dict())[:4000],
            iterations=sum(r.iterations for _, r in results),
            failure_history=[{"subtask": st,
                              "ok": r.ok,
                              "iterations": r.iterations}
                             for st, r in results])
        trace.add("decision",
                  f"goal {'ACHIEVED' if all_ok else 'NOT achieved'} "
                  f"after {len(results)} sub-tasks")
        self._say(f"goal {'ACHIEVED' if all_ok else 'FAILED'}")
        return all_ok, trace

    @staticmethod
    def _parse_plan(plan_text):
        """Extract numbered sub-tasks from plan text."""
        lines = []
        for ln in plan_text.strip().split("\n"):
            ln = ln.strip().lstrip("0123456789.)- ").strip()
            if ln and len(ln) > 10:
                lines.append(ln)
        return lines or [plan_text.strip()]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from .forge import ForgeTask, python_unittest_verifier

    # stub thinker: returns canned plans (the real one is XIIV itself)
    plans = {"plan": "1. Write an add function\n2. Write a multiply function",
             "reflect": "Remember to check operator correctness."}

    def think_fn(prompt):
        if "revised plan" in prompt or "different sub-tasks" in prompt:
            return "1. Write an add function\n2. Write a multiply function"
        if "what lesson" in prompt:
            return plans["reflect"]
        return plans["plan"]

    def act_fn(subtask):
        if "add" in subtask.lower():
            code = {"n": 0}

            def gen(prompt, ctx):
                code["n"] += 1
                # first attempt wrong, then corrects itself
                if code["n"] == 1 and "multiply" not in subtask.lower():
                    return "def add(a,b):\n    return a - b"
                return "def add(a,b):\n    return a + b"
            task = ForgeTask(
                "add", subtask,
                {"tests": [{"name": "t", "call": "print(add(2,3))",
                            "expect": "5"}], "must_pass": 1},
                python_unittest_verifier, max_iterations=4)
            return task, gen
        else:
            def gen2(prompt, ctx):
                return "def mul(a,b):\n    return a * b"
            task = ForgeTask(
                "mul", subtask,
                {"tests": [{"name": "t", "call": "print(mul(2,3))",
                            "expect": "6"}], "must_pass": 1},
                python_unittest_verifier, max_iterations=4)
            return task, gen2

    db = "/tmp/forge_agent_test.db"
    if os.path.exists(db):
        os.remove(db)
    agent = ForgeAgent(think_fn, act_fn, memory_path=db, verbose=True)
    ok, trace = agent.run("Create add and multiply functions")
    print("\nAGENT RESULT:", "ACHIEVED" if ok else "FAILED")
    print("trace steps:", len(trace.steps),
          "| kinds:", [s["kind"] for s in trace.steps])
    os.remove(db)
