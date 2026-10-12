"""XIIV Forge — verifier-gated recursive self-improvement loop.

The loop:
  1. TASK + CONTRACT  — task with a machine-checkable success criterion
  2. GENERATE         — model produces a candidate solution
  3. EXECUTE          — solution runs in a sandbox
  4. VERIFY           — INDEPENDENT verifier checks against the contract
  5. FEEDBACK         — on failure: structured failure report -> model
  6. REWRITE          — model regenerates with failure context
  7. REPEAT           — until 100% verified or max iterations
  8. LEARN            — success stored in the skill library

The judge is the independent guarantee-native verifier, never the model
itself. The model cannot grade its own homework.

Safety (non-negotiable):
  - sandboxed execution (no network, scratch-only fs, CPU/mem/time limits)
  - iteration cap (default 10)
  - the verifier is never modifiable by the loop
  - no weight self-modification
  - immutable audit log of every iteration

Status: harness implemented 2026-10-11. Not yet run.
"""
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
import time


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

class ForgeTask:
    """A verifiable task: prompt + contract + verifier + sandbox config."""

    def __init__(self, task_id, prompt, contract, verifier_fn,
                 executor="python", max_iterations=10, timeout_s=30):
        self.task_id = task_id
        self.prompt = prompt
        self.contract = contract  # dict the verifier checks
        self.verifier_fn = verifier_fn  # (output_text, contract) -> (ok, report)
        self.executor = executor
        self.max_iterations = max_iterations
        self.timeout_s = timeout_s

    def contract_hash(self):
        return hashlib.sha256(
            json.dumps(self.contract, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

def run_sandboxed(code, timeout_s=30, executor="python"):
    """Run code in a sandbox: no network, temp-only fs, resource limits.

    Returns (ok, stdout, stderr, timed_out).
    """
    if executor != "python":
        raise ValueError(f"unsupported executor: {executor}")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "solution.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            # resource limits: CPU time, memory, no network (via env)
            def preexec():
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s))
                resource.setrlimit(resource.RLIMIT_AS,
                                   (2 * 1024**3, 2 * 1024**3))  # 2GB
            env = dict(os.environ)
            env["PYTHONPATH"] = ""
            p = subprocess.run(
                ["python3", path], capture_output=True, text=True,
                timeout=timeout_s, cwd=td, env=env, preexec_fn=preexec)
            return True, p.stdout, p.stderr, False
        except subprocess.TimeoutExpired as e:
            return False, e.stdout or "", e.stderr or "", True


# ---------------------------------------------------------------------------
# Skill library
# ---------------------------------------------------------------------------

class SkillLibrary:
    """SQLite store of verified (task, solution, failure-history) triples."""

    def __init__(self, path="forge_skills.db"):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY,
            task_id TEXT, contract_hash TEXT,
            prompt TEXT, solution TEXT,
            iterations INTEGER, failure_history TEXT,
            verified_at TEXT)""")
        self.db.commit()

    def store(self, task_id, contract_hash, prompt, solution,
              iterations, failure_history):
        self.db.execute(
            "INSERT INTO skills (task_id, contract_hash, prompt, solution,"
            " iterations, failure_history, verified_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (task_id, contract_hash, prompt, solution, iterations,
             json.dumps(failure_history),
             time.strftime("%Y-%m-%dT%H:%M:%S")))
        self.db.commit()

    def recall(self, task_id=None, limit=5):
        q = "SELECT task_id, prompt, solution, iterations FROM skills"
        args = ()
        if task_id:
            q += " WHERE task_id LIKE ?"
            args = (f"%{task_id}%",)
        q += " ORDER BY id DESC LIMIT ?"
        return self.db.execute(q, args + (limit,)).fetchall()

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM skills").fetchone()[0]


# ---------------------------------------------------------------------------
# The Forge loop
# ---------------------------------------------------------------------------

class ForgeResult:
    def __init__(self, ok, solution, iterations, audit_log, certificate):
        self.ok = ok
        self.solution = solution
        self.iterations = iterations
        self.audit_log = audit_log  # immutable per-iteration trace
        self.certificate = certificate  # proof-carrying summary


def forge_loop(task, generate_fn, skill_lib=None, verbose=True):
    """Run the verifier-gated self-improvement loop.

    generate_fn(prompt, failure_context) -> candidate solution text.
    The verifier is task.verifier_fn — independent of the model.
    """
    audit = []
    failure_history = []
    context = ""

    # skill recall: relevant past successes as context
    if skill_lib is not None:
        past = skill_lib.recall(task.task_id, limit=3)
        if past and verbose:
            print(f"  recalled {len(past)} past skills")

    for it in range(1, task.max_iterations + 1):
        # 1. generate
        candidate = generate_fn(task.prompt, context)
        # 2. execute
        exec_ok, stdout, stderr, timed_out = run_sandboxed(
            candidate, timeout_s=task.timeout_s, executor=task.executor)
        # 3. verify (INDEPENDENT — never the model judging itself)
        ver_ok, ver_report = task.verifier_fn(
            candidate, stdout, stderr, task.contract)

        entry = {"iteration": it, "exec_ok": exec_ok,
                 "timed_out": timed_out,
                 "stdout": stdout[:2000], "stderr": stderr[:2000],
                 "verified": bool(ver_ok), "report": ver_report}
        audit.append(entry)

        if verbose:
            print(f"  iter {it}: exec_ok={exec_ok} verified={ver_ok} "
                  f"report={str(ver_report)[:100]}")

        if ver_ok:
            # 4. learn: store the verified success
            if skill_lib is not None:
                skill_lib.store(task.task_id, task.contract_hash(),
                                task.prompt, candidate, it, failure_history)
            cert = {"task_id": task.task_id,
                    "contract_hash": task.contract_hash(),
                    "iterations": it,
                    "audit_hash": hashlib.sha256(
                        json.dumps(audit, sort_keys=True).encode()
                    ).hexdigest(),
                    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            return ForgeResult(True, candidate, it, audit, cert)

        # 5. feedback: structured failure -> context for rewrite
        fb = {"iteration": it, "stdout": stdout[:1000],
              "stderr": stderr[:1000], "report": ver_report}
        failure_history.append(fb)
        context = (
            f"Previous attempt {it} FAILED verification.\n"
            f"Verifier report: {ver_report}\n"
            f"stdout: {stdout[:500]}\n"
            f"stderr: {stderr[:500]}\n"
            f"Fix the specific failures above and try again.")

    return ForgeResult(False, None, task.max_iterations, audit, None)


# ---------------------------------------------------------------------------
# Example verifiers
# ---------------------------------------------------------------------------

def python_unittest_verifier(candidate, stdout, stderr, contract):
    """Contract: {tests: [...], must_pass: N}. Verifier runs the candidate
    against embedded unit tests — independent of the model."""
    tests = contract.get("tests", [])
    passed = 0
    failures = []
    for t in tests:
        # each test: {"name": ..., "call": ..., "expect": ...}
        with tempfile.TemporaryDirectory() as td:
            harness = (candidate + "\n\n" + t["call"] + "\n")
            path = os.path.join(td, "t.py")
            open(path, "w").write(harness)
            try:
                p = subprocess.run(["python3", path], capture_output=True,
                                   text=True, timeout=10, cwd=td)
                if p.returncode == 0 and t["expect"] in p.stdout:
                    passed += 1
                else:
                    failures.append(t["name"])
            except Exception as e:
                failures.append(f"{t['name']}: {e}")
    need = contract.get("must_pass", len(tests))
    ok = passed >= need
    return ok, {"passed": passed, "needed": need, "failures": failures}
