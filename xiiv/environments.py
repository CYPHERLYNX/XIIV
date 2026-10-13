"""
Autonomous Task Environments for XIIV (System 4 of the Adaptive Intelligence
Training System).

Isolated environments where the model plans, acts, observes results, and
recovers from errors. Every interaction is recorded as a trajectory for
later training.

  - SandboxedEnvironment: base class (reset / step / is_done / get_reward).
  - CodeEnvironment:     model writes Python -> runs in a subprocess with
                         timeout, CPU/memory limits, cwd=/tmp -> reward is
                         the test pass rate. stdout/stderr become the next
                         observation.
  - MathEnvironment:     model outputs an answer string -> checked against
                         the expected answer -> reward 1.0 / 0.0.
  - GameEnvironment:     wraps the game harnesses in xiiv/games/
                         (Sudoku today; Chess when a chess harness lands)
                         as a steppable environment.
  - TrajectoryRecorder:  appends every (observation, action, reward, done)
                         tuple to JSONL for later training.

SAFETY
  - Model output is NEVER eval()ed or exec()ed in-process. Code actions run
    only inside a child subprocess with a wall-clock timeout, CPU-time
    limit, and address-space limit, with cwd forced to /tmp.
  - No network access is granted to the child (it inherits a scrubbed
    environment; socket use is not explicitly enabled).
  - Validators and reward functions live in their own modules
    (xiiv/games/*, task definitions passed in by the caller). The model
    receives only observations and reward scalars - it has no handle to
    the validator objects and cannot modify them.
"""

import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("xiiv.environments")

# Directory the model-controlled subprocess is allowed to touch.
SANDBOX_DIR = "/tmp/xiiv_env_sandbox"
# Hard caps for a single code execution.
CODE_TIMEOUT_SECS = 5
CODE_CPU_SECS = 4
CODE_MAX_RSS_BYTES = 512 * 1024 * 1024  # 512 MB


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class SandboxedEnvironment:
    """Base class. Subclasses implement a task loop."""

    def reset(self) -> str:
        """Start a new episode. Returns the initial observation (text)."""
        raise NotImplementedError

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Apply a text action.

        Returns (observation, reward, done, info). Reward is always a float.
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        raise NotImplementedError

    def get_reward(self) -> float:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def _child_limits():
    """Applied inside the child process before exec (preexec_fn)."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (CODE_CPU_SECS, CODE_CPU_SECS))
        resource.setrlimit(resource.RLIMIT_AS,
                           (CODE_MAX_RSS_BYTES, CODE_MAX_RSS_BYTES))
        # No core dumps from model code.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass  # best effort; the wall-clock timeout still applies


def run_untrusted_code(code: str, timeout: int = CODE_TIMEOUT_SECS) -> dict:
    """Run model-produced Python in a locked-down subprocess.

    Never raises on model code errors: they become a failed observation.
    Returns dict(stdout, stderr, returncode, timed_out).
    """
    os.makedirs(SANDBOX_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=SANDBOX_DIR,
                                     delete=False) as f:
        f.write(code)
        path = f.name
    try:
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": SANDBOX_DIR,
        }
        proc = subprocess.run(
            [sys.executable, "-I", "-c",
             "import runpy,sys; runpy.run_path(sys.argv[1], run_name='__main__')",
             path],
            capture_output=True, text=True, timeout=timeout,
            cwd=SANDBOX_DIR, env=env, preexec_fn=_child_limits,
        )
        return {"stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:],
                "returncode": proc.returncode, "timed_out": False}
    except subprocess.TimeoutExpired as e:
        return {"stdout": (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else "",
                "stderr": "TIMEOUT after %ss" % timeout,
                "returncode": -1, "timed_out": True}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@dataclass
class CodeTask:
    """A coding task: hidden tests the model's program must pass.

    tests: list of {"stdin": str, "expected_stdout": str} dicts.
    The model's code is run once per test with stdin piped in.
    """
    description: str
    tests: list[dict]
    task_id: str = ""


class CodeEnvironment(SandboxedEnvironment):
    """Model writes a full Python program; reward = fraction of tests passed."""

    def __init__(self, task: CodeTask, max_steps: int = 5):
        self.task = task
        self.max_steps = max_steps
        self._step = 0
        self._done = False
        self._last_reward = 0.0
        self._last_observation = ""

    def reset(self) -> str:
        self._step = 0
        self._done = False
        self._last_reward = 0.0
        obs = ("Write a complete Python program that reads from stdin and "
               "writes to stdout.\nTask: " + self.task.description +
               "\nOutput ONLY the Python code, no explanations.")
        self._last_observation = obs
        return obs

    def _run_tests(self, code: str) -> tuple[float, str]:
        passed = 0
        details = []
        for i, t in enumerate(self.task.tests):
            os.makedirs(SANDBOX_DIR, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", suffix=".py",
                                             dir=SANDBOX_DIR,
                                             delete=False) as f:
                f.write(code)
                path = f.name
            try:
                env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": "",
                       "PYTHONDONTWRITEBYTECODE": "1", "HOME": SANDBOX_DIR}
                proc = subprocess.run(
                    [sys.executable, "-I", path],
                    input=t.get("stdin", ""), capture_output=True, text=True,
                    timeout=CODE_TIMEOUT_SECS, cwd=SANDBOX_DIR, env=env,
                    preexec_fn=_child_limits)
                out = proc.stdout.strip()
                ok = (proc.returncode == 0
                      and out == t["expected_stdout"].strip())
                if ok:
                    passed += 1
                details.append(f"test {i}: {'PASS' if ok else 'FAIL'} "
                               f"(got {out!r}, want {t['expected_stdout']!r}, "
                               f"rc={proc.returncode})")
            except subprocess.TimeoutExpired:
                details.append(f"test {i}: TIMEOUT")
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        reward = passed / max(1, len(self.task.tests))
        return reward, "\n".join(details)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on finished episode; call reset()")
        self._step += 1
        reward, test_report = self._run_tests(action)
        self._last_reward = reward
        done = reward >= 1.0 or self._step >= self.max_steps
        self._done = done
        obs = (f"Tests: {test_report}\n"
               f"Pass rate: {reward:.2f} "
               f"({'SOLVED' if reward >= 1.0 else 'keep trying' if not done else 'out of steps'}).\n"
               f"Revise the full program and output ONLY code.")
        self._last_observation = obs
        logger.info(json.dumps({"env": "code", "task": self.task.task_id,
                                "step": self._step, "reward": reward,
                                "done": done}))
        return obs, reward, done, {"pass_rate": reward}

    def is_done(self) -> bool:
        return self._done

    def get_reward(self) -> float:
        return self._last_reward


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

@dataclass
class MathTask:
    question: str
    answer: str            # canonical expected answer, as text
    task_id: str = ""
    tolerance: float = 0.0  # >0 allows numeric near-matches


def _normalize_math(s: str) -> str:
    return "".join(s.strip().lower().split())


class MathEnvironment(SandboxedEnvironment):
    """Single-step: model outputs an answer string; exact (or tolerant)
    match against the expected answer. Reward 1.0 / 0.0."""

    def __init__(self, task: MathTask):
        self.task = task
        self._done = False
        self._reward = 0.0

    def reset(self) -> str:
        self._done = False
        self._reward = 0.0
        return ("Answer the following. Output ONLY the final answer, "
                "no explanation.\n" + self.task.question)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on finished episode; call reset()")
        given = _normalize_math(action)
        want = _normalize_math(self.task.answer)
        correct = given == want
        if not correct and self.task.tolerance > 0:
            try:
                correct = abs(float(given) - float(want)) <= self.task.tolerance
            except ValueError:
                correct = False
        self._reward = 1.0 if correct else 0.0
        self._done = True
        obs = "Correct." if correct else f"Incorrect. Expected: {self.task.answer}"
        logger.info(json.dumps({"env": "math", "task": self.task.task_id,
                                "reward": self._reward}))
        return obs, self._reward, True, {"correct": correct}

    def is_done(self) -> bool:
        return self._done

    def get_reward(self) -> float:
        return self._reward


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

class GameEnvironment(SandboxedEnvironment):
    """Wraps a xiiv/games harness as a steppable environment.

    adapter: object exposing
        new_episode() -> prompt_text
        score(action_text) -> (reward: float, done: bool, info: dict)
    This keeps game rules/validators inside xiiv/games (read-only to the
    model); the environment only passes text back and forth.
    """

    def __init__(self, adapter, max_steps: int = 1, game_name: str = ""):
        self.adapter = adapter
        self.max_steps = max_steps
        self.game_name = game_name
        self._step = 0
        self._done = False
        self._reward = 0.0
        self._prompt = ""

    def reset(self) -> str:
        self._step = 0
        self._done = False
        self._reward = 0.0
        self._prompt = self.adapter.new_episode()
        return self._prompt

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on finished episode; call reset()")
        self._step += 1
        reward, done, info = self.adapter.score(action)
        self._reward = float(reward)
        done = bool(done) or self._step >= self.max_steps
        self._done = done
        obs = info.get("observation", "")
        if not done:
            obs += "\nTry again."
        logger.info(json.dumps({"env": "game", "game": self.game_name,
                                "step": self._step, "reward": self._reward,
                                "done": done}))
        return obs, self._reward, done, info

    def is_done(self) -> bool:
        return self._done

    def get_reward(self) -> float:
        return self._reward


class SudokuAdapter:
    """Adapter binding xiiv/games/sudoku.py into GameEnvironment.

    The validator (is_valid_solution) stays in xiiv/games/sudoku.py;
    this adapter only calls it. The model never receives the validator.
    """

    def __init__(self):
        from xiiv.games import sudoku as _sudoku
        self._sudoku = _sudoku
        self._puzzle = None

    def new_episode(self) -> str:
        puzzle, _answer = self._sudoku.generate_extreme_sudoku()
        self._puzzle = puzzle
        grid_text = self._sudoku.grid_to_text(puzzle)
        return ("Solve this EXTREME Sudoku. Output ONLY the completed 9x9 "
                "grid, one row per line, digits separated by spaces.\n\n"
                + grid_text)

    def score(self, action_text: str) -> tuple[float, bool, dict]:
        solution = self._sudoku.parse_solution(action_text)
        valid, msg = self._sudoku.is_valid_solution(self._puzzle, solution)
        reward = 1.0 if valid else 0.0
        return reward, True, {"observation": msg, "valid": valid}


# ---------------------------------------------------------------------------
# Trajectory recording
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    observation: str
    action: str
    reward: float
    done: bool
    info: dict
    step_index: int
    timestamp: float


class TrajectoryRecorder:
    """Appends every (observation, action, reward, done) to JSONL.

    One file per episode; a manifest lists all episodes. Used as training
    data for later supervised / RL updates.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self.log_dir / "manifest.jsonl"
        self._current: list[TrajectoryStep] = []
        self._episode_id: Optional[str] = None
        self._env_name = ""

    def start_episode(self, env_name: str, episode_id: Optional[str] = None):
        self._current = []
        self._env_name = env_name
        self._episode_id = episode_id or f"{env_name}_{int(time.time()*1000)}"

    def record(self, observation: str, action: str, reward: float,
               done: bool, info: dict):
        self._current.append(TrajectoryStep(
            observation=observation, action=action, reward=float(reward),
            done=bool(done), info=info, step_index=len(self._current),
            timestamp=time.time()))

    def end_episode(self) -> Path:
        if self._episode_id is None:
            raise RuntimeError("end_episode() with no start_episode()")
        total_reward = sum(s.reward for s in self._current)
        path = self.log_dir / f"{self._episode_id}.jsonl"
        with open(path, "w") as f:
            for s in self._current:
                f.write(json.dumps(asdict(s)) + "\n")
        with open(self._manifest, "a") as f:
            f.write(json.dumps({
                "episode_id": self._episode_id,
                "env": self._env_name,
                "steps": len(self._current),
                "total_reward": round(total_reward, 4),
                "path": str(path),
            }) + "\n")
        logger.info(json.dumps({"recorder": "end_episode",
                                "episode": self._episode_id,
                                "steps": len(self._current),
                                "total_reward": total_reward}))
        self._episode_id = None
        self._current = []
        return path


def run_episode(env: SandboxedEnvironment, policy: Callable[[str], str],
                recorder: Optional[TrajectoryRecorder] = None,
                env_name: str = "") -> float:
    """Run one full episode with a text-in/text-out policy. Returns total reward."""
    if recorder:
        recorder.start_episode(env_name or type(env).__name__)
    obs = env.reset()
    total = 0.0
    while True:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        total += reward
        if recorder:
            recorder.record(obs, action, reward, done, info)
        if done:
            break
    if recorder:
        recorder.end_episode()
    return total
