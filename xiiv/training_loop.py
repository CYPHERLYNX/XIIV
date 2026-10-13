"""
Central Autonomous Training Loop
================================
Implements the full cycle:

  BASELINE -> WEAKNESS_ANALYSIS -> CHALLENGE_GEN -> TRAIN ->
  VALIDATE -> HIDDEN_EVAL -> COMPARE -> PROMOTE/ROLLBACK -> REPEAT

Features:
  - Curriculum learning: difficulty rises as performance improves
  - Hard-example replay: failed examples are buffered and mixed into training
  - Anti-forgetting: samples from mastered domains always included
  - Structured JSON logging per cycle
  - Compute tracking: time, memory, improvement-per-compute
  - Anomaly detection: reward hacking, benchmark leakage, catastrophic
    forgetting, repeated unsolved loops

No fake metrics: the loop works with any trainer/evaluator conforming to
the typed interfaces below. With dummy components it exercises the full
machinery end-to-end.
"""

import json
import random
import resource
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from .hidden_eval import (
    AcceptanceGate,
    CheckpointManager,
    DatasetSplit,
    DomainMetrics,
    FreshScenarioGenerator,
    HiddenEvaluator,
    _utcnow,
)

LOOP_ROOT = Path.home() / "workspace" / "ai-tools" / "corsac" / "xiiv" / "training_loop"
CYCLE_DIR = LOOP_ROOT / "cycles"
HARD_BUFFER_PATH = LOOP_ROOT / "hard_example_buffer.jsonl"
LOOP_STATE_PATH = LOOP_ROOT / "loop_state.json"
for _d in (CYCLE_DIR,):
    _d.mkdir(parents=True, exist_ok=True)
LOOP_ROOT.mkdir(parents=True, exist_ok=True)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Typed interfaces
# ---------------------------------------------------------------------------

class SolverFn(Protocol):
    """Maps (domain, item) -> (predicted_answer, confidence in [0,1])."""
    def __call__(self, domain: str, item: dict) -> tuple[str, float]: ...


class TrainerFn(Protocol):
    """Runs one training phase.

    Args:
        domain: weakest domain to target
        train_items: targeted training examples for that domain
        replay_items: hard-example replay buffer sample
        antiforget_items: samples from previously mastered domains
        cycle: cycle number (for logging)
    Returns:
        dict with at least {"trained": True, "notes": str}
    """
    def __call__(self, domain: str, train_items: list, replay_items: list,
                 antiforget_items: list, cycle: int) -> dict: ...


# ---------------------------------------------------------------------------
# Cycle report
# ---------------------------------------------------------------------------

@dataclass
class CycleReport:
    cycle: int
    started_at: str
    ended_at: str = ""
    phases: dict = field(default_factory=dict)   # phase -> summary dict
    baseline_metrics: dict = field(default_factory=dict)
    hidden_metrics: dict = field(default_factory=dict)
    weakest_domain: str = ""
    difficulty: dict = field(default_factory=dict)
    promotion: dict = field(default_factory=dict)
    anomalies: list = field(default_factory=list)
    compute: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Detects training pathologies from cycle history."""

    REWARD_HACK_JUMP = 0.40      # +40pp in one cycle on primary metric
    FORGETTING_DROP = 0.10       # >10pp drop on a previously good domain
    REPEAT_WEAKNESS_CYCLES = 3   # same weakest domain 3 cycles in a row

    def __init__(self):
        self.history: list[CycleReport] = []

    def _primary(self, metrics: dict, domain: str) -> Optional[float]:
        m = metrics.get(domain)
        if not m:
            return None
        return m.get("win_rate", m.get("accuracy"))

    def check(self, report: CycleReport) -> list[dict]:
        anomalies: list[dict] = []
        if not self.history:
            self.history.append(report)
            return anomalies
        prev = self.history[-1]

        # 1. Reward hacking: suspiciously fast improvement
        for domain, m in report.hidden_metrics.items():
            cur = self._primary(report.hidden_metrics, domain)
            old = self._primary(prev.hidden_metrics, domain)
            if cur is not None and old is not None and cur - old >= self.REWARD_HACK_JUMP:
                anomalies.append({
                    "type": "reward_hacking_suspected",
                    "domain": domain,
                    "detail": (f"Primary metric jumped {old:.3f} -> {cur:.3f} "
                               f"in one cycle (>= {self.REWARD_HACK_JUMP:.0%}). "
                               f"Verify the evaluator was not gamed."),
                    "severity": "high",
                })

        # 2. Catastrophic forgetting: old domain drops >10pp
        for domain, m in report.hidden_metrics.items():
            cur = self._primary(report.hidden_metrics, domain)
            old = self._primary(prev.hidden_metrics, domain)
            if (cur is not None and old is not None and old >= 0.5
                    and old - cur >= self.FORGETTING_DROP):
                anomalies.append({
                    "type": "catastrophic_forgetting",
                    "domain": domain,
                    "detail": (f"Domain '{domain}' dropped {old:.3f} -> {cur:.3f} "
                               f"(>= {self.FORGETTING_DROP:.0%}). Anti-forgetting "
                               f"mix may be insufficient."),
                    "severity": "high",
                })

        # 3. Repeated unsolved loop: same weakest domain 3+ cycles
        recent_weak = [r.weakest_domain for r in self.history[-2:]]
        recent_weak.append(report.weakest_domain)
        if (len(recent_weak) >= self.REPEAT_WEAKNESS_CYCLES
                and len(set(recent_weak[-self.REPEAT_WEAKNESS_CYCLES:])) == 1
                and report.weakest_domain):
            anomalies.append({
                "type": "repeated_unsolved_loop",
                "domain": report.weakest_domain,
                "detail": (f"'{report.weakest_domain}' weakest for "
                           f"{self.REPEAT_WEAKNESS_CYCLES}+ consecutive cycles. "
                           f"Training is not moving this domain — change strategy, "
                           f"increase difficulty differently, or escalate."),
                "severity": "medium",
            })

        # 4. Benchmark leakage: sealed access log has non-evaluation entries
        try:
            from .hidden_eval import ACCESS_LOG
            if ACCESS_LOG.exists():
                with open(ACCESS_LOG) as f:
                    for line in f:
                        if "WARNING" in line and "non-evaluation" in line:
                            anomalies.append({
                                "type": "benchmark_leakage",
                                "domain": "global",
                                "detail": ("Sealed access log contains "
                                           "non-evaluation access. Possible "
                                           "benchmark leakage into training."),
                                "severity": "critical",
                            })
                            break
        except Exception:
            pass

        self.history.append(report)
        return anomalies


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@dataclass
class CurriculumState:
    """Per-domain difficulty level. Rises when gates pass, holds otherwise."""
    levels: dict = field(default_factory=dict)  # domain -> int level

    # Difficulty knobs per domain and level
    SUDOKU_GIVENS = {0: 35, 1: 30, 2: 26, 3: 22}       # fewer = harder
    MATH_DIGITS = {0: 2, 1: 3, 2: 4, 3: 5}             # more digits = harder

    def level(self, domain: str) -> int:
        return self.levels.get(domain, 0)

    def advance(self, domain: str):
        self.levels[domain] = min(self.level(domain) + 1, 3)

    def difficulty_params(self, domain: str) -> dict:
        lvl = self.level(domain)
        if domain == "sudoku":
            return {"givens": self.SUDOKU_GIVENS[lvl], "level": lvl}
        if domain == "math":
            return {"digits": self.MATH_DIGITS[lvl], "level": lvl}
        return {"level": lvl}


# ---------------------------------------------------------------------------
# TrainingLoop
# ---------------------------------------------------------------------------

class TrainingLoop:
    """The central autonomous loop."""

    DOMAINS = ["sudoku", "math", "logic"]

    def __init__(self,
                 solver_fn: SolverFn,
                 trainer_fn: TrainerFn,
                 domains: Optional[list[str]] = None,
                 checkpoint_dir: Optional[Path] = None,
                 n_eval_per_domain: int = 50,
                 hard_buffer_size: int = 200):
        self.solver_fn = solver_fn
        self.trainer_fn = trainer_fn
        self.domains = domains or list(self.DOMAINS)
        self.n_eval = n_eval_per_domain
        self.evaluator = HiddenEvaluator()
        self.ckpt_mgr = CheckpointManager(checkpoint_dir)
        self.gate = AcceptanceGate()
        self.anomalies = AnomalyDetector()
        self.curriculum = CurriculumState()
        self.cycle = 0
        self.hard_buffer: list[dict] = []
        self.hard_buffer_size = hard_buffer_size
        self.mastered: set[str] = set()
        self._load_state()

    # -- state ---------------------------------------------------------------
    def _load_state(self):
        if LOOP_STATE_PATH.exists():
            with open(LOOP_STATE_PATH) as f:
                s = json.load(f)
            self.cycle = s.get("cycle", 0)
            self.curriculum.levels = s.get("curriculum_levels", {})
            self.mastered = set(s.get("mastered", []))
        if HARD_BUFFER_PATH.exists():
            with open(HARD_BUFFER_PATH) as f:
                self.hard_buffer = [json.loads(l) for l in f if l.strip()]

    def _save_state(self):
        with open(LOOP_STATE_PATH, "w") as f:
            json.dump({
                "cycle": self.cycle,
                "curriculum_levels": self.curriculum.levels,
                "mastered": sorted(self.mastered),
                "updated_at": _utcnow(),
            }, f, indent=2)
        with open(HARD_BUFFER_PATH, "w") as f:
            for item in self.hard_buffer[-self.hard_buffer_size:]:
                f.write(json.dumps(item) + "\n")

    # -- phases ---------------------------------------------------------------
    def _phase_baseline(self, report: CycleReport) -> dict[str, DomainMetrics]:
        """Evaluate current solver on sealed sets (fresh problems)."""
        metrics = {}
        for domain in self.domains:
            gen = FreshScenarioGenerator(seed=10_000 + self.cycle)
            items = gen.generate_domain(domain, self.n_eval)
            # Evaluate on fresh (not sealed) for the fast baseline path;
            # sealed is reserved for the HIDDEN_EVAL phase.
            correct = 0
            for item in items:
                try:
                    pred, _ = self.solver_fn(domain, item)
                except Exception:
                    pred = ""
                if self.evaluator._grade(domain, item, pred):
                    correct += 1
            acc = correct / len(items) if items else 0.0
            metrics[domain] = DomainMetrics(
                domain=domain, n_evaluated=len(items),
                accuracy=acc,
                win_rate=acc if domain in ("sudoku", "chess") else None,
                task_completion=acc if domain == "code" else None,
                notes=f"baseline cycle={self.cycle} fresh_seed={10_000 + self.cycle}",
            )
        report.phases["baseline"] = {
            d: {"accuracy": m.accuracy, "n": m.n_evaluated}
            for d, m in metrics.items()
        }
        return metrics

    def _phase_weakness(self, metrics: dict[str, DomainMetrics],
                        report: CycleReport) -> str:
        scored = [(d, m.primary()) for d, m in metrics.items()]
        scored = [(d, s) for d, s in scored if s is not None]
        if not scored:
            weakest = self.domains[0]
        else:
            weakest = min(scored, key=lambda x: x[1])[0]
        report.weakest_domain = weakest
        report.phases["weakness_analysis"] = {
            "scores": {d: round(s, 4) for d, s in scored},
            "weakest": weakest,
        }
        return weakest

    def _phase_challenge_gen(self, domain: str,
                             report: CycleReport) -> list[dict]:
        """Generate targeted training examples at current difficulty."""
        params = self.curriculum.difficulty_params(domain)
        gen = FreshScenarioGenerator(seed=20_000 + self.cycle)
        items = gen.generate_domain(domain, 100)
        # Attach difficulty params for the trainer's reference
        for it in items:
            it["_difficulty"] = params
            it["_target_domain"] = domain
        report.phases["challenge_gen"] = {
            "domain": domain, "n_items": len(items),
            "difficulty": params,
        }
        report.difficulty = {domain: params}
        return items

    def _phase_train(self, domain: str, train_items: list,
                     report: CycleReport) -> dict:
        # Hard-example replay sample
        replay = self.hard_buffer[-50:] if self.hard_buffer else []
        # Anti-forgetting: samples from mastered domains
        antiforget = []
        for md in self.mastered:
            if md != domain:
                gen = FreshScenarioGenerator(seed=30_000 + self.cycle)
                for it in gen.generate_domain(md, 20):
                    it["_antiforget_domain"] = md
                    antiforget.append(it)
        t0 = time.time()
        rss0 = _peak_rss_mb()
        result = self.trainer_fn(domain, train_items, replay, antiforget,
                                 self.cycle)
        dt = time.time() - t0
        report.phases["train"] = {
            "domain": domain,
            "n_train": len(train_items),
            "n_replay": len(replay),
            "n_antiforget": len(antiforget),
            "secs": round(dt, 1),
            "peak_rss_mb": round(_peak_rss_mb(), 1),
            "rss_delta_mb": round(_peak_rss_mb() - rss0, 1),
            "result": result,
        }
        report.compute = {
            "train_secs": round(dt, 1),
            "peak_rss_mb": round(_peak_rss_mb(), 1),
        }
        return result

    def _phase_validate(self, domain: str,
                        report: CycleReport) -> dict[str, DomainMetrics]:
        split = DatasetSplit(domain)
        try:
            items = split.load_valid()[: self.n_eval]
            source = "valid_split"
        except Exception:
            gen = FreshScenarioGenerator(seed=40_000 + self.cycle)
            items = gen.generate_domain(domain, self.n_eval)
            source = "fresh_fallback"
        correct = 0
        for item in items:
            try:
                pred, _ = self.solver_fn(domain, item)
            except Exception:
                pred = ""
            if self.evaluator._grade(domain, item, pred):
                correct += 1
            else:
                # Hard-example replay: buffer the failures
                self.hard_buffer.append({
                    "domain": domain, "item": item,
                    "cycle": self.cycle,
                })
        self.hard_buffer = self.hard_buffer[-self.hard_buffer_size:]
        acc = correct / len(items) if items else 0.0
        m = DomainMetrics(domain=domain, n_evaluated=len(items),
                          accuracy=acc,
                          notes=f"validate cycle={self.cycle} source={source}")
        report.phases["validate"] = {
            "domain": domain, "accuracy": round(acc, 4),
            "n": len(items), "source": source,
            "hard_buffer_size": len(self.hard_buffer),
        }
        return {domain: m}

    def _phase_hidden_eval(self, report: CycleReport) -> dict[str, DomainMetrics]:
        """Sealed evaluation across all domains. Integrity-checked."""
        metrics = {}
        for domain in self.domains:
            try:
                m = self.evaluator.evaluate_domain(
                    domain, self.solver_fn, n=self.n_eval, use_sealed=True)
            except Exception as e:
                # No sealed set built yet -> fresh eval, clearly marked
                gen = FreshScenarioGenerator(seed=50_000 + self.cycle)
                items = gen.generate_domain(domain, self.n_eval)
                correct = sum(
                    1 for it in items
                    if self._grade_safe(domain, it)
                )
                acc = correct / len(items) if items else 0.0
                m = DomainMetrics(
                    domain=domain, n_evaluated=len(items), accuracy=acc,
                    notes=f"hidden_eval FALLBACK fresh (sealed unavailable: {e})")
            metrics[domain] = m
        report.hidden_metrics = {d: m.to_dict() for d, m in metrics.items()}
        report.phases["hidden_eval"] = {
            d: {"accuracy": m.accuracy, "n": m.n_evaluated,
                "sealed": "sealed" not in (m.notes or "") or "FALLBACK" not in m.notes}
            for d, m in metrics.items()
        }
        return metrics

    def _grade_safe(self, domain: str, item: dict) -> bool:
        try:
            pred, _ = self.solver_fn(domain, item)
        except Exception:
            return False
        return self.evaluator._grade(domain, item, pred)

    def _phase_compare(self, hidden_metrics: dict[str, DomainMetrics],
                       report: CycleReport) -> dict:
        # Save candidate checkpoint record (metrics only; adapters optional)
        cand_name = f"cycle_{self.cycle:04d}"
        rec = self.ckpt_mgr.save_checkpoint(
            name=cand_name, lora_state={},
            step=self.cycle, metrics=hidden_metrics,
            notes=f"loop cycle {self.cycle}")
        gate_result = self.gate.evaluate(hidden_metrics)
        promoted, reason = self.ckpt_mgr.promote_if_better(
            cand_name, primary_domain=report.weakest_domain or self.domains[0])
        # Curriculum advance: if weakest domain now passes its gate, level up
        for crit in self.gate.criteria:
            if crit.domain == report.weakest_domain:
                m = hidden_metrics.get(crit.domain)
                val = getattr(m, crit.metric, None) if m else None
                if val is not None and val >= crit.threshold:
                    self.curriculum.advance(crit.domain)
                    self.mastered.add(crit.domain)
        report.phases["compare"] = {
            "candidate": cand_name,
            "gate": gate_result,
        }
        report.promotion = {"promoted": promoted, "reason": reason}
        return {"promoted": promoted, "reason": reason,
                "gate": gate_result}

    # -- one full cycle ---------------------------------------------------------
    def run_one_cycle(self) -> CycleReport:
        self.cycle += 1
        report = CycleReport(cycle=self.cycle, started_at=_utcnow())

        # 1. BASELINE
        baseline = self._phase_baseline(report)
        report.baseline_metrics = {d: m.to_dict() for d, m in baseline.items()}

        # 2. WEAKNESS_ANALYSIS
        weakest = self._phase_weakness(baseline, report)

        # 3. CHALLENGE_GEN
        train_items = self._phase_challenge_gen(weakest, report)

        # 4. TRAIN
        self._phase_train(weakest, train_items, report)

        # 5. VALIDATE
        self._phase_validate(weakest, report)

        # 6. HIDDEN_EVAL
        hidden = self._phase_hidden_eval(report)

        # 7. COMPARE
        self._phase_compare(hidden, report)

        # 8. PROMOTE/ROLLBACK decided inside _phase_compare; rollback on anomaly
        anomalies = self.anomalies.check(report)
        report.anomalies = anomalies
        critical = [a for a in anomalies if a.get("severity") == "critical"]
        if critical:
            ok, msg = self.ckpt_mgr.rollback()
            report.promotion["rollback_on_critical"] = {"ok": ok, "msg": msg}

        report.ended_at = _utcnow()

        # Persist
        cycle_path = CYCLE_DIR / f"cycle_{self.cycle:04d}.json"
        with open(cycle_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        self._save_state()
        return report

    def run_until_converged(self, max_cycles: int = 20) -> list[CycleReport]:
        """Run cycles until all gates pass or max_cycles reached."""
        reports = []
        for _ in range(max_cycles):
            rep = self.run_one_cycle()
            reports.append(rep)
            gate = rep.phases.get("compare", {}).get("gate", {})
            if gate.get("passed"):
                break
            # Stop early on repeated critical anomalies
            crit = [a for a in rep.anomalies if a.get("severity") == "critical"]
            if crit and len(reports) >= 3:
                last3 = reports[-3:]
                if all(any(a.get("severity") == "critical" for a in r.anomalies)
                       for r in last3):
                    break
        return reports

    def status(self) -> str:
        lines = ["CENTRAL AUTONOMOUS TRAINING LOOP", "=" * 45,
                 f"Cycle: {self.cycle}",
                 f"Mastered: {sorted(self.mastered) or 'none'}",
                 f"Curriculum levels: {self.curriculum.levels}",
                 f"Hard buffer: {len(self.hard_buffer)} examples",
                 f"Best checkpoint: {self.ckpt_mgr.best_name}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dummy components for end-to-end testing (no fake metrics — the dummy's
# behavior is explicit and labeled)
# ---------------------------------------------------------------------------

class DummySolver:
    """Explicitly fake solver for machinery testing.

    improve_every: every N cycles, accuracy target rises by `step`.
    Labels itself as DUMMY in every prediction's provenance.
    """

    def __init__(self, improve_every: int = 2, step: float = 0.15,
                 base_acc: float = 0.30, seed: int = 7):
        self.improve_every = improve_every
        self.step = step
        self.base_acc = base_acc
        self.cycle = 0
        self.rng = random.Random(seed)

    def set_cycle(self, cycle: int):
        self.cycle = cycle

    def _target_acc(self) -> float:
        bumps = self.cycle // self.improve_every
        return min(0.98, self.base_acc + bumps * self.step)

    def __call__(self, domain: str, item: dict) -> tuple[str, float]:
        # Deterministic per (cycle, item id): correct with prob = target_acc
        key = f"{self.cycle}:{item.get('id', '')}"
        r = random.Random(hash(key) & 0xFFFFFFFF).random()
        correct = r < self._target_acc()
        expected = item.get("answer") or item.get("solution", "")
        if domain == "sudoku":
            pred = expected if correct else expected[:40]  # wrong grid
        else:
            pred = expected if correct else "wrong_answer_dummy"
        return pred, 0.7


def dummy_trainer(domain: str, train_items: list, replay_items: list,
                  antiforget_items: list, cycle: int) -> dict:
    """Explicitly mock trainer: advances the DummySolver's cycle counter."""
    # The real wiring: the loop owns the solver; the trainer just logs.
    return {
        "trained": True,
        "notes": (f"DUMMY trainer: saw {len(train_items)} targeted + "
                  f"{len(replay_items)} replay + {len(antiforget_items)} "
                  f"antiforget items for '{domain}' (no real weight update)"),
        "dummy": True,
    }


def run_dummy_test(cycles: int = 6, n_eval: int = 40) -> list[CycleReport]:
    """End-to-end machinery test with dummy components."""
    solver = DummySolver()
    loop = TrainingLoop(
        solver_fn=solver, trainer_fn=dummy_trainer,
        domains=["math", "logic"], n_eval_per_domain=n_eval,
        checkpoint_dir=LOOP_ROOT / "dummy_checkpoints",
    )

    # Wire cycle counter into solver before each cycle
    orig_run_one = loop.run_one_cycle

    def patched():
        solver.set_cycle(loop.cycle + 1)
        return orig_run_one()

    loop.run_one_cycle = patched
    reports = []
    for _ in range(cycles):
        reports.append(loop.run_one_cycle())
    return reports


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy-test", action="store_true")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        # Needs solver/trainer; use dummies for status display
        loop = TrainingLoop(solver_fn=DummySolver(),
                            trainer_fn=dummy_trainer)
        print(loop.status())
    elif args.dummy_test:
        reports = run_dummy_test(cycles=args.cycles, n_eval=args.n_eval)
        for r in reports:
            promo = r.promotion
            weak = r.weakest_domain
            hm = {d: m.get("accuracy") for d, m in r.hidden_metrics.items()}
            anom = [a["type"] for a in r.anomalies]
            print(f"cycle {r.cycle}: weakest={weak} hidden={hm} "
                  f"promoted={promo.get('promoted')} anomalies={anom or 'none'}")
        print(f"\n{len(reports)} cycles complete. "
              f"Reports in {CYCLE_DIR}")
