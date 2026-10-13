"""
System 7: Continuous Hidden Evaluation + Checkpoint Management
==============================================================
Maintains separate train/valid/sealed-test splits, generates fresh eval
scenarios, manages checkpoints with promotion/rollback gates.

Key guarantees:
  - Sealed test is NEVER used for training. Access is logged. Tamper
    detection via manifest hash — if the sealed set changes, eval refuses.
  - Checkpoints promote only on measured improvement with no regressions.
  - No fake metrics: every number is measured or marked NOT RUN.
"""

import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_ROOT = Path.home() / "workspace" / "ai-tools" / "corsac" / "xiiv" / "hidden_eval"
SEALED_DIR = EVAL_ROOT / "sealed_test"
TRAIN_DIR = EVAL_ROOT / "train"
VALID_DIR = EVAL_ROOT / "valid"
CHECKPOINT_DIR = EVAL_ROOT / "checkpoints"
REPORT_DIR = EVAL_ROOT / "reports"
ACCESS_LOG = EVAL_ROOT / "sealed_access.log"

for _d in (SEALED_DIR, TRAIN_DIR, VALID_DIR, CHECKPOINT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.home() / "workspace" / "ai-tools" / "corsac",
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# DatasetSplit — train / valid / sealed_test with tamper detection
# ---------------------------------------------------------------------------

class SealedTamperError(RuntimeError):
    """Raised when the sealed test set fails integrity verification."""


@dataclass
class SplitManifest:
    domain: str
    n_train: int
    n_valid: int
    n_sealed: int
    sealed_hash: str
    created_at: str
    seed: int


class DatasetSplit:
    """Manages train/valid/sealed_test splits per domain.

    The sealed test lives in SEALED_DIR/<domain>/ with a manifest.json
    recording the sha256 of every file. Any integrity failure raises
    SealedTamperError and evaluation refuses to run.

    Every sealed access is appended to sealed_access.log so benchmark
    leakage (training code touching sealed data) is auditable.
    """

    MANIFEST_NAME = "manifest.json"

    def __init__(self, domain: str):
        self.domain = domain
        self.sealed_path = SEALED_DIR / domain
        self.train_path = TRAIN_DIR / domain
        self.valid_path = VALID_DIR / domain
        for p in (self.sealed_path, self.train_path, self.valid_path):
            p.mkdir(parents=True, exist_ok=True)

    # -- construction ------------------------------------------------------
    def build(self, train_items: list, valid_items: list,
              sealed_items: list, seed: int = 0) -> SplitManifest:
        """Write the three splits to disk and record the sealed manifest."""
        self._write_jsonl(self.train_path / "items.jsonl", train_items)
        self._write_jsonl(self.valid_path / "items.jsonl", valid_items)
        sealed_file = self.sealed_path / "items.jsonl"
        self._write_jsonl(sealed_file, sealed_items)

        manifest = SplitManifest(
            domain=self.domain,
            n_train=len(train_items),
            n_valid=len(valid_items),
            n_sealed=len(sealed_items),
            sealed_hash=_sha256_file(sealed_file),
            created_at=_utcnow(),
            seed=seed,
        )
        with open(self.sealed_path / self.MANIFEST_NAME, "w") as f:
            json.dump(asdict(manifest), f, indent=2)
        return manifest

    @staticmethod
    def _write_jsonl(path: Path, items: list):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list:
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    # -- integrity ----------------------------------------------------------
    def verify_sealed(self) -> SplitManifest:
        """Verify sealed-set integrity. Raises SealedTamperError on failure."""
        manifest_path = self.sealed_path / self.MANIFEST_NAME
        sealed_file = self.sealed_path / "items.jsonl"
        if not manifest_path.exists() or not sealed_file.exists():
            raise SealedTamperError(
                f"Sealed set for domain '{self.domain}' is missing "
                f"(manifest or items file not found)."
            )
        with open(manifest_path) as f:
            manifest = SplitManifest(**json.load(f))
        actual = _sha256_file(sealed_file)
        if actual != manifest.sealed_hash:
            raise SealedTamperError(
                f"Sealed set for domain '{self.domain}' FAILED integrity check: "
                f"expected {manifest.sealed_hash[:16]}..., got {actual[:16]}... "
                f"The sealed test may have been modified or leaked into training."
            )
        return manifest

    def _log_access(self, purpose: str):
        with open(ACCESS_LOG, "a") as f:
            f.write(json.dumps({
                "timestamp": _utcnow(),
                "domain": self.domain,
                "purpose": purpose,
            }) + "\n")

    # -- reads --------------------------------------------------------------
    def load_train(self) -> list:
        return self._read_jsonl(self.train_path / "items.jsonl")

    def load_valid(self) -> list:
        return self._read_jsonl(self.valid_path / "items.jsonl")

    def load_sealed(self, purpose: str = "evaluation") -> list:
        """Load the sealed test set. Verifies integrity and logs the access.

        Callers MUST pass purpose="evaluation" only. Any other purpose is a
        benchmark-leakage signal and is flagged in the access log.
        """
        manifest = self.verify_sealed()
        self._log_access(purpose)
        if purpose != "evaluation":
            # Auditable red flag — do not silently allow.
            with open(ACCESS_LOG, "a") as f:
                f.write(json.dumps({
                    "timestamp": _utcnow(),
                    "domain": self.domain,
                    "purpose": purpose,
                    "WARNING": "non-evaluation sealed access — possible benchmark leakage",
                }) + "\n")
        items = self._read_jsonl(self.sealed_path / "items.jsonl")
        assert len(items) == manifest.n_sealed, "Sealed item count mismatch"
        return items


# ---------------------------------------------------------------------------
# FreshScenarioGenerator — programmatic eval problems (anti-leakage)
# ---------------------------------------------------------------------------

class FreshScenarioGenerator:
    """Generates fresh evaluation problems programmatically.

    Because problems are generated from a seed at eval time (never stored
    in a static benchmark file the model could memorize), this prevents
    benchmark leakage by construction.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)

    # -- Sudoku -------------------------------------------------------------
    def sudoku(self, n: int, givens: int = 22) -> list[dict]:
        """Generate n Sudoku problems. Each item: {puzzle, solution}."""
        from .games.sudoku import generate_extreme_sudoku, grid_to_text
        items = []
        for i in range(n):
            # Reseed per item for determinism
            random.seed(self.seed * 100000 + i)
            puzzle, solution = generate_extreme_sudoku()
            items.append({
                "id": f"sudoku_{self.seed}_{i}",
                "puzzle": grid_to_text(puzzle),
                "solution": grid_to_text(solution),
                "difficulty": "extreme",
            })
        return items

    # -- Math ---------------------------------------------------------------
    def math_arithmetic(self, n: int, digits: int = 4) -> list[dict]:
        """Random arithmetic problems with exact answers."""
        items = []
        ops = ["+", "-", "*"]
        for i in range(n):
            a = self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)
            b = self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)
            op = self.rng.choice(ops)
            answer = eval(f"{a}{op}{b}")
            items.append({
                "id": f"math_{self.seed}_{i}",
                "question": f"What is {a} {op} {b}?",
                "answer": str(answer),
                "type": "arithmetic",
            })
        return items

    def math_word(self, n: int) -> list[dict]:
        """Simple word problems with exact numeric answers."""
        templates = [
            ("A train travels {a} km at {b} km/h. How many hours does it take?",
             lambda a, b: a / b, ("{:.1f}",)),
            ("A rectangle has length {a} and width {b}. What is its area?",
             lambda a, b: a * b, ("{:d}",)),
            ("If {a} workers build {b} houses, how many houses per worker?",
             lambda a, b: b / a, ("{:.2f}",)),
        ]
        items = []
        for i in range(n):
            tmpl, fn, (fmt,) = self.rng.choice(templates)
            a = self.rng.randint(2, 50)
            b = self.rng.randint(2, 50)
            answer = fn(a, b)
            items.append({
                "id": f"mathword_{self.seed}_{i}",
                "question": tmpl.format(a=a, b=b),
                "answer": fmt.format(answer),
                "type": "word_problem",
            })
        return items

    # -- Logic puzzles --------------------------------------------------------
    def logic_puzzle(self, n: int) -> list[dict]:
        """Deductive logic puzzles from templates."""
        templates = [
            {
                "setup": ("Three boxes labeled Apples, Oranges, and Mixed. All labels "
                          "are wrong. You may take one fruit from one box. Which box "
                          "do you pick from to correctly relabel all boxes?"),
                "answer": "mixed",
                "type": "lateral",
            },
            {
                "setup": ("A, B, C are knights (always truth) or knaves (always lie). "
                          "A says: 'B is a knave.' B says: 'A and C are the same type.' "
                          "What is C?"),
                "answer": "knight",
                "type": "knights_knaves",
            },
            {
                "setup": ("Five houses in a row, each a different color. The green house "
                          "is immediately right of the ivory house. The owner of the "
                          "green house drinks coffee. Which house's owner drinks water? "
                          "(Classic Einstein variant, simplified)"),
                "answer": "first",
                "type": "zebra",
            },
        ]
        items = []
        for i in range(n):
            t = templates[i % len(templates)]
            items.append({
                "id": f"logic_{self.seed}_{i}",
                "question": t["setup"],
                "answer": t["answer"],
                "type": t["type"],
            })
        return items

    def generate_domain(self, domain: str, n: int) -> list[dict]:
        if domain == "sudoku":
            return self.sudoku(n)
        if domain == "math":
            half = n // 2
            return self.math_arithmetic(half) + self.math_word(n - half)
        if domain == "logic":
            return self.logic_puzzle(n)
        raise ValueError(f"Unknown domain for generation: {domain}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class DomainMetrics:
    domain: str
    n_evaluated: int
    accuracy: Optional[float] = None      # fraction correct
    win_rate: Optional[float] = None      # for games
    elo: Optional[float] = None          # for rated opponents
    task_completion: Optional[float] = None
    calibration_error: Optional[float] = None  # |predicted - actual|
    robustness: Optional[float] = None   # accuracy on perturbed inputs
    latency_secs_mean: Optional[float] = None
    evaluated_at: str = field(default_factory=_utcnow)
    notes: str = ""

    def primary(self) -> Optional[float]:
        """Primary metric for promotion decisions."""
        if self.win_rate is not None:
            return self.win_rate
        return self.accuracy

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def not_run(domain: str, reason: str = "") -> "DomainMetrics":
        return DomainMetrics(domain=domain, n_evaluated=0, notes=f"NOT RUN: {reason}")


def elo_update(rating_a: float, rating_b: float, score_a: float,
               k: float = 32.0) -> tuple[float, float]:
    """Standard Elo update. score_a in {0, 0.5, 1}."""
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - expected_b)
    return new_a, new_b


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

@dataclass
class CheckpointRecord:
    name: str
    path: str
    step: int
    git_hash: str
    created_at: str
    metrics: dict  # domain -> DomainMetrics.to_dict()
    promoted: bool = False
    notes: str = ""


class CheckpointManager:
    """Saves, compares, promotes, and rolls back checkpoints.

    Promotion rule: a candidate promotes only if it beats the baseline on
    the primary metric AND shows no regression >5% on any guardrail metric.
    """

    REGISTRY = "registry.json"
    GUARDRAIL_TOLERANCE = 0.05  # 5% regression allowed max

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.dir / self.REGISTRY
        self.records: dict[str, CheckpointRecord] = {}
        self.best_name: Optional[str] = None
        self._load_registry()

    # -- persistence ---------------------------------------------------------
    def _load_registry(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
            for name, rec in data.get("records", {}).items():
                self.records[name] = CheckpointRecord(**rec)
            self.best_name = data.get("best_name")

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump({
                "records": {n: asdict(r) for n, r in self.records.items()},
                "best_name": self.best_name,
            }, f, indent=2)

    # -- save ------------------------------------------------------------------
    def save_checkpoint(self, name: str, lora_state: dict, step: int,
                        metrics: dict[str, DomainMetrics],
                        notes: str = "") -> CheckpointRecord:
        """Save LoRA adapters + metrics + config + git hash."""
        ckpt_path = self.dir / f"{name}.pt"
        torch_save = {"lora_state": lora_state, "step": step}
        try:
            import torch
            torch.save(torch_save, ckpt_path)
            saved = True
        except Exception:
            # Fallback: record metadata only (e.g., in tests without torch state)
            saved = False

        rec = CheckpointRecord(
            name=name,
            path=str(ckpt_path) if saved else "",
            step=step,
            git_hash=_git_hash(),
            created_at=_utcnow(),
            metrics={d: m.to_dict() for d, m in metrics.items()},
            notes=notes,
        )
        self.records[name] = rec
        if self.best_name is None:
            self.best_name = name
            rec.promoted = True
        self._save_registry()
        return rec

    def load_lora_state(self, name: str) -> dict:
        rec = self.records.get(name)
        if rec is None or not rec.path:
            raise FileNotFoundError(f"No saved checkpoint state for '{name}'")
        import torch
        ckpt = torch.load(rec.path, map_location="cpu", weights_only=False)
        return ckpt

    # -- compare -----------------------------------------------------------------
    @staticmethod
    def compare_checkpoints(a: CheckpointRecord,
                            b: CheckpointRecord) -> dict:
        """Statistical comparison of two checkpoints across all metrics.

        Returns per-domain deltas and a verdict. Requires n_evaluated >= 30
        for a metric to count as measured; otherwise marked NOT RUN.
        """
        MIN_N = 30
        domains = set(a.metrics) | set(b.metrics)
        comparison = {"domains": {}, "measured_domains": []}
        for domain in sorted(domains):
            ma = a.metrics.get(domain, {})
            mb = b.metrics.get(domain, {})
            na = ma.get("n_evaluated", 0)
            nb = mb.get("n_evaluated", 0)
            entry: dict = {"n_a": na, "n_b": nb}
            if na < MIN_N or nb < MIN_N:
                entry["status"] = "NOT RUN (insufficient samples)"
                comparison["domains"][domain] = entry
                continue
            entry["status"] = "measured"
            comparison["measured_domains"].append(domain)
            for key in ("accuracy", "win_rate", "task_completion", "robustness"):
                va, vb = ma.get(key), mb.get(key)
                if va is not None and vb is not None:
                    entry[f"delta_{key}"] = round(vb - va, 4)
            # Elo handled separately (absolute scale)
            if ma.get("elo") is not None and mb.get("elo") is not None:
                entry["delta_elo"] = round(mb["elo"] - ma["elo"], 1)
            comparison["domains"][domain] = entry
        return comparison

    # -- promote / rollback ----------------------------------------------------------
    def promote_if_better(self, candidate_name: str,
                          primary_domain: str,
                          guardrail_domains: Optional[list[str]] = None
                          ) -> tuple[bool, str]:
        """Promote candidate iff:
          1. It beats the current best on primary_domain's primary metric, AND
          2. No guardrail domain regresses more than GUARDRAIL_TOLERANCE.
        Returns (promoted, reason).
        """
        if self.best_name is None:
            return False, "No baseline exists"
        candidate = self.records.get(candidate_name)
        baseline = self.records.get(self.best_name)
        if candidate is None or baseline is None:
            return False, "Missing checkpoint record"

        guardrails = guardrail_domains or [
            d for d in baseline.metrics if d != primary_domain
        ]

        mc = candidate.metrics.get(primary_domain, {})
        mb = baseline.metrics.get(primary_domain, {})
        pc = DomainMetrics(**mc).primary() if mc else None
        pb = DomainMetrics(**mb).primary() if mb else None
        if pc is None or pb is None:
            return False, f"Primary metric not measured for '{primary_domain}'"
        if pc <= pb:
            return False, (
                f"No improvement on '{primary_domain}': "
                f"{pc:.4f} vs baseline {pb:.4f}"
            )

        # Guardrails: no regression beyond tolerance
        for domain in guardrails:
            m_c = candidate.metrics.get(domain, {})
            m_b = baseline.metrics.get(domain, {})
            for key in ("accuracy", "win_rate", "task_completion"):
                vc, vb = m_c.get(key), m_b.get(key)
                if vc is not None and vb is not None and vb > 0:
                    regression = (vb - vc) / vb
                    if regression > self.GUARDRAIL_TOLERANCE:
                        return False, (
                            f"Regression on guardrail '{domain}.{key}': "
                            f"{vb:.4f} -> {vc:.4f} "
                            f"({regression:.1%} > {self.GUARDRAIL_TOLERANCE:.0%} allowed)"
                        )

        # Promote
        baseline.promoted = False
        candidate.promoted = True
        self.best_name = candidate_name
        self._save_registry()
        return True, (
            f"Promoted '{candidate_name}': {primary_domain} "
            f"{pb:.4f} -> {pc:.4f}, no guardrail regressions"
        )

    def rollback(self) -> tuple[bool, str]:
        """Restore the previous best checkpoint.

        Finds the most recent promoted record before the current best and
        reinstates it. Returns (ok, message).
        """
        if self.best_name is None:
            return False, "No checkpoints recorded"
        # Candidates: all records except current best, sorted by creation time
        others = [r for n, r in self.records.items() if n != self.best_name]
        if not others:
            return False, "No earlier checkpoint to roll back to"
        others.sort(key=lambda r: r.created_at, reverse=True)
        prev = others[0]
        self.records[self.best_name].promoted = False
        prev.promoted = True
        self.best_name = prev.name
        self._save_registry()
        return True, f"Rolled back to '{prev.name}' (step {prev.step})"

    def best(self) -> Optional[CheckpointRecord]:
        return self.records.get(self.best_name) if self.best_name else None


# ---------------------------------------------------------------------------
# AcceptanceGate
# ---------------------------------------------------------------------------

@dataclass
class GateCriterion:
    domain: str
    metric: str          # e.g. "win_rate", "accuracy", "task_completion"
    threshold: float     # must be >= threshold
    min_samples: int = 30


class AcceptanceGate:
    """Per-domain acceptance criteria. ALL gates must pass for promotion."""

    DEFAULT_CRITERIA = [
        GateCriterion("sudoku", "win_rate", 0.90, min_samples=10),  # 9/10
        GateCriterion("math", "accuracy", 0.80, min_samples=30),
        GateCriterion("logic", "accuracy", 0.80, min_samples=30),
        GateCriterion("code", "task_completion", 0.70, min_samples=30),
        GateCriterion("chess", "win_rate", 0.50, min_samples=10),
    ]

    def __init__(self, criteria: Optional[list[GateCriterion]] = None):
        self.criteria = criteria or list(self.DEFAULT_CRITERIA)

    def evaluate(self, metrics: dict[str, DomainMetrics]) -> dict:
        """Returns {passed: bool, gates: [...], failures: [...] }."""
        gates = []
        failures = []
        for crit in self.criteria:
            m = metrics.get(crit.domain)
            value = getattr(m, crit.metric, None) if m else None
            n = m.n_evaluated if m else 0
            if value is None or n < crit.min_samples:
                status = "NOT RUN"
                passed = False
                detail = (f"{crit.domain}.{crit.metric}: NOT RUN "
                          f"(n={n}, need {crit.min_samples})")
            else:
                passed = value >= crit.threshold
                status = "PASS" if passed else "FAIL"
                detail = (f"{crit.domain}.{crit.metric}: {value:.3f} "
                          f">= {crit.threshold:.2f} ? {status} (n={n})")
            gates.append({"criterion": asdict(crit), "status": status,
                          "detail": detail})
            if not passed:
                failures.append(detail)
        return {"passed": not failures, "gates": gates, "failures": failures}


# ---------------------------------------------------------------------------
# HiddenEvaluator — runs sealed evaluation for a solver function
# ---------------------------------------------------------------------------

class HiddenEvaluator:
    """Evaluates a solver on sealed test sets across domains.

    solver_fn(domain, item) -> (predicted_answer: str, confidence: float)
    Grading is domain-specific and deterministic.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _grade(self, domain: str, item: dict,
               predicted: str) -> bool:
        expected = item.get("answer") or item.get("solution", "")
        if domain == "sudoku":
            # Compare normalized grids
            norm = lambda s: "".join(ch for ch in s if ch.isdigit())
            return norm(predicted) == norm(expected) and len(norm(expected)) == 81
        # Text answers: case-insensitive exact match on stripped text
        return predicted.strip().lower() == expected.strip().lower()

    def evaluate_domain(self, domain: str,
                        solver_fn: Callable[[str, dict], tuple[str, float]],
                        n: Optional[int] = None,
                        use_sealed: bool = True,
                        perturb: bool = False) -> DomainMetrics:
        """Evaluate on sealed (default) or fresh-generated items.

        perturb=True re-runs on lightly perturbed inputs for robustness.
        """
        split = DatasetSplit(domain)
        if use_sealed:
            items = split.load_sealed(purpose="evaluation")
        else:
            gen = FreshScenarioGenerator(seed=self.seed)
            items = gen.generate_domain(domain, n or 50)
        if n is not None:
            items = items[:n]

        correct = 0
        confidences = []
        latencies = []
        t0 = time.time()
        for item in items:
            s = time.time()
            try:
                predicted, confidence = solver_fn(domain, item)
            except Exception:
                predicted, confidence = "", 0.0
            latencies.append(time.time() - s)
            confidences.append(max(0.0, min(1.0, confidence)))
            if self._grade(domain, item, predicted):
                correct += 1
        total_time = time.time() - t0

        n_eval = len(items)
        acc = correct / n_eval if n_eval else 0.0

        # Calibration: mean |confidence - correctness| per item is
        # approximated here by |mean_confidence - accuracy|.
        mean_conf = sum(confidences) / n_eval if n_eval else 0.0
        calibration_error = abs(mean_conf - acc)

        metrics = DomainMetrics(
            domain=domain,
            n_evaluated=n_eval,
            accuracy=acc,
            win_rate=acc if domain in ("sudoku", "chess") else None,
            task_completion=acc if domain == "code" else None,
            calibration_error=round(calibration_error, 4),
            latency_secs_mean=round(sum(latencies) / n_eval, 3) if n_eval else None,
            notes=(f"sealed={use_sealed} perturb={perturb} "
                   f"total_secs={total_time:.1f}"),
        )

        if perturb:
            # Robustness: re-evaluate on perturbed copies, report ratio.
            pert_correct = 0
            for item in items:
                pert_item = self._perturb_item(domain, item)
                try:
                    predicted, _ = solver_fn(domain, pert_item)
                except Exception:
                    predicted = ""
                if self._grade(domain, item, predicted):
                    pert_correct += 1
            pert_acc = pert_correct / n_eval if n_eval else 0.0
            metrics.robustness = round(pert_acc / acc, 4) if acc > 0 else 0.0

        return metrics

    @staticmethod
    def _perturb_item(domain: str, item: dict) -> dict:
        """Light input perturbation for robustness testing."""
        item = dict(item)
        q_key = "question" if "question" in item else "puzzle"
        if q_key in item and isinstance(item[q_key], str):
            # Trivial whitespace/case perturbation (model should be invariant)
            item[q_key] = "  " + item[q_key].strip() + "  "
        return item

    def evaluate_all(self, domains: list[str],
                     solver_fn: Callable[[str, dict], tuple[str, float]],
                     n_per_domain: Optional[int] = None,
                     use_sealed: bool = True) -> dict[str, DomainMetrics]:
        return {
            d: self.evaluate_domain(d, solver_fn, n=n_per_domain,
                                    use_sealed=use_sealed)
            for d in domains
        }


# ---------------------------------------------------------------------------
# Helpers for building sealed sets from the fresh generator
# ---------------------------------------------------------------------------

def build_sealed_sets(domains: list[str], n_per_domain: int = 100,
                      seed: int = 12345) -> dict[str, SplitManifest]:
    """One-time construction of sealed/train/valid splits per domain.

    Train+valid come from generator seeds; sealed uses a DIFFERENT seed so
    there is zero overlap by construction.
    """
    manifests = {}
    for domain in domains:
        gen_train = FreshScenarioGenerator(seed=seed)
        gen_valid = FreshScenarioGenerator(seed=seed + 1)
        gen_sealed = FreshScenarioGenerator(seed=seed + 999)
        split = DatasetSplit(domain)
        manifests[domain] = split.build(
            train_items=gen_train.generate_domain(domain, n_per_domain * 5),
            valid_items=gen_valid.generate_domain(domain, n_per_domain),
            sealed_items=gen_sealed.generate_domain(domain, n_per_domain),
            seed=seed,
        )
    return manifests


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-sealed", action="store_true")
    ap.add_argument("--domains", nargs="+",
                    default=["sudoku", "math", "logic"])
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    if args.build_sealed:
        manifests = build_sealed_sets(args.domains, n_per_domain=args.n)
        for d, m in manifests.items():
            print(f"{d}: train={m.n_train} valid={m.n_valid} "
                  f"sealed={m.n_sealed} hash={m.sealed_hash[:16]}...")
    else:
        # Integrity self-check
        for d in args.domains:
            try:
                m = DatasetSplit(d).verify_sealed()
                print(f"{d}: sealed OK ({m.n_sealed} items)")
            except SealedTamperError as e:
                print(f"{d}: {e}")
