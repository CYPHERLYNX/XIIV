"""
Adaptive Test-Time Reasoning for XIIV (System 3 of the Adaptive Intelligence
Training System).

Provides inference-time search strategies that trade extra compute for
higher accuracy:

  - BestOfN:          generate N candidates, rank with an independent
                      verifier, return the best.
  - BranchingSearch:  multi-step problems: K candidates per step, prune to
                      top-B, backtrack when a step yields no viable branch.
  - AdaptiveCompute:  estimate problem difficulty from cheap heuristics,
                      allocate the search budget accordingly, and log
                      accuracy-vs-compute.

Every strategy works with ANY verifier: a plain callable
    verifier(text: str) -> float   # score in [0.0, 1.0]

Loop guards everywhere: max iterations, max total forward passes, early
stop when a candidate scores 1.0. No uncontrolled reasoning loops.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

logger = logging.getLogger("xiiv.test_time_search")

# A candidate generator: given the prompt (and optional step context),
# produce one candidate solution as text. Each call = one forward pass.
CandidateFn = Callable[..., str]
# A verifier: text -> score in [0.0, 1.0]. Must be independent of the
# generator (no shared weights / no self-grading).
VerifierFn = Callable[[str], float]


@dataclass
class SearchResult:
    """Outcome of a test-time search run."""
    best_text: str
    best_score: float
    n_candidates: int          # total forward passes spent
    compute_cost: int          # alias for n_candidates (explicit budget unit)
    elapsed_secs: float
    stopped_early: bool        # True if a perfect (1.0) candidate was found
    strategy: str = ""
    log: list = field(default_factory=list)  # structured per-step records

    def to_dict(self):
        return asdict(self)


def _check_score(score: float, where: str) -> float:
    if not isinstance(score, (int, float)) or math.isnan(score):
        raise ValueError(f"verifier returned non-numeric score in {where}: {score!r}")
    return max(0.0, min(1.0, float(score)))


class BestOfN:
    """Generate N candidates, return the highest-scoring one.

    Args:
        n: number of candidates (forward passes).
        verifier: independent scoring function text -> [0,1].
        early_stop: stop immediately when a candidate scores 1.0.
    """

    def __init__(self, n: int, verifier: VerifierFn, early_stop: bool = True):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.verifier = verifier
        self.early_stop = early_stop

    def run(self, prompt: str, generate: CandidateFn) -> SearchResult:
        t0 = time.time()
        log = []
        best_text, best_score = "", -1.0
        stopped_early = False
        spent = 0

        for i in range(self.n):
            cand = generate(prompt)
            spent += 1
            score = _check_score(self.verifier(cand), f"BestOfN candidate {i}")
            log.append({"candidate": i, "score": round(score, 4)})
            logger.info(json.dumps({"strategy": "best_of_n", "candidate": i,
                                    "score": score, "prompt_len": len(prompt)}))
            if score > best_score:
                best_text, best_score = cand, score
            if self.early_stop and score >= 1.0:
                stopped_early = True
                break

        return SearchResult(
            best_text=best_text,
            best_score=best_score,
            n_candidates=spent,
            compute_cost=spent,
            elapsed_secs=round(time.time() - t0, 3),
            stopped_early=stopped_early,
            strategy=f"best_of_n(n={self.n})",
            log=log,
        )


class BranchingSearch:
    """Step-wise search with pruning and backtracking.

    At each step, generate K candidate continuations from each surviving
    beam, score the resulting partial solutions, keep the top-B beams.
    If every candidate at step t fails (score below min_score), backtrack
    to step t-1 and expand the next-best branches that were pruned.

    Args:
        steps: maximum number of steps (hard cap).
        k: candidates generated per beam per step.
        beam_width: number of beams kept after pruning.
        verifier: scores a (possibly partial) solution text -> [0,1].
        min_score: branches scoring below this are treated as failed.
        max_total_candidates: hard cap on total forward passes.
    """

    def __init__(self, steps: int, k: int, beam_width: int,
                 verifier: VerifierFn, min_score: float = 0.0,
                 max_total_candidates: int = 500):
        if steps < 1 or k < 1 or beam_width < 1:
            raise ValueError("steps, k, beam_width must be >= 1")
        self.steps = steps
        self.k = k
        self.beam_width = beam_width
        self.verifier = verifier
        self.min_score = min_score
        self.max_total_candidates = max_total_candidates

    def run(self, prompt: str, generate_step: CandidateFn) -> SearchResult:
        """generate_step(prompt, partial_solution) -> next-step text."""
        t0 = time.time()
        log = []
        spent = 0
        stopped_early = False

        # beams: list of (partial_text, score)
        beams = [("", 0.0)]
        # backtrack stack: pruned branches we can revisit, per step
        pruned: dict[int, list] = {}

        best_text, best_score = "", -1.0
        step = 0

        while step < self.steps:
            new_beams = []
            for beam_text, _ in beams:
                for _ in range(self.k):
                    if spent >= self.max_total_candidates:
                        break
                    cand = generate_step(prompt, beam_text)
                    spent += 1
                    full = (beam_text + "\n" + cand).strip() if beam_text else cand
                    score = _check_score(self.verifier(full),
                                         f"BranchingSearch step {step}")
                    new_beams.append((full, score))
                    if score > best_score:
                        best_text, best_score = full, score
                    if score >= 1.0:
                        stopped_early = True
                if stopped_early or spent >= self.max_total_candidates:
                    break

            # split into viable / failed
            viable = [(t, s) for t, s in new_beams if s >= self.min_score]
            failed = [(t, s) for t, s in new_beams if s < self.min_score]
            viable.sort(key=lambda x: x[1], reverse=True)

            log.append({
                "step": step,
                "candidates": len(new_beams),
                "viable": len(viable),
                "best_score": round(max([s for _, s in new_beams], default=0.0), 4),
            })
            logger.info(json.dumps({"strategy": "branching", "step": step,
                                    "candidates": len(new_beams),
                                    "viable": len(viable)}))

            if stopped_early:
                break

            if not viable:
                # Backtrack: revisit pruned branches from the previous step.
                backtracked = False
                for prev in range(step - 1, -1, -1):
                    if pruned.get(prev):
                        beams = pruned[prev][: self.beam_width]
                        pruned[prev] = pruned[prev][self.beam_width :]
                        log.append({"step": step, "backtrack_to": prev})
                        logger.info(json.dumps({"strategy": "branching",
                                                "backtrack_to": prev}))
                        backtracked = True
                        break
                if not backtracked:
                    log.append({"step": step, "terminated": "no_viable_no_backtrack"})
                    break
                continue  # retry current step index with restored beams

            pruned[step] = viable[self.beam_width :]
            beams = viable[: self.beam_width]
            step += 1

            if spent >= self.max_total_candidates:
                log.append({"terminated": "compute_budget_exhausted"})
                break

        return SearchResult(
            best_text=best_text,
            best_score=best_score,
            n_candidates=spent,
            compute_cost=spent,
            elapsed_secs=round(time.time() - t0, 3),
            stopped_early=stopped_early,
            strategy=(f"branching(steps={self.steps},k={self.k},"
                      f"beam={self.beam_width})"),
            log=log,
        )


class AdaptiveCompute:
    """Allocate search budget based on estimated problem difficulty.

    Difficulty heuristic (cheap, no model call):
        difficulty = alpha * normalized_problem_length
                   + beta  * verifier_uncertainty

    verifier_uncertainty is measured by scoring a small probe set of
    candidates and taking the variance: high variance => the verifier
    discriminates between candidates => harder problem, worth searching.

    The total budget (max forward passes) is split across problems
    proportionally to difficulty. Accuracy vs compute is logged so the
    cost/benefit of extra search is measurable.
    """

    def __init__(self, total_budget: int, alpha: float = 0.5,
                 beta: float = 0.5, probe_candidates: int = 3,
                 ref_length: int = 2000):
        if total_budget < 1:
            raise ValueError("total_budget must be >= 1")
        self.total_budget = total_budget
        self.alpha = alpha
        self.beta = beta
        self.probe_candidates = probe_candidates
        self.ref_length = ref_length
        self.history: list[dict] = []  # accuracy-vs-compute records

    def estimate_difficulty(self, prompt: str, generate: CandidateFn,
                            verifier: VerifierFn) -> float:
        """Return difficulty in [0, 1]. Costs probe_candidates forward passes."""
        length_term = min(1.0, len(prompt) / self.ref_length)

        probe_scores = []
        for _ in range(self.probe_candidates):
            cand = generate(prompt)
            probe_scores.append(_check_score(verifier(cand), "difficulty probe"))
        mean = sum(probe_scores) / len(probe_scores)
        var = sum((s - mean) ** 2 for s in probe_scores) / len(probe_scores)
        # variance of scores in [0,1] is at most 0.25 -> normalize
        uncertainty_term = min(1.0, var / 0.25)

        difficulty = self.alpha * length_term + self.beta * uncertainty_term
        return max(0.0, min(1.0, difficulty))

    def allocate(self, prompts: list[str], generate: CandidateFn,
                 verifier: VerifierFn) -> list[int]:
        """Split total_budget across prompts proportional to difficulty.

        Returns per-prompt candidate counts summing to <= total_budget.
        Each prompt gets at least 1 candidate.
        """
        difficulties = [self.estimate_difficulty(p, generate, verifier)
                        for p in prompts]
        probe_cost = self.probe_candidates * len(prompts)
        remaining = max(len(prompts), self.total_budget - probe_cost)

        total_d = sum(difficulties) or 1.0
        allocs = [max(1, round(remaining * d / total_d)) for d in difficulties]

        # fix rounding so the sum does not exceed the remaining budget
        while sum(allocs) > remaining:
            idx = max(range(len(allocs)), key=lambda i: allocs[i])
            if allocs[idx] > 1:
                allocs[idx] -= 1
            else:
                break
        logger.info(json.dumps({"strategy": "adaptive_compute",
                                "difficulties": [round(d, 3) for d in difficulties],
                                "allocations": allocs,
                                "probe_cost": probe_cost}))
        return allocs

    def run_best_of_n(self, prompts: list[str], generate: CandidateFn,
                      verifier: VerifierFn) -> list[SearchResult]:
        """Adaptive BestOfN across a batch; records accuracy-vs-compute."""
        allocs = self.allocate(prompts, generate, verifier)
        results = []
        for prompt, n in zip(prompts, allocs):
            res = BestOfN(n=n, verifier=verifier).run(prompt, generate)
            results.append(res)
            self.history.append({
                "prompt_len": len(prompt),
                "allocated": n,
                "compute_cost": res.compute_cost,
                "best_score": round(res.best_score, 4),
                "stopped_early": res.stopped_early,
            })
        return results

    def cost_benefit_report(self) -> dict:
        """Accuracy vs compute across all runs so far. Real numbers only."""
        if not self.history:
            return {"runs": 0}
        costs = [h["compute_cost"] for h in self.history]
        scores = [h["best_score"] for h in self.history]
        return {
            "runs": len(self.history),
            "mean_compute": round(sum(costs) / len(costs), 2),
            "mean_best_score": round(sum(scores) / len(scores), 4),
            "total_compute": sum(costs),
            "score_per_compute": round(sum(scores) / max(1, sum(costs)), 4),
        }
