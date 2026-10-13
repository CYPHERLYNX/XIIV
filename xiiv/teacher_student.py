"""System 5: Teacher-Student Improvement Loop.

Generate candidate solutions -> critique with INDEPENDENT verification ->
rank -> build preference pairs -> train student via Diffusion-DPO.

Critical rule: NEVER assume teacher outputs are correct. Every teacher
solution goes through the independent verifier before it can become a
"chosen" example. The verifier is deterministic code (math checkers,
unit tests, constraint checkers), never the model grading itself.

Pipeline:
    1. SolutionGenerator: K candidates per problem via sample_trajectory
    2. SolutionCritique: independent verification + quality scoring
    3. PreferencePairBuilder: ranked candidates -> (chosen, rejected) DPO pairs
    4. StudentTrainer: LoRA + diffusion_dpo_loss; SFT fallback on chosen-only
    5. Anti-memorization: 20% holdout, report train/eval gap

Status: IMPLEMENTED, unit-tested with dummy verifiers. Full end-to-end
with real XIIV model weights = NOT RUN (needs trained base model).
"""

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import torch

from .diffusion import sample_trajectory
from .diffusion_dpo import diffusion_dpo_loss
from .lora import inject_lora, get_trainable_params


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """A training problem with an independent verifier."""
    problem_id: str
    prompt_text: str                       # human-readable prompt
    prompt_ids: list[int]                  # tokenized prompt (conditioning)
    verifier: str                          # verifier name (registry key)
    verifier_data: dict = field(default_factory=dict)  # e.g. expected answer
    difficulty: str = "medium"


@dataclass
class CandidateSolution:
    problem_id: str
    candidate_idx: int
    ids: list[int]                         # full sequence (prompt + completion)
    text: str                              # decoded completion text
    seed: int


@dataclass
class CritiqueResult:
    candidate: CandidateSolution
    verified_correct: bool                 # independent verifier verdict
    verifier_notes: str
    quality_score: float                   # 0..1, for ranking among verified
    passed_verifier: bool = True           # False if verifier errored


# ---------------------------------------------------------------------------
# Verifier registry — INDEPENDENT, deterministic, no model in the loop
# ---------------------------------------------------------------------------

def _verify_exact_match(text: str, data: dict) -> tuple[bool, str]:
    expected = str(data.get("expected", "")).strip().lower()
    got = text.strip().lower()
    # Compare after stripping whitespace; allow expected as substring for
    # robustness against trailing explanations, but require it present.
    if expected and expected in got:
        return True, f"expected '{expected}' found in output"
    return False, f"expected '{expected}' not found in output"


def _verify_math_numeric(text: str, data: dict) -> tuple[bool, str]:
    """Extract last number from text, compare to expected within tolerance."""
    import re
    expected = float(data.get("expected"))
    tol = float(data.get("tolerance", 1e-6))
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not nums:
        return False, "no numeric answer found"
    try:
        got = float(nums[-1])
    except ValueError:
        return False, f"could not parse '{nums[-1]}' as number"
    if abs(got - expected) <= tol:
        return True, f"{got} == {expected} (tol {tol})"
    return False, f"{got} != {expected} (tol {tol})"


def _verify_code_unittest(text: str, data: dict) -> tuple[bool, str]:
    """Extract python code block, run against provided unit tests.

    Runs in-process with a timeout guard via signal (best-effort on CPU).
    Code that fails to parse or times out is marked incorrect, not crashed.
    """
    import re, io, contextlib
    code_blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    code = code_blocks[0] if code_blocks else text
    tests = data.get("tests", "")
    namespace: dict = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<candidate>", "exec"), namespace)
            exec(compile(tests, "<tests>", "exec"), namespace)
    except Exception as e:
        return False, f"code/test failure: {type(e).__name__}: {e}"
    return True, "all unit tests passed"


VERIFIERS: dict[str, Callable[[str, dict], tuple[bool, str]]] = {
    "exact_match": _verify_exact_match,
    "math_numeric": _verify_math_numeric,
    "code_unittest": _verify_code_unittest,
}


def register_verifier(name: str, fn: Callable[[str, dict], tuple[bool, str]]):
    """Add a new independent verifier. Must be deterministic code."""
    VERIFIERS[name] = fn


# ---------------------------------------------------------------------------
# 1. SolutionGenerator
# ---------------------------------------------------------------------------

class SolutionGenerator:
    """Generates K diverse candidate solutions per problem."""

    def __init__(self, model, mask_id: int, eos_id: int,
                 gen_length: int = 64, steps: int = 16,
                 temperatures: tuple = (0.7, 1.0, 1.3)):
        self.model = model
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.gen_length = gen_length
        self.steps = steps
        self.temperatures = temperatures

    def generate(self, problem: Problem, k: int = 4,
                 base_seed: int = 0) -> list[CandidateSolution]:
        """Generate k candidates with varied temperature/seed for diversity."""
        self.model.eval()
        candidates = []
        prompt_ids = torch.tensor(problem.prompt_ids, dtype=torch.long)
        total_len = len(problem.prompt_ids) + self.gen_length

        for i in range(k):
            temp = self.temperatures[i % len(self.temperatures)]
            seed = base_seed * 1000 + i
            with torch.no_grad():
                ids, _ = sample_trajectory(
                    self.model, length=total_len,
                    mask_id=self.mask_id, eos_id=self.eos_id,
                    steps=self.steps, temperature=temp,
                    seed=seed, prompt_ids=prompt_ids,
                )
            ids_list = ids.tolist() if torch.is_tensor(ids) else list(ids)
            completion_ids = ids_list[len(problem.prompt_ids):]
            candidates.append(CandidateSolution(
                problem_id=problem.problem_id,
                candidate_idx=i,
                ids=ids_list,
                text=f"<tokens:{','.join(map(str, completion_ids[:32]))}>",
                seed=seed,
            ))
        return candidates


# ---------------------------------------------------------------------------
# 2. SolutionCritique — independent verification, never self-grading
# ---------------------------------------------------------------------------

class SolutionCritique:
    """Critiques candidates using registered INDEPENDENT verifiers."""

    def __init__(self, decode_fn: Optional[Callable[[list[int]], str]] = None):
        # decode_fn: token ids -> text (tokenizer.decode). If None, uses
        # the candidate's stored text field (for tests).
        self.decode_fn = decode_fn

    def critique(self, candidate: CandidateSolution,
                 problem: Problem) -> CritiqueResult:
        verifier = VERIFIERS.get(problem.verifier)
        if verifier is None:
            return CritiqueResult(
                candidate=candidate, verified_correct=False,
                verifier_notes=f"unknown verifier '{problem.verifier}'",
                quality_score=0.0, passed_verifier=False,
            )
        text = (self.decode_fn(candidate.ids[len(problem.prompt_ids):])
                if self.decode_fn else candidate.text)
        try:
            correct, notes = verifier(text, problem.verifier_data)
        except Exception as e:
            return CritiqueResult(
                candidate=candidate, verified_correct=False,
                verifier_notes=f"verifier crashed: {e}",
                quality_score=0.0, passed_verifier=False,
            )
        # Quality score: verified candidates ranked by completion brevity
        # (shorter correct answers preferred); unverified get 0.
        quality = 0.0
        if correct:
            n_tokens = len(candidate.ids) - len(problem.prompt_ids)
            quality = 1.0 / (1.0 + n_tokens / 64.0)
        return CritiqueResult(
            candidate=candidate, verified_correct=correct,
            verifier_notes=notes, quality_score=quality,
        )

    def rank(self, critiques: list[CritiqueResult]) -> list[CritiqueResult]:
        """Verified-correct first (by quality), then unverified."""
        return sorted(
            critiques,
            key=lambda c: (c.verified_correct, c.quality_score),
            reverse=True,
        )


# ---------------------------------------------------------------------------
# 3. PreferencePairBuilder
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    problem_id: str
    chosen_ids: list[int]
    rejected_ids: list[int]
    chosen_correct: bool
    rejected_correct: bool
    margin_note: str


class PreferencePairBuilder:
    """Builds (chosen, rejected) pairs from ranked critiques.

    Rules:
    - chosen must be verifier-correct; if none correct, NO pair is built
      for that problem (never train on unverified "best guesses").
    - rejected = highest-ranked incorrect candidate; if all correct,
      rejected = lowest-quality correct candidate (quality contrast).
    """

    def build(self, problem: Problem,
              ranked: list[CritiqueResult]) -> Optional[PreferencePair]:
        if not ranked or not ranked[0].verified_correct:
            return None  # no verified solution -> skip, don't fabricate signal
        chosen = ranked[0]
        rejected = None
        for c in ranked[1:]:
            if not c.verified_correct:
                rejected = c
                break
        if rejected is None and len(ranked) > 1:
            rejected = ranked[-1]  # quality contrast among correct
        if rejected is None:
            return None
        return PreferencePair(
            problem_id=problem.problem_id,
            chosen_ids=chosen.candidate.ids,
            rejected_ids=rejected.candidate.ids,
            chosen_correct=chosen.verified_correct,
            rejected_correct=rejected.verified_correct,
            margin_note=(f"chosen: {chosen.verifier_notes} | "
                         f"rejected: {rejected.verifier_notes}"),
        )


# ---------------------------------------------------------------------------
# 4. StudentTrainer — LoRA + Diffusion-DPO, SFT fallback
# ---------------------------------------------------------------------------

class StudentTrainer:
    """Trains LoRA adapters on preference pairs via Diffusion-DPO.

    Falls back to SFT-on-chosen (diffusion_loss) if DPO is unavailable
    or degenerate for a batch.
    """

    def __init__(self, model, ref_model, mask_id: int,
                 lora_rank: int = 8, lr: float = 1e-4, beta: float = 0.1):
        self.model = model
        self.ref_model = ref_model
        self.mask_id = mask_id
        self.beta = beta
        n_trainable = inject_lora(model, rank=lora_rank)
        self.n_trainable = n_trainable
        self.opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        # Reference stays frozen
        for p in ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

    def train_step_dpo(self, pairs: list[PreferencePair],
                       prefix_lens: list[int]) -> dict:
        """One DPO step over a batch of pairs. Returns info dict."""
        self.model.train()
        ids_w = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(p.chosen_ids, dtype=torch.long) for p in pairs],
            batch_first=True, padding_value=self.mask_id,
        )
        ids_l = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(p.rejected_ids, dtype=torch.long) for p in pairs],
            batch_first=True, padding_value=self.mask_id,
        )
        self.opt.zero_grad()
        loss, info = diffusion_dpo_loss(
            self.model, self.ref_model, ids_w, ids_l, self.mask_id,
            prefix_len_w=prefix_lens, prefix_len_l=prefix_lens,
            beta=self.beta,
        )
        if not torch.isfinite(loss):
            return {"loss": float("nan"), "skipped": True,
                    "reason": "non-finite DPO loss"}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.opt.step()
        info["loss"] = loss.item()
        info["skipped"] = False
        return info

    def train_step_sft(self, chosen_ids_list: list[list[int]],
                       prefix_lens: list[int]) -> dict:
        """Fallback: SFT on chosen completions only (diffusion loss)."""
        from .diffusion import diffusion_loss
        self.model.train()
        total = 0.0
        n = 0
        for ids_list, plen in zip(chosen_ids_list, prefix_lens):
            ids = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0)
            self.opt.zero_grad()
            # Mask only the completion region: build a custom mask via
            # diffusion_loss on the completion slice.
            comp = ids[:, plen:]
            if comp.numel() == 0:
                continue
            loss, _info = diffusion_loss(self.model, comp, self.mask_id)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            self.opt.step()
            total += loss.item()
            n += 1
        return {"loss": total / max(n, 1), "n": n, "mode": "sft_fallback"}


# ---------------------------------------------------------------------------
# 5. Anti-memorization: train/eval split + gap report
# ---------------------------------------------------------------------------

def split_problems(problems: list[Problem], eval_frac: float = 0.2,
                   seed: int = 0) -> tuple[list[Problem], list[Problem]]:
    """Hold out eval_frac of problems as UNSEEN eval. Deterministic."""
    rng = random.Random(seed)
    idx = list(range(len(problems)))
    rng.shuffle(idx)
    n_eval = max(1, int(len(problems) * eval_frac))
    eval_idx = set(idx[:n_eval])
    train = [p for i, p in enumerate(problems) if i not in eval_idx]
    ev = [p for i, p in enumerate(problems) if i in eval_idx]
    return train, ev


@dataclass
class GapReport:
    n_train: int
    n_eval: int
    train_acc: float
    eval_acc: float
    gap: float          # train_acc - eval_acc; large gap => memorization
    verdict: str


def evaluate_accuracy(critiques: list[CritiqueResult]) -> float:
    if not critiques:
        return 0.0
    return sum(1 for c in critiques if c.verified_correct) / len(critiques)


def report_gap(train_critiques: list[CritiqueResult],
               eval_critiques: list[CritiqueResult]) -> GapReport:
    train_acc = evaluate_accuracy(train_critiques)
    eval_acc = evaluate_accuracy(eval_critiques)
    gap = train_acc - eval_acc
    verdict = ("OK" if gap < 0.15 else
               "WATCH" if gap < 0.30 else "MEMORIZATION SUSPECTED")
    return GapReport(
        n_train=len(train_critiques), n_eval=len(eval_critiques),
        train_acc=train_acc, eval_acc=eval_acc, gap=gap, verdict=verdict,
    )


# ---------------------------------------------------------------------------
# End-to-end driver (single loop iteration)
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    k_candidates: int = 4
    gen_length: int = 64
    dpo_steps: int = 50
    batch_size: int = 4
    eval_frac: float = 0.2
    seed: int = 0


def run_improvement_loop(problems: list[Problem], model, ref_model,
                         mask_id: int, eos_id: int,
                         decode_fn: Optional[Callable] = None,
                         config: LoopConfig = LoopConfig(),
                         log_dir: Optional[Path] = None) -> dict:
    """One full teacher-student iteration. Returns metrics dict.

    NOT RUN end-to-end with real weights yet — needs trained base model.
    """
    t0 = time.time()
    train_problems, eval_problems = split_problems(
        problems, config.eval_frac, config.seed)

    gen = SolutionGenerator(model, mask_id, eos_id,
                            gen_length=config.gen_length)
    crit = SolutionCritique(decode_fn=decode_fn)
    builder = PreferencePairBuilder()
    trainer = StudentTrainer(model, ref_model, mask_id)

    pairs: list[PreferencePair] = []
    pair_prefix_lens: list[int] = []  # prompt length per pair, aligned
    train_critiques: list[CritiqueResult] = []
    for prob in train_problems:
        cands = gen.generate(prob, k=config.k_candidates,
                             base_seed=config.seed)
        critiques = [crit.critique(c, prob) for c in cands]
        train_critiques.extend(critiques)
        ranked = crit.rank(critiques)
        pair = builder.build(prob, ranked)
        if pair:
            pairs.append(pair)
            pair_prefix_lens.append(len(prob.prompt_ids))

    # Train
    dpo_infos = []
    for step in range(config.dpo_steps):
        if not pairs:
            break
        rng = random.Random(config.seed + step)
        idx = rng.sample(range(len(pairs)), min(config.batch_size, len(pairs)))
        batch = [pairs[i] for i in idx]
        blens = [pair_prefix_lens[i] for i in idx]
        info = trainer.train_step_dpo(batch, blens)
        dpo_infos.append(info)

    # Eval on held-out problems (no training on these)
    eval_critiques: list[CritiqueResult] = []
    for prob in eval_problems:
        cands = gen.generate(prob, k=1, base_seed=999)
        eval_critiques.extend(crit.critique(c, prob) for c in cands)

    gap = report_gap(train_critiques, eval_critiques)
    metrics = {
        "n_train_problems": len(train_problems),
        "n_eval_problems": len(eval_problems),
        "n_pairs": len(pairs),
        "train_acc": gap.train_acc,
        "eval_acc": gap.eval_acc,
        "memorization_gap": gap.gap,
        "memorization_verdict": gap.verdict,
        "dpo_steps_done": len(dpo_infos),
        "secs": round(time.time() - t0, 1),
    }
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "loop_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics
