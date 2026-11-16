# Phase 2 — Learned Composition Policy (2026-11-16, step 1: supervised warmup)

**Spec:** `research/composition_operator_v1.md` §6 — "Learning the composition
policy (from specification to primitive sequence) becomes the next problem."
Options named there: supervised on (spec, sequence) pairs, RL with
certificate-validity as a dense reward, or LLM-guided synthesis with the
certificate as a correctness oracle. This step is the first option.

**Status:** training (25 epochs, CPU). Results below are filled from
`results/eval_step1_*.json` once the run finishes — nothing here is claimed
before measurement.

## Part-A grounding (2026-11-16 screen)

The recipe shape comes from three November-2026 papers, all reported from
search-result abstracts, not read papers; their numbers are the papers' own
claims, not XIIV measurements:

- **TIPS** (arXiv:2610.22099): a ~600K-param tiny inference-time policy that
  learns unmasking + remasking around a *frozen* backbone, trained by
  supervised warmup followed by RL from terminal correctness rewards. Our
  policy is the same scale (~669K params) and the same two-stage shape:
  supervised warmup now (this step), RL from certificate-validity reward
  next. The difference is the oracle: TIPS's reward is terminal task
  correctness; ours is the independent certificate verifier + exact symbolic
  target equality.
- **What Makes a Good Unmasking Policy?** (arXiv:2610.31576): supervised
  policies are distillation of a decoding procedure through a teacher; the
  winning supervision is *trajectory-contraction targets* — accept only
  commitments that stay on a reference trajectory. Our supervision is
  exactly that: the deterministic VRC planner's plans are the reference
  trajectory, and the policy is only ever shown plans that stay on it.
- **Verifier-Grounded Agentic RL for Provably Correct Code**
  (arXiv:2610.19298): a verifier-as-reward must be *dense* (weighted
  fraction of proof obligations discharged, not binary accept/reject) and
  needs an *integrity audit that outranks the verifier* (reject any
  candidate that alters the spec or assumes the conclusion). Adopted as the
  design for the RL stage: dense reward = fraction of certificate
  obligations discharged; integrity audit = exact symbolic target equality.
- **Verified but Vacuous** (arXiv:2611.05590, IBM): 99.1% verifier
  acceptance, only 1.9% non-vacuous — RL against a binary verifier reward
  learns to eliminate proof obligations. This is why our evaluation's
  honest metric is `verified_success` (certificate verifies AND concrete ==
  symbolic AND final value equals the target), never certificate validity
  alone. A policy that "verifies" a trivial plan that misses the target
  scores 0 here.

Also screened and shelved (needs a generating base first; XIIV-1B is 0/5
generation at 20 evaluated checkpoints): PURL (2611.11551v2, unmasking +
revision joint learning), PBPO (2611.07783, dLLM RL), Learning Unmasking
Policies (2512.09106, Apple GRPO unmasking policy), Left-to-Right Training
(2601.15165). See `research/daily_2026-11-16.md`.

## What this step builds

| File | Role |
|---|---|
| `gen_dataset.py` | Random affine specs -> VRC planner plans (8,000 train / 1,500 test, seed 20261116). Filters counted, never silent: plan-rejected, plan-too-long (>16), probe-out-of-bound, duplicate. |
| `policy.py` | Tiny encoder-decoder transformer, d=128, 2+2 layers, 668,683 params. Char-level spec input, op-level output. Greedy decode. |
| `train_policy.py` | Supervised imitation, Adam 1e-3, cosine, batch 64, 25 epochs, CPU. Saves `checkpoints/policy_step1.pt` + `results/train_log.json`. |
| `eval_policy.py` | PRE-REGISTERED evaluation (see below). |
| `test_phase2.py` | 7 tests, all green (dataset determinism, closure rejection, tokenizer roundtrip, model shapes + greedy termination, gold-plan verification oracle). |

## Pre-registered evaluation (eval_policy.py docstring)

1. **exact_match** — predicted ops == planner's ops.
2. **verified_success** (the honest metric) — the predicted sequence SOLVES
   the spec: ProofMachine execution on 5 probes with concrete == symbolic,
   independent certificate verifies, AND final symbolic value == realized
   target at every probe. `alt_valid` = verified but not exact (an
   alternative valid composition — genuine generalization beyond imitation).
3. **invalid** — empty predicted plan.
4. Reported overall AND by planner-plan-length bucket {1-3, 4-6, 7-9, 10-12,
   13-16}. The deterministic planner is 100% on both metrics by
   construction — the reference trajectory, not a competitor. The learned
   policy is judged on generalization, not on beating it.

## Incidental fix (same run): `canon` mod-precedence bug

`certificates.canon` rendered `mod(add(x0,3),7)` as `"(x0+3 mod 7)"`, but
the VRC parser binds `mod` tighter than `+` (term level), so that string
parses as `x0 + (3 mod 7)` — a different expression. ~20% of generated
specs round-tripped to a different target. Fixed by parenthesizing the mod
argument unless atomic (`"((x0+3) mod 7)"`); regression test
`test_canon_parse_roundtrip` asserts `parse(canon(e)) == e` on 300 random
exprs. All 4 Phase-1 suites (`test_vrc`, `test_conditional`,
`test_logic`, `test_noise`) still pass — `canon` is display-only everywhere
(certificate strings, verifier diagnostics).

## Results (measured 2026-11-16, pre-registered protocol)

Training: 25 epochs, 28.5 min CPU, final train loss 0.107, val token
accuracy 0.975 (500 held-out). Eval: n=1500 held-out specs, greedy decode,
`results/eval_step1_20261116_015457.json`.

| Metric | Overall |
|---|---|
| exact_match (== planner plan) | **81.9%** |
| verified_success (solves the spec: executes, certificate verifies, target equality) | **83.3%** |
| alt_valid (verified but not exact — alternative valid composition) | 1.4% |
| invalid (empty plan) | 0.0% |

By planner-plan-length bucket (exact / verified): 1–3: 90.9% / 95.5%
(n=22); 4–6: 90.9% / 92.4% (n=540); 7–9: 87.8% / 88.7% (n=556); 10–12:
71.3% / 72.4% (n=254); 13–16: 38.3% / 41.4% (n=128).

All 250 failures are `target-mismatch`: the predicted plan executes and its
certificate verifies, but it computes a different function than the target.
Zero `neural-divergence`, zero `cert-invalid`, zero empty plans. The oracle
held — no vacuous certificate shipped; target equality caught every failure.

**Reading (honest):** supervised imitation of the planner works well to
~9 ops, then degrades steeply (41% at 13–16 ops). The learned policy shows
its own compositionality wall — the same shape Phase 1 measured for the
transformer baseline, at a much higher level (83% vs the baseline's 1.5% at
its long end; not directly comparable — different depth scales — but the
shape is the signal). The deterministic planner stays 100% by construction
at every length; the learned policy is judged on generalization, not on
beating it. The 1.4% alt_valid shows the policy sometimes finds genuinely
different valid plans — the seed the RL stage builds on. Scope note:
verified_success compares values at 5 probes (5, -3, 42, 1000, -1000), not
full symbolic equality — a plan differing from the target only outside the
probe range would pass; probes span ±1000 so this is a bounded risk, and
the stricter symbolic check would wrongly reject alt_valid plans.

## Next

- RL stage (spec §6 option 2): policy-gradient on the trained warmup
  policy with the dense certificate-obligation reward + target-equality
  integrity audit (per 2610.19298), using the VRC harness as the
  environment. Not started.
- Conditional synthesis policy (`synthesize_conditional` targets) — same
  recipe, IFZ programs. Not started.
- Integration question (DECISION.md): how the policy meets the diffusion
  1B once generation exists. Untimed — the 1B has no generation to compose
  yet.
