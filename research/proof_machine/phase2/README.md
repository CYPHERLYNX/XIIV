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

## Step 2 — GRPO with dense certificate-obligation reward (2026-11-17)

**Status:** DONE. 150 steps, 8.3 min CPU, from the step-1 checkpoint.
`rl_train.py` (GRPO, G=4, T=1.0, LR 1e-5, β_KL=0.02, no critic);
`reward.py` (dense reward = fraction of the 5 probes whose full
obligation is discharged — execute + certificate verifies + symbolic
result == realized target; target equality is the integrity audit, so a
verified-but-wrong plan scores ~0; no length term). Training specs
stratified compound-heavy (4–6: 15%, 7–9: 25%, 10–12: 30%, 13–16: 30%;
1–3 held out of RL) per the decomposed-to-composed asymmetry
(arXiv:2609.19465). Reward uses the pure symbolic path (cheap, exact);
full neural + certificate audits at steps 50/100/150: cert-valid 1.0
every time. Training signals: mean rollout reward 0.694 → 0.763
(first 10 vs last 10 steps), groups solved 0.825 → 0.878, KL ~0.05
flat, entropy 0.164 → 0.141.

**Pre-registered evaluation** (`results/eval_step2_20261117_013534.json`,
same protocol as step 1, n=1500 held-out, greedy):

| Metric | Step 1 | Step 2 | Δ |
|---|---|---|---|
| verified_success overall | 83.33% | **84.87%** | +1.54pp |
| verified 13–16 ops | 41.41% | **49.22%** | +7.81pp |
| verified 10–12 ops | 72.44% | 77.95% | +5.51pp |
| verified 4–6 ops | 92.41% | 91.85% | −0.56pp |
| verified 1–3 ops | 95.45% | 95.45% | 0.00pp |
| alt_valid | 1.40% | 2.53% | +1.13pp |

Failures 250 → 227, all `target-mismatch`; zero cert-invalid, zero
neural-divergence, zero empty plans. **Pre-registered verdict: PRIMARY
PASS, GUARDRAIL PASS** (both primary legs true; short buckets within
5pp). Honest reading: the gain is modest (+23 net successes on 1,500)
and neither leg alone clears a 95% significance bar — what makes it
credible is the length-monotone shape (0.0 / −0.6 / +0.4 / +5.5 /
+7.8pp from short to long), exactly the designed pattern, with no
atomic degradation. The 13–16 wall is dented (49%), not broken. Full
detail: `research/daily_2026-11-17.md`.

## Step 3 — longer GRPO run, 500 steps (2026-11-18): PLATEAU

**Status:** DONE. 500 steps, 28.1 min CPU, from the step-2 checkpoint
(`rl_train.py --tag step3` — the tag names the checkpoint
`policy_step3.pt` and the log `rl_log_step3.json`, so step 2's
artifacts are never clobbered; the step-2 log was renamed
`rl_log.json` → `rl_log_step2.json`). Same recipe as step 2 — the only
variable under test is the step count. `results/rl_log_step3.json`;
`results/eval_step3_20261118_014901.json`.

Training signals (logged, not evaluation): mean rollout reward
first-10 0.718 → last-10 0.745; groups solved 0.875 → 0.863; KL
0.059 → 0.066; entropy 0.149 → 0.134 — all flat. Audit cert-valid
1.0 at all 10 audits.

**Pre-registered evaluation** (same protocol, n=1500 held-out, greedy):

| Metric | Step 2 | Step 3 | Δ |
|---|---|---|---|
| verified_success overall | 84.87% | **85.47%** | +0.60pp |
| verified 13–16 ops | 49.22% | **49.22%** | 0.00pp |
| verified 10–12 ops | 77.95% | 78.35% | +0.40pp |
| verified 4–6 ops | 91.85% | 92.78% | +0.93pp |
| verified 1–3 ops | 95.45% | 95.45% | 0.00pp |
| alt_valid | 2.53% | 3.67% | +1.13pp |

Failures 227 → 218, all `target-mismatch`; zero cert-invalid, zero
neural-divergence, zero empty plans. **Pre-registered verdict:
PRIMARY FAIL — PLATEAU, GUARDRAIL PASS.** The overall leg is true
(+0.60pp) but z = 0.46, not significant; the 13–16 leg is false —
the bucket is literally identical (63/128 verified in both runs).
Step 2's long-bucket gain did not compound with 350 more GRPO steps.
Honest reading: response-level GRPO saturates at ~85% overall /
~49% at 13–16; more steps of the same recipe buy nothing at the long
end. The wall is a capability wall for this reward/policy scale, not a
training-time wall. One real signal inside the flat top line:
exact_match fell (−0.53pp) while verified rose and alt_valid rose —
RL moves the policy toward alternative valid compositions, not toward
more correct long plans. Full detail: `research/daily_2026-11-18.md`.

## Next

- **NOT a still-longer run of the same recipe** — step 3 closed that
  direction (plateau, pre-registered).
- Token-level credit assignment (arXiv:2608.27906v3) as a step-4
  refinement — the reward is response-level today and long plans fail
  late. Not started.
- Conditional synthesis policy (`synthesize_conditional` targets) —
  same recipe, IFZ programs. Not started.
- Integration question (DECISION.md): how the policy meets the diffusion
  1B once generation exists. Untimed — the 1B has no generation to compose
  yet.
