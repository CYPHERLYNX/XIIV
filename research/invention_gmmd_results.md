# Guarantee-MMD Post-Training — Pilot Results (2026-10-22)

**Candidate:** post-train a diffusion LM with an added MMD term pulling the
sampler's output distribution (frozen-model feature space) toward the
verifier-accepted region — training objective = intervention rate, not
perplexity. Queued from the 2026-10-21 Part-A pick.

**Two independent questions, two independent answers:**
- **Novelty:** HOLDS — dedicated prior-art sweep
  (`~/workspace/goals/xiiv-133m-training-run/hidden_files/research_scout_2026-10-22_mmd_priorart.md`,
  10 items, all arXiv IDs verified). Every *pair* of the three elements
  exists (MMD+diffusion-LM training: Yandex arXiv:2610.06648, D-MMD
  arXiv:2603.20155; verifier+diffusion-LM training: Seed Diffusion
  arXiv:2508.02193, dInfer arXiv:2510.08666), but the **triple combination
  (MMD training objective + diffusion LM + target = verifier-accepted
  region) has no prior art**. Open ground.
- **Mechanism (this probe):** PILOT FAIL at toy scale — the MMD term is
  mechanically active but does not significantly reduce the intervention
  rate. Per the pre-registered verdict clause: **do NOT advance the
  mechanism claim without a stronger testbed.** Filed status:
  *novelty HOLDS / mechanism unproven* — the same honest split as
  Invention #1 (gate 1 PASS signal / gate 2 FAIL payoff).

## Method

`research/invention_gmmd_probe.py` (CPU-only, toy scale; mirrors the
gate-1/2 harness). Tiny XIIV (d=64, 2 layers), template data, 250 brief
training steps, family-A constraints, 48-token canvas, frozen anchors
every 12 positions.

- **Phase 0:** confirmed the calibrated family on 30 fresh drafts: base
  intervention rate **0.63** — inside the pre-registered [0.15, 0.85] band.
- **Phase 1:** froze the brief-trained model; 512 drafts labeled by the
  external verifier → **134 accepted / 378 rejected** (accept 0.262).
  Accepted drafts' features (mean-pooled final_norm hidden states over
  non-frozen positions, standardized) form the reference bank.
- **Phase 2:** 4 arms, loss = CE(masked LM) + λ·MMD_RBF(feats(own fresh
  drafts), bank), 300 steps each, λ ∈ {0.0, 0.3, 1.0, 3.0}. Sampling stays
  outside the autograd graph; the MMD gradient flows through the model's
  representation of its own samples (distribution-level self-alignment).
  No structural enforcement anywhere — verifier stays external.
- **Phase 3:** held-out evaluation, n=200 fresh-seed free drafts per arm,
  paired seeds across arms. Guardrails: diversity (unique-token ratio
  within 20% of control) and held-out CE (within 2× of control).

## Results

| arm (λ) | intervention rate | McNemar vs control | p |
|---|---|---|---|
| 0.0 (control) | 0.985 | — | — |
| 0.3 | 0.980 | b01=3, b10=2 | 1.0000 |
| 1.0 | 0.985 | b01=2, b10=2 | 0.6171 |
| 3.0 | 0.970 | b01=5, b10=2 | 0.4497 |

- **Verdict: PILOT FAIL** — no λ arm significantly beats the control
  (best p=0.45). Guardrails held (diversity 0.614–0.697 vs 0.659 control;
  held-out CE 3.58–3.68 vs 3.67 control) — the MMD term did not win by
  collapsing the generator.
- The MMD term was **mechanically active**: final-batch MMD distance fell
  with λ (0.1258 → 0.0804 → 0.0268 → 0.0194). Feature-space pull happened;
  it did not translate to the discrete constraint outcome.

## Honest diagnosis

1. **The testbed's dominant effect was drift, not MMD.** The pilot base
   rate was 0.63, but after post-training *every* arm — including the
   pure-CE control — sat at ~0.97–0.985. Continued CE training sharpens
   the distribution toward the most-frequent tokens, and the forbidden
   token *is* the 2nd-most-frequent template token: sharpening increases
   forbidden-token violations. The constraint-satisfying behavior of the
   brief-trained model was fragile; the probe measured the model's drift
   under continued training more than the MMD term's effect.
2. **Same bottleneck as gate 2 and CSM Phase C.** Distribution-level
   nudges (allocation policy, routing, MMD pull) don't fix a generator
   whose competence under constraints is the binding constraint.
   Feature-space alignment ≠ discrete constraint satisfaction at toy
   scale with a barely-competent model.
3. **Memorization, not generalization.** Training-batch CE sat at
   ~0.5–0.7 while held-out-template CE was ~3.6 for all arms: the tiny
   model memorized the 10 training templates. A fairer testbed needs a
   base model whose constraint satisfaction is *stable* under continued
   training.

## What would be a fairer test

(a) A base model trained long enough (or constraint-aware from the start)
that its intervention rate is mid-band *after* post-training, so the MMD
term has headroom against a stable control; (b) per-input
accepted/rejected pairs rather than a marginal bank (tests whether the
pairwise signal beats the marginal one); (c) a model competent enough
that extra distributional pull can actually change discrete outcomes.

## Standing

- Novelty: HOLDS (open ground, dedicated sweep on file).
- Mechanism: PILOT FAIL at toy scale (this probe).
- Next: do NOT compose or file as a working invention until (a)–(c)
  above is run. The idea stays in the candidate queue behind a stronger
  testbed — same discipline as Invention #1.

*Run: 50.8 min wall-clock, CPU-only. Checkpoints:
`research/checkpoints_gmmd/arm_lam{0.0,0.3,1.0,3.0}.pt`. Full log:
`research/invention_gmmd_run.log`.*

## UPDATE (2026-10-22, later) — stronger testbed run

The (a)-(c) fairer testbed specified above was built and run:
research/gmmd_testbed2_probe.py / gmmd_testbed2_results.md.
Stability gate passed (RSFT control 0.290, no drift recurrence).
**Marginal MMD arm: 0.190 vs control 0.290, McNemar p=0.0175,
guardrails intact — PASS at toy scale.** Paired arm: FAIL
(0.320, p=0.519) at 75 pairs. Mechanism standing updated from
"unproven" to "toy-scale PASS (marginal); replication and
scale-up owed". This file's pilot record is unchanged history.
