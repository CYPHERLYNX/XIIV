# Invention #1 — Gate 2 Results (2026-10-21)

**Invention #1:** proof-difficulty-adaptive diffusion sampling — allocate denoising compute by
predicted certificate-construction difficulty. Prior art: `invention01_prior_art.md`
(NOVEL conditionally). Gate 1 (2026-10-20): PASS — signal distinct from confidence
(Spearman 0.123; confident-but-hard quadrant 10.0%).

**Gate 2 question (pre-registered):** does allocating refinement steps by certificate difficulty
actually reduce the intervention rate vs confidence-based allocation at equal compute budget?

## Method

`research/invention01_gate2_probe.py`. Tiny XIIV (d=64, 2 layers, same recipe as gate 1),
template-structured data, 250 brief training steps, CPU-only. Per trial: free draft (no structural
enforcement — model generates, external `check_constraints` verifies; violations = interventions),
then 3 refinement arms on the SAME draft (paired design), identical budget (2 rounds × 15% remask),
identical mechanics (re-denoise, no enforcement — only the targeting signal differs):

- **random** — control
- **confidence** — remask lowest-confidence positions (mirrors `xiiv.diffusion.self_refine`;
  the CadLLM/SAID/CHLG-like baseline)
- **load** — INVENTION #1: remask highest `guarantee_load_priority` positions (static GLOD predictor)

**Protocol amendment (documented in the probe):** the pre-registered difficulty knob
(n_required ∈ {3,4,5} over uniform-random tokens) gave base rate 1.00 at every level —
unmeasurable. Recalibrated to a distribution-aware family (family A: required=[most-frequent
template token], forbidden=[2nd most-frequent], at_most={3rd: 3}; base rate 0.70 over 40 drafts),
re-confirmed on 30 fresh drafts (0.80, inside the pre-registered [0.15, 0.85] band). Calibration
used drafts only; arms never ran before calibration.

## Results (n=200 paired trials)

| arm | intervention rate |
|---|---|
| draft (no refinement) | 0.800 (pilot) |
| random | 0.730 |
| confidence | 0.735 |
| load (Invention #1) | 0.720 |

- **McNemar (load vs confidence, paired):** b01 (conf-fail/load-pass) = 7, b10 = 4,
  χ² = 0.36, **p = 0.5465** — not significant.
- Violation breakdown is near-identical across arms (forbidden_absent ≈ 134–136/200;
  required_present ≈ 24–25; caps_met ≈ 4–6).
- Predictor cost: 0.99 ms/trial vs 0.11 s/trial refinement — the static predictor is
  ~100× cheaper than the compute it directs (prior-art caveat 3's cost question: predictor
  cost is NOT the problem).
- Quadrant-concentration analysis was uninformative: all 200 trials landed above the
  quad ≥ 0.10 split (every draft contained confident-but-hard positions), so no
  high/low contrast was available. Reported as-is, not as evidence.

## Verdict: GATE 2 FAIL — payoff unproven at toy scale

Allocating refinement compute by (static) certificate difficulty does **not** significantly
reduce interventions vs confidence-based allocation (p = 0.55, Δ = 1.5pp on n=200 paired).
Per the pre-registered verdict clause: **do NOT advance the novelty claim on payoff grounds.**

## Honest diagnosis (why it failed — and what it does/doesn't imply)

1. **Refinement itself barely helps** (0.80 → 0.72–0.735 for ALL arms). The bottleneck is the
   model's generative competence on this constraint family, not the allocation policy:
   re-denoising with the same weak model mostly regenerates the same forbidden (high-frequency)
   tokens. When the repair operator cannot act, the targeting signal is irrelevant.
2. **This does NOT kill Invention #1.** Gate 2 tested the *static* GLOD predictor at toy scale
   with a barely-competent model. The gate-1 result (the signal exists and diverges from
   confidence exactly where guarantees are cheapest to lose) still stands.
3. **What would be a fairer test:** (a) a model competent enough that extra steps CAN fix
   violations (longer training / larger toy model); (b) refinement with teeth — verifier-guided
   resampling or structural enforcement during refinement, so the allocation decides *where* a
   capable repair operator works; (c) the learned difficulty head (Patch-Forcing-style) the
   prior-art report already flags as the mature form of the predictor.
4. Standing implication for the invention's file: status = **gate 1 PASS (signal distinct),
   gate 2 FAIL (payoff unproven)**. No payoff-based novelty claim until (a)–(c) is run.

## Relation to today's Part-A pick

The queued Guarantee-MMD post-training candidate (today's research note) is now *more*
interesting, not less: gate 2 suggests the payoff bottleneck is model competence under
constraints, and MMD-against-verifier-acceptance is a training-time (not allocation-time)
attack on exactly that bottleneck. The two ideas compose; neither is claimed as working yet.
