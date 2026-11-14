# Compositional Proof Machine — Phase-1 Decision: PASS

**Date:** 2026-11-14
**Spec:** `research/composition_operator_v1.md`, §5.7 Weeks 5–6 deliverable (analysis, robustness tasks, write-up, decision: PASS or KILL).
**Decision: PASS.** The kill gate was honored literally and nothing triggered it.

## Evidence (all measured, pre-registered metrics)

| Gate | Result file | Outcome |
|---|---|---|
| Primary — certified arithmetic chains, depths 1→100 × 1000 instances | `results/killgate_20261017_013853.json`, `results/verdict_2026-10-17.json` | **PASS.** A_PM = 100% at every depth (1, 2, 4, 8, 16, 32, 64, 100); V = 100% at every depth; slope b = 0.0 (t = 0.00, flat); A_TF(100) = 1.5%; separation **98.5pp** ≥ 40pp. All three pass criteria true, zero kill triggers. |
| §5.6 task 1 — logic chains (boolean primitives on bit-vectors) | `results/killgate_logic_20261020_034921.json` | **PASS.** A_BOOL(100) = 100%, V = 100%, slope 0.0, A_TF(100) = 1%, separation **99pp**. Zero kill triggers. |
| §5.6 task 2 — conditional composition (IFZERO branching) | `results/killgate_cond_20261019_025556.json` | **PASS (PM-only).** A_CIF(100) = 100%, V = 100%, slope 0.0, 0 loud halts. Transformer-baseline separation QUEUED — not part of this PM-only gate (baseline needs retraining on the conditional program encoding). |
| §5.6 task 3 — noisy inputs (interval abstraction) | `results/killgate_noise_20261023_012645.json` | **PASS.** `flat_all_valid` + `cond_all_valid`; validity 1.0 at radii {0, 1, 3, 5}; n_flat 200, n_cond 120, seed 20261023. |

The transformer baseline degrades with depth in the expected way (58% → 1.5% from L=1 to L=100 on the primary gate) while the proof machine stays flat at 100% with mechanically verified certificates at every depth. That is the architectural claim the spec set out to test — exact composition beats approximate composition at depth — and it holds in this regime.

## What PASS does and does not mean (honest scope)

**It means:** the composition operator ∘, with its certificate-composition rule, scales flat to depth 100 with mechanically verified certificates in the toy regime. The kill gate was designed to kill this idea if it degraded like a transformer; it did not.

**It does not mean:**
- Toy regime only: ~1M parameters, Presburger arithmetic, 8 primitives, composition policy *given*, not learned (spec §6 non-goals). No natural-language proof-generation claim.
- Nothing about the 1B diffusion model's generation gap. The 1B scores 0/5 generation at 20 evaluated checkpoints — a separate problem whose binding constraint is token scale + LR (see `research/generation_gap_diagnosis_2026-11-13.md`). The proof machine does not fix that, and this PASS must not be read as progress on it.
- The conditional result is PM-only until the transformer baseline is retrained on the conditional encoding; the separation leg of that gate is still owed.
- Baseline caveat carried from the verdict JSON: the primary baseline is small (352K params, 10k steps, in-distribution accuracy modest at 58% at L=1) and had ~5x the proof machine's neural parameter count. The depth-degradation signature appears *within* training depths, so it reads as a compositionality wall rather than pure out-of-distribution failure — but a larger/longer baseline could narrow the gap.

## Next (Phase 2, per spec §6)

- **Learned composition policy** — discovering compositions rather than being given them (the spec's stated Phase-2 step; VRC harness scaffolding exists in `vrc_harness.py`, the learned-policy work itself is not yet run).
- **Richer logic fragments** beyond Presburger arithmetic, and a larger primitive library.
- **Integration question** — how the proof machine meets the diffusion 1B once generation exists. Untimed: the 1B has no generation to compose yet, so this stays a design question, not an implementation task.

No Phase-2 work is claimed here. This document closes Phase 1 only.
