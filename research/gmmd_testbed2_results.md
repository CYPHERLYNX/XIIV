# Guarantee-MMD — Stronger Testbed Results (2026-10-22)

Pre-registered in research/gmmd_testbed2_probe.py before the run.
Confounds from the pilot (invention_gmmd_results.md) removed by
construction: (a) all arms post-train on the base model's own
ACCEPTED drafts (RSFT) instead of raw data, so CE reinforces rather
than fights the constraint — with a registered stability gate
(control >= 0.90 intervention = testbed invalid); (b) per-input
accepted/rejected pairs from identical canvases (75 discordant
pairs collected) enable a paired-MMD arm beside the pilot-style
marginal arm; (c) MMD features standardized with bank statistics
on BOTH sides (the pilot standardized the bank only). Runtime 188s.

## Results (eval: n=200 paired-seed free drafts per arm)

| arm | intervention rate | vs base |
|---|---|---|
| base (pre-post-training) | 0.720 | — |
| rsft control | 0.290 | −0.430 |
| **mmd_marginal (λ=1.0)** | **0.190** | −0.530 |
| mmd_paired (λ=1.0) | 0.320 | −0.400 |

- Stability gate: PASSED (control 0.290, nowhere near the 0.90
  invalidation line). The drift confound is gone: RSFT alone cuts
  intervention by 43 points — itself a strong toy-scale result.
- **mmd_marginal vs rsft: McNemar b01=42, b10=22, p=0.0175,
  guardrails intact (diversity 0.781 vs 0.789; held-out CE 3.897
  vs 3.840) → pre-registered PASS.** The marginal MMD term adds a
  significant −10pp (−34% relative) on top of RSFT. Guarantee-MMD's
  mechanism claim advances one notch: from "unproven" to
  **toy-scale PASS under a confound-controlled testbed**.
- mmd_paired vs rsft: b01=27, b10=33, p=0.519 → **FAIL**. The
  per-input paired target does not beat control as implemented.
  Plausible mechanism: 75 pairs is a thin target set — batch-32
  draws recycle the same mates constantly, making the paired
  target noisy and over-local, while the marginal bank is a smooth
  distribution-level target. The pair idea is not refuted at
  adequate pair counts; this implementation at this scale is.

## Standing after this run

- Novelty: HOLDS (unchanged; dedicated sweep on file).
- Mechanism: marginal Guarantee-MMD — toy-scale PASS (this run,
  pre-registered, confounds controlled). Paired variant — FAIL
  at 75 pairs.
- Owed before any stronger claim: replication across seed families,
  a λ sweep, pair-count scaling for the paired arm, and a run at
  SGBD-block scale when the trained 1B exists. The toy PASS licenses
  exactly that follow-up program, not a production claim.
