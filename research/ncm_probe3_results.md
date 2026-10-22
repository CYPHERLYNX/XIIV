# NCM Probe 3 — The Documented Fair Follow-Ups (2026-10-22)

Runs attacks #1, #2, #4 from ncm_probe2_results.md's closing list.
Script: research/ncm_probe3_followups.py (348s). Setup identical to
probes 1–2 by construction (imports probe 1; baselines re-run
in-process: flat 0.671, hard-C2F 0.650 @ concept 0.790, oracle 0.850 —
all within 0.001 of the recorded values).

## Headline (25% visible, 16 steps, token accuracy)

| variant | token acc | concept acc |
|---|---|---|
| flat | 0.671 | — |
| C2F hard (probe-1 recipe) | 0.650 | 0.790 |
| **A: matched-condition head training (#1)** | **0.678** | **0.825** |
| **B: end-to-end soft conditioning (#2)** | **0.681** | **0.827** |
| oracle concepts | 0.850 | 1.000 |

Both fair fixes work. Concept accuracy rises 0.790 → 0.825/0.827,
and predicted-concept C2F crosses from net-NEGATIVE vs flat
(0.650 vs 0.671) to net-POSITIVE (0.678/0.681 vs 0.671) for the
first time in the probe series. A and B land within noise of each
other — two different fixes removing the same train/inference
mismatches (head trained under inference conditions; soft
conditioning trained, not bolted on).

## Visible-fraction sweep (#4) — pricing the evidence limit

| frac | flat | c2f | A | B | oracle | concept acc (A/B) |
|---|---|---|---|---|---|---|
| 0.125 | 0.476 | 0.461 | 0.466 | 0.479 | 0.849 | 0.570 / 0.591 |
| 0.250 | 0.671 | 0.650 | 0.678 | 0.681 | 0.850 | 0.825 / 0.827 |
| 0.500 | 0.819 | 0.806 | 0.827 | 0.828 | 0.852 | 0.979 / 0.976 |
| 0.750 | 0.851 | 0.847 | 0.851 | 0.850 | 0.854 | 0.997 / 0.996 |

The sweep is the sharpest result in the series. Oracle accuracy is
flat (~0.85, the 15% noise ceiling) at EVERY fraction — oracle
concepts carry information the canvas does not. Predicted accuracy
tracks concept accuracy, which tracks visible evidence. At 0.75
visible, every method converges to the ceiling and concepts add
nothing; at 0.125, no procedure recovers what's missing.

## Verdict

- The predicted→oracle gap (0.200 at 25% visible) decomposes as
  roughly **15% procedural / 85% evidence-limited**: the two fair
  fixes together recover ~0.03 of it (0.650 → 0.681), and the sweep
  shows the remainder is information the input does not contain.
  Probe 2's sharpened conclusion stands, now with the procedural
  share measured and collected rather than merely asserted.
- Attack #3 (multiple codes per block / residual code) remains
  untested — the only procedural lever left. The sweep bounds its
  plausible upside at this visibility (~0.03–0.04 token accuracy);
  it attacks error *damage* (one wrong code poisoning a block),
  not error *rate*. Deprioritized accordingly, not closed.
- For XIIV: coarse-to-fine conditioning is net-positive when the
  concept pathway is trained under its inference conditions — a
  design constraint, now measured, for any future concept-level
  machinery. Vendor NCM claims remain unaddressed by design (toy
  testbed, mechanism only).
