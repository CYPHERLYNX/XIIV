# Selective-looping prototype — results (2026-10-22)

Implements the mechanism from Looped Uniform Diffusion (arXiv:2610.30066)
and LoopMDM (arXiv:2605.26106) on XIIV's stack and probes their central
claim (same val NLL for up to ~3x fewer training FLOPs — authors'
numbers, single-lab, unverified) at toy scale.

## What shipped

- `XIIVModel.forward(..., loop_group=(lo, hi), loop_count=k)`: re-applies
  a contiguous layer group k extra times (recurrent depth, fixed params);
  callers can vary the count per call (the papers' count-sampling).
  Default path bit-identical. `selective_loop_flop_ratio` analytic helper.
  Grad-checkpoint path supported. Tests: tests/test_looping.py, 6/6
  (manual-reapplication equivalence, default-path identity, gradient flow).
- Probe: research/selective_looping_probe.py (+ results JSON). Three arms,
  identical init/data order: base (ratio 1.0), fixed loop (0,2)x1 (1.5),
  sampled counts {0:.25, 1:.5, 2:.25} (expected 1.5). Tiny model
  (d=64, L=4), template corpus, 300 steps, val NLL every 50 steps.

## Measured (honest)

Val NLL by step — base: 4.520 @50 → 5.256 @300; fixed: 4.515 @50 →
5.083 @300; sampled: 4.552 @50 → 4.672 @300.

- In the non-overfit window (steps 50–100) the arms are within ~0.02
  NLL of each other — no separation.
- After ~step 100 ALL arms' val NLL rises: the tiny model overfits the
  1,024-sequence corpus. Looped arms end lower at step 300, but that
  gap tracks differential overfitting, not demonstrated FLOP
  efficiency; the naive iso-pass comparison (looped@200 vs base@300)
  is an artifact of exactly that and is NOT claimed.

Verdict: **claim NOT reproduced at toy scale; ledger factor stays
UNBANKED.** The mechanism is in place and tested, so a fair test is now
cheap: a corpus large enough to stay out of the overfit regime, finer
eval granularity, and a matched-layer-pass budget — queued against a
real SGBD block budget when the Kaggle lane opens (blocks 5–7 / 5B T1).
