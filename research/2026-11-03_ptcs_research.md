# 2026-11-03 research: PC-Sampler + SureLock -> PTCS

## Findings

### PC-Sampler (Huang et al., arXiv:2508.13021, v4 Oct 2026)
Position-Aware Confidence-Calibrated Sampling for masked diffusion models.
Diagnoses two weaknesses in uncertainty-based samplers that match our
sampler family exactly:
- (a) **no global trajectory control** -- DCMR/CGAU/CGEC/CDR all decide
  *which* position to commit inside a *fixed* outer schedule (constant
  budget/step, fixed cap). Nothing plans the trajectory.
- (b) **bias toward trivial tokens early in decoding** -- uncertainty
  samplers burn early steps on easy tokens.
Proposes position-aware weighting + calibrated confidence; reports +10%
average over existing MDM decoding strategies on 7 benchmarks, narrowing
the gap to AR models of comparable size.

### SureLock (Oba et al., ICLR 2026)
Stopping computation for converged tokens in masked diffusion-LM decoding:
when a token's posterior stabilizes (local-KL "sure condition"), lock the
position -- skip its query projection + FFN, cache K/V for others to
attend to. 30-50% algorithmic FLOP reduction on LLaDA-8B at comparable
quality, with a theoretical bound on final-token deviation.

## Fit assessment for XIIV
Both fit the diffusion + guarantee-native thesis, and both are
training-free decoding strategies -- implementable and verifiable on this
CPU-only box. Neither is adopted as a port:
- PC-Sampler's mechanism (learned calibration, confidence objective) is
  confidence-native; XIIV's substrate is certificate-native. We port the
  *diagnosis* (trajectory control + trivial-token bias) and re-derive the
  mechanism with the planning objective = certificate acceptance under the
  independent verifier.
- SureLock's convergence criterion (the model's own local KL) is circular
  under the Forge doctrine (the judge must not be the model grading
  itself). We port the *lock-and-skip* idea with a verifier-side criterion:
  settle only when the independent certificate margin holds for a streak.

## What was invented (xiiv/ptcs.py, this commit)
PTCS -- Planned-Trajectory Certificate Scheduler:
1. Calibrated informativeness with a decaying trivial-token discount
   (trivial commits late, never stranded).
2. A global budget planner reading the per-step informativeness
   distribution (expand on confident clusters, contract to 1 when nothing
   is commit-worthy, else finish-on-time base budget).
3. Verifier-side settling with minimal-repair un-settling (F4 documents the
   semantics: one remask dissolves the repeat; the verifier does not
   over-repair).
4. Pre-commit certificate gate with next-best fallback (no stranded
   positions -- the naive gate's stall, caught during test design).
All claims measured on CPU scripted stubs (13/13 tests); real-checkpoint
behavior untested, stated in the module docstring.
