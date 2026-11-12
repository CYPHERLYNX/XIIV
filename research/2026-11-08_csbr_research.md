# 2026-11-08 research: RDD + ProSeCo + BackPlay -> CSBR

## Findings

### RDD — Reversible Diffusion Decoding (arXiv:2602.00150, Feb 2026)
Block-wise dLLM decoding degrades quality, and irreversible commitments
can lead to stagnation: the reverse process fails to make further
progress under a suboptimal context ("stagnation trap"). RDD treats
stagnation as a state-dependent failure and backtracks to earlier blocks
without recomputation via cached model states; confidence-guided
re-masking selectively reinitializes uncertain tokens while preserving
reliable context; a cap rmax on rollbacks, then forced decoding.
Diagnosis absorbed: a stalled canvas is a failure of the *state*, not of
the step — the fix is to move backwards, not to push harder forwards.
Mechanism NOT ported: rollback to cached *model states* (a serving
optimization, not a guarantee), and confidence-guided re-masking
(confidence is not a certificate). XIIV's Nov-7 note already queued the
certificate-native version of this idea for a later sampler — today is
that sampler.

### ProSeCo — Progressive Self-Correction (arXiv:2602.11590, NeurIPS 2026)
Trains a model to both unmask and correct: a corrector loss term added
to the standard MDM objective, trained on the denoiser's own outputs as
corrupted inputs; at inference, corrective refinement steps interleave
with unmasking steps and can change already-decoded tokens. Up to ~4x
faster sampling at equal quality, up to ~1.2x benchmark gain with
inference-time compute scaling; SFT on LLaDA-8B beats vanilla MDM SFT.
Diagnosis absorbed: already-decoded tokens are the error-accumulation
surface, and revisiting them is where the quality lives. Mechanism NOT
ported: it needs corrector *training* (a second objective, GPU-scale
SFT on an 8B model) — no CPU-only path, and a trained corrector's
revisions carry no certificate. CSBR gets the revisiting without the
training: the verifier, not a learned head, decides what re-opens.

### BackPlay — Head-Only Look-Back Self-Correction (arXiv:2601.06428)
Frozen backbone + a lightweight correction head trained on errors from
the same frozen generator; Look-back Correction injects predictions
from earlier, more corrupted denoising states into later, richer
contexts; at inference, previously generated tokens are periodically
revisited through selective remasking and regeneration. Improves the
speed-quality trade-off on math reasoning and code.
Diagnosis absorbed: later context is the instrument that detects
earlier mistakes — a commit that looked fine at step t can be exposed
by step t+k. Mechanism NOT ported: the correction head (trained,
uncertified) and periodic revisiting on a schedule (revisits happen
whether or not anything is wrong). CSBR revisits only on stagnation,
and re-opens by certificate margin (lowest commit probability in the
snapshot), not by a learned detector.

### Also noted (watch tier, not adopted)
- Scout run 58 (06:02 IST today): no new arXiv batch (next Mon Nov 9);
  GPT-6.2 Astra still unreleased; Qwen 4 "next week" window closes
  today unfulfilled; prior-art watch clean; standing adoptable shelf
  unchanged (NCM, LoopMDM, ALoDLM+SPOT, T2Quant, softmax/CE fix).
- FlashProve contrast (Nov 7) still holds: certificates prove content
  properties from the token list alone; no 70x overhead, no weights.

## The pick: CSBR — Certificate-Snapshot Backtracking Rollback

RDD's diagnosis (stagnation is a state failure; move backwards) +
ProSeCo/BackPlay's diagnosis (already-decoded tokens are the error
surface; later context exposes earlier mistakes) + XIIV's
guarantee-native thesis, with a mechanism that is neither a port of
RDD nor a variant of CDR/CBRS/CCCA/CWTC:

- **Snapshots only at certificate-clean states.** After every step
  that commits >= 1 token, the canvas is re-checked by the independent
  verifier; only a passing canvas is pushed on the snapshot stack
  (bounded, last 8). A rollback can therefore only ever restore a
  canvas the verifier already accepted — rollback never touches a
  certificate-verified commit except by re-opening it, and any
  re-commit must re-pass the gate.
- **Stagnation trigger on certificate progress, not confidence.**
  `stall_window` consecutive steps with zero commits (all remaining
  candidates below threshold, or all gate/taboo-blocked) is the trap
  signal — the same failure RDD names, detected on what the sampler
  actually achieved.
- **Rollback with monotone taboo (VRC doctrine).** Restore the most
  recent snapshot; every (position, token) committed *after* that
  snapshot is tabooed, and the `reopen` lowest-margin committed
  positions inside the snapshot are re-masked with their tokens
  tabooed too. Taboo only ever grows, so a rollback can never replay
  the trajectory that stalled — this is RDD's "avoid repeated failure
  trajectories" goal, enforced by a guarantee-native mechanism rather
  than by confidence heuristics.
- **Escalation by popping.** The restored snapshot is consumed, so a
  second stall rolls back one snapshot further. Termination: monotone
  taboo growth + `max_rollbacks` cap + `max_steps` cap; fail-closed
  (completed=False, masks remaining) when the stack empties or the
  cap is hit.
- **The certificate is the audit trail.** The csbr-v1 certificate
  itemizes every rollback (step, snapshot step, taboo added, positions
  re-opened, reason) alongside every commit. A third party replays
  the claim from the token list alone.

Why this is the most promising of today's findings: it is the only
one that closes the loop the family has been building toward — every
earlier sampler commits monotonically and can only stall or give up;
CSBR is the first that can *retreat* without surrendering the
guarantee, and the retreat itself is certified. It is also the queued
shelf item from Nov 7, so the design debt is already paid.
