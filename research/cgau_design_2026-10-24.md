# CGAU — Certificate-Gated Adaptive Unmasking (design + v1 results, 2026-10-24)

**Status:** implemented (`xiiv/cgau.py`), 10/10 unit tests green
(`tests/test_cgau.py`), falsifier suite PASS on all pre-registered checks
(`research/cgau_results_2026-10-24.json`). Potential XIIV invention (see
novelty accounting below).

## What it is

A proof-certificate-gated sampler for masked diffusion LMs. At each reverse
step, unmask proposals are gated on the existing `partial_certificate_check`
(the sound-fail prefix check from the CGMS fourth gate): a proposal commits
only if the partial canvas remains extendable to a fully certified output.
Three mechanisms sit on top of the gate:

1. **Certificate-margin early exit** — a masked position whose top-1 is
   certifiable AND (model-confident OR the only certifiable choice among the
   top-k) commits immediately, skipping further denoising passes for that
   position. Adaptive compute driven by *proof margin*, not model confidence.
2. **Certified-state recovery** — a cache of the last certifiable canvas;
   after `stall_budget` consecutive zero-commit steps, roll back to it and
   taboo the tokens committed since (LAVE's recovery pattern, lifted from
   CFG syntax to proof state).
3. **Targeted re-mask backtracking (last resort)** — when nothing in the
   top-k is certifiable at a position (hard stall), taboo the top-1; only on
   *persistent* stall with a *canvas-guilty* reason (`forbidden_present` /
   `cap_exceeded` naming committed tokens) re-mask the lowest-confidence
   holder and taboo that token 3x longer (so the early exit can't re-commit
   it and churn — diagnosed and fixed 2026-10-24).

Shared sampler behavior (not CGAU mechanisms): forbidden-token logit
suppression and tight-regime slot reservation, mirroring DenoisingState.

## Honest novelty accounting

- The per-placement certificate gate is **CGMS's** (in-repo,
  `xiiv/diffusion.py::_cgms_commit`, 4th gate). CGAU reuses the same
  `partial_certificate_check`, it does not re-invent gating.
- The lookahead + verified-prefix-cache + recovery *pattern* is **LAVE's**
  (arXiv:2602.00612) — absorbed as prior-art inspiration; CGAU's recovery
  operates on semantic proof state, not CFG syntax.
- **LookUM** (arXiv:2511.05563, ICML 2026) does verifier-based *path
  selection* with a sequence-certainty verifier: no rollback, no proof
  certificates, no re-masking. Different mechanism, same problem family.
- **UnMaskFork** assumes monotonic unmasking (never re-masks); CGAU's
  backtracking explicitly breaks that assumption.
- **RL unmasking policies** (arXiv:2512.09106) learn the policy from a
  correctness+compute reward; CGAU is training-free and certificate-driven.

The claimable novel surface is the **triad at the semantic-certificate
level**: certificate-margin early exit + proof-state rollback recovery +
reason-targeted re-mask backtracking, in one sampler. No prior art found
(2026-10-24 sweep) combines these.

## Falsifier suite (pre-registered before the run)

60 synthetic contract instances (vocab 32, length 24; required/forbidden/
at-most/order/anchor constraints), deterministic adversarial stub LM
("confident but wrong" +4-logit spikes on forbidden/capped tokens at 45%
of positions — the failure mode gating exists for). Three policies:

| policy | final validity | fwd passes | cert-fail rate |
|---|---|---|---|
| vanilla (confidence only) | 0.050 | 12.00 | — (no gate) |
| gated (CGMS-4th-gate spirit: per-placement gate + top-k scan) | 1.000 | 12.42 | 0.189 |
| **cgau** | **1.000** | **10.43** | 0.201 |

Pre-registered checks: **P1** validity ladder ✓ (1.000 ≥ 1.000 > 0.050);
**P2** gate fires ✓; **P3** cgau compute ≤ gated ✓ (**1.19x fewer forward
passes**, 12.42 → 10.43); **P4** mechanisms fire ✓ (230 early exits);
**KILL** (cgau ≤ vanilla) not triggered. **Verdict: PASS.**

Spike-rate sensitivity: at spike 0.15, cgau 12.62 vs gated 12.65 (parity);
at 0.30, 13.37 vs 12.75; at 0.45, 14.28 vs 12.87. The early-exit win is
largest when violations are sparse; under heavy adversarial pressure the
gated scan is cheaper. Rollback and tier-2 backtracking never fired on the
suite (no hard stalls) — they are unit-tested safety nets, value unproven.

## Two design bugs found and fixed during the run (honest log)

1. **Zero-probability commits.** `_topk_for_pos` kept 0.0-prob entries, so
   the top-k scan committed filler tokens the model assigned zero
   probability (exposed by scripted-stub tests). Fixed: strictly-positive
   filter. This was a real sampler bug, not a test artifact.
2. **Early-exit vs backtrack churn.** An earlier commit-top-1-then-backtrack
   design fought the early exit (79 remasks, run never completed on a
   pathological stub). Fixed: budgeted pass scans top-k like the gated
   core; backtracking is last-resort only, with 3x victim taboo.

## What this does NOT prove

- Toy scale only (24-token canvas, stub LM). Transfer to the real XIIV-1B
  unmasking loop is unproven; gate-eval cost vs forward-pass cost will
  differ at scale.
- The 1.19x is a sampler-efficiency micro-factor, banked in the 100x ledger
  with the toy-scale caveat — not a capability multiplier.
- Rollback's real-world value is untested (never fired on the suite).

## Next

- Wire CGAU as a `DenoisingState` stepping policy (currently standalone) so
  it can run against the real model when blocks 5–7 finish.
- Adversarial stall-regime suite where rollback is the *only* way out
  (persistent model blindness, not just spikes).
- Real-model A/B on contract tasks once the 1B assembles.
