# 2026-11-05 research: PACT + self_refine + change-envelope certs -> CBRS

## Findings

### PACT (arXiv:2611.04152, Nov 2026)
Training-free "certify the commit set" parallel-decoding rule for masked
diffusion LMs -- the first paper to frame a certificate over a *set* of
positions in dLLM decoding. Inference-only, token-reliability framing,
not a proof of generation correctness.
Diagnosis absorbed: a certificate can scope a *set*, not just a single
commit. Mechanism NOT ported: PACT certifies the commit set (what gets
written); it says nothing about localizing faults after the fact.

### self_refine (xiiv/diffusion.py, in-repo)
Diffusion-native revision: remask the lowest-confidence spans and
re-denoise in place. Model-confidence-driven, whole-span, non-localized.
Diagnosis absorbed: *rework* is a valid decoding shape for masked
diffusion (the model can revise its own canvas). Mechanism NOT ported:
the driver is the model's own confidence (circular under the Forge
doctrine), the rework region is a confidence span rather than a
verifier-localized fault, and there is no final-canvas certificate.

### "Where Did the Update Go?" (arXiv:2611.04192, Nov 2026)
Certifies declared change envelopes for model updates (decision-like
outputs only, no open-ended chat); policy cert, not generation proof.
Diagnosis absorbed: a *change envelope* (affected region + margin) is a
certifiable object. Mechanism NOT ported: the envelope is
declared-by-the-updater and the certified claim is update scoping, not
generation soundness.

## Fit assessment for XIIV
All three fit the diffusion + guarantee-native thesis and are
training-free decoding strategies -- implementable and verifiable on
this CPU-only box. None is adopted as a port:
- From PACT we take the *certify a set* shape, but invert the object:
  the certified set is the *repair* set (fault windows + halo),
  localized by the verifier itself, not a commit batch.
- From self_refine we take the *rework* shape, but replace the driver:
  not model confidence over spans, the independent verifier's fault
  localization -- and the rework is frozen-draft-local (untouched
  regions are never remasked).
- From the change-envelope paper we take the *envelope* shape (fault
  windows + halo margin), but the envelope is verifier-computed on the
  canvas, and the certified claim is generation soundness (final-canvas
  no-repeat), not update scoping.

## What was invented (xiiv/cbrs.py, this commit)
CBRS -- Certificate-Backed Repair Scheduling:
1. Phase 1 DRAFT: fast unverified speculation -- one forward pass per
   round, the frontier (top-`frontier` masked positions by top-1
   probability) commits its model top-1 outright. The draft carries NO
   certificate claim.
2. Phase 2 REPAIR: the INDEPENDENT verifier (`verify_canvas`,
   canvas-only, no model) localizes every certificate fault. The repair
   zone = union of fault-window positions + `halo` positions each side,
   frozen positions excluded. The zone is remasked; everything outside
   it is FROZEN for the rest of the run (locality -- a no-regress
   invariant by construction).
3. Repair rounds resample the zone with per-position candidate depth
   (`repair_depth`), accepting the first candidate that is
   certificate-safe against the working canvas. The zone is re-localized
   every round from a fresh verification.
4. Fail-closed: completed=True only when the canvas is fully unmasked
   AND the independent verifier confirms it clean, unless the breach
   pre-existed in frozen text (reported, not hidden). The cbrs-v1
   certificate is a claim about the FINAL canvas only.
Novelty claim (honest): draft-then-verify exists (SSD), per-position
candidate fallback exists (PTCS's gate), and span rework exists
(self_refine); the combination of an *unverified full draft* with
*verifier-localized, halo-enveloped repair scheduling* and *frozen-draft
locality* is new, and it is guarantee-preserving by construction -- the
certificate is recomputed on the final canvas by the independent
verifier, never trusted from the run. All claims measured on CPU
scripted stubs (8/8 tests); real-checkpoint behavior untested, stated in
the module docstring.
