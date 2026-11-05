# 2026-11-04 research: SSD + EB-Sampler -> SCV

## Findings

### Self Speculative Decoding for dLLMs (Gao et al., SJTU, arXiv:2510.04147)
The dLLM itself as both speculative drafter and verifier, via hierarchical
verification trees in a single forward pass -- lossless w.r.t. stepwise
decoding, up to 3.46x speedup on LLaDA and Dream, no auxiliary draft model.
Diagnosis absorbed: speculation is a valid acceleration axis for masked
diffusion (draft-then-verify works without an extra model). Mechanism NOT
ported: the verifier is the model grading its own drafts (circular under
the Forge doctrine -- the judge must not be the model grading itself), and
"lossless" is measured against stepwise decoding, which carries no
certificate at all.

### EB-Sampler (Ben-Hamu et al., Meta, Aug 2026)
Entropy-bounded unmasking: decides *which* tokens and *how many* to unmask
per step, using an entropy bound that approximately limits inter-token
dependence, under a predefined error tolerance. 2-3x on coding/math
reasoning benchmarks without quality loss.
Diagnosis absorbed: a parallel commit batch needs a principled budget, and
the budget should bound the *joint* risk of the batch, not per-position
confidence. Mechanism NOT ported: the bound is entropy-native (the model's
own uncertainty, circular again) and it is an error *tolerance*, not a
guarantee.

### LESS (Mohamed et al., arXiv:2606.16908, Jun 2026)
Mutual-stability sampling: commit = online stopping problem; a position is
eligible only when top-1 is confident AND persists across recent reverse
steps AND the predictive distribution is stable under top-K inter-step
JS divergence. 72.1% fewer reverse steps on Dream-7B / LLaDA-8B / 1.5-8B.
Not used as the invention base: its stability rule overlaps the family's
existing CDR (certificate-drift remasking) and PTCS (verifier-side
settling) axes -- no fresh axis, and its criteria are model-native.

## Fit assessment for XIIV
SSD and EB-Sampler both fit the diffusion + guarantee-native thesis and
are training-free decoding strategies -- implementable and verifiable on
this CPU-only box. Neither is adopted as a port:
- From SSD we take the *draft-then-verify* shape, but the verifier is the
  INDEPENDENT certificate verifier (canvas-only, no model), never the
  model itself.
- From EB-Sampler we take the *batch-size-as-a-bound* idea, but the bound
  is not an entropy error tolerance -- it is a hard certificate gate: the
  batch that commits is exactly the longest verifier-endorsed prefix of
  the speculation. The verifier, not a confidence heuristic, decides how
  many positions commit per forward pass.

## What was invented (xiiv/scv.py, this commit)
SCV -- Speculative Certificate Verification:
1. One forward pass per round; the frontier (top-`frontier` masked
   positions by top-1 probability) is drafted at depth `draft_depth`
   (top-k token candidates per position) -- the speculation.
2. Position-major prefix walk: positions in confidence order; each
   position's candidates tried in rank order against the INDEPENDENT
   verifier (`dcmr._would_repeat` on the working canvas -- canvas-only,
   no model). The first certificate-safe candidate is accepted.
3. A position whose whole candidate list is verifier-rejected is
   speculation-exhausted: the round stops (prefix property -- the tail
   waits for denser context next round). No fixed per-step budget; the
   batch size is whatever prefix the verifier endorses, up to the
   frontier -- a certificate-native dynamic batch size.
4. Fail-closed: a canvas that is not fully unmasked, or that violates the
   no-repeat certificate, is never presented as certified
   (completed=False), unless the breach pre-existed in frozen text.
Novelty claim (honest): draft-then-verify exists (SSD) and per-position
candidate fallback exists (PTCS's gate); the combination of an
*independent-verifier-arbitrated* speculation with *prefix commit* and
*verifier-decided batch size* is new, and it is guarantee-preserving by
construction -- every accepted token was certificate-endorsed at accept
time. All claims measured on CPU scripted stubs (7/7 tests);
real-checkpoint behavior untested, stated in the module docstring.
