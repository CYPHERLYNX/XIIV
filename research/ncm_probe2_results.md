# NCM Probe 2 — Iterative Refinement & Soft Concepts (2026-10-21)

**Question:** Probe 1 located the bottleneck precisely: coarse-to-fine
concept conditioning fails only because concept *inference* is weak
(79% concept accuracy; hard-predicted 0.650 vs oracle 0.850). Can
inference-time fixes close the predicted→oracle gap?

Script: `research/ncm_probe2_refinement.py`. **Setup identical to
probe 1 by construction** — the script imports probe 1's data
generator, model class and training function, so templates, seeds
(data seed 0, train seed 1234, eval seed 99), d=128, 2 layers, 500
training steps, bs=64, and the inpainting eval (25% visible, n=256)
all match exactly. Both new variants are inference-time procedures on
the same trained C2F model; baselines were re-run in-process under
the same weights for direct comparability. Primary budget: 16
denoise steps (8 also reported — accuracies were identical at both).

**RAM deviation (noted per constraints):** MemAvailable was
~0.4–0.8GB during the run (< the ~1GB shrink threshold). Footprint
was shrunk operationally — torch threads capped at 2, flat and C2F
models trained/evaluated sequentially and freed between phases —
while the comparability-critical config was NOT shrunk, since
changing d or training steps would void the probe-1 comparison.
Full run completed without OOM in 118s.

## Results (token accuracy, inpainting)

| Variant | acc @16 steps | acc @8 steps | Concept acc |
|---|---|---|---|
| Flat masked diffusion (re-run) | 0.671 | 0.670 | — |
| C2F hard argmax (probe 1 variant, re-run) | 0.651 | 0.651 | 0.790 |
| **(a) Iterative refinement, K=3** | **0.650** | **0.650** | 0.790 → 0.791 (round 1 → round 3) |
| **(b) Soft concept embeddings** | **0.644** | **0.644** | 0.790 (argmax of the distribution) |
| C2F ORACLE concepts (re-run) | 0.850 | 0.850 | 1.000 |

Train loss (last-50 avg): flat 1.178, C2F 1.132 — matching probe 1
exactly (1.178 / 1.132), confirming the reproduction is faithful.
Probe 1's original numbers were flat 0.672 / hard 0.650 / oracle
0.850; the re-runs land within 0.001 of them.

## Verdict — neither variant closes the gap. At all.

The predicted→oracle gap is 0.199 (0.651 → 0.850). Iterative
refinement closes **0.000** of it (0.650, −0.001 vs hard). Soft
embeddings close **none** either (0.644, −0.007 vs hard — slightly
worse). Both fixes fail, and the traces show why:

- **Refinement fails because re-inference learns nothing new.**
  Concept accuracy moves 0.790 → 0.791 across three rounds. The
  partially denoised canvas gives the coarse head no evidence beyond
  what the 25%-visible canvas already provided — the denoiser fills
  masked positions consistently with its current beliefs (wrong
  concept included), so re-inference largely re-reads the model's
  own commitments. Concept errors are self-confirming, not
  self-correcting, in this loop.
- **Soft conditioning fails on a train/test mismatch (a real
  confound, stated plainly).** The model was trained with hard
  one-hot (or "unknown") concept embeddings; a probability-weighted
  blend is a point in embedding space the trunk never saw in
  training. With a 79%-accurate, fairly confident coarse head, the
  blend also still injects a substantial wrong-template component.
  This run therefore kills *inference-only* soft conditioning on a
  hard-trained model — it does **not** fairly test soft conditioning
  trained end-to-end, which remains untested.

**Sharpened conclusion:** the bottleneck is not the conditioning
interface (hard vs soft vs iterated). It is that all usable concept
evidence lives in the visible tokens, and the coarse head already
extracts essentially all of it in a single pass. The remaining
concept errors are evidence-limited, not procedure-limited. That
also explains probe 1's oracle result: the 20pp headroom is real
but unreachable from the visible evidence alone — oracle concepts
smuggle in information the input does not contain.

## What would actually attack it (not run here)

- Train the coarse head on masked / partially-denoised canvases
  (it currently trains on full-canvas hidden states — a train/
  inference mismatch of its own), so inference conditions match.
- Train with soft or sampled concept conditioning end-to-end, making
  the soft variant a fair test.
- Richer concepts: multiple codes per block or a residual code, so
  one wrong code cannot poison a whole block.
- Accept more visible evidence: concept accuracy vs visible-fraction
  sweep would price the information limit directly.

## Honest scope

Toy data, one seed family, inpainting, inference-time variants on a
single trained model. This kills two specific fixes for the located
bottleneck and sharpens where the information limit sits; it says
nothing new about NCM's published benchmark claims, which remain
single-lab and unreplicated.
