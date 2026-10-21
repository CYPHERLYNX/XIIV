# NCM Prototype Probe — Results (2026-10-20)

**Question:** Does coarse-to-fine generation (block-level "concept" codes
first, tokens conditioned on them — the NCM idea, arXiv:2610.14162)
improve masked-diffusion denoising? Scout-recommended prototype probe:
`research/ncm_prototype_probe.py`.

**Setup (toy scale, synthetic):** sequences len 32, 4 blocks of 8, each
block drawn from one of 8 random templates + 15% token noise. Tiny
2-layer transformer (d=128). Flat baseline vs C2F (concept embedding +
coarse head). Eval = inpainting: 25% of tokens visible, fill the rest.
500 training steps, CPU.

## Results

| Variant | Inpainting accuracy |
|---|---|
| Flat masked diffusion | 0.672 |
| C2F, predicted concepts (79.0% concept acc) | 0.650 |
| C2F, ORACLE concepts (true block ids) | **0.850** |

Data noise ceiling ≈ 0.85 (15% of tokens are random replacements).
Train loss (last-50 avg): flat 1.178 vs C2F 1.132.

## Verdict

- **The mechanism is real.** With correct concepts, accuracy rises from
  0.672 to 0.850 — the entire achievable gap, closed. Global structure
  first, details second works exactly as the coarse-to-fine thesis says.
- **Naive hard concept conditioning fails.** At 79% concept accuracy,
  predicted concepts are net NEGATIVE (−2.2pp vs flat): one wrong
  concept misleads its whole block, outweighing the gains on correct
  blocks. Argmax concepts with no uncertainty propagation are not
  viable.
- **The bottleneck is precisely located:** concept inference quality.
  Predicted-to-oracle headroom is 20pp. Nothing else in the pipeline
  is the constraint.

## Next probe (sharpened, not yet run)

Iterative concept refinement: concepts and tokens co-evolve —
denoise a few steps → re-infer concepts from the partially-resolved
canvas → continue. This maps naturally onto diffusion steps and
directly attacks the located bottleneck. Soft concept conditioning
(probability-weighted concept embeddings instead of argmax) is the
second variant to test.

## Honest scope

Toy data, one seed family, inpainting (not unconditional generation).
This validates the *mechanism* and locates its bottleneck; it says
nothing about NCM's published benchmark claims (10–31× inference,
4–5× fewer training tokens), which remain single-lab and unreplicated.
