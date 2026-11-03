# XIIV-1B Model Card

## Model Details

- **Name:** XIIV-1B
- **Architecture:** Masked discrete diffusion (NOT autoregressive)
- **Parameters:** 1,000,000,000 headline — 857,429,760 runnable parameters in the checkpoint file (embeddings 41,943,040 + 40 blocks × 20,318,720 + final norm 1,280 + E-block 2,736,640) plus 142,570,240 contract scaffold that was designed but **excluded from the file by design**. 284 tensors, all fp32, zero non-finite. Stitch audited 2026-10-28 (whole-file SHA-256 `6948350434842af66fb0f7a0d1e02e9f2834958d6760a3023bc058bf32ba52aa`). Full counting in `research/expansion_exact_1b.md`.
- **Lineage:** Grown from XIIV-500M (single lineage: 133M → 500M → 1B exact)
- **Training status:** EXACT-1B STITCHED AND AUDITED 2026-10-28. Phase B (new params only, FineWeb-Edu): held-out 12.048 → 8.401 (−30.3%). Full pretraining trial: 17,715,200 FineWeb-Edu tokens, all 857,429,760 runnable params, held-out 8.141 → 6.091 (−25.2%). **Generation gates 0/5 at every checkpoint** — the model does not yet generate factual text; this is an experimental research artifact, not a usable model.

### Architecture Spec

| Component | Value |
|-----------|-------|
| Hidden dim (d) | 1280 |
| Layers | 40 (36 dense + 4 grown) |
| Attention heads | 20 |
| FFN dim (SwiGLU) | 3584 |
| Vocabulary | 32768 |
| Context length | 128 |

### Guarantee-Native Design

XIIV is built guarantee-native from the ground up:
- Generation under proof (contract-native compute)
- Deterministic bricks with proof-carrying certificates
- Independent verification (not self-grading)
- Intervention rate as the core metric

Verification is performed by code independent of the model, and the
intervention rate (verifier interventions per generation) is tracked as a
primary quality metric.

## Growth Details

- **Source:** `checkpoints_500m_gn/ckpt_8218.pt` (503M params)
- **Method:** Net2Net-style function-preserving growth
  - Width: 1024 → 1280 (new dims replicate, 1% noise for symmetry breaking)
  - Depth: 27 → 40 layers (36 dense + 4 grown as exact identity)
  - Heads: 16 → 20 (new heads replicate head 0)
- **Drift gate:** < 0.6 ratio (grown vs random-init) — REQUIRED
- **Seed:** 0
- **Exact-1B expansion:** 4 new SGBD blocks (layers 36–39) + 1 residual E-block after layer 39, grown identity-init; only the 84,011,520 new params trained (Phase B). See `research/expansion_exact_1b.md`.

## Intended Use

**This checkpoint is a research artifact, not a production model.** It has
not passed its generation tests (0/5 gates at every checkpoint). It should
not be used for downstream tasks.

## Training Requirements

- **Hardware:** GPU with ≥6GB VRAM (RTX 4050 or better)
- **Data:** FineWeb-Edu (99M tokens, real data only — no TinyStories)
- **Config:** fp32, 8-bit Adam, gradient checkpointing
- **LR:** 1e-5 (pretrain probe winner)
- **Trial:** 8,650 steps, 17,715,200 tokens (Kaggle T4, 12h session)

## Limitations

- Generation quality unproven — 0/5 on factual generation gates at every checkpoint; loss reductions have not translated into generation ability
- Not instruction-tuned — no chat template; interaction is via contracts and prompts
- Not safety-aligned — no RLHF or safety training has been applied
- Training corpus is small relative to frontier practice for a 1B model (17.7M tokens in the full trial)
- Context length is 128 tokens in the current configuration

## Lineage

```
XIIV-133M (133M, trained on stories, step 8218)
    → XIIV-500M (503M, grown, embeddings initialized)
        → XIIV-1B assembly (915,988,480 headline, Oct 2026)
            → XIIV-1B exact (1,000,000,000 headline, 857,429,760 runnable, stitched + audited 2026-10-28)
```

Single grown lineage. No random re-initialization. No cherry-picked checkpoints.
