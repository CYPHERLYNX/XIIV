# XIIV-1B Model Card

## Model Details

- **Name:** XIIV-1B
- **Architecture:** Masked discrete diffusion (NOT autoregressive)
- **Parameters:** 915,988,480 unique trainable (957,931,520 serialized elements, 406 tensors) — 0.92B. Verified 2026-10-13 by reading `checkpoints_1b_gn/ckpt_8218.safetensors` header and tensor values (`embed.weight` ≡ `lm_head.weight`, value-identical; all weights finite).
- **Lineage:** Grown from XIIV-500M (single lineage: 133M → 500M → 1B)
- **Training status:** IN TRAINING — grown from the XIIV-500M checkpoint; block-wise training (SGBD) in progress. Blocks 0–4 of 8 complete (1,000 steps each, 622.5M parameters trained); blocks 5–7 pending. The assembled 1B has not yet been evaluated end to end.

### Architecture Spec

| Component | Value |
|-----------|-------|
| Hidden dim (d) | 1280 |
| Layers | 36 |
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
  - Depth: 27 → 36 layers (new blocks as exact identity)
  - Heads: 16 → 20 (new heads replicate head 0)
- **Drift gate:** < 0.6 ratio (grown vs random-init) — REQUIRED
- **Seed:** 0

## Intended Use

**This checkpoint is a research artifact, not a production model.** Block
training is still in progress (see Training status above), and the assembled
1B has not been evaluated. It should not be used for downstream tasks.

## Training Requirements

- **Hardware:** GPU with ≥6GB VRAM (RTX 4050 or better)
- **Data:** FineWeb-Edu (99M tokens, real data only — no TinyStories)
- **Config:** BF16, 8-bit Adam, gradient checkpointing
- **LR:** 1.65e-4 (inverse-width scaled from 133M reference)
- **Steps:** 8218 → 48218 (40k training steps)

## Limitations

- Training incomplete — blocks 5–7 untrained; assembled-model quality is unmeasured
- Not instruction-tuned — no chat template; interaction is via contracts and prompts
- Not safety-aligned — no RLHF or safety training has been applied
- Training corpus is small relative to frontier practice for a 1B model
- Context length is 128 tokens in the current configuration

## Lineage

```
XIIV-133M (133M, trained on stories, step 8218)
    → XIIV-500M (503M, grown, embeddings initialized)
        → XIIV-1B (0.92B, grown, NOT trained)
```

Single grown lineage. No random re-initialization. No cherry-picked checkpoints.
