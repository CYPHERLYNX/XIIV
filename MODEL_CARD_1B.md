# XIIV-1B Model Card

## Model Details

- **Name:** XIIV-1B
- **Architecture:** Masked discrete diffusion (NOT autoregressive)
- **Parameters:** 1,019,230,720 (1.02B)
- **Lineage:** Grown from XIIV-500M (single lineage: 133M → 500M → 1B)
- **Training status:** GROWN, NOT TRAINED — grown initialization only

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

This is the differentiator frontier labs haven't reached.

## Growth Details

- **Source:** `checkpoints_500m_gn/ckpt_8218.pt` (503M params)
- **Method:** Net2Net-style function-preserving growth
  - Width: 1024 → 1280 (new dims replicate, 1% noise for symmetry breaking)
  - Depth: 27 → 36 layers (new blocks as exact identity)
  - Heads: 16 → 20 (new heads replicate head 0)
- **Drift gate:** < 0.6 ratio (grown vs random-init) — REQUIRED
- **Seed:** 0

## Intended Use

**This checkpoint is NOT ready for use.** It requires training on real data
(FineWeb-Edu) on GPU hardware before it can generate coherent text.

## Training Requirements

- **Hardware:** GPU with ≥6GB VRAM (RTX 4050 or better)
- **Data:** FineWeb-Edu (99M tokens, real data only — no TinyStories)
- **Config:** BF16, 8-bit Adam, gradient checkpointing
- **LR:** 1.65e-4 (inverse-width scaled from 133M reference)
- **Steps:** 8218 → 48218 (40k training steps)

## Limitations (Honest)

- NOT trained — output will be incoherent until trained
- NOT instruction-tuned — no chat/Q&A ability
- NOT safety-aligned — no RLHF or safety training
- 99M training tokens is small for 1B params (frontier models use 100B+)
- This is a research artifact, not a production model

## Lineage

```
XIIV-133M (133M, trained on stories, step 8218)
    → XIIV-500M (503M, grown, embeddings initialized)
        → XIIV-1B (1.02B, grown, NOT trained)
```

Single grown lineage. No random re-initialization. No cherry-picked checkpoints.
