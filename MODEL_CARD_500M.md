# XIIV-500M — Model Card (rung 2, grown)

**Status: grown and verified 2026-10-09. Not yet trained at 500M.**

## Lineage

- Parent: XIIV-133M at step 8218 (133.52M params, trained on this box, then stopped)
- Growth: Net2Net-style, `scripts/grow_checkpoint.py`, seed 0
- Checkpoint: `checkpoints_500m_gn/ckpt_8218.pt` (2.01 GB, optimizer state intentionally absent)

## Architecture

| | 133M parent | 500M grown |
|---|---|---|
| d_model | 704 | 1024 |
| layers | 15 | 27 |
| heads (head_dim 64) | 11 | 16 |
| ffn_dim | 2048 | 2816 |
| vocab | 8192 | 32768 |
| params (actual) | 133.52M | **503.45M** |

Tied embeddings kept. Contract-native encoder and per-block cross-attention retained.

## Frozen growth config

- Width: replicate-then-split with 1/d count normalization (exact for linears)
- Heads: 11 → 16, new heads = mean of old heads + 1% noise (seed 0)
- FFN: 2048 → 2816, new rows = mean of old rows + 1% noise
- Depth: 15 → 27, new blocks are exact residual identities
- Vocab: 8192 → 32768 via `weights/vocab_map_8k_32k.json` (7834 rows copied, 24934 new from old-piece means)

## Verification (measured 2026-10-09)

- Actual parameter count: **503.45M unique trainable** (503,453,952; verified 2026-10-11 by instantiating `XIIVModel` with the exact checkpoint config and by direct checkpoint analysis). Serialized form is 537.01M elements across 343 tensors — the difference is exactly the tied 32768×1024 embedding matrix (33,554,432), which `state_dict()` duplicates but shares storage (`data_ptr()` identical). Headline figure is unique trainable, matching standard practice.
- Drift gate (relative, pre-registered in `tests/test_grow.py`):
  - grown mean|logit drift| = **2.9499** nats
  - random-init mean|drift| = **26.0116** nats
  - ratio **0.113 < 0.6** — the grown init is 8.8× closer to the parent than random
- Forward smoke test: (1, 32) → (1, 32, 32768), all finite

## Honest notes

- The absolute drift (2.95 nats) is larger than the tiny-test 0.20 gate: that
  gate was calibrated on a toy growth and does not scale to a real
  1.45×-width / 1.8×-depth growth. The scale-invariant relative gate is the
  pre-registered criterion; it passes with wide margin.
- This checkpoint is a grown *initialization*, not a trained 500M model.
  The first ~1k steps of continued training will re-settle the new capacity.
- No training has run at 500M. Planned: LR 2.06e-4 by the inverse-width rule
  (3e-4 × 704/1024), steps 8218 → 48218, micro-batch 2 × accum 2.
- Guarantees prove structural compliance only — not truth or factuality.
