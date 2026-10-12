# XIIV-1B — Architecture Specification (rung 3, planned)

**Status:** Spec frozen 2026-10-11. Growth not yet executed. Training not yet started.

**Parent:** XIIV-500M grown checkpoint `checkpoints_500m_gn/ckpt_8218.pt` (503.45M unique trainable params; 537.01M serialized elements — the serializer duplicates the tied embed/lm_head matrix; see "Parameter counting convention" below).

**Method:** Net2Net-style single-lineage growth via `scripts/grow_checkpoint.py` (same method as 133M→500M). No random initialization. The guarantee-native thesis and masked discrete-diffusion architecture are preserved.

## Architecture

| | 500M parent | 1B target |
|---|---|---|
| d_model | 1024 | **1280** |
| layers | 27 | **36** |
| heads (head_dim 64) | 16 | **20** |
| ffn_dim | 2816 | **3584** |
| vocab | 32768 | 32768 (unchanged) |
| params (unique trainable) | 503.45M | **1019.23M** |
| params (serialized, tied dup) | 537.01M | 1061.17M |

Growth ratios: width 1.25×, depth 1.33×, FFN 1.27×. Head dim preserved at 64 (1280/20). FFN/d ratio preserved at ~2.8.

Parameter count verified by instantiating `XIIVModel` with this exact config on a meta device and summing `model.parameters()` (deduplicates tied weights): **1,019,230,720** unique trainable parameters — a true 1B model. Serialized form is 1,061,173,760 elements (the tied 32768×1280 embedding matrix appears twice in `state_dict()`). The 1B estimate lands in the 0.9–1.1B target band on the unique-trainable definition.

### Parameter counting convention

XIIV publishes **unique trainable parameters** (tied weights counted once). This matches standard practice (e.g., GPT-2 124M, Llama counts). Rationale, verified 2026-10-11 on the 500M checkpoint:
- `embed.weight` and `lm_head.weight` share the same storage (`data_ptr()` identical, values identical) — truly one parameter matrix used twice.
- Naive `state_dict()` summation gives 537.01M (500M) / 1061.17M (1B) — inflated by exactly one embedding matrix (33.55M / 41.94M).
- Model cards report both numbers; the headline figure is always unique trainable.

## Growth procedure (frozen)

Same Net2Net operators as the 133M→500M growth (documented in MODEL_CARD_500M.md):

- **Width** (1024→1280): replicate-then-split with 1/d count normalization (exact for linears)
- **Heads** (16→20): new heads = mean of old heads + 1% noise (seed 0)
- **FFN** (2816→3584): new rows = mean of old rows + 1% noise (seed 0)
- **Depth** (27→36): 9 new blocks as exact residual identities
- **Contract attention** (q, kv, out projections): same width-growth operator as main attention
- **Contract encoder**: width-grown with the main model
- **Vocab**: unchanged (32768) — no vocab growth needed this rung
- **Embeddings/lm_head**: width-grown (1024→1280 dim)

## Drift gate (pre-registered)

Same relative gate as the 500M growth (`tests/test_grow.py`):

- grown mean|logit drift| vs random-init mean|drift|, ratio must be < 0.6
- 500M achieved 0.113; 1B must pass the same bar
- Absolute drift will be larger than 500M's 2.95 nats (bigger growth step); the relative gate is the criterion

## Training plan (post-growth)

- **Hardware:** RTX 5090 (32GB VRAM). All 4050-era 6GB estimates are obsolete.
- **Data:** Real data only. FineWeb-Edu (99.3M tokens, `data_fineweb/`) as base. TinyStories (`data_32k/`) is EXCLUDED from all future training per 2026-10-10 directive.
- **LR:** Inverse-width rule from the 133M reference (3e-4 × 704/1280) = **1.65e-4**, cosine decay
- **Batch:** To be profiled on the 5090 (VRAM headroom is large; exact micro-batch × accum TBD by smoke test)
- **Precision:** BF16 (to be validated for finite loss/gradients on the 5090)
- **Corpus gaps:** Code (StarCoder blocked on `hf auth login`), books, math/STEM, instruction data — all needed before the 1B run is truly "real data"

## What this rung does NOT include

- No training has been run at 1B. This spec is the architecture and growth plan only.
- The 24,934 untrained 32k-vocab embeddings from the 500M growth remain untrained until real-data training begins.
- The diffusion benchmark scorer audit (methodology caveat) is separate work, still pending.
- Comparison table and improvement plan are separate deliverables, in progress.

## Lineage

```
XIIV-133M @ step 8218 (133.52M, trained on TinyStories)
  → grown → XIIV-500M @ step 8218 (537.01M, grown init, not trained)
    → grown → XIIV-1B @ step 8218 (1019.23M, planned)
```

Single lineage preserved. No random initialization at any rung.
