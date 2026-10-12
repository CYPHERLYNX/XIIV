# XIIV-700M Model Card — guarantee-native rung

**Lineage:** grown directly from XIIV-133M (133.52M, d=704/L=15/h=11/ffn=2048)
at step 8218 via Net2Net-style growth. The 200M intermediate rung was
skipped per Fahad's call on 2026-10-08 ("make it to 700M with this model").
One lineage, never a from-scratch restart.

## Architecture

| field | 133M parent | 700M grown |
|---|---|---|
| params (exact) | 133,520,000 | 697,930,000 |
| d_model | 704 | 1152 |
| n_layers | 15 | 30 |
| n_heads | 11 | 18 (head_dim 64 preserved) |
| ffn_dim (SwiGLU) | 2048 | 3200 |
| vocab | 8192 (tied) | 32768 (tied) |
| contract encoder | d=256, L=2, h=4, ffn=512, max_tokens=96 | unchanged (grown proj only) |
| per-block contract cross-attention | yes | yes, widened |
| objective | masked diffusion, seq 128 | same |

## Growth method (`scripts/grow_checkpoint.py`)

- Width 704→1152: new dims replicate dims [0, 448) (whole heads);
  new heads replicate head 0; outgoing weights split by replication count.
- FFN 2048→3200: new intermediate rows replicate rows [0, 1152);
  `down` columns split by replication count (function-preserving).
- Depth 15→30: new blocks are exact identities (all output projections
  zeroed; pre-norm residuals pass through unchanged). Verified zero drift.
- Vocabulary 8192→32768 (`scripts/expand_vocab.py` + `--vocab-map`):
  7834 surface forms survived with trained rows; 24,934 genuinely new
  tokens initialized from the mean of their old-token pieces + noise.
- Symmetry breaking: 1% Gaussian noise on all new rows/columns.
- Honest limit: width growth is approximate through RMSNorm (the rms
  statistic changes with dimensionality). Drift is measured, not claimed
  zero — see verification below.

## Growth verification

- Parent: `checkpoints_100m_gn/ckpt_8218.pt` (step 8218, best_val 4.1989)
- Transfer artifact: `transfer/xiiv-133m-gn-step8218-weights.pt`
  (sha256 `47fbf8d95c3c52850fc10e7e1df8b7f3023ebcb183bb4a946aec85b388dae34e`,
  534MB) — optimizer stripped for transfer; weights bit-identical to the
  parent (spot-checked). Full-scale growth needs ~4-6GB RAM, so it runs on
  his PC via `grow_xiiv_700m.ps1`, not on this box.
- Grown: `checkpoints_700m_gn/ckpt_8218.pt`
- Parent-vs-grown logit drift on random inputs: max|d|=___, mean|d|=___
  nats (gate: grown/random-init drift ratio < 0.6, the scale-invariant
  criterion from `tests/test_grow.py` — the old absolute 0.20 bar was
  calibrated on a toy growth and does not scale to real growth sizes)
- Grown init is ___× closer to the parent than a random init
  (measured, not asserted).

## Training config (frozen)

- Data: `data_32k/` — 41.3M train tokens / ~1.5M valid tokens, 32k BPE.
  Same documents as the 133M corpus, same order, retokenized.
- Steps 8218 → 48218 (40,000 steps × 512 tok/step = 20.48M tokens —
  the same budget as the 133M rung; far below Chinchilla-optimal,
  documented as hardware-constrained).
- LR: 1.833e-4 peak → 1.833e-5 min, cosine from step 8218, warmup 500.
  (Inverse-width rule from the 133M rung's 3e-4: ×704/1152.)
- Optimizer: AdamW 8-bit (bitsandbytes), betas (0.9, 0.95), wd 0.05,
  grad-clip 1.0. **Fresh optimizer at growth** — momentum does not
  transfer (shapes changed).
- Batch: micro-batch 2 × grad-accum 2 = 512 tok/step (identical to 133M).
- Precision: bf16 autocast + gradient checkpointing.
- Guarantee-native: contract_prob 0.4, adversarial_frac 0.3, lock_prob 0.4,
  contract_max_tokens 96 — identical to the 133M rung.
- Hardware: NVIDIA RTX 4050 6GB (`train_xiiv_700m_gpu.ps1`).
  VRAM budget ≈ 5.3GB: params 1.4 + grads 1.4 + 8-bit states 1.4 +
  activations/checkpointing ~0.3 + CUDA overhead ~0.8.

## Eval gates (identical every rung)

- Val NLL every 500 steps (8 batches), full curve kept.
- Intervention-rate benchmark vs the 52.5M baseline (1.00): promotion
  needs val NLL trending down AND intervention rate improving.
- Sampler/lock-span regression tests (byte-exact).

## Known limitations (honest)

- Token budget is ~0.03 tok/param — two orders of magnitude below
  Chinchilla-optimal. This rung tests the guarantee-native thesis at
  scale, not pretraining scaling laws.
- Training corpus is still TinyStories/Gutenberg/Wikipedia-derived;
  the writing-quality bar needs the planned source-quality upgrade.
- Width growth is approximate (RMSNorm); the first ~1k steps after
  growth re-settle the new capacity.
- Guarantees prove structural compliance only — not truth, factuality,
  or safety beyond what the certificates prove.
