# XIIV-500M Kit — README

> **2026-10-09 — real-data regime:** the heavy path now exists as
> `train_xiiv_500m_heavy.ps1` (one command; FineWeb-Edu + StarCoder + books
> mix, no children's stories, 195k steps, contract training stays on).
> Use it instead of `train_xiiv_500m_gpu.ps1` below — the story-data run is
> kept only as the 500M smoke-test baseline.

This folder continues the XIIV lineage on your PC: the 133.52M rung
(trained to step 8218) was grown to **503.45M params** ON THIS BOX
(verified: drift ratio 0.113 < 0.6 gate, forward smoke finite — see
`MODEL_CARD_500M.md`), and this kit trains it to step 48218 on your
RTX 4050.

Unlike the 700M path, no growth step is needed on your PC — the grown
checkpoint already exists and is verified.

## What's inside

| path | what |
|---|---|
| `checkpoints_500m_gn/ckpt_8218.pt` | grown 500M weights (1.9GB, no optimizer state — starts training fresh from step 8218) |
| `weights/tokenizer_32k.json` | 32k BPE tokenizer (mask=2, eos=3, decoder fixed — no more mojibake) |
| `data_32k/` | retokenized corpus: 41.3M train / 1.5M valid tokens, same documents as the 133M run (82MB) |
| `train_xiiv_500m_gpu.ps1` | one command: RTX 4050 training, crash-resilient, VRAM smoke test first |
| `requirements-gpu.txt` | CUDA torch + bitsandbytes (8-bit AdamW) + tokenizers |
| `MODEL_CARD_500M.md` | the rung's model card (grown + verified, training section blank) |
| `xiiv/` | model + training source (audited 2026-10-09: 36/36 tests, 5 real wrongs fixed) |

Total transfer: ~2GB. Zip this folder, unzip on the PC, one command.

## Run it (one command)

Requirements: Python 3.10+ from python.org, NVIDIA driver installed.

```powershell
.\train_xiiv_500m_gpu.ps1
```

The script runs a 20-step VRAM smoke test first (real measured peak —
expect ~4-5GB of your 6GB, 500M sits lighter than 700M), asks for
confirmation, then trains to step 48218 with a crash-resilient loop.
Safe to re-run any time — it resumes from the latest checkpoint. It
STOPS (no auto-relaunch) on the non-finite-loss abort so a diverging
run never burns VRAM silently.

Expected wall clock: 1-3 s/step at 500M on a 4050 → 40,000 steps ≈
11-33 hours. The full 41.3M-token corpus is 80,000+ steps (2 epochs of
this budget); this rung trains the first 20.48M tokens.

## Frozen config (do not hand-tune)

- Arch: d=1024, L=27, h=16, ffn=2816, vocab=32768, tied embeddings.
- Growth (already done): width/head/ffn replication + 1% noise, 15→27
  identity depth blocks, drift gate grown/random-init ratio < 0.6 (the
  scale-invariant bar from `tests/test_grow.py`).
- LR 2.0625e-4 → 2.0625e-5 cosine from step 8218 (inverse-width rule
  from the 133M rung's 3e-4: ×704/1024), warmup 500, fresh 8-bit AdamW.
- Batch: micro-batch 2 × grad-accum 2 = 512 tok/step (identical to 133M).
- Steps 8218 → 48218 = 40,000 steps × 512 tokens = 20.48M tokens.
- Guarantee-native: contract_prob 0.4, adversarial_frac 0.3, lock_prob 0.4.

## Honest notes

- Token budget is ~0.04 tok/param — far below Chinchilla-optimal. This
  rung tests the guarantee-native thesis at scale, not scaling laws.
- The 32k vocabulary's new tokens (24934) have seen zero training; the
  first ~1k steps re-settle the new capacity and warm the new
  embeddings. Expect early outputs to stay parent-vocab-leaning.
- The 133M run's seeded-val fix (bit-identical evals) ships in this
  stack — val numbers you watch will be measurement, not noise.
- Guarantees prove structural compliance only — not truth or factuality.
