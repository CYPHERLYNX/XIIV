# XIIV-700M Kit — README

This folder continues the XIIV lineage on your PC: the 133.52M rung
(trained to step 8218) grows to **697.93M params**, then trains to step
48218 on your RTX 4050.

## What's inside

| path | what |
|---|---|
| `transfer/xiiv-133m-gn-step8218-weights.pt` | 133M parent weights (534MB, optimizer stripped — growth never reads optimizer state) |
| `scripts/grow_checkpoint.py` | Net2Net-style growth (width/head/ffn replication + noise, identity depth, vocab map) |
| `weights/tokenizer_32k.json` | 32k BPE tokenizer (mask=2, eos=3, same as the 8k one) |
| `weights/vocab_map_8k_32k.json` | 32768-entry map: 7834 surviving rows copied, 24934 new tokens from old-piece means |
| `data_32k/` | retokenized corpus: 41.3M train / 1.5M valid tokens, same documents as the 133M run |
| `grow_xiiv_700m.ps1` | step 0: CPU growth 133M → 700M (frozen config, drift gate) |
| `train_xiiv_700m_gpu.ps1` | step 1: RTX 4050 training, crash-resilient, VRAM smoke test first |
| `requirements-gpu.txt` | CUDA torch + bitsandbytes (8-bit AdamW) + tokenizers |
| `MODEL_CARD_700M.md` | the rung's model card (fill in the drift numbers after growth) |
| `xiiv/` | model + training source |

## Run it (two commands, in order)

Requirements: Python 3.10+ from python.org, NVIDIA driver installed.

```powershell
# 1. Grow 133M -> 700M on CPU (~4-6 GB RAM, a few minutes).
#    Writes checkpoints_700m_gn/ckpt_8218.pt and prints drift numbers.
.\grow_xiiv_700m.ps1

# 2. Copy the printed drift numbers (max|d|, mean|d|) into
#    MODEL_CARD_700M.md's verification section, then train:
.\train_xiiv_700m_gpu.ps1
```

The training script runs a 20-step VRAM smoke test first (real measured
peak, ~5.3GB of your 6GB), asks for confirmation, then trains to step
48218 with a crash-resilient loop. Safe to re-run any time — it resumes
from the latest checkpoint. It STOPS (no auto-relaunch) on the
non-finite-loss abort so a diverging run never burns VRAM silently.

## Frozen config (do not hand-tune)

- Arch: d=1152, L=30, h=18, ffn=3200, vocab=32768, tied embeddings.
- Growth: width/head/ffn replication + 1% noise, 15→30 identity depth
  blocks, drift gate grown/random-init ratio < 0.6 (the scale-invariant
  bar from `tests/test_grow.py`).
- LR 1.833e-4 → 1.833e-5 cosine from step 8218 (inverse-width rule from
  the 133M rung's 3e-4: ×704/1152), warmup 500, fresh 8-bit AdamW.
- Batch: micro-batch 2 × grad-accum 2 = 512 tok/step (identical to 133M).
- Steps 8218 → 48218 = 40,000 steps × 512 tokens = 20.48M tokens.
- Guarantee-native: contract_prob 0.4, adversarial_frac 0.3, lock_prob 0.4.

## Honest notes

- Token budget is ~0.03 tok/param — far below Chinchilla-optimal. This
  rung tests the guarantee-native thesis at scale, not scaling laws.
- Width growth is approximate through RMSNorm; the first ~1k steps after
  growth re-settle the new capacity.
- Guarantees prove structural compliance only — not truth or factuality.
