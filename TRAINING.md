# XIIV training recipe (reproducible)

Everything below was run on a 2-core CPU-only Linux box with 7 GB RAM.
No GPU was used at any point. Total wall-clock: roughly one working day.

## 1. Data

- Source: `roneneldan/TinyStories` (TinyStoriesV2-GPT4 split), CC-BY-flavored
  research dataset of short children's stories.
- Files: `TinyStoriesV2-GPT4-train.txt` (2.2 GB), `TinyStoriesV2-GPT4-valid.txt`.
- Tokenizer: byte-level BPE, 8192 pieces + `[UNK] [PAD] [MASK] [EOS]`,
  trained on a 50 MB sample (`tokenizers` library). Saved as
  `data/tokenizer.json`; the special-token ids (`mask_id`, `eos_id`) are
  recorded in `data/meta.json` and threaded through training — never
  hardcoded.
- Packed into uint16 memmap blocks of 128 tokens, `[EOS]` between stories:
  **~38M train tokens**, **~1.2M validation tokens** (frozen split).

```bash
.venv/bin/python scripts/prepare_data.py --out-dir data \
    --train-chars 150000000 --valid-chars 5000000
```

## 2. Model

Bidirectional transformer (no causal mask anywhere):

| hyperparam | value |
|---|---|
| d_model | 320 |
| layers | 6 |
| heads | 5 (head_dim 64) |
| FFN | SwiGLU, 864 |
| positions | RoPE θ=10000 |
| norm | RMSNorm |
| embeddings | tied in/out |
| vocab | 8192 + specials |
| seq_len | 128 |
| **params** | **10.06M** |

## 3. Objective

LLaDA-style absorbing-state masked diffusion: per batch sample
*t* ∼ Uniform(0.15, 1.0), mask each token independently w.p. *t*,
mean cross-entropy over masked positions only.

## 4. Optimization

| hyperparam | value |
|---|---|
| optimizer | AdamW (β 0.9/0.95, wd 0.05) |
| LR | 3e-4 → 3e-5 cosine, 500 warmup steps |
| batch | 48 × 128 = 6144 tokens |
| steps | 5000 (~30.7M tokens, <1 epoch) |
| precision | bf16 autocast on CPU (AVX512-BF16) |
| grad clip | 1.0 |
| throughput | ~2000 tok/s on 2 cores → ~4.3 h |

```bash
.venv/bin/python -m xiiv.train --data data --ckpt-dir checkpoints \
    --total-steps 5000 --batch-size 48 --bf16
```

Checkpoints (`ckpt_<step>.pt` + `best.pt`) are full training state —
Ctrl-C safe, resume picks up the latest automatically. A JSONL log
(`checkpoints/train_log.jsonl`) records loss/tokens/sec per 25 steps.

## 5. Evaluation & export

```bash
.venv/bin/python -m xiiv.evaluate --weights weights/xiiv.safetensors  # → eval_results.json
.venv/bin/python scripts/export_weights.py --ckpt checkpoints/best.pt     # → weights/
```

`eval_results.json` from the reference run is committed at
[`eval_results.json`](eval_results.json). The numbers in the README
come straight from that file — no other source.

## 6. Scaling notes

The 10M config is sized so the full recipe runs on a laptop CPU in an
afternoon. To scale toward the ~50M research target on a GPU: set
`d_model=512, n_layers=12, n_heads=8, ffn_dim=2048`, raise `--total-steps`
to ~60000 on the full 2.2 GB corpus, keep the objective and sampler
unchanged. Nothing in the code is CPU-specific.
