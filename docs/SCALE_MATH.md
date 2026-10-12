# XIIV Scale Math — what "powerful" actually costs

No hype. Every number below is arithmetic you can check.

## Where we are

| | XIIV-500M (now) | GPT-5 / Opus 4.6 class |
|---|---|---|
| Params | 503M | 100B+ (est.) |
| Training tokens seen | ~4.2M | trillions |
| Data | children's stories | web + code + books + papers, filtered at scale |
| Chinchilla-optimal tokens | ~10B | — |

The gap is not 2x or 10x. It is ~1,000,000x in data and ~200x in params.

## What one RTX 4050 (6GB) can actually do

Measured on this project: 500M bf16, batch 512 tok/step, grad-checkpoint +
8-bit AdamW → roughly **1–3 s/step** (the smoke test in
`train_xiiv_700m_gpu.ps1` measures the real number on your card).

| Training budget | Steps (512 tok) | Wall clock (4050) | What it buys |
|---|---|---|---|
| 41M tokens (all of data_32k) | 80k | ~1–3 days | Working 500M: trained 32k embeddings, fluent stories. **Do this first.** |
| 1B tokens | 2M | ~1–2 months | Genuinely capable small model *if the data is good* (see below) |
| 4.2B tokens ("1000x" today's) | 8.2M | ~6 months | Serious small-model territory, needs the data mix below |
| 10B tokens (Chinchilla-optimal) | 20M | ~1–2 years | Not feasible on one card |

**The binding constraint is not the model — it is the data.** Training
longer on children's stories makes a better story model, not a smarter
one. 41M story tokens ≠ 41M FineWeb tokens.

## The data upgrade (biggest lever, in order)

1. **FineWeb-Edu** (1.3T tokens available; take 1–5B): general knowledge,
   the single highest-value corpus for "powerful".
2. **StarCoder / The Stack** (code): the 133M/500M cannot code today
   because it has never seen code. Non-negotiable for benchmarks.
3. **Books + papers**: keep what we have, add more.
4. **Mix ratio** (frontier practice): ~60% web-edu, ~20% code, ~20% books/stories.

`scripts/prepare_wiki.py` and `scripts/prepare_books.py` are the pattern;
a `prepare_fineweb.py` follows the same memmap format (`train.bin` /
`valid.bin` / `meta.json`, 128-token blocks).

## What "benchmarks near GPT-5" really requires

- 10B+ quality tokens, 7B+ params, 100s of GPU-days. That is a funded
  cluster, not a laptop. Saying otherwise would be lying to you.
- What the 4050 *can* produce: a strong, honest, open 500M with
  guarantee-native contracts — a model with a real differentiator
  (proof-carrying outputs) that no 500M model in the open has.
- Staged path: 500M done right → data upgrade → 1–5B tokens →
  measured evals (MMLU-lite, HumanEval-lite probes) → then decide
  whether the results justify bigger compute.

## The evals that keep us honest

- Every rung reports: val NLL (seeded, comparable), generation samples
  (fixed prompts/seeds), contract-compliance rate, intervention rate.
- No benchmark is claimed without the exact probe set, seed, and
  command that produced it. Numbers without provenance are marketing.
