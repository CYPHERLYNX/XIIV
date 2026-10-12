# XIIV improvement plan — grounded in what training/compute can actually change

Written 2026-10-11. No hype, no fabricated projections. Each step states what
it changes, what it costs, and what we honestly expect.

Standing constraints: RTX 5090 on Fahad's PC (the old 6GB-VRAM/4050 planning
is obsolete). TinyStories is out of all future training (real data only).
Single grown lineage: 133M → 500M → 1B. Guarantee-native thesis preserved.

---

## Step 0 — Fix the scorer (blocks all claims)

**Problem:** `xiiv/lm_eval_wrapper.py` may leak candidate-answer information —
it masks only sampled continuation positions while other continuation tokens
stay visible. Until audited, every XIIV benchmark number is suspect, including
the "before" in COMPARISON_TABLE.md.

**Work:**
1. Audit the wrapper: verify exactly which positions are masked vs visible
   during scoring. Write the finding down.
2. Replace with a defensible diffusion pseudo-likelihood estimator
   (e.g., full-continuation masking, multiple fixed-seed samples, mean NLL).
3. Add deterministic scoring/ranking regression tests — same inputs must give
   same rankings across runs.
4. Re-run the 133M baseline with the fixed scorer. Record whether numbers
   move; if they do, the old numbers are retracted, not revised.
5. Freeze the scorer version. All future rungs use the identical scorer.

**Cost:** CPU-only work, days not weeks. No GPU needed.
**Expected movement:** unknown direction, possibly ±several points. This step
doesn't improve the model — it makes the ruler honest.

---

## Step 1 — Real-data pretraining at 500M

**Problem:** the 133M was trained on 4.2M children's-story tokens. The 500M
checkpoint (`checkpoints_500m_gn/ckpt_8218.pt`) is a grown init with story
weights and 24,934 untrained expanded-vocab embeddings. It has never seen
real data.

**Available now:**
- FineWeb-Edu: **99,329,024 train tokens** (776,008 blocks × 128),
  `data_fineweb/` in train.py memmap format, meta.json written.
  Caveat: resume restarted the HF stream without persisting doc hashes —
  ~1/3 of the corpus may duplicate the first 62M-token partial run. Usable
  for the first heavy run; rebuild cleanly if pristine provenance is needed.
- StarCoder: **blocked** — gated behind HF auth. Fahad runs `hf auth login`
  on his PC. No code corpus until then. Open alternatives (The Stack
  dedup, permissively licensed subsets) still to be investigated.

**Work:**
1. Train the grown 500M on FineWeb-Edu on the 5090. This is the first rung
   that sees real data.
2. Track: train/val loss, and the 24,934 new embeddings' gradient norms
   (confirm they actually train).
3. Re-run benchmarks with the frozen scorer (Step 0). This is the first
   honest "after" number.

**Cost:** GPU time on the 5090. 99M tokens is small by frontier standards
(~0.2 tokens/param at 500M — far from Chinchilla-optimal ~20 tok/param).
**Honest expectation:** loss will drop and outputs will stop sounding like
children's stories. Benchmark movement is genuinely uncertain — 99M tokens
is ~24× the story corpus but still ~100× less than GPT-2's data. Direction:
up. Magnitude: unknown. Anyone promising a number is guessing.

---

## Step 2 — Grow to 1B, train on the 5090

**Status:** 1B is the confirmed target (RTX 5090 correction, 2026-10-10).
Exact 1B config, growth gate, and training recipe are not yet frozen —
that spec work is in progress.

**Work:**
1. Freeze the ~1B XIIV diffusion config programmatically (verify parameter
   count, head_dim divisibility, memory fit on 24GB VRAM).
2. Implement 500M→1B growth with the pre-registered drift gate
   (same protocol as the 133M→500M growth: drift ratio < 0.6 vs random init).
3. Write the 1B model card before training.
4. Train on the 5090 with real data. Target corpus: FineWeb-Edu (rebuilt
   clean if needed) + code + books/reference + math/STEM as available.
   TinyStories stays out.

**Cost:** the main GPU campaign. 1B at Chinchilla-optimal would want ~20B
tokens — we will have ~0.1–1B. This is the fundamental constraint: we cannot
match frontier data scale, so we compete on the guarantee-native
differentiator, not on raw benchmark parity.
**Honest expectation:** the 1B should clearly beat the 133M "before" and
should beat chance by a wide margin on ARC-Easy/HellaSwag if the data mix is
right. It will not match 7B models trained on trillions of tokens. The goal
is the best 1B-scale diffusion model with proof-carrying guarantees — a
category with exactly one entrant.

---

## Step 3 — Instruction tuning (currently zero)

**Problem:** no XIIV rung has ever seen instruction-response pairs. The base
models can't follow instructions, which caps every benchmark that rewards
following the question format.

**Work:**
1. Assemble an instruction corpus with provenance (licensed/instruction
   datasets — OpenHermes, Alpaca-style, or similar permissively licensed
   sets; record exact versions and licenses).
2. Add a preference/alignment stage only after SFT works — do not skip to
   RLHF on an untuned base.
3. Re-benchmark. Instruction tuning is historically the largest single
   jump on these tasks for small models (LLaDA instruct: 47.9 → 88.5 on
   ARC-C, though at 8B scale with SFT).

**Cost:** small GPU cost relative to pretraining; the cost is mostly data
curation.
**Honest expectation:** this is the highest-leverage step per GPU-hour for
benchmark movement. But "highest leverage" ≠ a promised number. At 1B scale
with limited SFT data, expect clear improvement over the base, not
frontier parity.

---

## Step 4 — Corpus expansion (ongoing, parallel)

The 1B needs more than 99M FineWeb-Edu tokens. In priority order:

1. **Rebuild FineWeb-Edu cleanly** — fix the resume-duplication issue,
   persist doc hashes, write the provenance manifest.
2. **Code** — unblock StarCoder via Fahad's `hf auth login`, or pivot to
   openly licensed alternatives. Code data is the biggest missing piece
   for reasoning-adjacent tasks.
3. **Books/reference** — investigate genuinely open, permissively licensed
   sources. No piracy, no gray-area scraping.
4. **Math/STEM** — same licensing bar.
5. **Instruction/preference** — for Step 3.

Each addition gets: source, license, doc count, token count, hash manifest,
contamination check against eval sets. No dataset enters training without
this.

---

## What each step is expected to move (honestly)

| Step | Moves | Doesn't move |
|---|---|---|
| Scorer fix | Trust in the numbers; maybe ±points | Model quality |
| 500M real-data train | Loss, output coherence, "after" baseline exists | Data-scale gap vs frontier (99M ≪ 10B) |
| 1B growth + train | Capacity, best honest benchmark rung | The ~20B-token Chinchilla gap |
| Instruction tuning | Benchmark scores (highest leverage/GPU-hr) | Pretraining knowledge gaps |
| Corpus expansion | Ceiling of every step above | Nothing by itself |

**What we will not do:** publish projected benchmark numbers, claim
parity with any frontier model, or present diffusion-NLL scores as
comparable to AR-logprob scores without the caveat. The comparison table's
"after" column gets filled with measured numbers or stays empty.
