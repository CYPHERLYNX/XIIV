# XIIV — Model Card

## Model details
- **Name:** XIIV v0.1 — a tiny masked-diffusion language model
- **Parameters:** 10,059,840 (exact, `count_parameters()`); the 52,508,736-param
  XIIV run completed 2026-10-02 (see Status). Both are masked-diffusion
  transformers; the 52.5M is d=576, L=12, h=9, SwiGLU-1536, same vocab/canvas
- **Architecture:** bidirectional transformer, d=320, L=6, h=5, SwiGLU-864,
  RMSNorm, RoPE, tied embeddings, no causal mask
- **Tokenizer:** byte-level BPE, vocab 8,192, trained on the TinyStories corpus
- **Context:** 128 tokens (fixed-length canvas)
- **Objective:** masked diffusion — mask ratio ~U(0.15, 1.0), CE on masked
  positions only; validation reports the masked-LM NLL upper bound
- **Training data:** TinyStoriesV2-GPT4 (36.5M train / 1.2M valid tokens),
  pinned snapshot with SHA hashes, 0 block-level train/valid overlaps
  (block-level audit; does not prove zero semantic leakage)
- **Compute:** single CPU, 2 threads, BF16, batch 48, 5,000 steps,
  30.7M tokens seen (completed 2026-10-01 ~15:52 IST after a resume from
  step 4,520 — an earlier interruption was recovered, not hidden)
- **Release:** weights (safetensors) + tokenizer + config, MIT

## Intended use
Research and demonstration of **constraint-native generation at tiny scale**:
exact-length infill, forbidden-token exclusion, locked-region preservation,
required-token inclusion, parallel iterative de-masking, plus diffusion-native
reasoning (scratchpad thinking, self-revision). Not a general chat model;
trained on single-domain children's stories.

## What is genuinely different here
Autoregressive models generate left-to-right; revising an emitted token means
regenerating from scratch. XIIV denoises the whole canvas in parallel and
**revises natively in-canvas**: low-confidence
spans are remasked and re-denoised in place with full bidirectional context.
Constraints (required/forbidden tokens, locked spans, exact length) are
enforced structurally during denoising — not filtered after generation.
These are architectural properties, verified by tests (27/27), not benchmark
claims against frontier models. XIIV is not and does not claim to be
GPT-4/Opus-class; it is a different *kind* of machine at a tiny budget.

## Metrics (measured, this machine)
- Best held-out masked NLL bound: **3.4021 nats/token** (final, step 5,000)
  — an upper bound on sequence NLL, **not comparable to AR perplexity**
- Constraint mechanics: 19/19 sampler tests (required/forbidden/locked/
  exact-length/impossible-rejection/order presets/stateful resume)
- Reasoning APIs: 8/8 tests (scratchpad determinism + region accounting,
  self-refine guarantee preservation, long-form exact length +
  determinism, vision shapes/errors, multimodal frozen prefix,
  snapshot round-trips old and new)
- CPU generation (measured this machine, 2 threads, parallel de-masking): 128
  tokens in 6.61 s = **19.4 tok/s**, 16 forward passes, 400.2 MB peak RSS
- Infill benchmark (200 held-out 128-token blocks, 32-token masked span):
  recovery accuracy **0.0938** (600/6400 tokens exact match), 4.977 s/block,
  16 passes/block
- Constraint benchmarks (measured on the exported weights):
  exact_length 1.000, forbidden_words 1.000, edit_region_preservation 1.000,
  must_include_words 1.000 — see `bench/constraints/results.json`
- Promotion gates (declared before training finished): **8/9 PASS**.
  G7 (`cpu_tok_s>=50`) **FAILS** at 19.4 tok/s — the diffusion model is
  ~2.6x below the declared CPU-throughput bar, so promotion is **blocked**
  pending re-review of that bar. All other gates pass.

## Metrics — XIIV-52.5M (measured, this machine, completed 2026-10-02 ~20:29 IST)
- Best held-out masked NLL bound: **3.3414 nats/token** (train-log eval at
  step 4,500 of 5,000; final step-5,000 eval 3.3473). Independent post-export
  eval (`python -m xiiv.evaluate`, 64 held-out batches) on the exported
  weights: **3.34** — an upper bound on sequence NLL, **not comparable to
  AR perplexity**
- 10M vs 52.5M on the same masked objective: 3.4021 → 3.3414 (0.06 nats).
  A modest gain; 20.48M TinyStories tokens is a tiny budget for 52.5M
  parameters — expect repetition and limited vocabulary
- Exported weights `weights_52m/xiiv-52m.safetensors` (210 MB, fp32) load
  strict via the API (`XIIV("weights_52m/xiiv-52m.safetensors")`); a
  generation smoke test produced coherent masked-span infill
- CPU generation (2 threads, parallel de-masking): 128 tokens in 6.38 s =
  **20.0 tok/s** at 16 denoise steps; **37.9 tok/s** at 8 steps.
  Peak RSS 814 MB
- Against the declared G7 bar (`cpu_tok_s>=50`): 20.0 < 50 — the bar fails
  at this size too; promotion stays **blocked** on G7, not waived
- Test suite: **31/31 pass** on this code+weights snapshot, including the
  19/19 structural constraint tests (required/forbidden/locked/exact-length,
  50 seeds per constraint type) and 8/8 reasoning tests

## Limitations (honest)
- 10M params on TinyStories: no broad knowledge, no reasoning benchmarks.
- Masked-NLL is not comparable to autoregressive perplexity; do not rank them.
- English children's-story domain only; byte-level BPE is inefficient.
- Constraint guarantees are structural (logit −inf, frozen mask, slot
  reservation) — verified over 50 seeds per constraint type, not proven
  for all possible inputs.
- Multimodal scaffold (PatchEmbed, image prefix, vision.py) is
  **architecture-only and untrained** — shape-tested, never trained on
  image-text data. Do not claim XIIV understands images.
- `generate_long` chains chunks past the 128-token training window;
  cross-chunk coherence is not guaranteed by training.

## Status (2026-10-01, ~17:20 IST)
- XIIV-10M: finished, exported (`weights/xiiv.safetensors`, 40.2 MB),
  forward parity vs best checkpoint verified (max diff 0.0).
- Post-training pipeline complete 2026-10-01 ~17:30 IST: constraint bench
  all four satisfaction=1.000; efficiency bench 19.4 tok/s, infill
  accuracy 0.0938; **8/9 promotion gates pass — G7 (cpu_tok_s>=50) fails
  (19.4), so promotion is blocked pending re-review of that bar**.
- XIIV-52.5M (d=576, L=12, h=9, ffn=1536; 52,508,736 params): **completed**
  2026-10-02 ~20:29 IST — 5,000/5,000 steps, 20.48M tokens seen, 2 CPU
  threads, bf16. Survived 6 VM reboots mid-run; checkpoints held every time
  (resumed from newest ckpt each time, RNG state included). Best val NLL
  3.3414 (step 4,500); step-4,500 checkpoint was pruned by retention before
  export, so the exported weights are the step-5,000 final (val 3.3473) —
  0.006 nats off the run's best, recorded honestly. Post-export
  `xiiv.evaluate`: NLL bound 3.34, 20.0 tok/s @16 steps (37.9 @8 steps).
  Weights: `weights_52m/xiiv-52m.safetensors` (210 MB fp32). G7 bar (50
  tok/s) not met at this size; promotion remains blocked on G7.
- Publication to `CYPHERLYNX/XIIV` pending: benchmarks, gates, and power
  review still to run. Nothing ships until gates pass or it is explicitly
  marked an experimental prerelease. Repo will be **private**, per the
  user's 2026-10-01 instruction (not public).

## Provenance
- Frozen run config hash: `3497b90c82b3fe5d`
- Data snapshot: `data/data_provenance.json` (SHA256 per file)
- Training log: `checkpoints/train_log.jsonl` (JSONL, resumable)
- 52.5M run: `checkpoints_50m/` (log `checkpoints_50m/train_log.jsonl`,
  resume logs `logs/train_50m_resume*.log`) — completed 2026-10-02, exported
  `weights_52m/`
- Promotion gates (declared before training finished): `PROMOTION_GATES.md`
- Prior-art boundaries: `NOVELTY.md` (no overclaims on diffusion/infilling)
