# Clover-35B-A3B-Preview (DiffuSpace) — research note 2026-10-31

Source: DiffuSpace launch blog + HF model card + GitHub (via secondary
coverage, 2026-10-28..31). All benchmark/throughput figures below are
VENDOR CLAIMS — no independent evaluation exists yet. Treat as
prior-art/inspiration awareness, not as adopted results.

## What it is

- First production-oriented open diffusion LLM from DiffuSpace (commercial
  continuation of HKU NLP's Dream team; Dream 7B shipped 2025-04).
- 35B total params, ~3B active/token, MoE: 40 layers, 256 routed experts
  with top-8 routing + 1 shared expert per layer. Hybrid Gated DeltaNet /
  full-attention backbone. 128K context. Apache 2.0 (weights + code).
- Conversion, not from-scratch: built on Alibaba Qwen3.6-35B-A3B —
  DiffuSpace adapted its memory mechanism (Gated DeltaNet) to read context
  bidirectionally within each block, plus learnable block-size calibration
  and a zero-initialized right-context convolution branch.
- Decoding: BLOCK diffusion — each 32-token block starts fully masked;
  positions are filled in parallel when the model is >=0.95 confident;
  the block closes on confidence, not on a fixed step count, then joins
  the context and the next block begins.
- Vendor claims: SWE-bench Verified 73.6 (vs 73.4 they re-ran for the AR
  base), Terminal-Bench 2.1 51.7, GPQA Diamond 83.8, 680 tok/s on one
  GB300 (BF16, bs=1, internal agent data). INT4-on-RTX-5070Ti claims
  reference weights/quant recipes that are NOT published. Runtimes:
  Transformers path official; SGLang PR unmerged; llama.cpp "will be
  added"; no GGUF/Ollama/LM Studio builds. ~69GB BF16 weights.

## Prior-art position vs XIIV

- Diffusion MoE at 35B/3B-active with a linear-attention (Gated DeltaNet)
  backbone surviving bidirectional decoding is the closest public
  precedent to XIIV's diffusion + sparse direction. It does NOT touch
  our moat axis: Clover ships zero certificates, zero guarantees, zero
  verifiable-reward training — it is a capability play, not a
  guarantee-native one.
- The 256-expert / top-8 / 1-shared granularity is a second public
  reference point for the 4B MoE expansion spec (research/
  expansion_4b_moe.md: 8 experts/layer, top-2) alongside Mistral Large 4's
  config. Reference only — not adopted; our granularity decision stays
  with our own compute/quality measurements.
- Confidence-gated commit policy (>=0.95 closes a block) is the sampler-side
  idea most adjacent to our DCMR sampler (xiiv/dcmr.py): DCMR gates
  REMASKING by certificate margin; Clover gates COMMITTING by confidence.
  The two are complementary, not duplicative — a confidence-gated
  block-parallel commit with a per-commit certificate is a candidate
  future sampler invention, documented here, not implemented today.

## What we do NOT do

- No port of any Clover component. No adoption of vendor benchmark claims
  into any XIIV record (XIIV rows are filled only from real runs).
- No AR->diffusion conversion of our own: our 1B is grown from the 133M
  seed by SGBD and the failure modes under study (premature-commitment,
  step-0 void) are being addressed by BASC/CSC/combined training steps,
  not by converting an AR teacher.

## Watch items

- Independent evals of Clover (zbruceli-style open evals like Mercury 3
  got on 2026-10-21) — if real, the confidence-gated commit numbers
  become design inputs for the sampler-side candidate above.
- The promised "decision model" variant (independent vs joint mode, up
  to 8 related questions in one denoising pass) — joint multi-question
  denoising is adjacent to our joint-prior (CSC) work; watch for a paper.
