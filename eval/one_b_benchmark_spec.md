# 1B-Segment Benchmark — Spec v1.0

Status: HARNESS ONLY. No run has been published from this harness.
Every score in this harness is UNMEASURED until a timestamped run
file in `eval/results/` proves it. This spec does not claim XIIV
beats anything.

## Model under test

XIIV 1B, assembled (not the untrained parent, not a tiny rung):
- `checkpoints_1b_sgbd/xiiv_1b_sgbd.safetensors` — 3,093,698,487 bytes,
  fp32, 254 tensors, verified SHA prefix `cf548ec4` (full SHA in the
  assembly manifest).
- Unique parameters: 915,988,480 (773,418,240 runnable + 142,570,240
  contract, reconciled to the reference). Contract is OFF in the
  stitched file by design; the parent `contract_attn.kv` shape defect
  is still open, so no contract claim is made here.
- Config: d_model 1280, 36 layers, 20 heads, vocab 32768, mask 2,
  eos 3, seq_len (canvas) 128, tied embeddings.
- Sampler: masked discrete diffusion, `xiiv.diffusion.sample_trajectory`.

## Segment definition (fixed before any run)

"1B segment" = publicly released decoder models with roughly 0.5B to
1.7B parameters, runnable on a single consumer GPU. The range is
stated now, in full, because market "1B" labels run from 1.0B to
1.7B; narrowing the range after seeing results would be
cherry-picking. The competitor set, with sources, is in
`eval/one_b_competitors.json`. Every row there is UNMEASURED.

Competitor set (name — HF repo — class):
- Llama-3.2-1B — meta-llama/Llama-3.2-1B — ~1.2B class, causal
- Gemma-3-1B-IT — google/gemma-3-1b-it — ~1B class, causal, instruct
- Qwen2.5-1.5B-Instruct — Qwen/Qwen2.5-1.5B-Instruct — ~1.5B, causal, instruct
- SmolLM2-1.7B-Instruct — HuggingFaceTB/SmolLM2-1.7B-Instruct — ~1.7B, causal, instruct
- Qwen3-0.6B — Qwen/Qwen3-0.6B — ~0.6B, causal, lower-bound control
- OLMo-1B — allenai/OLMo-1B-hf — ~1B, causal, base (confirm repo name at download)

Caveats that must stay in any published table:
- Instruct-tuned competitors vs a base diffusion model is NOT a
  like-for-like tuning comparison; label it.
- Size advantages (1.5B/1.7B vs 915,988,480) must be stated, not hidden.
- Gated repos (Llama, Gemma) are not runs until their terms are
  accepted and weights are actually downloaded.
- A competitor score may only come from a harness run file with the
  same prompt file and fixed spec. Never from a vendor blog, a model
  card number, or memory. The old `eval/compare_frontier.py`
  hardcoded frontier scores are NOT used here and must not be copied
  into this harness.

## Fixed prompt set and scoring

- File: `eval/one_b_prompts.json` — 100 prompts, 10 categories x 10.
- Fixed spec: length 64, steps 16, seed 7, temperature 0.8, top_p 0.9.
  Prompt tokens must be < 64 and <= 24, so >= 40 tokens are generated.
  The canvas is 128, so length 64 is inside it.
- Checks run ONLY on the generated portion (tokens after the prompt).
  Scoring the full text would let a keyword already in the prompt
  pass trivially.
- Automated rubric v0 per prompt: `non_empty`, `contains:<word>`,
  `not_contains:<word>`, `max_word_repeat_ratio` (most frequent word
  / total words, fails above 0.5 when >= 10 words — the glitch screen).
  These checks are crude by design and are labelled as such.
- Each item also needs a human rating before publication. A category
  score from automated checks alone is PROVISIONAL. A "beat" claim
  needs: same hardware, same prompts, same spec, raw outputs in the
  run files, human-rated scores, and the timing side-by-side.

Categories (10, each 10 items):
1. factual_explanation
2. general_knowledge_qa
3. arithmetic_reasoning
4. code_short
5. instruction_following
6. constraint_compliance (include / do-not-use a word)
7. coherence_no_glitch (story continuation, repetition screen)
8. summarisation
9. proof_carrying (fact + reason/evidence, not just an assertion)
10. latency_reliability (10 timed repeats of the quantum prompt)

## The 9/10 gate, translated into this harness

Standing directive: 9/10 in ALL categories; failure in ANY category
is a push trigger, not a release.

- Per category: score = number of passing items out of 10.
- Gate PASS requires every category >= 9/10. No averaging across
  categories can rescue a failing one.
- Gate FAIL in any category -> the gate fails, full stop. The next
  step is improvement work (sampler, consolidation, training), then
  a fresh timestamped run — never editing an old run file.
- Speed is reported, not gated on an invented floor. The latency
  category scores reliability (10 timed runs completing with
  non-empty output). A speed WIN additionally requires XIIV
  tokens/second > the competitor's tokens/second on the same
  hardware, in the same run set, with both raw timings published.
- Gate status in a run file is computed by `score_run()` and is
  marked `automated_only: true` + `human_rating_required: true`
  until a human rating exists. An automated PASS is not a verdict.

## Runner and outputs

- Runner: `eval/one_b_benchmark.py` (see its docstring).
- `--dry-run` validates prompts/spec only, loads no model, writes
  no scores. This is the only mode safe on a RAM-constrained box.
- Model load is refused when MemAvailable < 2.0 GB. There is no
  force flag. An OOM failure is not a result and must not be recorded
  as one.
- Outputs: `eval/results/one_b_<model>_<device>_<timestamp>.json`,
  never overwriting a previous run (a suffix is added on collision).
  Each file carries: spec, weights path/bytes/SHA, config, every
  item's prompt, full text, generated text, token ids, seconds,
  tokens/second, and the computed gate report.
- Competitor runs require `--competitor-adapter` (a file defining
  `generate(prompt, spec)`) and a `--competitor-name` already in
  `one_b_competitors.json`. Without an adapter the runner refuses
  rather than fabricating a competitor row.

## What is still needed before the first honest comparison

1. A GPU run of the assembled XIIV through this harness (the Kaggle
   GPU route currently uploading; this box is RAM-guarded out).
   The v1 GPU job's single quantum prompt is NOT a harness run.
2. A competitor adapter + a real competitor run on the same hardware
   with the same prompt file and spec, for at least one competitor.
   Until then, every comparison table stays UNMEASURED.
3. Human rating of both sides' outputs for every category.
4. Only then: a comparison table, with raw run files cited, size and
   tuning caveats stated, and the gate verdict (PASS/FAIL per
   category) shown in full — not just the winning rows.
