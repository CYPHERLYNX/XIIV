# Generation-gap diagnosis — first findings (2026-11-13, ~10:00 IST)

Status: diagnosis in progress, NOT a root cause yet. Two findings are
measured and verified from files already on this box; the rest are
hypotheses with a discriminating test specified below.

## Finding 1 — VERIFIED: Leg 3's "held-out loss" is 100% cold-start loss, not denoising loss

`lever3_train.py: held_fw_loss()` reseeds `torch.manual_seed(1234)`
before EVERY held-out sequence, then calls `general_loss`, which in
Leg 3 is `cold_start_loss(..., p_cold=0.15)` (`research/lever3/csc.py`).

Reproduced with the repo venv (`~/workspace/ai-tools/corsac/.venv/bin/python`):

```
torch.manual_seed(1234); torch.rand(())  ->  0.028979241847991943  (< 0.15)
```

So the cold branch is taken for **all 200 held-out sequences**. The
reported Leg 3 held-out loss (7.5907 -> 7.5722) is the loss of
predicting all 128 tokens from an all-mask canvas — the joint prior —
not the standard masked-denoising loss.

Consequences:
- The 01:20 IST report's sentence "BASC actually made the standard
  masked loss *worse* (6.091 at step 8000 -> 7.591 at step 12000)" is
  INVALID. 6.091 was measured under the pre-CSC trainer (plain
  diffusion/BASC loss); 7.591 is a cold loss under a different
  function. Checkpoint AND metric both changed. Whether BASC/CSC
  regressed denoising is UNKNOWN — it was never measured on a common
  metric across checkpoints.
- The train-loss (~6.2-6.5 in `pretrain.log`) vs held-loss (7.57) gap
  is likewise explained by metric mixing, not by a generalisation gap:
  train steps are 85% denoise + 15% cold; held is 100% cold.
- Within Leg 3 only, the comparison is valid: cold loss 7.5907 ->
  7.5722 (-0.24%). CSC barely moved the joint prior it was built to fix.
- Generation 0/5 stands untouched by this correction — it is measured
  directly by sampling, not via any loss. The December release is
  still gated on generation.

Fix for the metric (for the next eval job, not a code change made here):
measure BOTH, separately, on the same checkpoints, with the seed set
ONCE outside the loop (or a dedicated generator per sequence):
  (a) denoise loss: `xiiv.diffusion.diffusion_loss`, t ~ U(0.15, 1.0),
      fixed seed, 200 held seqs;
  (b) cold loss: all-mask canvas, same 200 seqs.
Report (a) and (b) as two numbers, never one blended number.

## Finding 2 — MEASURED from pretrain_gates.json: generation is generic-function-word salad, prompt-type-locked at the first token

Analysed all 8 gates x 7 prompts in
`~/workspace/xiiv-kaggle-gpu/lever3-out/xiiv-lever3-out/pretrain_gates.json`:

- Prompt is always preserved verbatim at the start (7/7 prompts, 8/8 gates).
- Continuation is fluent-shaped word salad: ~45 words per output,
  27-40 unique words, dominated by function words. Top words across
  all 56 outputs: the (339), of (150), to (123), and (107), as (84),
  a (70), from (57), in (54), is (52). Content words never appear:
  "Paris" 0 times in all 56 outputs; "four" 0; "12" 0; "def" 1 (inside
  a word, not as code). Non-words from subword merges recur:
  "scientification", "doaborents", "gradally", "passial", "theial".
- First generated token is locked by prompt TYPE, not content, at
  every gate (8/8): all three factual prompts continue " in ..."
  ("The capital of France is in ...", "2 plus 2 equals in ...",
  "3 times 4 equals in ..."); both code prompts continue " A...";
  "Once upon a time" continues ", ...". The model is predicting the
  most generic continuation for the prompt shape, with no detectable
  conditioning on the factual content.
- Outputs change substantially gate to gate (word-set overlap
  step-13000 vs step-19600 only ~12-23 of ~55-62 words per prompt),
  except "3 times 4 equals" at 19000 vs 19600, which is byte-identical
  — consistent with a near-flat loss surface and sampling noise
  dominating, not with a model converging onto answers.
- Held (cold) loss 7.57 vs uniform ln(32000) = 10.37: the model is
  better than uniform, but perplexity ~1940 on the cold task is still
  very weak.

## Hypotheses (ranked, unproven)

1. **Metric confusion hid the real state.** The denoise loss of the
   step-19600 checkpoint may be far better (or worse) than 7.57; nobody
   has measured it. If denoise loss is ~6, the model knows local
   structure but the sampler/prompt-conditioning path is broken. If it
   is also ~7.5, the base genuinely has not learned general text at
   this token scale.
2. **Prompt conditioning is weak/broken in sampling.** The
   type-locked first token + generic continuations point at the
   sampler committing high-confidence generic tokens first
   (confidence-order unmasking, 16 steps for 64 tokens) before the
   prompt can steer content — or at the model never having learned
   prompt-conditioned denoising at low mask rates (training t ~ U(0.15,
   0.90) always leaves most context masked; inference freezes the
   prompt and denoises the rest, a regime rarely seen in training).
3. **Token scale is simply too small for facts.** 41.7M cumulative
   tokens cannot inject "Paris" etc. at 1B scale if the SGBD-grown base
   never broadly pretrained on general text (the standing reading in
   lever3_train.py's header). This stays live, but Finding 1 removes
   the "BASC made it worse" evidence for it.

## Discriminating test (needs GPU — cannot run on this box)

This box has 605 MB RAM available (2026-11-13 10:00 IST); the 3.4 GB
fp32 checkpoint cannot be loaded here. Run on Kaggle (or the 5090
package) as one short eval job on step-8000, step-12000 and
step-19600 checkpoints (step-12000 safetensors is not local; only its
log is — re-download or use the 5090 package's copy):

1. Denoise held loss + cold held loss, separately, same 200 held seqs,
   seed set once (see fix above).
2. Generation on the 7 fixed prompts with: (a) current settings
   (steps=16, temp 0.8, top_p 0.9, confidence order); (b) temp 0,
   steps=64, left-to-right order; (c) one-token-at-a-time.
   If (b)/(c) ever finds Paris/4/12, the sampler is the bottleneck,
   not the weights. If all settings stay 0/5 while denoise loss is
   ~6, conditioning is broken. If denoise loss is ~7.5 too, it is a
   training-scale/knowledge problem.
3. One cheap probe: single-step denoising accuracy at fixed mask
   rates t = 0.15 / 0.5 / 0.9 on held text — shows whether the model
   can use context at all, and at which mask rate it collapses.

## Sources checked in this run

- `~/workspace/xiiv-kaggle-gpu/lever3-out/xiiv-lever3-out/pretrain_gates.json`
  (8 gates, 56 outputs), `pretrain_summary.json`, `pretrain.log`
  (242 lines, train loss 5.88-6.81, LR 1e-5 -> 0)
- `~/workspace/xiiv-kaggle-gpu/lever3-jobs/lever3_train.py` lines
  219-241 (general_loss / held_fw_loss), `lever2-jobs/lever2_train.py`
  lines 220-241 (BASC general_loss, same seed-1234 pattern)
- `~/workspace/ai-tools/corsac/research/lever3/csc.py` lines 72-100
  (cold_start_loss), `xiiv/diffusion.py` mask_batch lines 43-78
- RNG reproduction: corsac .venv python, seed 1234 -> u = 0.02898

## Update — discriminating tests COMPLETE (2026-11-13 ~12:00 IST, CPU, this box)

Correction to the note above: the checkpoint CAN be loaded on this box.
bf16 load peaks at ~4.5 GB RSS and a forward pass (batch 1 x 128) takes
~1.3 s, so the full test ran locally — no GPU needed. Scripts
`/tmp/xiiv_gen_gap_eval.py`, `/tmp/xiiv_gen_gap_eval2.py`; results
`/tmp/xiiv_gen_gap_results.json`, `/tmp/xiiv_gen_gap_results2.json`.

### Measured: standard loss improved at every leg — BASC did NOT regress it

Same-procedure standard loss (Leg-1 procedure: seed-1234 fixed mask draw,
t = 0.175 on CPU; N=24 held seqs, identical seqs/masks at every checkpoint;
bf16) plus cold loss (same procedure as the Leg-3 gate):

| checkpoint | standard loss | cold loss |
|---|---|---|
| step-8000 (end Leg 1) | 5.728 | 7.747 |
| step-12000 (end Leg 2/BASC) | 5.647 | 7.767 |
| step-19600 (end Leg 3/CSC) | 5.402 | 7.742 |

Fresh-mask cross-check (t ~ U(0.15, 0.90), N=12): standard 6.630
(step-12000) -> 6.430 (step-19600). The delivered claim "BASC made
standard loss worse, 6.091 -> 7.591" is therefore REFUTED: it compared
Leg 1's standard loss against Leg 3's cold loss. On a common metric the
standard loss improved monotonically; Leg 2 left it essentially flat
(Δ −0.08, within noise), Leg 3 improved it a little (Δ −0.25). Absolute
values differ from Kaggle's CUDA numbers (different fixed draw, bf16,
N=24 vs 200) — the trajectory, not the absolute value, is the evidence.

### Measured: cold loss flat — CSC did not move the joint prior

Cold loss 7.747 -> 7.767 -> 7.742 across the three checkpoints. 15% cold
batches at LR 1e-5 -> 0 did not measurably teach the joint prior. Kaggle's
7.5907 -> 7.5722 sits in the cold-loss range (7.7), not the standard-loss
range (5.4–6.4) — confirming Finding 1's identification.

### Measured: the model does not know the probed facts — sampler exonerated

Masked next-token probe on step-19600 (mask ONLY the next position, full
prompt as context; rank/prob of the expected token):

| prompt | expected | rank | p | top prediction |
|---|---|---|---|---|
| The capital of France is | " Paris" | 1307 / 32768 | 0.00006 | " the" (0.135) |
| 2 plus 2 equals | " 4" | 29 | 0.0054 | "," (0.042) |
| 3 times 4 equals | " 12" | 101 | 0.0012 | "," (0.046) |

The conditional is near-marginal: the model has not learned the facts,
so no sampler setting (temperature, steps, order) can surface them.
Hypothesis 2 (sampler/prompt-conditioning broken) is refuted as the
primary cause; Hypothesis 1 is answered (denoise loss ~5.4–6.4, i.e. the
model knows local structure better than the 7.57 cold number suggested,
but perplexity ~220–620 is still ~20–50x worse than a trained 1B at
~2.5–3.5). Hypothesis 3 (token scale) stands as the binding constraint:
40,140,800 tokens total = 0.047 tokens/param vs ~20 Chinchilla-optimal;
LR 1e-5 cosine-to-0 in all three legs, with Leg 3's last 600 steps at
LR < 2e-7 (effectively frozen — gate outputs at 19000 vs 19600 are
word-for-word identical for one probe).

### Implications (unchanged: release stays gated on generation)

- Do not spend more legs on samplers or cold-start variants until the
  conditional itself learns: the next lever must be training scale and
  LR, not decoding.
- Fix the gate metric before the next leg: report standard diffusion
  loss and cold loss as TWO separate numbers, never one `held_fw`
  whose meaning changes with the leg's loss function. Add the masked
  next-token probe (one forward per prompt) at each gate — it
  distinguishes knowledge failure from sampler failure immediately.
- Scale arithmetic: Leg 3 ran at ~1.47M tokens/hour on Kaggle, so 1B
  tokens ≈ 680 GPU-hours ≈ 23 Kaggle quota-weeks (30 h/week). The
  December path is the 5090 package (already built) running
  continuously plus Kaggle legs in parallel, targeting 500M–1B+ tokens.
- The three XIIV app spaces (Command Center, Training Legs, Benchmark
  Board) were last refreshed 2026-11-08 with the Leg 2 v5 verdict and
  are stale against the Leg 3 verdict + this diagnosis.

### Leg 4 sketch (prepared, not launched)

Objective: standard diffusion loss only + CSC cold term as auxiliary if
kept; metric fix above applies. LR 1e-4 (10x current; standard 1B
pretraining range 1e-4–3e-4), cosine to 0 over the leg; no 10-step probe
(it cannot distinguish learning rates) — compare the loss trajectory
against Leg 3's 5.65 -> 5.40 instead. Data: continue sequential order
from step 19600 (next unseen seqs). Honest expectation: one 15M-token
leg will not make generation coherent.
