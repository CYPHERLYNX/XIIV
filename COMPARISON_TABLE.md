# XIIV-133M benchmark comparison — final numbers + honest gap analysis

Compiled 2026-10-09 (research); final XIIV-133M numbers filled 2026-10-10
(full runs finished ~23:07 UTC Oct 9, ~6h on CPU). Updated 2026-10-11 with
exact statistics and gap analysis.

---

## ⚠️ METHODOLOGY CAVEAT — READ FIRST

**XIIV's scores are NOT apples-to-apples with the published rows below.**

- Published models (GPT-2, Mistral, LLaMA, LLaDA, GPT-4) are scored with the
  stock lm-eval harness: **autoregressive teacher-forcing logprobs**.
- XIIV is a **masked discrete-diffusion LM** — it has no next-token distribution,
  so stock lm-eval scoring is meaningless for it. XIIV's numbers come from a
  custom scorer (`xiiv/diffusion_scorer.py`, via `xiiv/lm_eval_wrapper.py`):
  each (context, continuation) is scored by **leave-one-out diffusion
  pseudo-likelihood** — each continuation token is masked individually with
  all other tokens visible; mean NLL; rank choices by NLL (lower = better).
  Deterministic, no random masking samples.
- Diffusion-NLL ranking and AR-logprob ranking are **different estimators of
  different quantities**. They agree on chance-level (25%) but there is no
  validated calibration between them above chance. A 3-point gap between XIIV
  and GPT-2 on this table does **not** mean the same thing as a 3-point gap
  between two AR models.
- **Scorer history:** the pre-2026-10-11 scorer applied random masking across
  the full sequence then unmasked context, leaving the continuation's
  effective masking ratio uncontrolled (sometimes zero masked tokens =
  degenerate scores). Fixed 2026-10-11: masks only continuation positions,
  one at a time; 6/6 regression tests in `tests/test_diffusion_scorer.py`.
  Numbers before this fix are superseded.
- **Residual caveat (inherent):** bidirectional attention sees future
  continuation tokens during scoring, unlike AR teacher-forcing. Treat
  cross-architecture numbers as **directional, not comparable**.
- What transfers across the paradigm gap: **rank vs chance, and the
  before/after delta** once the same scorer is used on a trained rung.

Do not present XIIV's numbers beside published AR numbers without this caveat.

---

## XIIV-133M final numbers (the honest "before")

Model: `checkpoints_100m_gn/ckpt_8218.pt` — 133.52M params, d=704/L=15/h=11,
FFN 2048, vocab 8192. Trained on **4,207,616 TinyStories tokens**
(children's stories). **Zero instruction tuning.** 0-shot.

| Task | n | acc | acc ± 1σ | acc_norm | acc_norm ± 1σ |
|---|---|---|---|---|---|
| ARC-Easy (full) | 2376 | 24.92% | ±0.89% | **25.55%** | ±0.89% |
| ARC-Challenge (full) | 1172 | 24.83% | ±1.26% | **28.50%** | ±1.32% |
| HellaSwag (200-subset) | 200 | 26.00% | ±3.11% | **22.50%** | ±2.96% |
| Chance baseline | — | 25% | — | 25% | — |

Source files: `benchmark_133m_arc_easy.json`, `benchmark_133m_arc_challenge.json`,
`benchmark_133m_hellaswag.json`. Scorer: `xiiv_diffusion` (custom, see caveat).

**Reading:** all three tasks sit at chance. ARC-Challenge acc_norm (28.5%) is
1.3σ above chance — noise, not signal. HellaSwag acc_norm (22.5%) is 0.8σ
below chance — also noise. The HellaSwag 200-subset has ±3pp stderr; treat
any gap under ~6pp on that row as unresolvable.

---

## Published reference numbers

| Model | Params | ARC-Easy | ARC-Challenge | HellaSwag | Setting / source |
|---|---|---|---|---|---|
| GPT-2 124M (official) | 124M | 39.2% (acc_norm) | 22.5% (acc_norm) | 34.3% (acc_norm) | 0-shot, lm-eval; zzkai098/reproduce-gpt2 |
| LLaDA 8B base (diffusion) | 8B | — | 47.9% | 72.5% | 0-shot; LLaDA paper arXiv:2502.09992 Table 1 |
| LLaDA 8B instruct (diffusion, SFT) | 8B | — | 88.5% | 74.6% | 0-shot; LLaDA paper Table 2 |
| iLLaDA 8B (diffusion) | 8B | — | 60.8% | 76.6% | Improved diffusion LM paper, base table |
| Dream 7B (diffusion) | 7B | — | 59.8% | 73.3% | Diffusion fine-tune on Qwen2.5 |
| GPT-3 175B | 175B | 70.1% (few-shot) | 51.5% (few-shot) | 79.3% (few-shot) | GPT-3 paper (0-shot: 68.8 / 51.4 / 78.9) |
| LLaMA-2 7B | 7B | 68.7% | 43.2% | 77.1% | Mistral 7B paper table, 0-shot |
| Mistral 7B v0.1 | 7B | 80.0% | 55.5% | 81.3% | Mistral 7B paper table, 0-shot |
| GPT-3.5 | 175B | — (note 3) | — | 85.5% (10-shot) | GPT-4 tech report Table 1 |
| GPT-4 | — | — (note 3) | — | 95.3% (10-shot) | GPT-4 tech report Table 1 |

### Notes

1. Shot counts differ: GPT-4/GPT-3.5 HellaSwag are 10-shot; open-model paper
   tables are 0-shot; XIIV is 0-shot. Label shot counts, don't mix silently.
2. GPT-4's report bundles ARC-Easy + ARC-Challenge into one "ARC" number
   (25-shot: GPT-4 96.3%, GPT-3.5 85.2%) — not ARC-Easy or ARC-Challenge.
3. **LLaDA is the honest architecture peer** — same family (discrete
   diffusion), same tasks. But the scale gap is extreme: LLaDA base saw
   **2.3T tokens**; XIIV-133M saw **4.2M story tokens** (~500,000× less).
   The instruct-LLaDA ARC-C row (88.5%) is the "diffusion models can do this"
   proof point, not a target for a 133M.
4. GPT-2 124M is the honest **size peer** — but the data gap dominates:
   GPT-2 saw ~40GB WebText (~10B tokens); XIIV saw 4.2M story tokens
   (~2,400× less, narrow children's-story domain, zero instruction tuning).

---

## Honest gap analysis

### What the numbers actually say

XIIV-133M is at chance on all three tasks. That is exactly what the training
setup predicts: 4.2M children's-story tokens cannot teach grade-school science
(ARC), physical commonsense (HellaSwag), or anything outside the story domain.
A model that has never seen a science question has no reason to beat chance on
one. The "before" number is honest and uninteresting — its value is as the
baseline the "after" will be measured against.

### Decomposing the gap vs GPT-2 124M (size peer)

| Gap component | Evidence | Contribution |
|---|---|---|
| **Data scale** | GPT-2: ~10B WebText tokens. XIIV: 4.2M story tokens (~2,400× less) | Dominant. Explains nearly all of the ARC-Easy gap (25.5 vs 39.2). |
| **Data domain** | Stories vs broad web text. ARC/HellaSwag test science + commonsense, absent from stories | Large. Even at equal token counts, story-only data would underperform. |
| **Instruction tuning** | GPT-2: zero (base model). XIIV: zero | None — both untuned. Not a differentiator here. |
| **Scorer paradigm** | Diffusion-NLL vs AR-logprobs (see caveat) | Unknown magnitude. Could shift XIIV ±several points either way. |
| **Architecture** | Masked diffusion vs autoregressive | Unknown. LLaDA proves diffusion can reach 88.5 ARC-C at 8B/2.3T — architecture is not the blocker. |

**Bottom line:** the gap is overwhelmingly **data**, not architecture. Nothing in
these numbers suggests masked diffusion can't do these tasks — LLaDA already
does. Nothing suggests 133M is too small either — GPT-2 124M beats chance by
14pp on ARC-Easy with the same parameter budget but 2,400× the data.

### What we cannot claim

- We cannot claim XIIV-133M is "worse than GPT-2 by 14 points" — the scorer
  paradigm gap makes point comparisons invalid.
- We cannot claim the architecture is validated or invalidated — at 4.2M
  tokens, no architecture would beat chance here.
- We cannot project what the 500M or 1B will score — no trained rung above
  133M exists yet, and the 500M checkpoint is a grown init, not a trained
  model.

---

## Sources

- GPT-4 tech report Table 1: https://github.com/viktor-shcherb/msc_thesis/blob/HEAD/references/2023-03-gpt-4-technical-report/analysis.md
- GPT-2 124M head-to-head (0-shot acc_norm): https://github.com/zzkai098/reproduce-gpt2
- GPT-3 paper ARC/HellaSwag: https://github.com/viktor-shcherb/msc_thesis/blob/HEAD/references/2020-12-gpt-3-few-shot-learners/analysis.md
- Mistral 7B paper Table (arXiv:2310.06825): http://arxiv.org/pdf/2310.06825
- LLaDA paper Tables 1–2 (arXiv:2502.09992): http://arxiv.org/pdf/2502.09992
- iLLaDA / Dream diffusion rows: https://arxiv.org/pdf/2606.25331
- GPT-3 HellaSwag cross-check: https://github.com/LudwigStumpp/llm-leaderboard
