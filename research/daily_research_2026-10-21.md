# Daily Research Note — 2026-10-21

**Track:** Invention #1 gate 2 + diffusion backbone (Part A of xiiv-daily-model-improvement).
**Part B today:** Invention #1 GATE-2 probe (`research/invention01_gate2_probe.py`) — the owed payoff
measurement: does allocating refinement compute by certificate difficulty reduce the intervention
rate vs confidence-based allocation at equal budget? Pre-registered McNemar paired test, α=0.05.

## Research sweep (4 targeted searches; claims from search-result pages/arXiv abstracts)

### 1. Masked diffusion LMs
- **Yandex MMD post-training** (yandex-research/dlm-mmd, via aiweekly 2026-10): minimizes Maximum Mean
  Discrepancy between generated and reference distributions in the *feature space of a frozen
  pretrained DLM* — 17–21% generative-perplexity cut vs IDLM at matched entropy; DMax-LLaDA2.0
  variants gain 10.3–16.5% tokens-per-forward on math +2.4 HumanEval-Instruct pass@1. Post-training
  itself is cheap (hundreds of steps); the base model training is frontier-GPU. The *loss idea* is
  CPU-implementable — see today's pick.
- **D2F — Discrete Diffusion Forcing** (arXiv 2508.09192): distill D2F dLLMs from bidirectional dLLMs
  with an asymmetric distillation loss; denoises token blocks with monotonically increasing mask
  ratios in parallel; pipelined parallel decoding → 50×+ acceleration on LLaDA/Dream math+code,
  2.5× faster than LLaMA3-8B on GSM8K. Needs GPU distillation — no lever for this box, but the
  *block-monotone-masking* schedule is a sampler idea worth stealing for `xiiv/diffusion.py`.
- **Self Speculative Decoding for Diffusion LLMs** (arXiv 2510.04147): draft+verify structure moved
  into diffusion space. Conceptually CPU-implementable; overlaps Invention #1's adaptive-compute
  neighborhood (spend verify-budget where the draft is certificate-hard) — noted, not distinct
  enough to file separately.
- **DSL-LLaDA** (arXiv 2606.01024): continuous embedding-space denoising via Discrete Stochastic
  Localization, 1,000-step CPT of LLaDA-8B. GPU-only.
- **DREAM context-adaptive noise rescheduling**, **MMaDA** (multimodal diffusion) — background.

### 2. Guarantee-native / constrained generation / proof-carrying outputs
- **PCRLLM — Proof-Carrying Reasoning with LLMs under Stepwise Logical Constraints**
  (arXiv 2511.08392): constrain reasoning to single steps, each verified against a target logic →
  proof-carrying outputs enabling automatic scoring of reasoning chains; plus a benchmark schema that
  *generates large-scale step-level training data* for formal reasoning. **Closest to XIIV's thesis
  found this month.** Honest relation to our line: per-step verification is already VRC doctrine
  (verify-or-halt), so the portable part is their *data-generation schema* — we need
  certificate-annotated step data at scale for guarantee-native training, and a PCRLLM-style
  generator schema is the CPU-feasible way to get it. Logged as related work for the proof-machine
  line, not as a new invention.
- Constrained-decoding ecosystem (Outlines, XGrammar, Guidance, SynCode, Itergen, vLLM/SGLang
  structured generation) — mature, *syntactic*. The hackernoon classifier piece (Sep 2026 study:
  schema-valid ≠ correct; 8% misrouted tickets with 0% type-error rate) is a clean articulation of
  why XIIV's thesis is *semantic* guarantees, not syntactic validity. Thesis support, not a lever.

### 3. Parameter-efficient training / CPU training
- No new CPU-training breakthrough surfaced today. LoRA/QLoRA/DoRA explainers only; GaLore recap
  (2024). SCALE optimizer (yesterday's note) remains the queued training-economics lever for
  SGBD/LoRA on the 7GB box. Nothing to add.

### 4. Attention / architecture
- **Superlinear Multi-Step Attention** (arXiv 2601.18401, Jan 2026): reformulates causal attention as
  an N-step search problem, O(L^{1+1/N}) (N=2 → O(L^{3/2}), jump-search-like); 114 tok/s at 1M
  context on a modified 30B hybrid MoE (B200). Public paper, honest about limited evaluation.
  XIIV's diffusion blocks are bidirectional — not directly portable — but the *multi-step search*
  framing is prior art to cite if XIIV ever touches long-context attention.
- **SubQ** (Subquadratic, Miami): still no public paper; vendor claims only (12M context, 52×
  FlashAttention at 1M). Per the benchmarks-to-beat rule: beat on *verified guarantees +
  proof-carrying output at long context* (their efficiency claims remain unaffiliated-unverified),
  never by copying their mechanism. No implementation lever.

## THE PICK: Guarantee-MMD post-training (candidate invention, queued)

**What it is:** Port Yandex's MMD post-training idea with a guarantee-native twist. Instead of
matching generated vs reference distributions to cut perplexity, match the *frozen DLM's
feature-space distribution of certificate-VALID generations*: post-train (tiny model, CPU-feasible
at toy scale) with an added MMD term that pulls the sampler's output distribution toward the
region of feature space where the external verifier accepts. Perplexity is the wrong objective for
XIIV — *intervention rate* is; this makes the training objective match the metric.

**Why this one:**
1. Fits the thesis exactly: guarantee-native *training* (not just decoding) — the missing half of
   "every answer carries its proof."
2. Implementable with zero GPU at pilot scale: the MMD term is kernel arithmetic over frozen-model
   features; the tiny 2-layer CPU model from the gate-1/2 probes is the testbed. Measurable:
   intervention rate before/after on held-out constraint sets.
3. Honest novelty posture: Yandex owns MMD-for-perplexity in DLMs; SPDD owns constraint-guided
   *decoding*. MMD-against-verifier-acceptance as a *training* objective for diffusion LMs appears
   unpublished as of today's sweep — but that is a same-day-search verdict, not a full prior-art
   sweep. A dedicated sweep (arXiv cs.CL/cs.LG + web, last 2 weeks included) is OWED before this
   is filed as an invention, same bar as Invention #1.

**Queued behind:** yesterday's CGMS (certificate-gated stability unmasking) and today's gate-2
result. If gate 2 PASSES, the natural next step is composing all three: CGMS unmasking +
load-adaptive refinement + Guarantee-MMD post-training, measured on intervention rate.

## Fit-to-thesis scorecard (today's sweep)
| Finding | Diffusion fit | Guarantee fit | CPU-feasible | Action |
|---|---|---|---|---|
| Yandex MMD loss idea | ✓ (DLM post-training) | twist → ✓ | ✓ (pilot) | PICK (queued, sweep owed) |
| PCRLLM step-data schema | ○ (AR reasoning) | ✓✓ | ✓ | related work; data idea for proof-machine line |
| D2F block-monotone masking | ✓ (sampler) | — | idea only | sampler note for diffusion.py |
| Self-speculative dLLM decoding | ✓ | ○ | ✓ | overlaps Invention #1; no separate filing |
| Superlinear attention | ○ (causal AR) | — | — | cite-if-needed |
| SubQ | — (unverified) | beat-target | — | beat on guarantees, not mechanism |

## Uncertainty notes
- Yandex MMD numbers (17–21%, +2.4 HumanEval) are from the aiweekly summary of the HF abstract;
  the paper itself was not fetched. Treat as reported.
- arXiv IDs 2601.18401 (Superlinear), 2511.08392 (PCRLLM), 2508.09192 (D2F), 2510.04147
  (self-speculative) are as given by search results; abstracts not independently verified.
- The Guarantee-MMD novelty verdict is same-day-search only — full prior-art sweep required
  before any invention claim, per the standing novelty rule.
