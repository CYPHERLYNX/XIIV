# XIIV Daily Research — 2026-10-20

Part A of `xiiv-daily-model-improvement`. Text-search only (no code written, no GPU needed for any of this).

## 1. Masked diffusion LMs

- **LLaDA2.1 (Ant Group, arXiv 2602.08676, Feb 2026)** — "Draft-and-Edit": weaves Token-to-Token (T2T) editing into Mask-to-Token (M2T) decoding via dual thresholds (τ_mask for unmasking, τ_edit for editing). Also reports the **first large-scale RL framework specifically tailored for dLLMs** (stable gradient estimation for diffusion). Released 16B/100B variants; 892 TPS on HumanEval+.
  - Fit: STRONG. T2T editing is a sampler-level change — implementable on CPU (pure decoding logic). RL-for-dLLM framework matters for the future RLVR-style training of the 1B (with verifier rewards), but needs real compute later; sampler is implementable now.
- **Seed Diffusion (ByteDance, arXiv 2508.02193, Aug 2025)** — discrete-state diffusion LM with a Two-Stage Curriculum (80% mask-based, then edit-based corruption for calibration) + trajectory distillation + **verifier-gated on-policy RL**. Code-focused preview.
  - Fit: STRONG on the verifier-gated RL idea — "train the diffusion model against a programmatic verifier as reward" is exactly the guarantee-native training loop. Implementable on CPU in principle (RL loop is compute-bound; prototype the reward side with contracts.py).
- **DUEL (arXiv 2603.01367, 2026)** — exact likelihood for masked diffusion via deterministic unmasking; beats MDLM/ELBO bounds on PTB/Wikitext/LM1B/etc.
  - Fit: EVAL. Exact likelihood gives honest, non-gameable eval numbers for XIIV's diffusion model — implementable on CPU (evaluation-time only, needs a trained model but no training). Useful for the "never fabricate benchmarks" bar.
- **LLaDA 8B (NeurIPS 2025), MMaDA (arXiv 2505.15809v2), Mercury (Inception Labs, commercial), Gemini Diffusion (Google, experimental)** — landscape confirmation: diffusion LMs are now a serious parallel track; masked diffusion is no longer fringe.

## 2. Guarantee-native / constrained generation / proof-carrying outputs

- **Correctness-Guaranteed Code Generation via Constrained Decoding (arXiv 2508.15866, Aug 2025)** — context-sensitive constrained decoding with **token healing**: DFA built from a CFG regex at each step, Gumbel-top-k adaptive rejection sampling over model logits, parser advances as tokens are generated.
  - Fit: VERY STRONG. This is the closest published prior art to XIIV's proof-carrying thesis — generation *steered* by a verifier at every step. Token healing (partial-token prefix matching) is a genuinely clever CPU-implementable technique XIIV's contracts.py constrained sampler could adopt.
- **Constrained Decoding for Secure Code Generation (arXiv 2405.00218v3)** — constrained beam search improves secure-pass@1 across 7 code models (CodeLlama-34B secure-pass@1 57.97 vs 48.99 for beam).
  - Fit: MEDIUM. Empirical support for the guarantee-native thesis; method itself (constrained beam) is standard.
- **DeepProve-1 (Lagrange, 2025)** — first zkML proof of a full GPT-2 inference (cryptographic proof-carrying inference).
  - Fit: LOW for XIIV. zkSNARKs over transformer inference are far too heavy for CPU and philosophically heavier than needed — XIIV's certificates are cheap programmatic verifiers, not cryptographic proofs.
- **Dafny-based verified synthesis preprint (preprints.org, Apr 2026)** — LLMs + Dafny formal verifier; "self-healing" loop (compile→verify feedback→retry) reaches 90.91% verify@5 with Gemma 4-31B.
  - Fit: STRONG directionally — verifier-feedback self-healing loops are the training-time analogue of XIIV's certificates. Validates the self-improvement loop design in SELF_IMPROVEMENT_LOOP.md.

## 3. Parameter-efficient training

- **LoRA family 2025**: C-LoRA (continual learning, single shared backbone + routing matrix, orthogonality penalty — Feb 2025); LoRA-A² (federated, alternating-freeze, rank-1 robust — 2024); LoRA-XS (frozen SVD matrices, 100× storage cut); DoRA/AdaLoRA/DyLoRA/LoRA+ (established).
  - Fit: MEDIUM. XIIV already runs LoRA adapters; C-LoRA's orthogonality-regularized shared backbone is the most relevant for XIIV's coreset-anchored streaming (anti-forgetting) — but needs a GPU eval before adopting.
- **GaLore / BAdam** — full-param memory-efficient optimizers.
  - Fit: MEDIUM. Relevant if XIIV ever does full 1B training on limited memory; not today's priority (SGBD path owns block training).

## 4. CPU / low-memory training

- **MeZO (Malladi et al., 2023, actively extended through 2025)** — memory-efficient zeroth-order optimizer: gradient estimate from 2 forward passes via SPSA with in-place seed-resampling; **same memory footprint as inference**; trains 30B on a single A100-80GB; crucially, **can optimize non-differentiable objectives** (accuracy, F1).
  - Fit: VERY STRONG. A verifier is a non-differentiable objective. MeZO lets XIIV optimize *directly against certificate validity* with inference-level memory — feasible on the 7GB box for the smaller blocks/seed model.
- **FZOO (arXiv 2506.09034, Jun 2025)** — fast ZO optimizer, Adam-scale convergence speed, 18× fewer forward passes than MeZO on RoBERTa-large, formally equivalent to normalized-SGD.
  - Fit: STRONG as a future upgrade path if MeZO convergence is too slow; FZOO's batched one-sided estimates are CPU-friendly.
- **Sparse MeZO (NeurIPS 2025)** — sparsity-masked ZO updates; "less parameters for better performance."
  - Fit: MEDIUM — useful if combining MeZO with block-wise training (only perturb SGBD block params).

## 5. Attention / sampling / architecture

- **LESS (arXiv 2606.16908, Jun 2026)** — "LESS Is More": training-free, model-agnostic adaptive sampler. Treats token commitment as an **online stopping problem**: unmask a position only when (i) top-1 confidence is high, (ii) top-1 token persists across recent reverse steps, (iii) predictive distribution is stable under top-K inter-step Jensen–Shannon divergence. Evaluated on Dream-7B, LLaDA-8B, LLaDA-1.5-8B across 7 benchmarks; beats strong adaptive samplers while using **72.1% fewer reverse steps** than fixed-budget decoding.
  - Fit: VERY STRONG. Pure sampling algorithm, zero training, CPU-implementable (numpy-level ops on logits), directly improves diffusion generation quality AND speed. This is the day's top pick (see below).
- **CadLLM (arXiv 2512.07173)** — training-free confidence-aware calibration; adaptive compute allocation across blocks/steps vs Fast-dLLM's static thresholds.
- **Learning Unmasking Policies for dLLMs (arXiv 2512.09106)** — learned unmasking policies beat confidence heuristics on LLaDA-8B/GSM8K.
- **Fast-dLLM (Wu et al. 2025)** — confidence-threshold parallel block decoding; baseline everyone now beats.
- **Diffuser (arXiv 2210.11794v2, alphaxiv)** — multi-hop attention diffusion via Personalized PageRank on sparse attention matrices; linear complexity approximating full attention.
  - Fit: MEDIUM-LOW for XIIV now — architectural change requiring retraining; linear-attention family (Gated DeltaNet, Log-Linear Attention ICLR 2026) targets AR long-context, less relevant to masked diffusion's bidirectional full-attention design.
- **Gated DeltaNet (ICLR 2025)** — de-facto linear attention layer (Qwen3-Next, Kimi Linear, OLMo Hybrid).
  - Fit: LOW for XIIV's thesis; bidirectional diffusion blocks don't need causal linear attention.

---

## THE PICK: Verifier-Gated Mutual-Stability Unmasking ("STABILISS")

**What it is:** Port the core of LESS (2026) into XIIV's diffusion sampler (`xiiv/diffusion.py`): commit a masked position only when its top-1 prediction is high-confidence, persists across recent reverse steps, and has low inter-step JS divergence. Then add XIIV's genuinely novel extension: a **fourth gate — certificate validity**. Before committing a position, run the cheap partial-program verifier (`contracts.py` machinery) on the emerging partial sequence; unmask only if the partial sequence remains *certifiable* (can still complete to a valid proof). If a statistically stable token would break certifiability, hold the mask and force a re-draw.

**Why this is the one:**
1. Fits the thesis twice over: it is a *diffusion-sampling* innovation (the current state of the art in 2026) AND it makes generation proof-carrying by construction — "every answer carries its proof" enforced *during* unmasking, not as post-hoc filtering.
2. Implementable with zero GPU: pure numpy/Python logic over logits plus the existing verifier code; testable on the 133M seed model on CPU.
3. Honest incremental path: implement LESS's 3-gate stability first (a faithful, cited port), measure unmasking-step reduction vs fixed-budget decoding, then add the certificate gate and measure the validity-rate delta — no fabricated numbers, everything measured.
4. Prior art scan says the combination is new: LESS/CadLLM/unmasking-policy papers check only *statistical* stability; constrained-decoding papers (2508.15866) steer *AR* models with CFGs but never inside a diffusion unmasking schedule. Proof-gated diffusion unmasking appears unpublished.

**Working name:** **STABILISS** — *STABle Inference via LIcensed (certified) Selective Sampling* … honestly, the acronym is forced. Alternative working name: **Certificate-Gated Mutual-Stability (CGMS) unmasking**. Recommend CGMS as the working name in code; it describes what it is.

**Implementation sketch (for Part B / a future run):**
- In `xiiv/diffusion.py`'s unmasking loop, keep a per-position ring buffer of the last k predictive distributions.
- Compute: top-1 confidence c; persistence count; top-K JS divergence between step t and t−1.
- Eligible set E = {positions with c ≥ τ_c, persistence ≥ p_min, JS ≤ δ}.
- Certificate gate: for each position in E (ordered by stability), tentatively unmask with its top-1 token, run `contracts.partial_certificate_check(sequence)`; commit only if certifiable. If none pass, fall back to the single most-stable position (LESS-style progress guarantee), mirroring Fast-dLLM's anti-stall rule.
- Log unmasked-per-step and certificate-pass-rate; report both.

**Runner-up invention (also CPU-feasible, worth a later run):** MeZO-with-verifier-reward post-training — use MeZO (SPSA, inference memory) to fine-tune a diffusion block/adapters directly against certificate-validity as a non-differentiable reward. The optimizer is CPU-native; the reward is `contracts.py`. Novel combination for diffusion LMs as far as today's scan shows.

## Uncertainty notes
- Search results included some 2026-dated arXiv IDs (2602.08676, 2603.01367, 2606.16908); dates taken from the search-result pages themselves, not independently verified against arXiv. The NeurIPS-2025 LLaDA paper (published 2025-09-19) and LLaDA2.1 (Feb 2026) are consistent with the known timeline.
- The "LLaDA 2" colloquial naming note (spheron.network, Apr 2026) clarifies that community "LLaDA2" = LLaDA-8B checkpoints; keep separate from LLaDA2.1 (Ant Group's token-editing paper).
- The Diffuser alphaxiv entry (2210.11794) is a 2022 paper surfaced in a 2026 search — flagged as older work, not a 2026 breakthrough.
