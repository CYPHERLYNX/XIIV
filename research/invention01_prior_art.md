# Prior-Art Sweep — Invention 01: Proof-Difficulty-Adaptive Diffusion Sampling

**Date:** 2026-10-19
**Invention (one line):** A masked diffusion LM sampler that dynamically allocates denoising steps / compute per position (or region) based on *predicted difficulty of constructing a valid proof certificate* for that position — spend steps where the certificate is hardest to build, skip where it is trivial. XIIV is guarantee-native (every output carries a proof certificate; intervention rate is the metric), so the difficulty signal comes from the certificate constructor, not from token entropy.

**Kill bar (pre-registered):** prior art kills novelty ONLY if someone already allocates diffusion sampling compute using *proof/certificate construction difficulty* as the signal. Adaptive steps based on token confidence/entropy, or classifier/constraint-guided diffusion without compute adaptation, do NOT kill it — they are adjacent.

**Verdict: NOVEL — with strong, citable adjacent work.**

No paper found uses proof-certificate construction difficulty as a compute-allocation signal in any generative model, diffusion or otherwise. The adaptive-compute-in-diffusion-LM space is active (2025–2026) and must be cited honestly, but every existing method uses model-internal uncertainty (confidence/entropy/loss) as the difficulty signal — never a proof/certificate constructor. The certificate-guided diffusion work that exists (SPDD, LAVE, CFG decoding) steers *what* gets generated under constraints; none of it adapts *how much compute* each position gets.

---

## The 5 closest papers/techniques

### 1. CadLLM — Confidence-aware adaptive compute allocation (closest on mechanism)
- **Cite:** arXiv:2512.07173 (Dec 2025).
- **What it does:** A training-free controller for masked diffusion LMs with four adaptive knobs: *adaptive block size* (explicitly "compute allocation to easy vs. hard blocks"), *adaptive steps* (precision–speed dial), adaptive vocabulary size, and adaptive threshold — closing the loop between signals and compute to improve throughput.
- **Why it doesn't collide:** The difficulty signal is token *confidence* (a model-internal uncertainty measure). Proof-difficulty-adaptive sampling replaces that signal with an external, guarantee-native one: predicted cost of constructing a valid certificate. These signals diverge exactly where the invention matters — a position can be high-confidence yet certificate-hard (e.g., the model is sure of a token that breaks a global invariant), or low-confidence yet certificate-trivial. CadLLM would misallocate in both cases; the invention would not. Mechanism is adjacent; signal is novel.

### 2. SAID + CHLG — Scaffold-aware decoding with confidence-hierarchical step assignment (closest on per-position step allocation)
- **Cite:** arXiv:2606.04974 (2026). Evaluated on LLaDA-8B / LLaDA-1.5.
- **What it does:** Instead of uniform denoising, SAID builds a semantic "scaffold" first, then generates detail tokens with fewer steps. Its Confidence-Hierarchical Layered Generation (CHLG) explicitly "assigns additional denoising steps to low-confidence tokens while committing high-confidence tokens early" — per-position step allocation, like the invention.
- **Why it doesn't collide:** Same signal problem as CadLLM: the allocation key is predictive confidence. There is no notion of a certificate, a verifier, or proof-construction cost anywhere in the method. It also never adapts on *certificate* grounds — a confident-but-unverifiable token gets committed early under CHLG, which is precisely the failure mode the invention targets (XIIV's metric is intervention rate, i.e., how often the certificate constructor rejects and forces rework).

### 3. WavefrontDiffusion — Dynamic decoding schedule adapting to token difficulty
- **Cite:** arXiv:2511.19473 (2026).
- **What it does:** A "wavefront" decoding frontier whose velocity adapts to local token difficulty: it "accelerates through high-confidence (easy) regions and slows down to accumulate context for low-confidence (hard) tokens, effectively implementing a dynamic computation allocation." Replaces rigid block schedules.
- **Why it doesn't collide:** Again confidence-as-difficulty, and the adaptation is about *decoding order/pace along the sequence*, not about allocating refinement steps to maximize certificate validity. No proof or certificate concept. Adjacent mechanism, different objective and signal.

### 4. Patch Forcing ("Denoising, Fast and Slow") — per-region difficulty head with adaptive compute (closest on "learned difficulty signal")
- **Cite:** arXiv:2604.19141 (CompVis, LMU Munich, 2026). Code: github.com/CompVis/patch-forcing.
- **What it does:** Moves from global to patch-level noise scales/timesteps and adds "a lightweight per-patch difficulty head" enabling "adaptive samplers that allocate compute dynamically where it is most needed." Easier regions are advanced earlier to provide context for harder ones.
- **Why it doesn't collide:** Two gaps: (a) it is *image* diffusion, not language — no tokens, no certificates, no verifier; (b) the difficulty head predicts *denoising* difficulty (reconstruction error), which is still a property of the generative process itself, not of an external proof obligation. The invention's difficulty head would predict *certificate-construction* cost — a property of the (output, verifier) pair, potentially orthogonal to denoising difficulty. This is the paper to cite as "the same architectural pattern (learned difficulty head driving adaptive compute), applied to a different signal in a different domain."

### 5. SPDD — Primal-dual guided decoding for constrained discrete diffusion (closest on the "guarantee" axis)
- **Cite:** arXiv:2605.09749 (2026). Zero-shot transfers to LLaDA-8B.
- **What it does:** Adds a constraint-dependent logit bias at each denoising step via a primal-dual update, with theoretical guarantees on constraint satisfaction — guided diffusion decoding for constrained discrete diffusion.
- **Why it doesn't collide:** SPDD steers *what* the model generates (constraint satisfaction via biasing); it does not adapt *how much compute* any position receives. The step budget is fixed and uniform. The invention is orthogonal and composable: one could run SPDD-style guidance *inside* a proof-difficulty-adaptive step allocator. Cite as the nearest "guarantees meet diffusion decoding" work, and note the composition explicitly — it strengthens rather than weakens the novelty claim.

---

## Also checked (adjacent, not colliding)

- **MGDM** (arXiv:2410.14157): token-level difficulty reweighting in the *training* loss + easy-first TopK *decoding order*. Difficulty = cross-entropy loss. About order, not step allocation; no certificates.
- **AdaDiff** (arXiv:2311.14768): RL-learned per-*prompt* step counts for image diffusion. Wrong granularity (prompt, not position) and wrong signal.
- **BACD / Adaptive Test-Time Compute for Block Diffusion** (2026): difficulty-aware denoising with average confidence of decoded tokens as the signal. Confidence again.
- **LAVE** (arXiv:2602.00612) and **CFG constrained decoding for diffusion LLMs** (SRI/ETH Zurich, 2025): verify-then-accept constrained decoding for diffusion LMs. Verification-gated, but the compute budget is not adapted per position; no proof-difficulty signal.
- **Learning Adaptive LLM Decoding** (arXiv:2603.09065): RL adapters selecting decoding strategies with verifiable rewards — autoregressive LLMs, strategy selection, not diffusion step allocation.
- **ADDS (Adaptive Diffusion Denoised Smoothing)** (arXiv:2507.08163): "adaptive" + "certified" appear, but this is certified *robustness* (randomized smoothing, images) — "certificate" here means a robustness radius, not a proof-carrying output. Disambiguate if cited; unrelated.
- **LLaDA low-confidence remasking** (arXiv:2502.09992): the standard confidence-based remasking baseline everything above builds on. Not adaptive compute.

## Honest caveats (what could weaken the claim)

1. **Signal correlation risk.** If proof-construction difficulty turns out to be highly correlated with token entropy in practice, the invention reduces to "CadLLM with extra steps" and the novelty is thin. The discriminating experiment: find positions where the model is *confident but the certificate is hard* (global-invariant violations are the natural source) and show the allocator spends compute there while confidence-based allocators don't. If that divergence doesn't exist, don't file this as novel.
2. **"Difficulty head" pattern is taken.** Patch Forcing owns the learned-difficulty-head-for-adaptive-compute pattern in images. The claim must rest on the *signal* (certificate difficulty), not the *mechanism* (adaptive allocation), and the write-up should cite Patch Forcing as the direct architectural ancestor.
3. **No certificate-cost predictor exists yet.** The invention assumes a cheap predictor of per-position certificate-construction difficulty. If building that predictor costs more than the steps it saves, the idea is net-negative. A pilot measurement (predictor cost vs. step savings vs. intervention-rate delta) is the mandatory first experiment before any novelty claim is finalized.
4. **Search limits.** Sweep covered arXiv cs.CL/cs.LG, web search, and paper HTML/PDFs through Oct 2026. Non-English venues and very recent (last ~2 weeks) preprints could hold something missed; re-run the sweep before committing the claim to any publication.

## Bottom line

**NOVEL, conditionally.** The precise claim — *allocating diffusion sampling compute using predicted proof-certificate construction difficulty as the signal* — has no prior art found. The surrounding territory is well-populated (confidence-adaptive diffusion sampling is a 2025–2026 hot area; constraint-guided diffusion decoding exists), so the write-up must cite CadLLM, SAID/CHLG, WavefrontDiffusion, Patch Forcing, and SPDD as adjacent work and locate the novelty exactly at the signal: certificate difficulty, not model confidence. Proceed to the discriminating experiment in caveat (1) before treating this as a locked invention.
