# XIIV — Prior Art & Novelty Statement (2026-10-01)

Status: working document. No public novelty claim may be made until the checks
below are completed across web, GitHub, Hugging Face, Product Hunt, arXiv/Scholar,
and package registries.

## What we do NOT claim
The individual mechanisms are not independently unprecedented:
- Masked/diffusion language modeling (LLaDA-style absorbing diffusion)
- Arbitrary-order / configurable decoding order (cf. Dream 7B)
- Interactive token remasking / mid-generation intervention (cf. LLM Visualizer,
  https://github.com/jsturges28/diffusion-llm, and its interactive-remasking plan)
- Locking converged tokens to skip recomputation (cf. SureLock, arXiv:2602.06412v1)
- Insertion/deletion diffusion (arXiv:2603.23507v1)
- Length-adaptive masked-diffusion decoding (arXiv:2608.22274)

Any public writing must cite these and describe XIIV as building on them.

## What IS defensible as distinctive (the integrated package)
To our knowledge (pending the full checks above), no single shipped artifact
combines ALL of:
1. Constraint-native sampling where structural conditions (exact token length,
   locked spans, required/forbidden token sets, named generation orders) are
   enforced DURING the denoising loop with a stated invariant — not filtered after.
2. First-class mid-process intervention: pause, pin tokens, edit/unmask spans,
   resume — with the RNG + canvas state serialized, bit-reproducible.
3. A from-scratch small model (10M params, CPU-trainable) with released weights,
   frozen run config (config hash + code SHA), pinned data snapshot, and a
   leakage-checked data contract.
4. An honest eval harness that reports ONLY measured numbers, with the explicit
   caveat that masked-NLL is an upper bound not comparable to AR perplexity.
5. A reproducible CLI from data prep → training → sampling → eval → export.

## Claims we make (each gated by a test or measurement)
- Exact output token length: 100% compliance (test: exact-length suite).
- Locked-span preservation: bit-for-bit (test: frozen-mask suite).
- Required/forbidden token sets: enforced during denoising, impossible sets
  rejected with a clean error (tests: 50-seed adversarial suites).
- Fixed-seed determinism: byte-identical outputs per order preset (test).
- Efficiency: measured CPU tok/s, peak RSS (eval harness); weight integrity and
  param count verified by promotion gate G9.

## Claims we do NOT make
- Superior fluency, factuality, reasoning, or general language quality.
- Any comparison to named frontier models without same-data, same-scale,
  compatible-metric evidence (an AR baseline on identical data is queued to
  provide exactly this).
- Trademark/name clearance beyond a broad web search (full checks pending).

## To complete before publication
- [ ] Re-verify each prior-art link above is live and accurately described.
- [ ] GitHub/HF/Product Hunt/arXiv/PyPI search for overlapping integrated work.
- [ ] Name checks: GitHub, Hugging Face, Product Hunt, arXiv/Scholar, PyPI/npm,
      product + trademark databases (ordinary search ≠ legal clearance).
      - 2026-10-01 broad web search for "XIIV": no AI model found. One unrelated
        Android ADB toolkit repo (xiiv420/xiiv_phone_control_center) and a
        Talkie AI character ("Mark-XIIV") use the string; neither is an AI model.
      - 2026-10-01 ~17:00 IST dedicated checks: GitHub code search (only
        unrelated hits: ADB toolkit, an Eclipse-plugin username, a pixel-art
        game, roman-numeral "XIIV" references); Hugging Face API
        (api/models?search=xiiv → []); PyPI (404, no package); npm
        ("Not found"). No AI model named XIIV anywhere. Trademark databases
        still unchecked — ordinary search is not legal clearance, and the
        repo ships private, which bounds the exposure.
- [ ] AR baseline trained and measured before any comparative sentence is written.
