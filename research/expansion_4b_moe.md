# XIIV-4B: sparse MoE expansion of the 1B (side work, target Dec 24)

**Directive (Fahad, 2026-10-30):** push parameters + training from 1B to
4B by Dec 24, ready to use in opencode easily. Runs as SIDE work in
parallel with the Dec-10 1B campaign — architecture and code now, expert
training after the 1B backbone is final.

**Why MoE, not dense:** dense 4B training is not computable on available
hardware (Kaggle 30 h/week; this box OOMs past ~1B). Sparse MoE keeps the
trained 1B as a FROZEN backbone and trains only new experts — the same
SGBD-grown philosophy as the 1B itself.

## Sizing (exact-4B design)

Backbone (frozen): exact-1B, 284 tensors, runnable 857,429,760,
headline 1,000,000,000. Per-layer FFN (d=1280, ffn=3584): ~9.17M params.

Expansion: every layer's FFN becomes MoE with **8 experts**, top-2
routing, one shared router per layer (d=1280 → 8, ~10K params/layer,
negligible).
- New params/layer: 8 × 9.17M = 73.4M (replaces the 9.17M FFN)
- Net new total: 40 layers × (73.4M − 9.17M) ≈ 2.57B
- **Headline total ≈ 1.0B + 2.57B ≈ 3.57B → sized to 4B** with the
  router + a small capacity margin; exact tensor inventory fixed in
  the build script (same audit discipline as the exact-1B stitch:
  assert total == 4,000,000,000 before any release claim).
- **Active per token:** backbone 857M + 2 experts × 9.17M × 40 ≈
  **1.59B** — fits ~4 GB in fp16, easy in opencode on modest GPUs.

## Growth method (mirrors SGBD)

1. **Grown identity init:** each expert is initialized as an exact
   copy of the backbone FFN it replaces (like the exact-1B E-block
   identity gate). Router initialized near-uniform. Day-0 invariant:
   the MoE model produces character-identical outputs to the 1B.
2. **Expert training:** backbone frozen; only experts + routers train,
   one layer-group at a time (SGBD block discipline), on FineWeb-Edu
   with the winning Dec-10 training step (BASC or CSC or combined).
3. **Load balancing:** auxiliary router-balance loss (standard
   Switch-Transformer style, weight 0.01) so experts diversify instead
   of collapsing onto the identity copy.

## Opencode readiness

- Same `xiiv/openai_server.py` interface; `--model-id xiiv/xiiv-4b`.
- MoE routing runs in the forward pass (no external router service).
- fp16 weights for distribution (fp32 master kept for training).
- The Dec-24 package mirrors the 1B opencode zip: weights, tokenizer,
  config, expansion manifest, server, USAGE.

## Dependencies & risks (honest)

- The 1B backbone must be GOOD by Dec 10 — experts amplify the
  backbone; a broken backbone grows into a bigger broken model.
  If the Dec-10 1B fails its gates, the 4B schedule slips by the
  same mechanism, reported plainly.
- Expert training needs GPU hours AFTER the 1B campaign — Dec 10–24
  quota (~60 h) is the budget; 40 layer-groups will NOT all train.
  Priority: train expert groups for the layers where the 1B's gates
  are weakest, measured, not all 40.
- 4B headline is a parameter count, not a capability claim — the
  Dec-24 verdict is the same harness + human rating as the 1B.
