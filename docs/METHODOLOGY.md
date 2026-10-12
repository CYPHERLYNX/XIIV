# XIIV Methodology: Guarantees, Not Hopes

XIIV is a masked-diffusion language model with a constraint engine built
into the sampler itself. This document explains how it works, what it
guarantees (with proof sketches), and where it sits relative to prior
art. It is written for the curious researcher.

## 1. The core loop: denoising, not predicting

An autoregressive model writes left to right: each token is final the
moment it is emitted. XIIV starts from a blank canvas of `[MASK]` tokens
and resolves the most confident positions first, in parallel, over a
fixed number of denoising steps (typically 8-16). Nothing is final until
the last step. That single difference — a *revisable* canvas instead of
an *append-only* tape — is what makes structural guarantees possible.

## 2. The slot-reservation invariant (required tokens)

**Claim.** Every token id in `required_ids` appears at least once in the
final canvas. Not probably — always.

**Mechanism.** Let `m_s` be the number of masked positions and `q_s` the
number of still-missing required ids at the start of iteration `s`.
The sampler maintains the invariant `q_s <= m_s`:

- *Base.* Initialization raises unless `m_0 >= len(required_ids)`.
- *Step.* Normal top-k filling is capped at `m_s - q_s` positions, so at
  least `q_s` masked slots survive the step. Missing ids only shrink as
  tokens appear naturally.
- *Final step.* All still-missing ids are force-placed into the
  most-confident remaining masked positions — possible exactly because
  the invariant held.

Hence `q_end = 0`. The proof is ~10 lines; the implementation is
`DenoisingState` in `xiiv/diffusion.py`. Manual interventions
(pin/unmask/remask) re-validate the invariant and raise instead of
breaking it.

**Why this matters.** An autoregressive model *cannot* do this. Once it
has emitted 50 tokens without your required word, the opportunity is
gone — recovery means regeneration. Here, the future is still blank, so
the guarantee is enforceable.

## 3. The rest of the guarantee set

- **Forbidden tokens.** Logits forced to `-inf` at every step for every
  masked position. The token cannot be sampled. Structural exclusion,
  not a blocklist checked afterward.
- **Locked spans (frozen positions).** Context tokens are never
  re-masked and never touched — verified byte-exact after every run
  (`anchors_intact`, 19/19 sampler tests). Infill edits *around* locked
  legal text without being able to alter it.
- **Anchors.** Position-pinned tokens: pre-placed and locked before
  denoising begins. An exact phrase can be fixed at an exact position
  while the model writes around it.
- **Exact length.** The canvas is always exactly `length` tokens; EOS is
  an ordinary vocabulary token, never a truncator.

`check_constraints()` verifies all four on every finished sequence — the
guarantees are checked, not trusted.

## 4. Guaranteed-faithful retrieval: the "library card"

Standard RAG pastes retrieved facts into the prompt and *hopes* the
model uses them. XIIV's knowledge layer (`xiiv/knowledge/`) does
something structurally different:

1. Retrieve facts (Wikipedia; 2020-2026 news via the Current events
   archive).
2. **Anchor** the top fact sentence verbatim at the answer start —
   position-pinned, byte-exact, verifiable via `anchors_intact`.
3. Extract single-token key terms from the remaining facts and pass
   them as `required_ids` — the slot-reservation invariant force-places
   each into the answer.

Single-token restriction is deliberate and load-bearing: the sampler
enforces *token ids*, so a multi-token word's tokens would be placed
independently and scatter. We caught this live, restricted required
terms to single tokens, and documented it rather than hiding it.

When no verified facts are retrieved, the model says so instead of
guessing. The honest fallback is part of the design.

Measured on 12 factual questions, same model, same facts, same seeds —
see `bench/faithfulness.py`:

| arm | mean key-term recall | verified guarantee rate |
|---|---|---|
| prompt-only (facts in context) | **0.083** | n/a |
| structural enforcement | **1.000** | **1.000** (12/12 via `check_constraints`) |

The enforced arm is 1.000 *by construction* — that is the point. The
0.083 baseline is partly this 52.5M model's weakness (its unguided
prose is rough); a stronger model would score higher unprompted. What
the gap demonstrates is the mechanism's value: enforcement removes the
model's quality from the faithfulness equation entirely.

## 5. Honest prior art

- **Diffusion LMs** are an active field (LLaDA to 100B, Mercury 2.5,
  Gemini Diffusion, Dream; 356+ papers). XIIV does not claim the
  architecture.
- **Constrained decoding for AR models** is mature (Outlines, native
  structured outputs from OpenAI/Anthropic/Gemini). XIIV does not claim
  the concept of guarantees.
- What XIIV claims is the *combination*: a tiny, fully open,
  CPU-runnable diffusion model with constraint enforcement as the
  first-class API — required/forbidden/locked/anchored/exact-length —
  plus faithful-by-construction retrieval, all with measured numbers
  and reproducible code.

## 6. Open questions (for the curious)

- What does faithful generation look like at 7B, where the prose
  around the guarantees is actually good? The machinery is
  scale-independent; the 52.5M prototype proves the mechanism, not the
  product.
- Can the slot-reservation invariant extend to *ordered* constraints
  (phrase A before phrase B) or *count* constraints (exactly 3
  mentions)? The canvas is revisable — the invariant family may be
  larger than implemented.
- Diffusion training is any-order: is constraint-following *learnable*
  as a training objective, not just a sampling trick?
- What is the throughput-optimal schedule for few-step constrained
  decoding? (Current: 20 tok/s @16 steps, 37.9 @8, on CPU.)

If any of these makes you want to replicate it: the code is open, the
proofs are short, and the benchmarks are reproducible. That is the
point.
