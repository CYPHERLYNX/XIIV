# Guarantee-Native Generation: A New LLM Architecture

**Technical report — the XIIV program, October 2026.**
*For YC, research labs, and anyone who wants to know how it works.*

## 0. The discovery, in one paragraph

Every major lab treats output trustworthiness as a *behavior* of the
model: train it well, prompt it carefully, and hope the answer contains
the facts and avoids the forbidden. XIIV moves trustworthiness out of
behavior and into **physics** — the sampling mathematics itself refuses
to produce an output that violates the declared contract. Required facts
*will* appear (proven, not prompted). Forbidden content *cannot* appear
(logits at −inf). Locked text is byte-exact. The finished output is then
machine-verified, and the model reports the proof. No lab — not OpenAI,
Google, Anthropic, or any diffusion-LM team — ships this: a language
model whose generation loop is natively, provably contract-enforcing.
That is the newly discovered approach, and this report shows the
mechanism, the proofs, and the measurements.

## 1. Why this is a model, not an application

MiroFish-style systems are applications *on top of* other labs' models:
clever orchestration around an API. XIIV is the opposite — the
guarantee lives **inside the model**: in the diffusion canvas, in the
constraint compiler, in the sampler's invariants. You cannot bolt this
onto an autoregressive API, because an autoregressive model writes on
an append-only tape: once 50 tokens are out without your required fact,
the opportunity is gone. XIIV generates on a **revisable canvas** —
nothing is final until the last denoising step — so contracts are
enforceable by construction. The architecture *is* the product.

## 2. The mechanism

**The canvas.** Generation starts from a blank field of `[MASK]`
tokens. Over 8–16 denoising steps the model resolves the most confident
positions first, in parallel. Frozen prompt tokens are never re-masked;
anchored tokens are pre-placed and locked before denoising begins.

**The slot-reservation invariant (required content).** Let `m_s` be
masked positions and `q_s` still-missing required ids at iteration `s`.
The sampler maintains `q_s ≤ m_s`: normal filling is capped at
`m_s − q_s` positions per step, so enough blank slots always survive;
on the final step every still-missing id is force-placed into the
most-confident remaining slots. Initialization raises unless
`m_0 ≥ len(required_ids)`; manual interventions re-validate and raise
instead of breaking it. Hence every required id appears at least once —
always, not probably. (~10-line proof; `DenoisingState`.)

**The rest of the guarantee set**, all composable in one generation:
- *Forbidden*: logits at −inf every step — structural exclusion.
- *Anchors*: exact phrases pinned at exact positions, written around.
- *Locked spans*: byte-exact, verified after every run.
- *Exact length*: the canvas is always exactly N tokens.

**The verifier.** `check_constraints()` machine-checks every finished
sequence (`required_present`, `forbidden_absent`, `anchors_intact`).
Guarantees are checked, not trusted — and the model says so.

**The Contract interface** (`xiiv/contracts.py`) — the programming
interface of the model:

```python
c = Contract()
c.lock("Policy: ask first. Record each step.")  # byte-exact, untouchable
c.require("ask", "step")     # must appear — enforced by the sampler
c.forbid("cut")              # must not appear — logits at -inf
c.exact_length(128)
```

## 3. Measured results (52.5M prototype)

- **Model:** 52,508,736 params, d_model=576, 12 layers, 9 heads;
  trained from scratch, 5,000 steps / 20.48M tokens; best val NLL 3.3414.
- **Guaranteed-faithful Q&A** (12 factual questions, Wikipedia +
  2020–2026 news; same model, facts, seeds — `bench/faithfulness.py`):

  | arm | key-term recall | verified guarantee rate |
  |---|---|---|
  | prompt-only (facts in context, hoped for) | 0.083 | n/a |
  | structural enforcement | **1.000** | **1.000** (12/12) |

  The 1.000 is *by construction* — that is the point. Enforcement
  removes the model's quality from the faithfulness equation entirely.
- **Speed (CPU):** 20.0 tok/s @16 steps, 37.9 @8. **Tests:** 72/72.
- **Honest boundary:** the 52.5M weights trained on TinyStories-scale
  data; unguided prose is rough. The prototype proves the *mechanism*,
  not the product — fluent constrained generation is the funded
  1–8B stage.

## 4. Why the labs don't have this

- **Diffusion LMs** (Mercury, Gemini Diffusion, LLaDA): diffusion
  sampling, but no semantic hard-guarantee API.
- **Constrained decoding** (Outlines, native structured outputs):
  syntax-level schemas on autoregressive tapes — hoping at the
  semantic level.
- **Nobody** ships required + forbidden + anchored + locked +
  exact-length as one native sampling system with proof sketches and a
  machine verifier, fully open and CPU-runnable. Prior-art sweep
  (web + GitHub + Product Hunt) found no direct competitor.

## 5. The futuristic arc: contracts beyond tokens

The Contract system generalizes naturally as AI scales:

- **Agents:** required verification steps, forbidden destructive
  actions, locked safety policies — enforced in the plan, verified
  after (prototype: `xiiv/agents/`).
- **Multi-agent organizations:** machine-checked handoffs — a plan is
  only accepted if its contract verifies.
- **Self-improving systems:** the same invariant family applied to
  AI-generated successor code — required safety checks present,
  forbidden capabilities absent, *proven before execution*.

This is the architecture's long game: as models graduate from
chatbots to agents to systems that rewrite themselves, the need is not
smarter hoping — it is **guarantees that scale with capability**.
That is what guarantee-native generation is built for.

## 6. Open questions

- Ordered/count constraints (A before B; exactly 3 mentions)?
- Constraint-following as a *training* objective (constraint-aware
  pretraining), not just sampling?
- Fluent constrained generation at 1–8B — the funded stage.
- Throughput-optimal few-step constrained schedules.

## 7. What we need

Compute for the 1–8B constraint-native build, and researchers who want
to push the invariant family further. The code is open, the proofs are
short, the benchmarks reproduce. Come see the demo — 60 seconds is
enough.
