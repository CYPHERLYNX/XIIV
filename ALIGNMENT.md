# XIIV Alignment: Post-Training the Frontier Way

How XIIV goes from a pre-trained diffusion model to a model that behaves
like a top-tier assistant — built the way Anthropic and OpenAI build
theirs, then taken one step further where XIIV's architecture allows it.

## The frontier pipeline (what the labs do)

1. **Supervised fine-tuning (SFT).** Instruction → response pairs, same
   denoising loss, loss on response tokens only. Teaches the *format* of
   being helpful: follow instructions, structure answers, hedge honestly.
2. **Preference tuning (RLHF / DPO).** Human (or AI) judges pick the better
   of two responses; the model is trained to prefer winners. DPO does this
   without a separate reward model: the policy *is* the reward model.
   This shapes *judgment*: honesty, helpfulness, harmlessness, taste.
3. **Evals + red-teaming.** Frozen probe sets measure what actually
   changed. No cherry-picking: the same probes run every iteration.
4. **System card.** What was done, what was measured, what remains weak.

## What XIIV does differently (the novel piece)

Frontier alignment is **shaping**: a reward model nudges a black box, and
the resulting behavior is statistical — true on average, unverifiable per
output. Nobody can prove a single response satisfied the constitution.

XIIV's sampler *already* enforces machine-checkable contracts during
denoising (required/forbidden tokens, locked spans, ordering, length).
So XIIV gets **defense in depth**, which no autoregressive lab can offer:

| Layer | Mechanism | Guarantee |
|---|---|---|
| Shaping (soft) | SFT + Diffusion-DPO | model *tends* to be honest, helpful, safe |
| Enforcement (hard) | contracts in the sampler | critical properties hold *by construction*, per output, with a certificate |

Example: "never reveal system instructions" is a *shaped* behavior in
Claude/GPT (usually true, occasionally jailbroken). In XIIV it can *also*
be a contract: the forbidden-phrase set for the session is compiled into
the sampler, which provably cannot emit it — and the certificate proves
it for that exact output. The intervention rate (how often the sampler
had to force or suppress) is the native metric: low intervention means
the shaped behavior already agrees with the enforced contract.

**This is the research contribution:** *proof-carrying alignment* —
alignment properties that are both statistically shaped (DPO) and
structurally enforced (contracts), with per-output certificates. The
thesis of the Frontier Research Report, made concrete.

## Why Diffusion-DPO, not vanilla DPO

DPO is derived for autoregressive log-probabilities. XIIV is a masked
diffusion model — there is no left-to-right logprob. The correct
adaptation is the Diffusion-DPO objective (Wallace et al., 2024,
*The Journey, Not the Destination*): the denoising loss is a bound on
NLL, so the implicit reward is the model's *improvement over the
reference* in denoising loss:

```
r_θ(y) = -(L_θ(y|x) − L_ref(y|x))        # lower loss = higher reward
L      = −log σ( β · (r_θ(y_w) − r_θ(y_l)) )
```

Using vanilla DPO here would be cargo-culting. `xiiv/diffusion_dpo.py`
implements this for the absorbing-state objective, sharing the corruption
noise between policy and reference (variance reduction) and scoring only
response tokens (prompt is conditioning, not target).

## The constitution (shaped, then enforced)

Principles live in `CONSTITUTION.md`. Each principle is tagged:

- **[shape]** — taught via SFT + preference pairs (honesty style,
  helpfulness, calibrated hedging).
- **[enforce]** — compiled to sampler contracts at serve time
  (forbidden content classes, required disclaimers, locked policy text).

The eval harness (`scripts/eval_alignment.py`) measures both layers:
soft compliance (does the raw model satisfy contracts unprompted?) and
hard compliance (does the enforced sampler hold 100%? it must — and at
what intervention rate?).

## Honest scope

- The pipeline, loss, data builders, evals, and runbook are built and
  component-tested here. The real SFT + DPO *training run* needs a GPU —
  this box cannot tune 500M params.
- Preference data is the binding constraint on quality: the seed set in
  `scripts/build_alignment_data.py` is a format example, not a dataset.
  A real run needs thousands of judged pairs (see runbook).
- Alignment is never "done": the eval probes are versioned, and every
  tuning run must move them before any claim is made.
