# XIIV Release Experience Spec

**Date:** 2026-10-20
**Status:** Draft v1.0 — design spec, not a demo. Nothing here is a mockup of a mockup; every interaction maps to machinery that exists in `xiiv/` today.
**Fulfills:** the 2026-10-19 commitment ("full spec document — interaction design, visual system, the materialization mechanic, calibration layer. 2050 standard: cited, falsifiable, no slop").
**Companion research:** `research/frontier_ux_forensics.md` (2026-10-19) — the forensic sweep this spec is built on. Cited below as **UXF**.

## 1. Thesis

**No frontier product presents generation as anything but typing, and no frontier product shows uncertainty inline. Both absences are structural to autoregressive generation, not accidents of taste. XIIV's masked-diffusion backbone makes both possible for the first time. The release experience should be the honest interface the criticism literature is already asking for — formed, not typed; certain where certain, visibly honest where not.**

This thesis is falsifiable (see §11). If it is wrong, the correct move is to ship a better typewriter, not this document.

## 2. What the industry does, in one paragraph

Every frontier chat product — ChatGPT, Claude, Gemini — streams tokens over SSE and appends them to the DOM, producing a typewriter effect (UXF §1–3). Differentiation has occurred only in the *waiting* experience (Claude's poetic loading messages, Gemini's "Answer now" button, reasoning traces), never in the generation visual itself. The academic literature now treats the typewriter as suspect: a pre-registered study names the **Typewriter Effect** hypothesis — that streaming inflates perceived accuracy (UXF §5a, hypothesis, not a finding); CHI'26 documented deliberate latency theater as "positive friction" and the backlash against it (UXF §5b); the ELIZA-effect literature is unambiguous that humanlike presentation cues inflate trust independently of quality (UXF §5e). Meanwhile **no major consumer product ships inline visual uncertainty** — only research prototypes (TU Darmstadt's Confidence Meter thesis, UCSF's Colab demo, Sebiomo's April-2026 framework; UXF §6). The forensics verified three ownable, empty claims: non-typing generation presentation, inline visual uncertainty, honest-interface positioning (UXF §7).

XIIV does not need to win on this axis by being prettier. It wins by being the only interface that doesn't lie about its own mechanism.

## 3. Interaction 1 — Materialization ("formed, not typed")

### The mechanic
A masked-diffusion model does not write left-to-right. It begins with a canvas of `[MASK]` positions and resolves them in parallel over N denoising steps, in confidence order, with the whole canvas re-evaluated at every step. The interface shows exactly this:

1. **The canvas opens fully masked.** The user sees the answer's *shape* first — its length, its structure (paragraph breaks, list items, code blocks resolve as structure before content, because high-confidence structural positions unmask early under confidence-order decoding). This is real information the model genuinely has before any word is final.
2. **Positions resolve in the order the model resolves them** — highest confidence first, as `DenoisingState` already unmasks (`order="confidence"`). A word that the model committed at step 2 appears before a word it committed at step 14. The visual order is a truthful record of the model's own certainty trajectory, not a left-to-right lie.
3. **Resolution reads as focus, not typing.** A masked position transitions through a brief intermediate state (see §8, visual system) into its final token. Because multiple positions resolve per step (confidence-parallel unmasking, cf. the GACD adaptive decode already in `DenoisingState`), the canvas visibly *settles* rather than being typed. Whole clauses can land at once. That is the correct visual for what happened: the model did not produce the clause word by word.
4. **The trajectory is the honest "thinking" display.** `DenoisingState` captures a full per-step canvas trajectory (`return_trajectory=True`; serializable via `to_snapshot()`). Where Claude shows poetic loading messages and ChatGPT shows nothing, XIIV can offer a scrubbable trajectory — "watch it resolve" — with zero theater, because the trajectory is the actual computation, not a narration of it. Default: collapsed; available on demand.

### What we refuse
- No character-by-character or word-by-word sequential reveal. Ever. It would be a lie about the mechanism — the exact lie UXF §5 documents as the industry's default.
- No artificial delay to simulate deliberation. The CHI'26 "positive friction" finding (UXF §5b) is the anti-pattern named: if the model resolved in 400ms, the interface says so. Speed is honest; theater is not.
- No thinking dots, no spinner-as-personality, no "Marinating a flowchart." Waiting is shown as *canvas state* (how many positions remain masked, current step of N) — a progress truth, not a personality.

### Honesty note
Materialization is *slower to first meaningful paint* than streaming's first token — the industry consensus (UXF §4) is that a blank screen feels broken. The spec answers this with the shape-first canvas (§3.1): the user sees structure immediately, which is more informative than a first token and arrives just as fast. **Falsifiable claim:** at equal total latency, users rate the shape-first canvas as responsive as or more responsive than token streaming (test in §11).

## 4. Interaction 2 — Mid-generation steering (the revisable canvas)

This is the interaction no autoregressive product can offer by construction: once a token is emitted, it is history. A diffusion canvas is revisable until the final step. Three gestures, each mapping 1:1 to a method that exists today in `xiiv/diffusion.py`:

1. **Pin.** The user clicks/taps any masked or resolved position and types (or selects) the token they want. Maps to `DenoisingState.pin(pos, token_id)`: sets the token, freezes the position (`frozen[pos]=True`, `conf[pos]=1.0`), and it survives all remaining steps. The model re-resolves the *rest* of the canvas around the pin. This is "redirect mid-generation" made literal: the user's constraint becomes part of the canvas, and the guarantee machinery re-validates around it.
2. **Reconsider.** The user selects a span of resolved text and asks for alternatives. Maps to `unmask_positions(indices)`: turns the span back into editable `[MASK]` slots (`conf=0.0`, unfrozen), which the remaining steps re-fill. The rest of the answer is untouched.
3. **Regenerate region.** Maps to `remask_positions(indices)`: re-masks non-frozen positions for re-fill with a fresh draw.

### Honest limits (shown in the UI, not hidden)
- Pinning is rejected — loudly, with the reason — when it would break a guarantee: `pin()` raises if the token is `[MASK]` or forbidden, or if it would leave a required token with no masked slot (the slot-reservation invariant). The interface surfaces the refusal as what it is: "That word can't go there — the answer is contractually required to contain X." A refusal that names its reason is a feature; silent acceptance would be the lie.
- Steering is only available while denoising is in progress. Once the canvas is final, steering becomes a new generation seeded from the old canvas. The boundary is explicit: *revising* vs *re-asking* are different operations and look different.
- Every steering action is itself snapshotable (`to_snapshot()`/`from_snapshot()`): the user can undo a pin, because the pre-pin state is a real object, not a memory.

### Why this matters
The forensics found that the only universal interrupt control is the Stop button — "users often know mid-answer that the direction is wrong" (UXF §1). Steering is the structural answer: instead of stopping a wrong answer and starting over, the user corrects the canvas while it's still forming. **Falsifiable claim:** steering reduces full regenerations per session vs. stop-and-retry (test in §11).

## 5. Interaction 3 — Honest uncertainty, visible

### The problem with the naive version
Per-token probability heatmapping is the obvious implementation and the wrong one. The UCSF prototype's own notes record the confound: high probability often means "common collocation" ("disease" after "Alzheimer's"), not "true" (UXF §6). Per-token coloring would paint idioms green and facts amber — the opposite of honesty. The spec therefore aggregates **per claim, not per token**.

### The two patterns (Sebiomo's rule: pick two, not five)
The literature warns that calibrated signals cost visual weight — "if every response carries three signals, every response reads like a disclaimer" (UXF §6). This spec ships exactly two:

1. **Solid vs. held.** Text the model resolved at high confidence renders solid. Text resolved at low confidence — or resolved only after re-draws — renders in a *held* treatment: same typeface, same size, but visually provisional (see §8). This is a binary the model genuinely computes: every position carries a `conf` value in `DenoisingState`, updated at every step. The held treatment is not a vibe; it is a threshold on a measured quantity, and the threshold is documented in §7.
2. **Claim underlining with proof on demand.** Where `claim_verify.py` has verified a factual claim deterministically against a fact store, the claim carries a subtle underline. Hovering or tapping reveals the certificate: the claim text, the check performed, the expected vs. actual value, pass/fail, and latency in microseconds. This is the motto made interactive — **"Every answer carries its proof"** — and it is only ever shown for claims that were actually checked. Unchecked claims get no underline, no matter how confidently worded.

### What the interface does with uncertainty (the behavior, not just the paint)
- A held span is an invitation, not a warning. Hovering a held span shows *why*: the top competing candidates the model considered at that position (the model genuinely computed them), and — where applicable — which guarantee constraint forced the choice (e.g., "chosen to satisfy the required-term contract").
- Held spans are never used for voice hedging. The model does not say "I think maybe" — it states the claim and marks its confidence visually. Voice stays declarative; calibration lives in the visual channel. This is the inversion of the industry default (hedge words in voice, uniform confidence in text; UXF §6).
- Color is never the only signal. Every treatment pairs color with a second channel — weight, underline style, or an explicit label — per the accessibility constraint (UXF §6).

### Honesty notes
- The fact store bounds the proof. `verify_claims` checks claims against a *provided* fact store — no model in the loop, microseconds per claim. The interface must never imply a claim was verified against the world when it was verified against a store. The certificate panel names the store and its vintage.
- Confidence is about the model's distribution, not the world's truth. A confidently wrong model renders solid and wrong. The spec does not claim the held treatment catches hallucinations; it claims it shows what the model *computed about itself*. That distinction is stated in the product, not buried in docs. (This is also why the proof underline exists as the second, independent signal: self-reported confidence and external verification are different axes, shown as different treatments.)

## 6. The proof affordance — certificates as a first-class object

XIIV's guarantee-native thesis produces something no frontier product has: machine-checkable certificates that a generation satisfies its contract (structural compliance via `contracts.py`, factual compliance via `claim_verify.py`). The release experience makes the certificate a visible, inspectable object:

- Every completed answer carries a **certificate badge**: contract satisfied (which contract, by hash), claims checked (n passed / m checked / k unchecked), generation parameters (steps, seed). Clicking expands the full certificate — the same object the verifier computed, not a summary written for the occasion.
- Unchecked claims are counted, not hidden: "3 claims not checked against any store" is displayed alongside "11 claims verified." The absence of verification is itself information, and hiding it would be the industry's move.
- The certificate is exportable (JSON) and re-verifiable offline: the verifier is deterministic and model-free, so a third party can re-run it. "Don't trust our black box" is not a slogan here; it is a file format.

### Scope honesty
Certificates cover what they cover: contract compliance and fact-store checks. They do not cover judgment, taste, relevance, or "is this the right answer." The interface states the certificate's scope in one line next to the badge. Overclaiming the certificate would be the fastest way to destroy the thing that differentiates it.

## 7. The calibration layer — how the numbers are actually computed

This section exists so that no number in the interface is magic. Every visual treatment above traces to a computed quantity:

- **Per-position confidence (`conf`).** Maintained by `DenoisingState` at every step: the model's top-1 probability at that position given the current canvas. Solid/held threshold: `conf >= 0.85` renders solid; below renders held. The threshold is a documented default, adjustable per deployment, and shown in the interface's "how this works" panel. (Threshold choice is a product decision, not a scientific constant — labeled as such.)
- **Stability.** Under the STABILISS sampler direction (LESS-port + certificate gate; `research/daily_research_2026-10-20.md`), a position also carries inter-step stability: whether its top-1 persisted across recent reverse steps and the JS divergence of its predictive distribution. Positions that resolved confidently but *unstably* (flickered before settling) render held regardless of final confidence — flicker is information about the model's uncertainty, and the trajectory records it truthfully.
- **Certificate validity gate.** Where a contract applies, a position that would break certifiability is never committed (the verifier-gated unmasking rule). The interface can therefore state, truthfully: "every solid token in a contracted answer is contract-compatible." That sentence is checkable against the code.
- **What is NOT shown.** Raw logits, full distributions, attention maps. The interface shows what a user can act on (confidence, alternatives, proof), not what looks impressive. Developer mode exposes the rest.

### Refusals
The calibration layer also governs what the interface will not do: it will not invent a confidence number for a model that didn't compute one; it will not smooth the held treatment into a gradient of false precision (binary solid/held, with the threshold documented); it will not display a certificate badge for an answer that bypassed verification. A missing signal renders as missing, never as neutral.

## 8. Visual system

Standing constraints: dark theme (black/dark-grey, never the generic AI-template look — project design rule 2026-09-30); OpenDesign is the default design tool when this moves to implementation.

- **Canvas.** The answer surface is a canvas, not a chat bubble. Masked positions render as quiet blocks (the answer's shape, visible before its content). Resolution is a focus transition: block → emerging token (brief intermediate treatment, ~120ms, motion-safe alternative below) → final token. No cursor, no caret, no blinking insertion point anywhere — there is no insertion point because nothing is being inserted sequentially.
- **Solid vs. held.** Solid: full-opacity text. Held: reduced-opacity text at the same size and weight, with a dotted underline on interactive spans. The held treatment must read as "provisional," never as "disabled" or "error." Verified claims: solid underline (distinct from held's dotted underline — two underline styles, two meanings, documented in the legend).
- **Certificate badge.** Minimal: a small mark (suggest: ◈, the project's own sigil) with counts. Expands to the full certificate panel. Never a checkmark inside a green circle — that reads as "trust us," which is the sentence this interface exists to not say.
- **Motion.** All transitions are state changes with a motion-safe path: `prefers-reduced-motion` collapses every transition to an instant state change. No parallax, no spring physics on text, no decorative animation. Motion budget: the interface may animate *state*, never *personality*.
- **Screen readers and assistive tech.** Per the forensics' accessibility findings (UXF §1): never announce per-token or per-position updates. Live regions announce three events only — generation started (with canvas size), generation complete (with certificate summary), and steering confirmations. The trajectory scrubber is keyboard-navigable with text descriptions of canvas state at each step.
- **Legend.** A persistent, one-line legend explains the three treatments (solid / held / verified). The interface never relies on the user having read documentation to interpret what they see.

## 9. What the interface never does

The anti-pattern list, stated positively as refusals (each traces to a forensics finding):

1. **No typewriter.** No sequential character/word reveal under any setting, including a "classic mode." (UXF §5: the Typewriter Effect hypothesis.)
2. **No artificial latency.** No deliberate delays, no "thinking" pauses scaled to question gravity. (UXF §5b: the positive-friction backlash.)
3. **No thinking theater.** No reasoning traces presented as interiority, no poetic loading messages, no anthropomorphic status text. Waiting is shown as canvas state: positions remaining, current step. (UXF §2, §5e.)
4. **No uniform confidence.** Text never renders at a single confidence level. If the model computed confidence, the interface shows it; if it didn't, the interface says so. (UXF §6: "every AI product is either calibrated or lying.")
5. **No fake controls.** No "regenerate" button that re-rolls silently — regeneration is region-scoped and explicit (§4). No stop button that leaves a half-typed corpse — stopping a diffusion run leaves a *partial canvas*, which is a meaningful, resumable object (snapshot), not a broken stream.
6. **No certificate theater.** No badge without a backing verification run. No "verified" language for fact-store checks without naming the store.

## 10. Implementation map — UI events from real machinery

The interface is a thin client over `DenoisingState`. The event stream it consumes:

| UI event | Source in `xiiv/` | Notes |
|---|---|---|
| Canvas opened (length, structure) | `DenoisingState.__init__` (length, prompt layout) | Structure positions (breaks, list markers) are high-confidence early unmaskers — shape-first is free |
| Position resolved (token, conf, step) | `step()` unmasking batch + `self.conf` | Render order = model's unmask order, not left-to-right |
| Canvas progress (masked remaining, step k/N) | `self.masked`, `steps_remaining` | The honest waiting display |
| Trajectory frame | `self.trajectory` (`return_trajectory=True`) | Powers the scrubbable "watch it resolve" view |
| Pin / reconsider / regenerate | `pin()`, `unmask_positions()`, `remask_positions()` | Refusals surface the invariant violation verbatim |
| Undo steering | `to_snapshot()` / `from_snapshot()` | Pre-steering state is a real, restorable object |
| Certificate | `verify_generation` (contracts) + `verify_claims` (claim_verify) | Badge counts; panel shows the actual check records |
| Pause / resume | `to_snapshot()` serialized | Generation is interruptible at the process level, not just the UI level |

Latency honesty: the interface may batch UI updates for render performance, but it must never *invent* intermediate states. If the model resolved 40 positions in one step, the UI shows 40 positions resolving — it does not stagger them for drama. Batching is a render concern and is disclosed as such.

## 11. Falsifiable claims — the 2050 standard

Every load-bearing claim in this spec, with the experiment that would kill it:

1. **Responsiveness.** "At equal total latency, users rate the shape-first canvas as responsive as or more responsive than token streaming." *Kill test:* A/B with matched total latency; if streaming wins perceived responsiveness, the canvas needs a first-paint improvement, not a typewriter.
2. **Honesty effect.** "Inline held-treatment reduces acceptance of low-confidence claims vs. uniform-confidence text, without reducing acceptance of high-confidence claims." *Kill test:* the TU Darmstadt Confidence Meter protocol (UXF §6) adapted to the held treatment; if held spans don't move acceptance, the treatment is decoration.
3. **Steering efficiency.** "Mid-generation steering reduces full regenerations per session vs. stop-and-retry." *Kill test:* instrumented sessions; if steering doesn't reduce regenerations, it is a power-user feature, not a core interaction.
4. **Differentiation.** "First-time users describe the experience as categorically different from ChatGPT/Claude/Gemini (not 'a nicer chatbot')." *Kill test:* 5-second unprompted description test; if descriptions cluster on aesthetics rather than mechanism, the design has failed its thesis.
5. **Certificate trust.** "The exportable, re-verifiable certificate increases willingness to rely on contracted answers in high-stakes use vs. the same answer without a certificate." *Kill test:* task-completion study with/without certificate panel; if no effect, the badge is marketing.

If any claim dies, the spec changes. That is the standard the user set ("cited, falsifiable, no slop" — 2026-10-19), applied to the spec itself.

## 12. Open questions (labeled unknowns, not hidden gaps)

- **Multi-claim sentences.** A sentence containing one verified and one unverified claim: underline the verified span only, or hold the whole sentence? Unknown — needs a prototype and the honesty-effect test (§11.2).
- **Held threshold calibration.** 0.85 is a documented default, not a finding. The right threshold is an empirical question per deployment; the spec requires the threshold be shown, not that 0.85 is correct.
- **Trajectory scrubber depth.** Full per-step trajectory for long answers is heavy. Unknown whether users want the whole trajectory or keyframes (e.g., steps where the top-k candidate set changed). Prototype both.
- **Voice interface.** This spec covers the visual canvas. A voice XIIV cannot "materialize" — the honest audio analogue of formed-not-typed is an open design question, explicitly out of scope for v1.
- **The model's own uncertainty about structure.** Shape-first assumes structural positions resolve early and confidently. If measurement shows otherwise on real workloads, §3.1's claim weakens — measure on the trained 1B before release.

---

*Spec ends. Next step when authorized: prototype the canvas against the real `DenoisingState` trajectory output (no model needed — trajectory replay from a saved snapshot drives the full UI), then run kill-test #1.*
