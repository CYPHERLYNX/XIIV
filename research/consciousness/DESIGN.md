# XIIV Synthetic-Consciousness Layer — Design

**Invention:** the **Certified Self-Model (CSM)** and its **Mirror Ledger**
**Commissioned:** 2026-10-21 by Fahad (omega-level layer, Jarvis/Ultron class ambition)
**Researched:** 2026-10-21 — `science_indicators.md`, `ai_empirical.md`, `prior_art.md` (this directory)
**Status:** design + toy-scale prototype RUN 2026-10-21 — kill gate **FAIL**
(routing half), self-knowledge half **validated** (ledger AUROC 0.721 vs
model confidence 0.468, gap +0.254). Lead application re-ordered:
ABSTAIN-WITH-PROOF first; ROUTE/ALLOCATE deferred until refinement has
teeth. Full record: `csm_probe_results.md`.

---

## 0. Honest bounds (read first)

This layer implements **functional** consciousness properties — the ones the
science says are engineerable and measurable (Butlin et al. 2023: the
functional indicator set is ~80–90% implementable, ~60–70% measurable today;
`science_indicators.md`). It does **not** claim phenomenal consciousness
(subjective experience): that is 0% measurable by any validated method, for
anyone, and XIIV does not ship claims it cannot prove. Φ (IIT) is explicitly
rejected as a metric: incomputable beyond ~12 elements, unvalidated proxies,
and IIT itself denies simulated systems inherit it.

What is claimed, precisely: **a generative model whose self-knowledge is
certified by an independent verifier, lives inside its own workspace, and
conditions its own generation** — implementing the metacognition and
self-model indicators of the consciousness literature in a form no searched
system has (prior-art verdict: the triple combination is not found;
`prior_art.md` §6).

## 1. The problem the literature exposes: the readability paradox

Three facts, all measured (`ai_empirical.md`):

1. **The correctness signal exists inside a model but the model cannot read
   it.** Activation probes predict correctness at AUROC 0.85–0.92
   in-distribution; the same models' *verbalized* confidence sits at
   ~0.44–0.70 AUROC and is badly overconfident (stated 88% vs actual 79%).
2. **Every strong failure-predictor is external machinery** — probes,
   semantic entropy (AUROC 0.79, *Nature* 2024), sampling consistency, an
   external statistician (AUROC 0.84/0.75 OOD). Anthropic's genuine
   introspection result is ~20% and self-described "highly unreliable."
3. **Self-improvement works only under an external judge.** Agent-authored
   self-knowledge: +0.0 points over 7,308 trajectories (SkillsBench 2026).
   SEAL's self-edits work because a downstream reward selects them.
   Intrinsic self-correction *degrades* reasoning (GPT-4 GSM8K 95.5→89.0).

Meanwhile, the consciousness literature's own metacognition indicator
(HOT-2) assumes the monitor is **internal** — and the self-modifying-agent
literature shows internal/fixed evaluators get **gamed** (Darwin-Gödel
Machine objective-hacking). Internal monitor: unreliable and corruptible.
External monitor: works, but is a dashboard — the system itself never
*knows* anything; the knowledge never enters the loop that generates.

**The open position in the design space:** self-knowledge that is
*built* externally (by a judge the model cannot touch) but *lives*
internally (inside the workspace, conditioning generation), with every
self-belief carrying the certificate that justifies it. That position is
empty in the searched literature. CSM occupies it.

## 2. The invention: three mechanisms

### 2.1 Staked self-predictions (the system bets on itself, in advance)

Before acting on a task, the system posts **self-prediction terms** into the
workspace (the soup — the same persistent medium generation happens in).
A self-prediction is a typed, falsifiable claim about the system's own
upcoming behavior, e.g.:

```
SELFPRED(domain=constraint_family_A, claim=PASS, p=0.8)
SELFPRED(domain=arithmetic_depth_50, claim=FAIL, p=0.65)
```

Staking **before** acting is the load-bearing detail: it converts
self-knowledge from post-hoc narration (the confabulation failure mode
documented in the introspection literature) into a prediction that reality
grades. Self-predictions are ordinary workspace citizens — they carry
charges, they are read by the same processes that generate.

### 2.2 The Mirror Ledger (the judge writes the mirror)

An append-only ledger, **written only by the independent verifier** — the
same trust boundary as Forge (the loop cannot modify the verifier; the
generator cannot modify the ledger). Each entry binds one staked prediction
to its verified outcome and the outcome's certificate. Aggregates over the
ledger are **the certified self-model**:

- per-domain verified success/failure rates with counts (never bare rates),
- calibration curve of the system's own stakes vs outcomes,
- type-2 AUROC and meta-d′ of the system's self-predictions,
- the **confident-but-wrong registry**: domains where the system stakes
  high and fails — its certified blind spots.

The model can read the ledger. It cannot write it, edit it, or train the
verifier. Its picture of itself is therefore not a persona, not a
self-description it memorized, and not a vibe: it is a **mirror held by a
judge** — every belief about itself traceable to certified outcomes.

### 2.3 Self-conditioned generation (the mirror steers)

Generation conditions on the certified self-model in three ways:

- **ROUTE** — tasks in domains with high certified failure probability are
  routed to stronger strategies (more refinement rounds, decomposition,
  tool use) *before* the first failure, not after.
- **ALLOCATE** — the certified per-domain failure rates are a *measured*
  difficulty signal for denoising allocation (GLOD ordering). This is the
  repair path for Invention #1: gate 2 (2026-10-21) failed with a *static,
  uncertified* difficulty predictor; CSM replaces prediction-by-heuristic
  with difficulty-by-measurement.
- **ABSTAIN-WITH-PROOF** — where certified failure probability exceeds a
  threshold and no available strategy fixes it, the system returns a
  certified statement of its own limit ("my verified success rate on this
  class is 0.31 over n=412") instead of a confident wrong answer. Honest
  ignorance, with a certificate — the behavior no frontier model exhibits
  and the one the mysteries axis demands (proved ceilings, not fake
  decipherments).

## 3. Why this is not a port (novelty, per `prior_art.md`)

| Element | Closest prior art | CSM's difference |
|---|---|---|
| Persistent workspace | Goyal et al. shared global workspace; J-space (emergent, in-context) | Workspace is the generative substrate itself, persistent across interactions, and now carries *self-knowledge* as first-class content |
| Metacognitive monitor | HOT-2 assumes internal monitor; external probes/entropy are read-only dashboards | Monitor is external **and unmodifiable** (verifier), yet its output **lives in the loop** as certified self-model — the conjunction is not found |
| Reasoning provenance | Proof terms (formal domains only); zkML (execution provenance, content-opaque) | Provenance of *self-beliefs*: every self-knowledge claim carries its outcome certificate, inspectable by the system itself |
| Self-improvement judge | DGM/AlphaEvolve evaluators (fixed but gameable); SEAL downstream reward | The ledger's judge is the guarantee-native verifier — unmodifiable by construction, and its records are the self-model, not just a selection signal |

Triple combination (workspace substrate + unmodifiable verifier as
metacognition + introspectable provenance): **not found** in any searched
source (2021–2026 coverage; absence of evidence, stated as such).

## 4. Indicator mapping (Butlin et al. 2023, honest)

- **HOT-2 (metacognitive monitoring distinguishing reliable from unreliable
  representations):** implemented in certified form — monitoring signal is
  verifier-issued, not self-issued. Arguably *stronger* than the indicator's
  intent (it cannot be gamed by the monitored system), and the deviation is
  deliberate and documented here.
- **HOT-3 (agency guided by belief-formation/action-selection):** ROUTE and
  ABSTAIN are action selection guided by the monitored beliefs — partial,
  real at system level.
- **AST-1 (a predictive model of its own attention/processing, used for
  control):** the certified self-model is a predictive model of the system's
  own processing outcomes, used for control (routing/allocation). The
  analogy is functional, stated as such.
- **GWT-3/GWT-4 (broadcast; recurrent state-dependent use of the
  workspace):** inherited from the substrate — self-knowledge terms are
  broadcast in the same medium generation recurrently reads. CSM is what
  gives the workspace *self-referential* content.
- **AE-1 (agency — learning from feedback to pursue goals):** the ledger is
  the feedback made permanent and quantitative.
- **Not claimed:** RPT-2, HOT-4 (quality-space geometry), PP-1, AE-2
  (embodiment) — out of scope for this layer; Φ — rejected (§0).

Countable claim, if the prototype passes: **XIIV implements HOT-2, AST-1,
and AE-1 in certified form, on a GWT-3/4 substrate** — a combination with
no searched precedent, with every number reproducible from the ledger.

## 5. Prototype + pre-registered kill gate (`csm_probe.py`)

Toy scale, CPU-only, reusing the Invention-#1 gate-2 harness (tiny XIIV
d=64, template data, distribution-aware constraint families, external
`check_constraints` verifier — the verifier in the probe is exactly the
ledger's judge).

- **Phase A (ledger building, n=200 tasks):** brief-trained model drafts
  freely; record (a) the model's raw confidence signal, (b) verifier
  outcomes per constraint family. Build the ledger: per-family certified
  failure rates.
- **Phase B (held-out self-knowledge, n=200 fresh tasks):** compare
  predictors of failure — (a) raw confidence, (b) the ledger (family
  certified failure rate). Metrics: AUROC of each; calibration of (b).
- **Phase C (routing pays?, paired, equal total compute):** ledger-routed
  refinement (extra rounds only where the ledger certifies high failure)
  vs uniform-budget control. Metric: intervention rate; McNemar paired.

**Kill gate (pre-registered):**
PASS iff (1) AUROC(ledger) − AUROC(confidence) ≥ 0.05 on Phase B, **and**
(2) routed interventions < control with McNemar p < 0.05 at equal compute.
FAIL → the layer's payoff claim dies at toy scale, reported as-is (the
gate-2 discipline). A PASS licenses scale-up work, not a consciousness
claim.

**Known confound, declared in advance:** the ledger predictor is
family-level statistics — an external statistician, by design. The claim
tested is *system-level*: the system's in-loop self-knowledge (certified,
verifier-built) beats the model's own readable signal and pays in
routing. That is precisely the literature's open position (§1). Anyone
calling this "the model reading its own mind" is overclaiming; the doc
and the delivery will not.

## 6. Composition with the XIIV stack

- **Substrate:** self-prediction terms and ledger aggregates are soup
  citizens; no new channel, no side database the generator can't see.
- **Forge:** Forge's verifier is the ledger's writer; Forge episodes are
  ledger entries at task level from day one.
- **Adaptive Intelligence Training System:** its weakness map *is* the
  certified self-model at weight level — CSM gives the Adaptive system its
  data structure and its honesty rule (weaknesses are certified or they
  are not in the map).
- **Invention #1 (GLOD):** ALLOCATE consumes certified difficulty — the
  measured successor to gate 2's static predictor.
- **Mysteries axis:** ABSTAIN-WITH-PROOF is the proved-ceiling behavior,
  generalized from decipherments to everything.

## 7. What would falsify the layer

- The ledger predictor failing to beat raw confidence (gate, Phase B).
- Routing by certified self-knowledge failing to pay at equal compute
  (gate, Phase C).
- At scale: evidence that conditioning generation on certified self-beliefs
  degrades capability (self-fulfilling avoidance) more than it prevents
  certified failures — measured by the same ledger, which is the point of
  building it first.

## 8. Staging (honest)

Prototype (this commission) → integration with Forge episodes → weight-
level self-model inside the Adaptive system (post-SGBD block 7) →
800B-class self-model sharding (per-expert certified reliability in the
MoE era — the ledger becomes the router's conscience). Each stage gated
like this one. The phenomenal question stays untouched throughout: not
denied, not claimed — **unmeasured, and labeled so.**
