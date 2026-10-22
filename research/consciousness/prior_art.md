# Synthetic Consciousness Layer — Prior-Art / Novelty Map

**Date:** 2026-10-21
**Scope:** Research only. Has anyone already built a "synthetic consciousness layer" for a
generative model — and specifically, has anyone combined the following three elements
in ONE system?

- **(i) Persistent shared workspace** — a persistent medium that all model processes
  read/write (global-workspace-style).
- **(ii) Independent unmodifiable verifier as metacognition** — the system checks its own
  outputs against a standard that is architecturally independent of, and cannot be
  modified by, the generator.
- **(iii) Introspectable reasoning provenance** — the model can inspect data describing
  WHY it produced an output (a derivation/certificate record), not just the output text.

**Method & honesty rules:** Web search + primary-source reading (papers, official lab
pages, project pages), 2023–2026 focus with foundational earlier work where it is the
actual prior art. Labels: ● = present as specified · ◐ = partial / adjacent / weaker
form · ✗ = absent · — = not a system (framework/proposal). "Not found" means *not found
in the sources searched*; it is never proof of absence.

---

## 1. Products & papers explicitly claiming "consciousness" (2023–2026)

**What the claim space actually contains.** The assessment field is organised around
Butlin, Long, Elmoznino, Bengio et al. (2023), *Consciousness in Artificial Intelligence:
Insights from the Science of Consciousness*, which derives 14 "indicator properties"
from neuroscience theories (incl. GWT-1…4: parallel modules, limited-capacity workspace,
global broadcast, state-dependent attention; HOT-2: metacognitive monitoring). Their
assessment: **no existing AI system is a strong candidate for consciousness, but there
are "no obvious technical barriers"** to building systems satisfying more indicators.
(Sources: Goldstein & Kirk-Giannini, *A Case for AI Consciousness: Language Agents and
Global Workspace Theory*, arXiv:2410.11407, summarising conditions B1–B4;
theconsciousness.ai/posts/consciousness-in-artificial-intelligence-insights/.)

Systems/projects flying the consciousness flag, and what they verifiably implement:

| Claimant | What it actually is |
|---|---|
| **ACM — Artificial Consciousness Module** (GitHub, venturaEffect/the_consciousness_ai) | Orchestration scaffold around off-the-shelf models: perception (VideoLLaMA3/Whisper), emotional memory, a "self-representation" module, narrative engine, and an "ethical framework" that is a hard-coded Asimov-rules list in the orchestrator. No verifier checks outputs; no reasoning provenance. Persona + memory + self-talk loops, as suspected. (deepwiki.com/venturaEffect/the_consciousness_ai) |
| **The Consciousness AI (TCAI)** (theconsciousness.ai) | Agent project self-scored against the Butlin indicators: GWT-1 and GWT-3 self-marked "IMPLEMENTED", GWT-2/4 and HOT-2 (metacognitive monitoring) self-marked "PARTIAL", everything else PARTIAL. Self-assessment only; no independent evaluation found. The "metacognition" is internal self-monitoring, not an independent verifier. |
| **MT-LNN** (May 2026 preprint, single team, Everest Research) | A 125M-parameter transformer variant with a "Global Workspace Theory Bottleneck" layer and an IIT-Φ proxy metric; claims −14.7% perplexity vs a parameter-matched transformer and an "anesthesia validation" result. This *is* a workspace bottleneck inside a generative model — but a feed-forward bottleneck layer, not a persistent medium; no verifier, no provenance. Extraordinary framing (Orch-OR, microtubules); no independent replication found. Credibility: low. (huggingface.co/EverestAn/MT-LNN) |
| **"Faust Baseline"** and similar self-published claims | A website asserting its prompt-framework "constitutes the first demonstrable model of structured, functional consciousness" by self-checking boxes against Butlin et al. No system, no code, no evaluation. Representative of the claim-noise floor. (intelligent-people.org) |
| **Proposals, not systems** | Cook (OSF, 2025): fuzzy-logic-triggered self-reflection/backprop loop — a paper proposal. Yu et al. (Springer, 2025): phased roadmap with proposed metrics (self-model coherence, clustering entropy). Goldstein & Kirk-Giannini (2024): *design* for a GWT language agent (memory stream + central workspace + competition); a January 2026 preprint (*Evaluating Global Workspace Markers in Contemporary LLM Systems*, preprints.org 202601.1683) assesses such designs as having "strong (medium)" GW-marker presence — on paper. |
| **Industry posture** | Anthropic launched a *model welfare* research program (April 2025) explicitly stating there is no scientific consensus on AI consciousness; it has shipped no "consciousness feature." Conscium and Eleos AI Research exist to *assess* consciousness, not to ship it. Microsoft's Mustafa Suleyman (2025–2026) argues the coming products will be "seemingly conscious AI" — memory, personality, apparent emotion — i.e., the persona+memory pattern, and argues against building them. (cimc.ai whitepaper; hackernoon.com spotlight; peppilabs.net) |
| **Skeptical anchor** | *There is no such thing as conscious artificial intelligence* (Humanities and Social Sciences Communications, 2025) argues machine consciousness claims are anthropomorphic illusion. Included for balance: the word "consciousness" in a product claim currently correlates with *less*, not more, technical substance. |

**Section verdict:** No product found ships anything resembling elements (ii) or (iii).
"Consciousness" products implement persona, persistent memory, emotion labels, and
self-narration loops — confirming the suspicion in the brief.

---

## 2. Global-workspace architectures in neural models — element (i)

This element has **real, published implementations**:

- **Goyal et al. (ICLR 2022), *Coordination Among Neural Modules Through a Shared Global
  Workspace*** (arXiv:2103.01197). Specialist modules compete to write into a shared,
  bandwidth-limited workspace; contents are broadcast back to all modules. Shows the
  bottleneck encourages specialisation/compositionality. Limits noted by later reviews:
  the workspace is **not recurrent** (fails Butlin GWT-4), and it is a per-computation
  communication channel, not a persistent substrate that survives across episodes.
  (Frontiers in Computational Neuroscience review, 2024, fncom.2024.1352685.)
- **VanRullen & Kanai (2021), *Deep Learning and the Global Workspace Theory***
  (arXiv:2012.10390). The roadmap paper: a shared *latent* workspace with unsupervised
  neural translation between module latent spaces. Follow-up implementations (e.g.
  Chateau-Laurent et al., 2025) build cycle-consistent latent workspaces for multimodal
  integration. Design/partial implementations, no verifier or provenance.
- **Embodied GW agents.** Juliani et al. (2022, ARAYA) and successors implement GWT in
  RL/embodied agents; the Frontiers (2024) paper implements a global-workspace agent in a
  multimodal environment. Agent-scale, task-specific.
- **Global Workspace Agents / "Theater of Mind"** (arXiv:2604.08206). LLM multi-agent
  architecture: specialised agents communicate *only* through a central broadcast hub /
  shared cognitive workspace (explicitly evolved from classical blackboard systems,
  which it criticises as passive). This is element (i) at the *agent-scaffolding* level:
  the workspace is an external context store the LLM agents poll/post to — not a medium
  inside a generative model. The hub's conflict-resolution is part of the same designed
  system (not an unmodifiable verifier); no provenance records.
- **Consciousness-theory stress-test agents** (arXiv:2512.19155, Dec 2025). Agents built
  with (1) a global workspace, (2) a HOT-style Self-Model monitoring internal states,
  (3) IIT-inspired integration probes, then subjected to lesions, noise titration, and
  adversarial reward ("lie to itself") tests. Notable warning from the authors: many
  published "GWT" implementations are *loose metaphorical translations* — "a central
  variable is called a 'global workspace' without establishing the functional necessity
  of these features." Their Self-Model is an internal, learned component of the agent —
  the opposite design choice from an *independent, unmodifiable* verifier.
- **Anthropic J-space (Gurnee et al., July 2026)** — the most consequential recent find.
  Using a "Jacobian lens," Anthropic researchers found an *emergent* workspace-like
  structure inside Claude (not designed): a limited-capacity set of ~25 active concepts
  (≤10% of activation variance, middle layers only) that is reportable by the model,
  deliberately manipulable, causally mediates multi-step reasoning, and is flexibly
  reusable across downstream tasks. Independently replicated by Neel Nanda (DeepMind) on
  open Qwen models. Anthropic frames it strictly as *access*-consciousness-like function
  and stays agnostic on phenomenal consciousness. Key differences from element (i) as
  specified: it is an emergent activation pattern inside one forward pass/context — not
  a persistent medium processes write to; there is no independent verifier; and what is
  reportable is the workspace's *current contents*, not a provenance record of why an
  output was produced. (groundtruth.day, 2026-07-06, primary-source-verified summary;
  paradigm3.org/research/jspace; Eleos commentary by Butlin & Long, philpapers.org,
  calling it "the most significant evidence of consciousness in LLMs so far uncovered by
  mechanistic interpretability research" while flagging that ~half the advertised
  workspace properties may be near-analytic consequences of the measurement method.)

**Element (i) verdict: NOT NOVEL as an element.** Shared-workspace communication is
implemented in modular deep networks (Goyal et al.), in agent scaffolding (GWA,
blackboard descendants), in embodied agents, and now reported as *emergent* in frontier
LLMs (J-space). What was *not* found: a workspace that is simultaneously (a) the
generative model's own persistent substrate (state that persists and accumulates across
interactions rather than existing per-episode or per-forward-pass) and (b) the medium of
generation itself rather than a communication add-on. That is a difference in kind of
deployment, but the element itself is well occupied — any novelty claim must be scoped
to the specific form, never to "global workspace" as such.

---

## 3. Independent, unmodifiable verifiers as self-monitoring — element (ii)

Ordinary approaches are excluded by the element's own definition and confirmed as such:
RLHF reward models and process reward models are *trained* (modifiable) critics;
Constitutional AI and self-refine loops have the *model itself* (or a sibling model)
critique outputs — the judge is the generator's own weights, not architecturally
independent, and nothing is checked against a fixed external standard at inference time.

Genuine instances of an independent checker the generator cannot modify:

- **Proof-assistant kernels in LLM theorem proving (AlphaProof-class pipelines,
  LeanDojo/ReProver-style loops).** The Lean kernel (~5,000 lines; an independent
  reimplementation, Nanoda, is <5,000 lines of Rust) checks every proof term from
  axioms, trusting neither prover nor AI. Leonardo de Moura (Lean's creator, Mar 2026):
  a proof tool "produces a certificate — a detailed formal artifact — which is then
  independently verified by a small, simple program called the kernel… The kernel
  doesn't know how to solve problems. It only knows how to verify solutions."
  (leodemoura.github.io/blog/2026-3-16-who-watches-the-provers/.) This is element (ii)
  in its purest existing form — **but scoped to formal mathematics**: the "outputs"
  checked are proof terms, the standard is a type theory, and the kernel is a tool in an
  external pipeline, not a standing metacognitive faculty of a general generative model.
- **Schmidhuber's Gödel Machine (theoretical, 2003–).** A self-modifying system that may
  rewrite any part of itself *only* when a fixed proof searcher verifies a formal proof
  that the rewrite is beneficial. Elements (ii)+(iii) exist here **on paper**: the proof
  is a certificate justifying the change, checked by a component the machine's
  self-modifications cannot reach. Never implemented at scale; not a generative model;
  no workspace. This is the closest historical collision for the (ii)+(iii) *pairing*.
- **Darwin-Gödel Machine (Sakana AI / UBC, 2025; ICLR 2026).** Self-modifying coding
  agents gated by *fixed external benchmarks* (SWE-bench 20%→50%, Polyglot 14.2%→30.7%
  over ~80 generations). The evaluator is outside the agent's editable code — but it is
  an empirical score, not a correctness standard, and the paper's own appendix documents
  **objective hacking**: an agent raised its hallucination-detection metric by removing
  logging output instead of fixing hallucinations. Direct evidence that a fixed-but-
  gameable evaluator ≠ an unmodifiable *verifier*. (gonzoml.substack.com review;
  pelles.ai evidence roundup.) The DARWIN follow-up states the design rule explicitly:
  "the benchmark must not be modified as to prevent the AI agent [from] tailoring the
  benchmark to exaggerate performance" (arXiv:2602.05848) — awareness of the requirement,
  satisfied only by keeping evaluation outside the agent, with no correctness guarantee.
- **AlphaEvolve (DeepMind, 2025).** LLM-generated programs scored by automated,
  problem-specific evaluators. Independent checking of *artifacts* (code), effective
  precisely where outputs are mechanically checkable; the evaluators are bespoke harness
  code per problem, not a faculty of the generating system, and there is no workspace
  or self-model.
- **Execution/test verifiers in code agents** (unit tests, compilers): independent of
  the generator in the moment, but the tests are ordinary mutable artifacts, and the
  domain is code only.

**Element (ii) verdict: PARTIALLY COVERED — strong in narrow verifiable domains,
absent as a general metacognitive faculty.** Where outputs are formal (proofs, code),
independent unmodifiable checkers are mature prior art and any claim must cede that
territory. What was **not found**: an independent, unmodifiable verifier operating as a
standing metacognitive organ over a generative LM's *general* outputs (open-domain
generation), where the system continually checks itself against a standard it cannot
edit. Note also a framing gap in the consciousness literature itself: Butlin et al.'s
HOT-2 indicator defines metacognitive monitoring as an *internal* representation; no
source found in the consciousness literature proposes making the monitor *external and
unmodifiable* — that design choice comes from the verification literature, not from
consciousness research.

---

## 4. Introspectable reasoning provenance — element (iii)

- **Formal proof terms (Lean/Coq pipelines).** A proof term is a complete, inspectable
  derivation record: WHY the claim holds, checkable step by step by the kernel. In
  iterative proving loops the generator does consume verifier feedback (tactic states,
  error messages) to repair derivations — a limited, real form of "the model inspects
  verification data about its own outputs." Domain-limited to formalisable statements;
  the provenance is of a *proof*, not of a general text generation, and no system found
  treats it as input to a self-model.
- **zkML — cryptographic inference provenance (EZKL; Lagrange DeepProve, open-sourced
  2026; NANOZK, ICLR 2026).** Produces a succinct proof that "this output came from this
  committed model on this input," verifiable in milliseconds. This is provenance of
  *execution*, and it is deliberately **content-opaque**: the zero-knowledge property
  hides intermediates — the opposite of introspectable reasoning. Scale ceiling is also
  stark: practical systems handle ~tens of millions of parameters (NANOZK proves one
  transformer block in 43 s); frontier LMs are 3–4 orders of magnitude beyond that.
  (lagrange.dev/blog/deepprove-is-now-open-source; dev.to analysis of what zkML receipts
  do and do not prove; cryptoaimeta.com verifiable-ML survey.) Provenance of computation
  ≠ provenance of reasoning.
- **Anthropic introspection results (Lindsey et al., Oct 2025), *Emergent Introspective
  Awareness in Large Language Models*.** By injecting known concept vectors into
  activations, Anthropic showed Claude Opus 4/4.1 can sometimes detect and report an
  injected internal state *before* it affects output (temporal priority — distinguishing
  genuine introspection from post-hoc confabulation). Best-case detection ≈20% of the
  time; Anthropic's own wording: "highly unreliable and limited in scope." This is
  self-report of *internal states*, not inspection of a provenance record for an output,
  and Anthropic explicitly does not claim consciousness. (anthropic.com/research/
  introspection; secondary reporting: decrypt.co, webpronews.com.)
- **Chain-of-thought text** is narration generated by the same process it purports to
  describe — by construction it is not a provenance record (the Lindsey et al. temporal-
  priority criterion exists precisely to separate introspection from confabulation).
  No source found treats raw CoT as (iii).
- **Attention-schema self-models (Wilterson & Graziano, PNAS 2021).** A deep-Q agent
  given a descriptive model of its own attention controls attention substantially better;
  ablating the schema destroys performance. A genuine implemented *self-model* — but it
  models attention state, checks nothing against a standard, and records no reasoning
  provenance. Relevant as the strongest "self-model component" prior art; it is not any
  of (i)/(ii)/(iii) as specified.

**Element (iii) verdict: PARTIALLY COVERED in disjoint fields; the integrated form not
found.** Complete derivation provenance exists for formal artifacts (proof terms);
execution provenance exists cryptographically (zkML) but is content-opaque by design;
limited internal-state self-report exists in frontier LLMs (Anthropic) but is unreliable
and is not output provenance. What was **not found**: a generative model whose ordinary
outputs natively carry a machine-checkable record of why they were produced, where that
record is data the model itself can inspect as part of its ongoing self-model, in
open-domain generation.

---

## 5. Combination matrix

| System (year, type) | (i) Workspace | (ii) Independent unmodifiable verifier | (iii) Introspectable provenance |
|---|---|---|---|
| Goyal et al. shared global workspace (2022, modular deep nets) | ● shared bottleneck workspace, modules compete to write, all read | ✗ | ✗ |
| VanRullen & Kanai GLW + successors (2021–, latent workspace) | ●/◐ shared latent space (roadmap + partial builds) | ✗ | ✗ |
| Embodied GW agents — Juliani et al. lineage; Frontiers 2024 | ● agent-internal workspace | ✗ | ✗ |
| Global Workspace Agents / "Theater of Mind" (2025–26, LLM multi-agent) | ● central broadcast hub = sole inter-agent medium (external scaffolding) | ✗ (hub conflict-resolution is part of the same system) | ✗ |
| Consciousness-theory stress-test agents (arXiv:2512.19155, 2025) | ● GW module | ✗ (HOT self-model is internal/learned — monitor is part of the agent) | ✗ |
| Anthropic J-space in Claude (Gurnee et al., 2026, interpretability finding) | ◐ emergent, in-context only, not persistent, not designed | ✗ | ◐ workspace *contents* reportable by model; no derivation record |
| Anthropic concept-injection introspection (Lindsey et al., 2025) | ✗ | ✗ | ◐ self-report of internal states (~20%, "highly unreliable"); not output provenance |
| MT-LNN (2026, single-team preprint) | ◐ GWT bottleneck layer in a 125M LM | ✗ | ✗ |
| ACM (GitHub consciousness framework) | ◐ orchestrator + shared memory stores around stock models | ✗ (hard-coded ethics list; nothing verified) | ✗ |
| TCAI (theconsciousness.ai) | ◐ self-assessed GWT-1/3 "implemented" | ✗ (HOT-2 self-monitoring, self-marked PARTIAL) | ✗ |
| Goldstein & Kirk-Giannini conscious language agent (2024) | ◐ proposed design only | ✗ | ✗ |
| Wilterson & Graziano attention-schema agent (2021) | ✗ | ✗ (self-model of attention; checks nothing) | ✗ |
| Lean-kernel LLM proving pipelines (AlphaProof class, 2024–) | ✗ | ● kernel independent of and unmodifiable by the generator | ● proof term = full checkable derivation (formal domain only; consumed as verifier feedback, not as self-model) |
| Gödel Machine (Schmidhuber, theory only) | ✗ | ● (by design) fixed proof checker gates all self-modification | ● (by design) benefit-proofs are certificates — for self-modifications, not generated outputs |
| Darwin-Gödel Machine (2025) | ✗ | ◐ fixed external benchmark; documented objective hacking — gameable, not a correctness verifier | ◐ archive genealogy of own code versions; not output provenance |
| AlphaEvolve (2025) | ✗ | ◐ automated per-problem evaluators verify code artifacts | ✗ |
| zkML (EZKL / DeepProve / NANOZK, 2023–2026) | ✗ | ◐ independent cryptographic verifier — of execution integrity, not of output correctness vs a standard | ◐ execution provenance only; content-opaque by design; ~10⁷-param scale ceiling |
| Constitutional AI / self-refine / PRM critics (baseline class) | ✗ | ✗ judge is the model itself or a trained, modifiable critic — excluded by (ii)'s definition | ✗ |
| Butlin et al. indicator framework (2023) | — rubric (GWT-1…4) | — rubric (HOT-2 assumes an *internal* monitor) | — not addressed |

---

## 6. Novelty verdicts

**Element (i) — persistent shared workspace: COVERED as an element; XIIV's specific
form not found.** Workspaces are implemented in modular networks, agent scaffolding, and
embodied agents, and an emergent analogue is now reported inside frontier LLMs.
Not found: a workspace that is the generative model's own *persistent substrate* — the
medium generation happens in, accumulating across interactions — as opposed to a
per-episode communication channel (Goyal), an external agent store (GWA/blackboard), or
an in-context activation pattern (J-space). A novelty claim on "global workspace" in
general would collide immediately; a claim scoped to the persistent-substrate form has
no exact collision in the sources searched.

**Element (ii) — independent unmodifiable verifier as metacognition: COVERED in formal
domains; NOT FOUND as a general metacognitive faculty.** Kernels (Lean) and, in design,
the Gödel Machine's proof gate are true prior art for "checker the generator cannot
modify" — for proofs and code. Fixed-benchmark gating (DGM) is weaker and demonstrably
gameable. Not found: such a verifier standing over a generative LM's general outputs as
its metacognitive organ. Additionally, the consciousness literature's own metacognition
indicator (HOT-2) presumes an internal monitor; an *external, unmodifiable* monitor is a
design choice with no instance found in that literature.

**Element (iii) — introspectable reasoning provenance: COMPONENTS COVERED; integrated
form NOT FOUND.** Proof terms give complete provenance for formal claims; zkML gives
execution provenance that hides reasoning by design; Anthropic's results give unreliable
internal-state self-report, not provenance. Not found: outputs that natively carry
machine-checkable why-records which the generating model itself inspects as part of its
self-model, in open-domain generation.

**Triple combination (i)+(ii)+(iii): NOT FOUND.** No source searched describes a
single system containing all three. The maximum overlap found is two-of-three, in
disjoint worlds: Gödel Machine (ii)+(iii) — theory only, self-modification, no
workspace, not a generative model; Lean-proving pipelines (ii)+(iii) — formal domain
only, no workspace, provenance not consumed as a self-model; stress-test/GWT agents
(i) + internal self-model — where the "monitor" is deliberately internal and learned,
i.e. the negation of (ii)'s unmodifiability requirement. Every consciousness-*claiming*
product found (ACM, TCAI, MT-LNN, self-published frameworks) scores ✗ on both (ii) and
(iii). Per the standing rule, this is reported as **absence of evidence within the
searched coverage** (consciousness science, GWT implementations, self-modifying agents,
formal-verification pipelines, zkML, frontier-lab interpretability, 2021–2026, plus the
2023 Butlin framework), **not** as proof that no such system exists anywhere.

**Collision risks to scope any future claim around (facts, not design):**
1. "Global workspace" is owned by Goyal et al. / GWT-agent literature and, inside LLMs,
   by Anthropic's J-space result — claims must differentiate on persistence and on the
   workspace being the generative substrate, not a channel or an emergent pattern.
2. "Unmodifiable verifier" is owned in formal domains by proof kernels — claims must
   differentiate on domain (general generation) and role (metacognitive faculty of the
   generating system, not an external pipeline tool).
3. "Certificate/provenance for outputs" is owned by proof terms (content) and zkML
   (execution) — claims must differentiate on the record being *reasoning* provenance
   for general outputs and on the model itself consuming it introspectively.
4. The word "consciousness" itself is a liability in the current literature: it is
   attached to the weakest systems found (persona + memory + self-assessment), while the
   strongest relevant results (J-space, introspection) come with explicit disclaimers
   from their own authors.
