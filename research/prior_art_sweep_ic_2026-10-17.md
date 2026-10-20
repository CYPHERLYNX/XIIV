# Prior-Art Sweep: Differentiable / ML-Native Interaction Combinators
**Date:** 2026-10-17 (sweep for 8am IST verdict)
**Target question:** Has anyone published a differentiable, learned, or ML-native system built on Lafont's interaction nets / interaction combinators (1990/1997), or applied Lévy-optimal reduction to neural computation?

## VERDICT

**No published work makes Lafont interaction nets/combinators differentiable, learned, or ML-native. The novelty gap is REAL — but narrower and more qualified than "zero prior art."**

Two mandatory qualifications:
1. **HVM is the same substrate, minus learning.** Any claim must cite it as closest prior art and state precisely what is new (a learned/differentiable layer, which HVM does not attempt).
2. **Optimal reduction's empirical record is mixed.** The 2025 Optiscope benchmarks show Lévy-optimal reducers running ~60× slower than unoptimized Haskell on ordinary programs. The asymptotic advantage is real only for sharing-heavy workloads. Any "beats transformers" claim must be scoped to that.

---

## 1. CLOSEST PRIOR ART (same substrate, not ML — must be cited)

### HVM / HVM2 / HVM3 — Victor Taelin, HigherOrderCO
- **What:** Massively parallel evaluator for (extended) interaction combinators. HVM2 paper reports 400 MIPS single-thread (M3 Max) to 74,000 MIPS (RTX 4090).
- **Why not the same:** It is a deterministic evaluator for functional programs. No learning, no gradients, no trainable parameters, no neural component. The claimed invention adds exactly what HVM lacks: a differentiable/learned layer over IC evaluation.
- **Source:** github.com/HigherOrderCO/HVM ; HVM2 paper (docs.rs/crate/hvm)

### Bend — HigherOrderCO
- **What:** High-level massively parallel language compiling to HVM (Python-like syntax → interaction nets → GPU).
- **Why not the same:** A programming language, not a learning system. Programs are written, not trained.

### Interaction Calculus — Victor Taelin
- **What:** Minimal term-rewriting system matching Lamping's optimal λ-calculus reduction; textual syntax for symmetric interaction combinators.
- **Why not the same:** A model of computation / language, not ML. Reference implementations in C/Haskell; no differentiation.

### Optiscope (2025) — orbisai0security, GitHub
- **What:** Lévy-optimal lambda calculus reducer in C, extending Lambdascope.
- **Why not the same:** Not ML. BUT its benchmarks are the most important empirical counter-evidence in this sweep: **~60× slower than unoptimized Haskell on insertion sort**; optimal reduction wins only on Church-numeral-heavy code (exponential vs. polynomial sharing payoff) and loses badly on real-world programs with native datatypes (per Asperti's analysis, cited in plurigrid/nanoclj-zig's 2025 landscape doc).
- **Implication for the verdict:** The "asymptotic advantage" claim must be scoped to sharing-heavy workloads. A kill-gate probe on generic workloads would likely FAIL — and that failure would be consistent with published evidence, not a surprise.

### BOHM — Andrea Asperti et al. (1990s)
- **What:** Prior state-of-the-art optimal reducer (Lamping-style, interaction-net based).
- **Why not the same:** Not ML; same performance caveats as Optiscope.

### Taelin's "Fast Discrete Program Search" (gists, ~2023-2024)
- **What:** Uses HVM superposition (SUP) nodes to enumerate candidate programs against test cases — finds add-carry functions, bit functions, etc.
- **Why not the same:** This is **enumeration-based program search**, not gradient-based learning. No differentiable parameters, no backprop, no training. The closest in spirit (search over IC programs) but a different mechanism class entirely.

---

## 2. NAMING COLLISIONS (must be explicitly disambiguated)

### Battaglia et al., "Interaction Networks for Learning about Objects, Relations and Physics" — NIPS 2016
- **What:** Hugely cited graph neural network for physics simulation.
- **Why not the same:** Zero relation to Lafont. Different "interaction," different "network." The verdict must say "Lafont interaction nets (1990), NOT Battaglia's interaction networks (2016)" everywhere.

### Dietz (2020), "Interaction Networks: Using a Reinforcement Learner to train other ML algorithms"; InteractionNN (IJCAI 2019)
- **What:** ML papers reusing the phrase.
- **Why not the same:** No connection to Lafont's formalism.

### "Differential Interaction Nets" — Ehrhard & Regnier; Mazza & Pagani, LPAR 2007
- **What:** Extension of linear-logic proof nets with differential operators (Taylor expansion of λ-terms); separation theorems, denotational models.
- **Why not the same:** "Differential" = differential LINEAR LOGIC (resource calculus), NOT differentiable programming / gradients. A reader searching "differentiable interaction nets" will hit this and must not mistake it for ML.

### "Diff-Comb Explainer" — arXiv 2404.02625 (LREC-COLING 2024)
- **What:** Differentiable ILP solver for NLI via Differentiable BlackBox Combinatorial Solvers.
- **Why not the same:** "Combinatorial" = integer linear programming, not interaction combinators.

---

## 3. ADJACENT: GRAPH REWRITING + NEURAL (different formalism)

### Chemlambda — Marius Buliga (2013–; arXiv 1403.8046, 2003.14332)
- **What:** Turing-complete artificial-chemistry graph-rewriting system (local moves on trivalent "molecule" graphs), related to interaction combinators.
- **Why not the same:** Artificial life / distributed computing, not ML. No learning, no gradients. Random rewrite scheduling, not optimal reduction.

### GReNN — "Graph Rewriting for Graph Neural Networks" (lanterncode.github.io)
- **What:** Uses graph-rewriting rules to *implement* GNN training/inference (error/delta/infer rules with backprop inside the rewrite system).
- **Why not the same:** Rewriting is the implementation language for a conventional GNN; the neural substrate is not interaction combinators and there is no optimal-reduction claim.

### RuleGNNs — "Rule Based Learning with Dynamic (Graph) Neural Networks," arXiv 2406.09954
- **What:** Rule-based neural layers generalizing FC/conv layers; dynamic structure from predefined rules.
- **Why not the same:** Rule-based weight sharing, not Lafont agents; no interaction-net semantics.

### NeuraLogic / Lifted Relational Neural Networks
- **What:** Differentiable Datalog-derived language for expressing GNNs.
- **Why not the same:** Differentiable logic programming, unrelated to interaction combinators.

---

## 4. ADJACENT: PROOF-CARRYING + NEURAL (post-hoc, not substrate)

### "Proof-Carrying Neuro-Symbolic Code" — arXiv 2504.12031
- **What:** Certified (Imandra-proved) proof checker for neural network verification.
- **Why not the same:** Certifies a CHECKER that verifies NNs post-hoc. The neural execution itself does not carry proofs; proofs are external artifacts about the model, not the computation's data type.

### NeuralCert — Roeling, arXiv 2609.30296 (2026)
- **What:** Discovery-to-certification: neural optimization proposes candidates, independent exact certification verifies.
- **Why not the same:** Explicitly separates discovery from certification ("neural optimization serves only as a proposal mechanism"). The claimed invention unifies them: the reduction IS the proof.

### "Proof of Execution" — arXiv 2607.05397
- **What:** Runtime attestation for governed AI agent actions (contract + event stream + replay).
- **Why not the same:** Agent-governance auditing, not a neural execution substrate.

### Delta Semantics (scienmag, Oct 2026) — proof-carrying ML for versioned ontologies/KGs
- **What:** Neuro-symbolic KG reasoning where outputs carry machine-checkable proofs across ontology updates.
- **Why not the same:** Certifies outputs of a conventional neuro-symbolic pipeline; not interaction-combinator-based; proofs are attached, not intrinsic to reduction.

---

## 5. OTHER IC IMPLEMENTATIONS (runtimes/languages — none ML)

- **Vine** (vine.dev; HN discussion) — interaction-net language with imperative+functional support. Not ML.
- **Stellogen** (micahstubbs) — constellation-based system with formally proven IC encoding (Ragot). Not ML.
- **SICAM** (AHartNtkn) — parallel computer based on ICs (Haskell). Not ML.
- **Small GitHub runtimes** (szeiger/interact, ia0/lafont, Gadersd/ic, zicklag/sucre) — evaluators/visualizations. Not ML.
- **Tübingen theses (2024–2025)** — optimal effects, compiling effect handlers via Bend/HVM. PL research. Not ML.
- **anoma/geb** — Lean formalization of interaction nets (GSOS). Not ML.

## 6. PATENTS & PRODUCT HUNT

- **Patents:** Patsnap indexes interaction-net patents (e.g., implementations of λ-calculus, parallel reduction). None surfaced connecting interaction nets/combinators to machine learning or differentiable computing.
- **Product Hunt:** No product launches found for ML × interaction-nets. Bend/HVM appear in HN/Towards-Data-Science coverage as programming-language stories, never as ML products.

---

## HONEST RESTATEMENT FOR THE VERDICT

Replace "zero prior art, verified clean" with:

> "Clean for ML-native use. The substrate (interaction combinators) has a mature non-ML lineage — Lafont 1990/1997, HVM/HVM2/HVM3 (Taelin), BOHM, Optiscope (2025) — which must be cited as closest prior art; none of it is differentiable or learned. The name collides with Battaglia's 2016 Interaction Networks (unrelated GNN) and with 'differential interaction nets' (differential linear logic, unrelated to gradients); both must be disambiguated. Empirical caveat: Optiscope's 2025 benchmarks show optimal reducers ~60× slower than unoptimized Haskell on ordinary programs — the asymptotic sharing advantage is real only for sharing-heavy workloads, which scopes both the claim and the kill gate."

## SEARCH LIMITATIONS

- arXiv API and Semantic Scholar API both returned HTTP 429 (IP-throttled) during this sweep; arXiv coverage came via web search instead. A slower, credentialed arXiv/Scholar pass could surface niche workshop papers.
- ~12 independent search angles (web, GitHub API, patents, Product Hunt, adjacent terminology) were all consistent: no differentiable/ML-native IC work exists in indexed literature.
