# Beyond the Proof Machine: Hunting Genuinely New Computational Paradigms

**Date:** 2026-10-16
**Mission:** Find computational paradigms "from the future" — ideas that would make even a Compositional Proof Machine (function composition as primitive, proof certificates as data type) look ordinary.
**Method:** Three parallel deep-research tracks (physics+math, neuro+bio, complex systems+weird), ~60 targeted searches across 2024–2026 arXiv/GitHub/literature, plus independent verification. Ruthless standard: weak ideas killed, not hyped.
**Anchor exclusion:** Guarantee-native generation, proof certificates, function composition, constrained decoding, diffusion LMs, transformer-adjacent — all kept as backup, not proposed.

---

## TOP 3 — Ranked by "From the Future" Potential

### #1 — Interaction Nets as the Native AI Substrate

**Primitive (one sentence):** Computation is local graph rewriting — typed agents meet at principal ports and reduce by fixed rules, every redex reducing concurrently with no evaluation order, no layers, no global control.

**Prior art:** NONE for ML — verified clean by direct search. The entire interaction-net community is programming-languages/hardware: HVM2 (Taelin 2024), the Bend language, Δ-Nets (Salvadori, arXiv:2505.20314), NELA, Stellogen. Zero papers on differentiable learning, gradient descent, or trained models on interaction nets. This is a true gap — not "underexplored," unbuilt.

**Language viability:** Path exists; gap is large and concrete. Interaction combinators are Turing-complete (Lafont's 3 combinators), so any differentiable program can be encoded; HVM2 already runs on GPUs with one thread per redex. The core tension: interaction nets are *linear by default* (each port used exactly once), which fights the weighted-sum primitive — weights would live as agent state or multiplicities, and nobody has shown even a toy learned model. Realistic entry point is not "neural net on interaction nets" but "differentiable program reduced optimally."

**10000x argument (mechanism):** Two *asymptotic* — not constant-factor — advantages. (a) **Lévy-optimal sharing:** a layered net recomputes or stores a subcomputation used k times at depth d roughly k×d times; optimal reduction evaluates each redex *family* exactly once. Language and reasoning are deeply shared compositional structures, so total reduction steps scale with distinct subcomputations, not depth×reuse — a compute/Joule win that *compounds with model depth*, exactly where transformers get expensive. (b) **Parallelism is semantic, not scheduled:** no sequential layer bottleneck, no batch synchronization point; depth converts into width automatically.

**Why it makes the Proof Machine look ordinary:** This is the *execution dual* of the compositional proof machine. CPM makes composition the primitive with certificates; interaction nets make composition the execution model with optimal sharing. CPM says "compose exactly"; interaction nets say "compose optimally." If optimal sharing works for learned models, CPM's certificate composition is revealed as one layer above the real substrate.

**Kill experiment:** Build a minimal differentiable weighted interaction-net evaluator (extend HVM2 with weighted agents), train by gradient descent on a tiny task with heavy substructure sharing (recursive Boolean function or small sequence transduction), measure (i) whether usable gradients survive optimal reduction and (ii) whether reduction-step count scales sublinearly vs. layered evaluation as sharing depth grows. **Falsified if:** gradients vanish/explode through reduction, or sharing bookkeeping (Lamping's brackets/croissants) eats the asymptotic win on realistic structure. Known risk stated upfront: optimality is about family-steps; per-step overhead is the open question.

---

### #2 — Ephaptic Field Computing

**Primitive (one sentence):** Neurons influence each other through the shared extracellular electric field — no synapses, no wiring; a global, instantaneous broadcast medium where the field itself acts as a spatial filter on population activity.

**Prior art:** Essentially NOTHING built for AI. One 2025 startup whitepaper (ephapsys.com/Bloxtel, claiming 74% perplexity reduction — unverified, treat as marketing) and one 2025 Zenodo theory piece on ephaptic binding. No peer-reviewed AI architecture. But the *neuroscience* is fresh and strong: Pinotsis et al., *Cerebral Cortex* **2026** — Granger causality shows endogenous fields *drive* neural activity (not vice versa) trial-by-trial in memory tasks, acting as "control parameters"; Boroujeni & Kastner (2025) — a **self-generated** endogenous field *decorrelates* the population, *expands its dimensionality*, and *improves decoding*, with closed-form theory showing the field acts as a spatial filter on the shared slow component. Critically: imposed external fields do the *opposite* (synchronize, collapse). That asymmetry is the key design constraint.

**Language viability:** Speculative but mappable. Replace attention's O(n²) pairwise weight matrix with a global field layer — all units couple to a shared low-dimensional field state (O(n) parameters), where the field carries the "common mode" and units read prediction errors against it. This is a wiring-free attention analogue. No one has tried.

**10000x argument (mechanism):** Attention is fundamentally pairwise — n² interactions computed and stored. A field is **computed once, read by all**: global coordination (binding, common-mode suppression, dimensionality expansion) at O(n) cost with zero wiring. The Boroujeni result is the mechanism: the field *is* the whitening/decorrelation operator, for free, as a physical side effect of population activity. Silicon analogue: shared field state updated by population statistics (software) or capacitively-coupled global lines (hardware).

**Why it makes the Proof Machine look ordinary:** If field dynamics replace attention, the entire transformer paradigm — which CPM still partially inhabits via its diffusion backbone — becomes obsolete. CPM optimizes *within* the pairwise-computation world (exact composition of pairwise-derived representations); field computing says pairwise computation was never necessary. It's a deeper cut.

**Kill experiment:** Build a small attention-free sequence model with a global field-coupling layer; test on retrieval-heavy tasks (needle-in-haystack). **Falsified if:** it can't match linear-attention baselines, or if replacing the learned field with random noise doesn't degrade performance (proving the field carries no real signal).

---

### #3 — Immune Generalization (the 2026 Reframing)

**Primitive (one sentence):** Learn a decision boundary by *deleting* detectors that match a small sampled "self" set, where each detector generalizes across similar inputs (cross-reactivity) — covering an astronomically large input space from ~10% sampling.

**Prior art:** Classic Artificial Immune Systems died with a **quantified** cause — Kim & Bentley showed negative selection would take **1,429 years** to reach 80% detection on 20 minutes of traffic data (~6×10⁸ detectors needed, exponential coverage blowup). Clonal selection variants were relabeled evolutionary algorithms and never beat SVMs. The field withered in the 2000s. **BUT the revival hook is genuinely new:** Meyer & Navalkha, Cold Spring Harbor Laboratory, **August 2026** (*Science Advances*) showed the real immune system achieves ~90% deletion of self-reactive T cells while each T cell encounters only ~10% of self-peptides — because (a) the thymic training distribution matches the tissue test distribution, and (b) receptors are cross-reactive, giving generalization. Classical implementations used *random* detectors with no generalization model — they were missing the actual mechanism. Nobody has built "generalizing negative selection" for ML at scale.

**Language viability:** HIGH and direct. LLMs need exactly this: learn "everything acceptable" as a boundary from safe completions *without* enumerating every jailbreak/hallucination. Applications: guardrail models trained only on safe outputs (one-class), hallucination detection as anomaly detection on hidden-state trajectories, alignment verification via negative selection against a sampled self-set of aligned behaviors.

**10000x argument (mechanism):** The immune system covers a ~10¹⁴-scale antigen space with ~10⁷ receptors — coverage-per-sample efficiency no classifier matches — and is inherently robust to distribution shift *because the training distribution is the test distribution*. Backprop needs negative examples for everything; negative selection with generalization needs only the self-distribution. For AI safety/alignment this **inverts the data problem**: you don't need to enumerate harms, you need a good model of self. The 10000x is in data efficiency for the safety problem specifically.

**Why it (partially) makes the Proof Machine look ordinary:** CPM's verification angle says "prove correctness." Immune generalization says "bound acceptability" — a strictly weaker but vastly cheaper claim. If you can learn the boundary of safe behavior from safe data alone with immune-scale efficiency, the entire proof-certificate apparatus is overkill for the safety use case. It doesn't touch CPM's composition core, but it undercuts the motivation.

**Kill experiment:** Train cross-reactive detectors (RBF/kernel detectors with learned radii) on hidden states of safe LLM completions only; evaluate AUC on jailbreak/hallucination benchmarks. **Falsified if:** (a) detectors can't beat chance without negative examples, or (b) detector count grows exponentially with representation dimension (reproducing the Kim–Bentley blowup — generalization didn't transfer).

---

## STRONG CONTENDERS (didn't make top 3, with reason)

### Single-Neuron Predictive Processing
**Primitive:** Every neuron independently minimizes its own prediction error (dendritic input prediction vs. somatic output) via purely local plasticity — no global backward pass, no weight transport, no layer-wise lockstep.
**Prior art:** Theory-heavy, no scaled system. Senn et al. 2024 (eLife) formalized mismatch-energy learning; Buonomano et al., *Nature Communications* 2025 — first empirical single neurons showing prediction-error responses; Chao/U. Tokyo, **October 2026** — single recurrent spiking population learns event identity/timing/probability with strictly local rules, adapting within 50 trials.
**10000x:** Training cost O(1) in depth — deletes the backward pass entirely, natively online/continual.
**Why not top 3:** It's a *training paradigm*, not a new computational paradigm. It doesn't make CPM look ordinary — it makes backprop look ordinary. Different battle.
**Kill:** Senn-style mismatch-energy learning on next-token prediction, scale depth 10→100. Falsified if perplexity gap vs. backprop widens with depth.

### Phase-Transition-Native Training ("grokking, engineered")
**Primitive:** Learning is a two-phase process (fast fitting, slow compression); the *phase transition itself* is the training objective — deliberately drive the system across it instead of minimizing loss and hoping.
**Prior art:** 2025 theory (de Mello Koch & Ghosh frame grokking + double descent + information bottleneck as one phenomenon with renormalization-group analogy). Nobody has built the training paradigm.
**10000x:** Current training wastes compute in the wrong phase; direct compression-phase optimization could reach generalizing solutions orders of magnitude faster — and produce exact algorithms instead of statistical mush.
**Why not top 3:** Also a training paradigm, not architecture. And grokking may not generalize to language.
**Kill:** Grokking task with explicit compression-phase objective vs. AdamW. Falsified if no ≥10× speedup to test accuracy.

### Univalence as Computation (HoTT)
**Primitive:** The model's data type is a higher type in cubical type theory; univalence makes type equivalence a path, and *transport along that path computes* — identifying representations as equivalent literally rewrites data across them.
**Prior art:** None as architecture (verified). Only metatheory and hobbyist formalizations.
**10000x:** Equivalence becomes definitional — solve once, transport everywhere, analogy as O(1) amortized computation. Reasoning invariant under representation change.
**Why not top 3:** No language path whatsoever; reasoning-only. And the most likely death — equivalence discovery costing ≥ solving directly — is acknowledged by the mechanism itself. Beautiful but narrow.
**Kill:** Cubical Agda benchmark with isomorphic variants; compare transport cost vs. from-scratch solving. Falsified if finding the path costs as much as the problem.

### Differentiable Weightless Networks (LUT primitive)
**Primitive:** Computation is *table lookup*, not multiply-accumulate — RAM nodes (n-input LUTs), learning = writing table entries. Inference is O(1) memory reads, zero arithmetic.
**Prior art:** Genuine 2024–2026 revival (TsetlinWiSARD, Differentiable WNNs — 96%+ at 56 nJ/sample, claimed 926,000× energy savings on FPGA). All at edge-classifier scale.
**10000x:** Deletes the multiply-accumulate — five to six orders of magnitude inference energy reduction. Intelligence stops being gated by FLOPs.
**Why not top 3:** LUTs explode combinatorially with input width (2^n entries). Zero language results. The combinatorial wall for sequences is likely fatal — this is an edge-inference story, not a paradigm for ASI.
**Kill:** Differentiable weightless sequence model on copy/reverse over binary sequences. Falsified if table count scales exponentially in sequence length.

### Bioelectric Pattern Memory (Levin Lab)
**Primitive:** Gap-junction-coupled cell network stores rewritable, long-term-stable memories as distributed resting-voltage patterns; computation = field relaxation toward an attractor.
**Prior art:** None for AI. Levin lab (Tufts): planarian flatworms with rewritten bioelectric states regenerate as two-headed worms indefinitely with wild-type genomes — memory in the bioelectric layer, surviving tissue destruction, rewritable without touching "weights."
**10000x:** Damage-proof, un-forgettable memory — catastrophic forgetting structurally impossible.
**Why not top 3:** Furthest from language of all candidates. Developmental biology with no ML bridge. Fascinating, not actionable.
**Kill:** Continuous-state network with diffusive coupling storing voltage attractors; lesion 30% of units. Falsified if recall doesn't exceed Hopfield nets of equal size.

---

## KILLED — With Cause of Death

**Physics track:**
- Time crystals / DTC as substrate — Zhang et al., arXiv:2508.15230 (Aug 2025): DTC quantum reservoir with *experimental* MNIST on superconducting processors. Real but reservoir-at-MNIST; blocked on quantum hardware regardless.
- Morphological computation — field's own 2025 commentary argues the body *doesn't* compute; stuck in locomotion demos.
- Jarzynski/fluctuation theorems as learning rule — absorbed: Sohl-Dickstein 2015 (diffusion = non-equilibrium thermo) already happened. Analysis, not a rule.
- Quantum Boltzmann machines — very active 2024–2026 (Coopmans & Benedetti, Gerlach et al.). Substantive prior art.
- Dissipative QNNs — Beer et al. 2020 + Sutter et al. 2025. Done.
- Anyonic braiding as forward pass — TRUE GAP but blocked on physics (no topological QC at scale, no language path). Revisit if hardware materializes.
- Coupled learning / physical in-situ training — substantive (EPFL PhyLL 2023, photonic DNN 2024) but a training method for the same architectures, not a new paradigm.

**Math track:**
- String diagrams / optics / lenses as native computation — Backprop-as-Functor (2017), DisCoCat/lambeq (Coecke) for QNLP. Done, multiply.
- Sheaf theory for distributed reasoning — Sakana AI's Sheaf-ADMM (2025): consistency as learning signal, fully differentiable. Current and substantive.
- TDA / persistent homology per layer — PersLay, TopologyLayer, TopoModelX, PHINN (2026). Differentiability never solved; stuck in vision/graphs.
- Topos theory — descriptive (Belfiore & Bennequin), no computational primitive, no demonstrated advantage.

**Neuro track:**
- Dendritic deep learning (dendrite as unit) — DenRAM (*Nature Comm* 2024), "Dendrites endow ANNs" (*Nature Comm* 2025), Dendrify. Taken. (Dendritic *predictive coding* as learning rule survives — see contenders.)
- Astrocyte-augmented architectures — Kozachkov & Krotov, *PNAS* 2023 (tripartite synapse = attention). Incremental since.
- Myelin plasticity / delay learning — DenRAM already uses learnable delays. No distinct mechanism.

**Bio track:**
- Protein folding as reasoning — "reasoning as energy minimization" is diffusion LMs / EBMs. Prior art.
- Slime mold computing — real TSP results (2025–26) but Hopfield-adjacent algorithmically; wetware stuck in optimization toys.
- Quorum-sensing computation — Technion already built bacterial ANNs. Weak transferable primitive.
- Fungal/mycelium computing — Adamatzky's program since 2018; still at logic-gate level.
- DNA neural networks — Cherry et al., *Nature* **2025**: supervised learning in DNA nets (7-year journey). Genuinely new substrate, but 30–60 min inference in a wet lab on 8×8 images. Language viability ≈ 0.
- Neural Darwinism revival — no 2024–26 revival found; philosophy literature only.

**Complex systems track:**
- Reaction-diffusion as computation — empty room. (Nearest: CNNs with "RD" in the name.)
- Amorphous computing — dead, no 2024–26 revival. NCAs are its successor.
- Stigmergic architectures — dead since 2019.
- Google NCA reasoners (arXiv 2609.36126, Sept 2026) — 10K–160K params, 100% on 201×201 mazes, 98.5% Sudoku-Extreme, competitive on ARC-AGI-1, test-time space/time scaling. SUBSTANTIVE — killed as "new" but the test-time scaling mechanism is worth stealing.

**Weird track:**
- HRR/VSA revival — alive (100× retrieval improvement, GHRR non-commutative binding) but binding-is-composition puts it adjacent to the excluded list; nothing near language scale.
- Deep Boltzmann resurrection — real 2025–26 work (unbiased CD, parallel samplers) but binary-native; no language path.
- Backprop-free (Forward-Forward / Equilibrium Propagation) — substantive revival (EP: 93.92% CIFAR-10, comparable to backprop) but nothing past image classification; negative-data for language unsolved.
- Mechanism design as architecture — nothing (only MLOps auction routing).

---

## BOTTOM LINE — Honest Assessment

**The single strongest find: interaction nets.** Zero ML prior art (verified), Turing-complete, runs on GPUs today via HVM2, asymptotic advantages via optimal sharing, cheapest kill experiment. It's the only candidate that is simultaneously completely unbuilt, immediately actionable on existing hardware, and offering asymptotic rather than incremental wins. If I funded one experiment tomorrow, it's the differentiable interaction-net evaluator.

**The highest-variance bet: ephaptic fields.** No prior art at all, real 2026 mechanism, attacks attention's O(n²) at the root. Could be nothing; could obsolete the transformer.

**The most immediately useful: immune generalization.** The August 2026 reframing is new science on a quantified death. Cheapest kill experiment. Directly applicable to the alignment/safety problem — and "inverts the data problem" is the kind of asymmetry that produces real moats.

**What would make ALL of these (including the Proof Machine) look ordinary:** a paradigm where *learning, inference, memory, and safety* are the same operation — not four systems bolted together. None of the candidates above achieve this alone. The interaction-net substrate comes closest structurally (execution unifies everything), but it's the furthest from a working system. That unification — if it exists — is the real "from the future" idea, and nobody has it yet.

**Ruthless self-check:** None of these is a guaranteed 10000x. Interaction nets might die on gradient-through-reduction. Ephaptic fields might not transfer from spiking nets to sequence models. Immune generalization might reproduce the Kim–Bentley blowup in high dimensions. Every top candidate has a clean kill experiment — that's what makes them candidates instead of hype. The ranking reflects "from the future" potential, not certainty. If the parent wants certainty, none exists at this frontier — that's what makes it the frontier.
