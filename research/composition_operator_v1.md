# Composition Operator v1: Mathematical Foundation of the Compositional Proof Machine

**Date:** 2026-10-16
**Status:** Phase 1 formalization — the core operator defined. Prototype-ready.
**Verdict after flaw-hunting:** SURVIVES. Two serious objections identified and answered below. No fatal flaw found.

---

## 1. Certified Composable Function: Formal Definition

### 1.1 The Mathematical Object

A **certified composable function** over a many-sorted signature Σ is a tuple:

```
F = (f, P, Q, τ)
```

where:

- **f : D_A → D_B** is a computable function between concrete domains. In the neural realization, `f` is implemented by a neural module (weights + architecture), but its *semantics* are what the certificate constrains, not its implementation details.

- **P ∈ Presburger(Σ_A)** is a **precondition**: a formula in Presburger arithmetic over the input sort. Presburger arithmetic = first-order logic with `0`, `1`, `+`, `<` over integers. **Decidable.** No variable–variable multiplication. This is the maximal decidable arithmetic — a deliberate choice (see §1.3).

- **Q ∈ Presburger(Σ_A ∪ Σ_B)** is a **postcondition**: a Presburger formula relating input `x` and output `y = f(x)`.

- **τ** is a **certificate**: an abstract-interpretation trace `(a₀, a₁, …, aₙ)` in a sound abstract domain `D#` such that:
  - `γ(a₀) ⊇ { x | P(x) }` — the initial abstract state over-approximates all inputs satisfying `P`
  - `a_{i+1} = F#(a_i)` for each step, where `F#` is the abstract transformer sound w.r.t. `f`
  - `γ(aₙ) ⊆ { (x,y) | Q(x,y) }` — the final abstract state implies the postcondition

**Soundness theorem (inherited, not invented):** By the Cousot–Cousot soundness theorem for abstract interpretation (1977), the existence of such a trace implies:

```
⊢ {P} f {Q}    i.e.    ∀x. P(x) → Q(x, f(x))
```

The certificate τ is not a claim — it is a *witness* whose validity is mechanically checkable in time linear in `|τ|`.

### 1.2 What the Certificate IS, Concretely

Not a proof term in a general-purpose proof assistant. Not a natural-language justification. The certificate is an **abstract-interpretation trace**: the sequence of abstract states produced by pushing the precondition's abstraction through the function's abstract transformer.

Example. `F = ADD_3` on integers:
- `f(x) = x + 3`
- `P(x) := true`
- `Q(x,y) := (y = x + 3)`
- Abstract domain: symbolic expressions (exact for linear ops). Trace: `a₀ = [x]`, `a₁ = [x + 3]`. Verification: check `a₁` syntactically implies `y = x + 3`. Trivially valid.

Example with intervals. `F = MUL_2` with input precondition `P(x) := (0 ≤ x ≤ 10)`:
- `f(x) = 2x`
- `Q(x,y) := (y = 2x ∧ 0 ≤ y ≤ 20)`
- Interval trace: `a₀ = [0,10]`, `a₁ = MUL_2#([0,10]) = [0,20]`. Valid.

### 1.3 Why Presburger Arithmetic (the Minimal Logic)

The assertion logic must satisfy three constraints:

1. **Decidable.** If certificate verification is undecidable, the whole system collapses — we'd have replaced neural approximation with an unsolvable verification problem. Presburger arithmetic is decidable (Fischer–Rabin: doubly-exponential worst case, but fast in practice for our small formulas via modern SMT solvers like Z3).

2. **Expressive enough for the prototype domain.** Presburger handles: linear arithmetic (`x + 3y < 10`), bounds and ranges, counting, multiplication by constants (`3x` unfolds to `x+x+x`), and modular arithmetic (`y = x mod m` ⟺ `0 ≤ y < m ∧ ∃q. x = q·m + y` — the existential is first-order, fine).

3. **Avoids Gödel.** By staying below full Peano arithmetic (no variable–variable multiplication as a primitive), we sidestep incompleteness. Every true Presburger sentence has a proof; our certificates can in principle witness any true property in the fragment.

**Deliberate limitation:** Presburger cannot express nonlinear relationships between variables (`y = x²`). The prototype does not need them. Richer logics (with corresponding decidable abstract domains) are a Phase 3 concern, not a Phase 1 blocker. Starting with the minimal logic is a feature: it keeps the kill-gate experiment clean.

---

## 2. The Composition Operator ∘

### 2.1 Definition

Given:
```
F = (f, P_f, Q_f, τ_f)    with   ⊢ {P_f} f {Q_f}
G = (g, P_g, Q_g, τ_g)    with   ⊢ {P_g} g {Q_g}
```
where the output sort of `F` matches the input sort of `G`, define:

```
G ∘ F  =  (g∘f, P, Q, τ)
```

- **Composed function:** `(g∘f)(x) = g(f(x))` — ordinary mathematical composition.
- **Composed precondition:**
  ```
  P(x)  :=  P_f(x)  ∧  ∀y. (Q_f(x,y) → P_g(y))
  ```
  The input must satisfy `F`'s precondition, AND every possible output of `F` (as characterized by `Q_f`) must satisfy `G`'s precondition. This is the *interface condition* — it is where composition can fail loudly rather than silently.

- **Composed postcondition:**
  ```
  Q(x,z)  :=  ∃y. (Q_f(x,y) ∧ Q_g(y,z))
  ```
  There exists an intermediate value `y` witnessing both halves. The witness is `y = f(x)`.

- **Composed certificate:** `τ = τ_f ⌢ τ_g*`, the concatenation of `F`'s trace with `G`'s trace *re-rooted* at `F`'s output abstraction. Formally, if `τ_f = (a₀,…,aₘ)` and `τ_g = (b₀,…,bₙ)`, then `τ_g*` is computed by setting `b₀' = aₘ` (replacing `G`'s initial abstraction with `F`'s final abstraction, after checking the interface condition `γ(aₘ) ⊆ {y | P_g(y)}`) and re-running `G`'s abstract transformer: `b_{i+1}' = G#(b_i')`.

### 2.2 The Certificate Composition Rule (the Heart)

```
    τ_f ⊢ {P_f} f {Q_f}         τ_g ⊢ {P_g} g {Q_g}
    γ(last(τ_f)) ⊆ { y | P_g(y) }          [interface check]
—————————————————————————————————————————————————————————————————————— (COMP)
    τ_f ⌢ τ_g* ⊢ {P_f ∧ ∀y.(Q_f(x,y) → P_g(y))} g∘f {∃y. Q_f(x,y) ∧ Q_g(y,z)}
```

**What makes this rigorous:** This is not a new proof rule we invented and hope is sound. It is the *abstract-interpretation instance* of the Hoare logic sequential composition rule, whose soundness is textbook (Hoare 1969; soundness via Cousot–Cousot for the abstract-interpretation realization). We are instantiating a proven rule, not proposing one.

**What makes this powerful:** Certificate composition is **syntactic and mechanical**. Given `τ_f` and `τ_g`, constructing `τ` requires:
1. One interface check (abstract-domain inclusion — decidable for our domains)
2. Re-running `G`'s abstract transformer from `F`'s output state (linear in `|τ_g|`)
3. Concatenation

There is **no proof search**. There is **no re-verification from scratch**. The cost of composing certificates is `O(|τ_f| + |τ_g|)` in proof size, and verification of the composed certificate remains linear in its size. After `L` compositions, the certificate has size `O(L · c)` where `c` bounds component certificate size — linear, not exponential.

### 2.3 The Deep Point: Exact vs. Approximate Composition

This is where the architecture diverges from every neural baseline:

| | Transformer (implicit) | Proof Machine (explicit) |
|---|---|---|
| Composition mechanism | Stacked attention layers approximate composed function | ∘ operator constructs composed function + certificate |
| Per-step error | ε > 0 (neural approximation) | 0 (exact, by construction) |
| L-step error | Compounds: ~L·ε or worse | 0 (certificates compose exactly) |
| Certificate | None | Mechanical, linear-size |
| Failure mode | Silent degradation | Loud interface-check failure |

The transformer *approximates* each composition step with neural weights. Approximation errors compound with depth — this is the structural source of the compositionality wall. The proof machine *constructs* each composition step with a certificate. There is no approximation in the composition path, so there is nothing to compound. **This is the entire bet, stated in one sentence: exact composition does not degrade with depth.**

---

## 3. Why Transformer Impossibility Results Don't Apply

### 3.1 What the Lower Bounds Actually Say

The TC⁰ characterization and depth-width trade-offs (arXiv 2402.08164 and related) are statements about a **specific computational model**:

- **Model:** Log-precision transformer = constant-depth threshold circuit with self-attention as the gate mechanism, polynomial width, O(log n)-bit activations.
- **Result:** Computing L-step function composition in this model requires Ω(L) depth or n^Ω(1) width. Log-precision transformers are contained in TC⁰ and therefore cannot compute TC⁰-hard functions (certain linear systems, context-free recognition) at fixed depth.

These are theorems about **attention-based bounded-depth circuits**. They bind any system that computes via that mechanism.

### 3.2 Why We're Not Bound

The Compositional Proof Machine is **not a log-precision transformer**. It is not a bounded-depth threshold circuit. Its primitive operation is not self-attention — it is the certified composition operator ∘ defined in §2.

The analogy: the lower bounds say "a finite automaton cannot recognize {aⁿbⁿ}." This does not imply "no computational system can recognize {aⁿbⁿ}" — a pushdown automaton does it trivially. The bound is about the *model*, not the *problem*. We are proposing a different model.

Concretely:
1. **Our "depth" IS the composition depth.** Transformers must *simulate* L-step composition within fixed attention layers (hence the Ω(L) lower bound — they need linearly many layers). We don't simulate composition; composition is the native step. Asking how many "layers" our ∘ needs to compose L functions is a category error — it needs L applications of ∘, each O(1), each exact.

2. **We are not precision-bounded in the relevant sense.** The TC⁰ result depends on log-precision activations. Our certificates are symbolic (Presburger formulas, abstract-domain elements) — they grow linearly with depth (§2.2), not logarithmically bounded. We trade the transformer's fixed-precision parallelism for exact symbolic accumulation.

3. **The hardness is relocated, not eliminated.** We do not claim to solve TC⁰-hard problems at fixed cost. The *certificate* for an L-step composition has size O(L). What we claim — and what the prototype tests — is that **accuracy does not degrade with L**, because each step is exact. The transformer degrades because each step is approximate. This is a claim about *error scaling*, not *complexity class membership*.

### 3.3 The Honest Boundary

This argument does not prove the proof machine is *more computationally powerful* than transformers in the complexity-theoretic sense. It argues something narrower and more relevant: **the specific mechanism that causes transformer accuracy to degrade with composition depth (compounding approximation error) is absent from our computational model by construction.** The kill-gate prototype (§5) tests exactly this claim, and nothing more.

---

## 4. The Neural Bridge: How Networks Learn Certified Functions

### 4.1 The Problem

Neural networks are function approximators. Certificates are symbolic objects. The gap: how do we get from gradient descent on weights to a certified composable function `(f, P, Q, τ)`?

### 4.2 Evaluated Options

**(a) Network outputs (function, certificate) pairs with verifier loss.**
The network outputs both weights and a proof term; loss includes `λ·(1 − verifier_accepts(π))`. **Rejected.** Proof terms are discrete symbolic structures. Gradients don't flow through them. Workarounds (RL, straight-through estimators, Gumbel-softmax over proof skeletons) are fragile and have no track record at this task. This path conflates what networks are good at (continuous approximation) with what they're bad at (discrete symbolic construction).

**(b) Certificates as constraints; constructed by abstract interpretation, not output by the network. ← SELECTED.**
The network outputs only the function (as weights). The certificate is *derived* by a fixed, sound symbolic procedure (abstract interpretation) that analyzes the trained network. Training includes a loss term that keeps the network's behavior within the precision of its abstract transformer. The network does what it's good at; the symbolic system does what it's good at; they meet at the abstract domain.

**(c) Differentiable proof construction.**
Make the proof system itself continuous (proof terms as differentiable relaxations). **Rejected for Phase 1.** Elegant in theory, but no existing system demonstrates differentiable proof construction at the reliability we need. This is a research program in itself, not a bridge we can walk today.

### 4.3 The Selected Mechanism: Certified-by-Construction Training

Each primitive in the library is a pair:

```
Primitive_i  =  (M_i,  M_i#)
```

- **M_i**: a neural module (small MLP) with weights θ_i, implementing function f_i.
- **M_i#**: a FIXED symbolic abstract transformer, sound w.r.t. any function in M_i's hypothesis class. Not learned. Not approximated. Constructed once by the designer from the architecture of M_i and the chosen abstract domain.

**Training objective** for primitive `i` on task data `D`:

```
L(θ_i)  =  L_task(θ_i; D)  +  λ · L_tight(θ_i)
```

- `L_task`: standard supervised loss (e.g., MSE against target outputs on random inputs).
- `L_tight`: the *abstraction-tightness* loss. Let `a = M_i#(ᾱ(x))` be the abstract output for input abstraction `ᾱ(x)`, and let `width(a)` measure the imprecision (e.g., interval width, or symbolic expression complexity). `L_tight` penalizes the network when its concrete outputs stray from the center of the abstract output, i.e., it trains the network to be *well-abstracted* by its fixed transformer. Concretely:
  ```
  L_tight(θ_i) = E_x[ dist( M_i(x; θ_i),  center( M_i#(ᾱ(x)) ) ) / width( M_i#(ᾱ(x)) ) ]
  ```
  This keeps certificates tight: a network whose behavior is precisely captured by its abstract transformer yields small, informative certificates.

**Composition at inference:** Given a sequence of primitives `(M_{i_1}, …, M_{i_L})`:
1. Concrete execution: `x_{k+1} = M_{i_k}(x_k)` — standard forward passes.
2. Certificate construction (in parallel, symbolic): `a_{k+1} = M_{i_k}#(a_k)` — abstract transformer applications.
3. Final certificate `τ = (a_0, …, a_L)`. Verified by checking each step's soundness (mechanical) and the interface conditions (decidable inclusion checks).

**Why this is not "neural net + external verifier" (Gap 3 objection):** In post-hoc verification systems, the network produces an output and a separate system checks it — verification can *fail*, triggering retry or fallback. Here, the abstract transformers are part of the *forward pass itself*: every execution produces its certificate by construction. There is no verification-failure mode for the composition — the certificate always exists because it is built, not checked. The interface check (§2.2) can fail, but that is a *loud, localizable* failure (tells you exactly which composition step's precondition was violated), not a silent degradation.

### 4.4 What the Network Actually Learns

The network learns *functions whose behavior is precisely captured by a fixed symbolic abstraction*. This is a restricted hypothesis class — the restriction is the point (cf. the Colbrook impossibility sidestep: we train within a provably-tractable class rather than hoping arbitrary networks are reliable).

The composition *policy* (which primitive to apply when) is a separate learning problem:
- **Phase 1 (kill-gate):** The composition sequence is GIVEN explicitly in the task input. The prototype tests the architectural claim (exact composition doesn't degrade), isolating it from program synthesis.
- **Phase 2+:** Learning the composition policy (from specification to primitive sequence) becomes the next problem. Options: supervised on (spec, sequence) pairs, RL with certificate-validity as a dense reward signal, or LLM-guided synthesis with the certificate as a correctness oracle. Deliberately out of scope for the kill gate.

---

## 5. Micro-Prototype Spec: The Kill Gate

### 5.1 Objective

Test the single architectural claim: **explicit certified composition maintains accuracy with composition depth, while implicit neural composition (transformer baseline) degrades.** Pass → continue the bet. Fail → kill it.

### 5.2 Task Family: Certified Arithmetic Chains

**Primitive library** (K = 8 primitives, each a small MLP + fixed symbolic abstract transformer):

| Primitive | Function | Certificate (postcondition) | Abstract domain |
|---|---|---|---|
| ADD_k (k ∈ {1,3,7}) | x → x + k | y = x + k | Symbolic (exact) |
| MUL_k (k ∈ {2,5}) | x → x × k | y = k·x | Symbolic (exact) |
| MOD_m (m ∈ {7,13}) | x → x mod m | 0 ≤ y < m ∧ ∃q. x = q·m + y | Symbolic + bounds |
| NEG | x → −x | y = −x | Symbolic (exact) |

Inputs: integers in [−1000, 1000]. All operations are Presburger-expressible. Symbolic abstract transformers are EXACT for this op set (no precision loss — see §5.5).

**Task instance (depth L):** Given input `x₀` and an explicit op sequence `S = (op_1, …, op_L)`, compute `x_L = op_L(…op_1(x₀)…)` and produce the composed certificate `τ`.

Example (L=4): `x₀ = 42`, `S = (ADD_3, MUL_2, MOD_7, ADD_1)`.
Expected: `((42+3)×2 mod 7)+1 = (90 mod 7)+1 = 6+1 = 7`, with certificate tracing each step.

### 5.3 Architecture Under Test

**Proof machine:**
- Each primitive `M_i` trained independently (supervised, 10K random inputs each, `L_task` + `L_tight`).
- At test: read op sequence `S` from input (no synthesis — sequence is explicit).
- Execute concretely + symbolically in parallel. Output `(x_L, τ)`.
- Correctness: `x_L` matches expected AND `τ` verifies (mechanical check).

**Baseline (transformer):**
- Standard decoder-only transformer (~1M params, matched to proof machine's total primitive params).
- Trained end-to-end on `(x₀, S) → x_L` pairs, same training distribution.
- Must learn composition implicitly through its layers.

### 5.4 Depths, Metric, Pass/Kill

**Depths:** `L ∈ {1, 2, 4, 8, 16, 32, 64, 100}`. 1000 random task instances per depth. Fresh random seeds per depth (no memorization).

**Primary metric:** Task accuracy `A(L)` = fraction of instances where output equals expected value (exact integer match).

**Secondary metric:** Certificate validity `V(L)` = fraction of proof-machine outputs where `τ` mechanically verifies. (Expected: 100% at all depths — if this drops, something is wrong with the implementation, not the theory.)

**Pass criterion (ALL must hold):**
1. `A_PM(100) ≥ 90%` — proof machine accuracy at max depth.
2. Flat-curve test: fit `A_PM(L) = a − b·L` by least squares. Pass requires `b` not significantly different from 0 (two-sided t-test, p > 0.05).
3. Separation: `A_PM(100) − A_TF(100) ≥ 40 percentage points` — the proof machine decisively beats the transformer at depth, confirming the architectural (not just training) difference.

**Kill criterion (ANY triggers kill):**
1. `b` significantly negative (p < 0.05) with effect size ≥ 50% of the transformer's degradation slope — the proof machine degrades like a transformer, falsifying the core claim.
2. `A_PM(100) < 70%` — absolute failure regardless of curve shape.
3. `V(L) < 100%` at any depth — certificate construction is broken (implementation flaw, but kills this prototype iteration).

### 5.5 Why This Test Is Fair (Pre-registered Defenses)

**"The symbolic transformers are exact, so of course it doesn't degrade — this is trivial."**
The non-trivial claim is not that symbolic execution works (it does — that's textbook). The claim is that a *neural* system can be organized around certified composition such that (a) the neural components learn effectively under the tightness loss, (b) the composed system maintains accuracy where end-to-end neural systems degrade, and (c) this organization is a viable architectural primitive for scaling. The prototype tests whether the neural+symbolic *integration* holds up, not whether symbolic execution works in isolation.

**"The transformer baseline is under-parameterized / under-trained."**
Match total parameters. Train to convergence on the training distribution (depths 1–16). Give it the same explicit op sequence in the input. If the baseline still degrades at depths 32–100 (out-of-distribution depth generalization), that IS the compositionality wall manifesting — the inability to generalize compositional computation beyond training depth is precisely the phenomenon we're targeting.

**"Explicit op sequences dodge the hard problem (program synthesis)."**
Correct, and deliberate. The kill gate isolates ONE claim: exact composition doesn't degrade. Program synthesis (inferring the sequence from a specification) is a separate, harder problem for Phase 2. Conflating them would make the experiment uninterpretable — a failure wouldn't tell us which claim died.

### 5.6 Additional Robustness Tasks (if primary passes)

1. **Logic chains:** Boolean primitives (AND, OR, NOT, XOR) on bit-vectors. Tests the framework beyond arithmetic.
2. **Conditional composition:** IFZERO branching (tests precondition-dependent interface checks — the `∀y.(Q_f → P_g)` condition doing real work).
3. **Noisy inputs:** Input intervals instead of points (tests abstract-interpretation precision under uncertainty).

### 5.7 Timeline and Resources

- **Weeks 1–2:** Implement primitive library (MLPs + symbolic transformers), training pipeline, certificate verifier.
- **Weeks 3–4:** Train primitives, build transformer baseline, run depth sweep.
- **Weeks 5–6:** Analysis, robustness tasks, write-up. **Decision: PASS or KILL.**

Compute: trivial (CPU-only, <1M params total). This runs on the current box without contention.

---

## 6. Known Limitations and Non-Goals for Phase 1

1. **Expressivity.** Presburger arithmetic cannot represent natural-language semantics. This phase does not attempt to. The claim under test is architectural (composition scaling), not linguistic.
2. **Program synthesis.** The composition policy is given, not learned. Learning to *discover* compositions is Phase 2.
3. **Scale.** The prototype has ~1M parameters and 8 primitives. Scaling the primitive library to thousands and the logic to richer fragments is future work — but the composition operator itself is scale-free (it doesn't care how many primitives exist).
4. **Training paradigm completeness.** The tightness loss (§4.3) is a first proposal. It may need refinement if primitives don't learn well-behaved abstractions. The kill gate will surface this.

---

## 7. Decision Log

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Assertion logic | Presburger arithmetic | Full PA / Dependent types | Decidable; sufficient for prototype; avoids Gödel |
| Certificate form | Abstract-interpretation trace | Proof terms / NL justifications | Mechanical construction; linear verification; no proof search |
| Neural bridge | (b) Certificates as constraints via fixed abstract transformers | (a) Network outputs proofs / (c) Differentiable proofs | (a) gradients don't flow through discrete proofs; (c) no working precedent |
| Composition semantics | Hoare-style sequential composition | Category-theoretic / Type-theoretic | Directly matches the problem; proven sound; implementable |
| Kill-gate scope | Explicit sequences, arithmetic | Program synthesis, NL | Isolates the architectural claim; interpretable failure modes |

---

**Next action:** Implement the primitive library + symbolic transformers + certificate verifier (Weeks 1–2 of §5.7). The formalization is complete enough to build from. No further theory needed before code.
