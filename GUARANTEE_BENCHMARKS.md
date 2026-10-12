# XIIV Guarantee-Native Benchmarks — definitions

**Purpose:** These are the benchmarks where XIIV competes on its own axis — not general knowledge or reasoning scale (where frontier models win on parameters alone), but provable guarantees about model behavior. Frontier autoregressive models score ~zero here because they lack the architectural capability, not because they're weak.

**Status:** Definitions frozen 2026-10-11. No measurements yet. Scorer and harness are NOT BUILT — this document defines what to build.

**Honesty rule:** Every metric below must be measured by an independent verifier, not self-reported by the model. A benchmark the model grades itself is not a benchmark.

---

## Benchmark 1: Contract Compliance Rate (CCR)

**What it measures:** % of generations that provably satisfy a given formal contract.

**Protocol:**
1. Define a contract as a machine-checkable predicate over outputs (e.g., "output contains all required tokens," "output contains zero forbidden tokens," "output length within [a,b]," "JSON output parses and has required keys").
2. Generate N outputs under the contract (N ≥ 500).
3. An INDEPENDENT verifier (separate code, no model weights) checks each output against the contract.
4. CCR = verified-compliant / N.

**Why frontier models score ~zero:** They generate probabilistically and filter post-hoc. Post-hoc filtering can reject bad outputs but cannot *prove* compliance — and when the filter is the same model judging itself, it's circular. XIIV's sampler enforces contracts during generation by construction (required/forbidden tokens, locked spans — already tested 19/19 in the 52.5M).

**Target:** CCR ≥ 99% on a contract suite, verified independently. The suite must include adversarial contracts (tight constraints where naive generation fails often).

---

## Benchmark 2: Intervention Rate (IR)

**What it measures:** How often the model detects and corrects its own contract violations mid-generation, with an auditable log.

**Protocol:**
1. Run generation under contracts designed to be violable (the model must navigate around forbidden regions).
2. Log every point where the generation process detected a potential violation and intervened (re-masked, re-sampled, or blocked a token).
3. IR = interventions / total generations. Also report: false-intervention rate (interventions on generations that would have been compliant anyway, measured by running without intervention and checking).

**Why this matters:** This is the "prover vs hoper" metric. A hoper generates and hopes. A prover monitors and corrects. No frontier lab publishes this because their architecture has no intervention mechanism to measure.

**Target:** Non-zero, calibrated IR with false-intervention rate < 5%. The absolute number matters less than: (a) it exists, (b) it's honest, (c) it correlates with higher CCR.

---

## Benchmark 3: Proof-Carrying Output Verification (PCOV)

**What it measures:** Can a third party, with no access to model weights, verify that an output satisfies its contract?

**Protocol:**
1. Model generates output + certificate (the certificate records: contract ID, generation seed, sampler constraints applied, intervention log hash).
2. Third-party verifier (published open-source, deterministic) takes (output, certificate, contract) and returns COMPLIANT / NON-COMPLIANT / CERTIFICATE-INVALID.
3. PCOV = % of outputs where verifier returns COMPLIANT, over N ≥ 500 generations.
4. Negative control: tamper with 100 outputs (flip a token to violate the contract); verifier must return NON-COMPLIANT on all 100.

**Why frontier models can't play:** Their outputs carry no proof. You can ask GPT to "follow these constraints" and check afterward, but there's no certificate chain — nothing to verify independently. This is the structural gap.

**Target:** PCOV ≥ 99%, negative control 100% detection.

---

## Benchmark 4: Locked-Span Fidelity (LSF)

**What it measures:** Byte-exact preservation of locked spans during masked infill.

**Protocol:**
1. Provide contexts with locked spans (substrings that must appear byte-identical in the output).
2. Model performs masked infill around the locked spans.
3. LSF = % of outputs where every locked byte is preserved exactly.

**Status:** Already 19/19 on sampler tests at 52.5M. This benchmark formalizes it at scale (N ≥ 1000, varied span lengths and positions).

**Why it matters:** Demonstrates deterministic control inside a probabilistic model — the "guarantee brick" primitive. Frontier models cannot lock spans; they can only be prompted and hoped.

---

## Benchmark 5: Generation-Order Control (GOC)

**What it measures:** Can the model generate in a specified non-left-to-right order while maintaining coherence?

**Protocol:**
1. Specify a generation order (e.g., "generate the conclusion first, then the premises" or a custom position permutation).
2. Model generates following the order constraint.
3. Independent coherence judge (or human eval on a subset) rates output quality vs left-to-right baseline.
4. GOC = % of outputs rated ≥ baseline quality while respecting the order constraint (verified by checking generation trace).

**Why it matters:** Diffusion LMs generate in parallel/any-order by architecture; autoregressive models are locked to left-to-right. This is a capability gap, not just a benchmark gap.

---

## What these benchmarks are NOT

- They are not general intelligence tests. A model can ace all five and still be bad at math.
- They are not comparable to MMLU/HellaSwag. Different axis, different meaning.
- They do not prove safety, truthfulness, or absence of hallucination. They prove *structural compliance* — the output has the shape the contract demanded. A compliant output can still be factually wrong.

## Frontier-model baseline

For the record, when we run these: we expect frontier models to score near-zero on CCR (no proof mechanism), zero on IR (no intervention mechanism), zero on PCOV (no certificates), zero on LSF (no span locking), and N/A on GOC (architecturally impossible for AR models). This isn't a dunk — it's the point. We're measuring something they weren't built to do.

The honest claim is: "On guarantee-native benchmarks, XIIV-1B does what frontier models cannot." Not: "XIIV-1B is smarter than GPT."

---

## Build order

1. Implement the independent verifier (separate module, no model deps).
2. Implement Benchmarks 1 and 4 (closest to existing tested capabilities).
3. Implement Benchmarks 2 and 3 (need intervention logging + certificate format).
4. Implement Benchmark 5 (needs generation-order tracing).
5. Run all five on the 133M baseline (expect low scores — it's story-trained, not contract-trained).
6. Re-run on 500M after real-data training, then on 1B.
7. Publish the benchmark code open-source so anyone can verify or replicate.

---

## Implementation status (2026-10-11)

All five benchmarks implemented in `bench/guarantee_bench.py`:
1. CCR — Contract Compliance Rate ✓
2. IR — Intervention Rate ✓ (uses `DenoisingState.interventions` in `xiiv/diffusion.py`; logs `forbidden_prevented` and `endgame_forced_placement`)
3. PCOV — Proof-Carrying Output Verification ✓ (with tamper control)
4. LSF — Locked-Span Fidelity ✓
5. GOC — Generation-Order Control ✓

Validation run 2026-10-11 on 133M (benchmark 2 only, n=12): mean 0.25 interventions/gen (3 forced placements, 0 forbidden attempts). Full suite not yet run on any checkpoint — pending 5090 GPU (CPU too slow).

Validation runs 2026-10-11 (133M checkpoint, n=10-12, CPU):
- Bench 1 (CCR): 0.0–0.1 (expected — unconstrained generation, story-trained model rarely emits required phrases naturally)
- Bench 2 (IR): mean 0.25 interventions/gen
- Bench 3 (PCOV): verified 0/10, tamper_rejected 100/100 — verifier correctly rejects ALL tampered outputs
- Bench 4 (LSF): 1.0 — locked spans preserved byte-exact in all 10 runs
- Bench 5 (GOC): 1.0 across all 4 generation orders

Bugs fixed during validation: tuple-unpacking in bench_ccr/bench_lsf/bench_pcov/bench_goc (sample_trajectory returns (ids, conf)); missing brick.allocate() before certify().
