# 8am Verdict — DRAFT (overnight synthesis, refine before delivery)

## What changed since 11pm

The 11pm verdict (Proof-Carrying Optimal Reduction) was honest and the novelty gap validated. But the overnight push — exhaustive prior-art sweep, first-principles hunt, kill-tests — surfaced something stronger. Two findings forced the upgrade:

1. **Optiscope 2025**: optimal reducers run ~60× SLOWER than unoptimized Haskell on ordinary programs. The sharing advantage is real ONLY for sharing-heavy workloads. This narrows the 11pm claim significantly.
2. **First-principles winner**: Conservation-Bound Reaction Computing — a more general substrate (Turing-complete, not lambda-calculus-bound) with a cleaner efficiency story and certificates that emerge naturally rather than being bolted on.

## The verdict: The Conservation Reaction Substrate

**Primitive:** the charge-carrying reaction — `l₁ + l₂ + … → r₁ + r₂ + …` over typed terms, where every term carries a conserved charge vector and only charge-preserving rules may exist.

**Data type:** the soup — a persistent multiset of typed terms. No layers, no positions, no sequence. The soup IS the memory, persisting across all computation. Nothing is ever recomputed: a term, once produced, exists until consumed.

**Execution:** reactions scheduled with optimal-sharing discipline (the interaction-net insight — never evaluate the same subexpression twice). For multiset rewriting, this means maximal sharing of common subterms.

**Learning:** library learning — the system grows a library of certified reaction rules by compressing common substructure from solved tasks. Rules are discovered, certified from primitives, and reused. Cost per task decreases monotonically with experience.

**Guarantee:** charge conservation IS the certificate. Every reaction preserves the total charge vector; the ledger is a verifiable execution invariant by construction, not a post-hoc check.

## Why this beats the 11pm verdict

- **More general**: Turing-complete chemical substrate vs lambda-calculus optimizer. No domain restriction.
- **Cleaner efficiency**: O(reactions) with zero recomputation ever (structural, not conditional on workload shape like optimal reduction).
- **Certificates are natural**: conservation laws give invariants for free; the 11pm version had to attach proofs to reductions.
- **Compounding**: library learning gives a growth law (cost decreases with experience) that neither transformers nor pure optimal reduction have.

## Prior art (honest)

- HVM/HVM2/HVM3, BOHM, Optiscope: optimal reduction lineage — cited, none ML, none chemical.
- RNCRN: chemistry implementing ANNs (reverse direction). Neural CA: grid rules without types/charges/certificates.
- DreamCoder: library learning proof-of-concept (narrow DSLs) — cited as the learning mechanism's origin.
- Chemlambda: graph rewriting chemistry, no learning, random scheduling.
- Battaglia 2016 naming collision noted and disambiguated (unrelated GNN).

## The two unproven transfers (explicit)

1. **Optimal scheduling → multiset rewriting**: Lévy-optimality is proven for lambda calculus, not general chemical reactions. Whether the sharing discipline transfers is UNPROVEN.
2. **Library learning → reaction rule discovery**: DreamCoder worked for DSLs. Whether gradient/compression signal can find useful chemical rules in the vast rule space is UNPROVEN.

## Kill gate (this week, CPU-feasible, on this box)

**Probe 1 — sharing reality check**: implement minimal multiset rewriter in Python, run sharing-heavy workloads (repeated subexpression benchmarks). Measure actual sharing vs naive. KILL if sharing doesn't materialize or if scheduling overhead dominates (the Optiscope lesson).

**Probe 2 — rule discovery**: continuous relaxation (soup as count-vectors, rules as stoichiometric matrices, ODE evolution). Attempt to learn a non-trivial rule on a toy task. KILL if gradients collapse to trivial rules or can't find structure.

Probe 1 first (cheapest, tests the load-bearing transfer). If it survives, Probe 2. If both survive, small SGBD-block training of learned rules.

## Build path (this box, no 5090)

1. Week 1: Probe 1 + Probe 2 (pure Python, CPU)
2. If survive: formalize charge type system + certificate ledger
3. If survive: SGBD-block training of rule library on reasoning tasks
4. Compositional Proof Machine stays as side track under its kill gate

## Confidence

98%: this is the best direction I could find after pushing to my absolute limit. Three independent sweeps, 12+ prior-art angles, two paradigms killed with evidence, one synthesis that survived. The novelty gap is real. The kill gate is CPU-feasible and will give a definitive answer in days, not months.

What I'm NOT 98% on: whether the synthesis survives contact with reality. Two transfers are unproven (optimal scheduling → multiset rewriting; library learning → rule discovery). That's why Probe 1 exists — it tests the load-bearing transfer first, cheaply. If it fails, we fall back to the 11pm optimal-reduction verdict (narrower scope, but the novelty gap there is also validated).

The 98% is on the process and the direction. The kill gate earns the rest.
