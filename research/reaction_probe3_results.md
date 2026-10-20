# Probe 3 results — Conservation Reaction Substrate: the SCALE kill gate

**Date:** 2026-10-18 ~21:05 +04. **Script:** `research/reaction_probe3.py`
(stdlib only, deterministic seed 20261018 + offsets, ~22s CPU total;
run with `nice -n 10` — box shared with SGBD block-2 training).
Run it to reproduce every number below. Not committed to git per probe
instructions.

## Verdict: KILL — per the pre-declared criterion, with analysis below

The kill criterion (decided before running, honored exactly as written):
*"if at alphabet ≤14 the search exceeds 1M raw candidates or 10 minutes wall
time without finding the exact rule, verdict = KILL."*

At n=14, raw = **1,395,360 > 1,000,000** → the criterion fires. **Verdict:
KILL.** No goalpost-moving: the number is what it is.

**Read the next section before acting on this verdict.** The kill fires on a
count that measures the *naive implementation's* enumeration strategy, while
every substantive scaling measurement in this probe is favorable. Both facts
are reported; the parent decides whether the kill generalizes.

## What the kill does and does not mean

**What died:** the naive enumerate-then-filter implementation (S1, faithful
to 2b: materialize the full |R|≤2 × |P|≤5 Cartesian product, then filter by
charge). Its materialized candidate count scales as ~n⁷ and crosses 1M at
n=14 (12.3M at n=20). As an implementation strategy it does not survive
scale-up. Killed cleanly, as specified.

**What did NOT die:** the method — discrete search + charge-conservation
pruning. Measured, not asserted:

| n | raw (exact) | pruned (evaluated) | prune factor | exact recovery |
|---|---|---|---|---|
| 6 | 12,936 | 128 | 101× | ✓ unique survivor |
| 10 | 198,198 | 288 | 688× | ✓ unique survivor |
| 14 | 1,395,360 | 512 | 2,725× | ✓ unique survivor |
| 20 | 12,273,030 | 968 | 12,679× | ✓ unique survivor |

- The **evaluated** candidate set (what the search actually costs) scales
  **quadratically** — closed form `pruned(n) = Σⱼ₌₀² C(n+j−2,j)·(3−j)·(6−j)`
  ~ 2n², asserted against brute force at every n. Discovery wall time at
  n=20: **0.01s** for 968 candidates × 64 train pairs.
- The prune factor **improves** with scale (101× → 12,679×): each added
  alphabet type adds a charge dimension that candidate rules must respect,
  so charge conservation gets *more* discriminative, not less.
- S3 (charge-guided direct enumeration — group P-multisets by charge, join
  each R only with charge-matching P's; standard database-join technique,
  no new machinery) **never touches the raw product**: 0.37s at n=20 vs
  9–12s for streaming filter (S2), and S1 was already skipped at n=14 by the
  RAM guard. The n⁷ term is an implementation artifact, removable with
  textbook technique — implemented and timed in this probe.

**Bottom line for the 8am verdict's learning leg:** discrete search +
charge pruning survives scale-up to n=20 with quadratic effective cost and
exact recovery throughout. The KILL applies to the naive enumerator per the
letter of the criterion. If the gate is re-run against a charge-guided
enumerator (S3), it passes. Whether to accept that redirect is the parent's
call — the numbers for both readings are above.

**Second scale limit found (real, not a kill of the approach):** flat JOINT
search over k-rule systems scales as pruned^k. For 3 rules at n=6:
pruned=1,100 → C(1100,3) = **221,228,700** triples, extrapolated to hours–
days (measured sample timing; infeasible as flat search). Compositional
discovery recovers the triple exactly (below). Multi-rule scaling needs
decomposed search, not flat joint enumeration — a finding, with a working
alternative demonstrated.

## Numbers

### Exp 1 — alphabet scale (2b T1 task: c-catalyzed doubling)

n=6 replicates 2b exactly (12,936 / 128 asserted). All n: unique train
survivor, exact rule `a+c -> a+a+c` on 32 held-out pairs. Timings vary with
box load (block-2 training shares the box); S1 at n=14 skipped by the
2.0GB RAM guard — its count is exact combinatorics, no materialization
needed for the kill clause.

Closed forms (1 neutral substrate, n−1 distinct unit charges):
- `raw(n) = M(n,≤2)·M(n,≤5)`, `M(n,≤k) = ΣⱼC(n+j−1,j)` → ~n⁷
- `pruned(n) = Σⱼ₌₀² C(n+j−2,j)·(3−j)·(6−j)` → ~2n²

### Exp 2a — two-rule system (JOINT search)

Target `{a+c→2a+c, b+c→2b+c}`, shared catalyst, single-pass firing
semantics (each rule fires at most once, canonical order — deterministic).
pruned=406 (2 neutral substrates) → C(406,2) = **82,215** pairs searched in
~2–7s. **Unique survivor = the exact true pair**, held-out exact (48 pairs).
Partial-reactive pairs (single substrate + catalyst) kill merged single
rules like `a+b→2a+2b`; substrate-without-catalyst negatives kill
catalyst-free rules. All four negative kinds verified load-bearing (see
data-discipline fixes).

### Exp 2b — two-catalyst rule

`a+c1+c2 → 2a+c1+c2` under |R|≤3 grammar. raw=38,808 → pruned=324
(**120×**). Unique survivor, exact on held-out.

### Exp 2c — three-rule system

Target `{a+c→2a+c, b+c→2b+c, s+c→2s+c}`.
- Flat joint search: 221M triples → **infeasible** (measured, extrapolated
  hours–days). This is the pruned^k wall, stated plainly.
- Compositional discovery: **unsupervised** change-set clustering (types
  whose counts changed; distractors/catalyst conserved by construction, so
  no generator labels used) → clusters `{a}:11, {b}:10, {s}:7, {a,b,s}:42,
  {}:26` → per-substrate single-rule discovery (train only, all
  empty-change pairs as negatives) → 3 exact rules → triple **jointly exact
  on all 96 train and 48 held-out pairs** under single-pass semantics.

### Exp 3 — prune-ratio curve

101× → 688× → 2,725× → 12,679× at n = 6, 10, 14, 20. Monotone increasing;
charge pruning strengthens with alphabet size (each new type = a new charge
dimension constraining the rule space).

## Semantics finding (during this probe, honest)

Naive FIXPOINT firing ("repeat rules until no change") **diverges** on
autocatalytic rules: `a+c→2a+c`'s product still contains its reactants, so
it re-fires to the iteration cap. This is not a probe bug — it is *why* the
substrate needs an explicit scheduling discipline, i.e. exactly Probe 1's
subject (optimal-sharing scheduling of reactions). The multi-rule
experiments use single-pass semantics (each rule fires at most once,
canonical order); the fixpoint observation is recorded as a constraint on
the execution layer, connecting Probes 1–3.

## Data-discipline fixes (transparent, method untouched)

Two experiment-design bugs found by the first run, fixed, re-run. The
discovery algorithm (enumerate + behavioral filter + pre-declared tie-break)
was not modified.

1. **Missing "substrates without catalyst" negatives** (exp2a/2c): the first
   `gen_pairs_multi` never generated substrate-present/catalyst-absent
   pairs, so catalyst-free rules (`a→2a`, 3 atoms) were behaviorally
   indistinguishable from catalyzed ones and won the atom-count tie-break
   (exp2a first run: 4 survivors, wrong selection; exp2c first run:
   degenerate `0→a` rules). Fixed by adding both negative kinds.
   Before: exp2a FAIL (surv=4, selected `a→2a+b→2b`).
   After: exp2a OK (surv=1, exact true pair).
2. **Per-substrate negatives filtered too narrowly** (exp2c): negatives were
   restricted to substrate-present pairs, omitting catalyst-without-
   substrate cases — so `c→a+c` (behaviorally identical to `a+c→2a+c` on
   reactive-only data, fewer atoms) won the tie-break.
   Before: compositional FAIL (selected `c→a+c` etc.).
   After: OK (3 exact rules, joint train+test exact).

Both are the same lesson as 2b's distractor discipline: the negative pairs
must cover every degenerate the tie-break would otherwise prefer. Exp 1
(the kill-gate experiment) was unaffected — it uses 2b's `gen_pairs`, and
all its rows replicate across runs.

## Prior art (honest, mandatory)

- **DreamCoder** (Ellis et al.): enumerative search over a DSL + library
  compression via anti-unification. This probe transfers that method to
  reaction-rule discovery and stress-tests its scaling; **no novelty claimed
  over DreamCoder** — the contribution, if any, is the charge-conservation
  pruning analysis (prune factor *increasing* with alphabet size), which has
  no DreamCoder analogue.
- **HVM/HVM2/HVM3, BOHM, Optiscope**: optimal-reduction lineage, cited;
  none are ML, none do rule discovery.
- **Chemlambda**: graph-rewriting chemistry, no learning component.
- **Battaglia et al. 2016 "Interaction Networks"**: an unrelated famous GNN
  — naming collision, disambiguated; not cited as prior art for this work.
- Correct claim: **"clean for ML-native use; substrate has mature non-ML
  lineage,"** never "zero prior art."

## Honest caveats — read before citing this probe

1. The KILL is per the pre-declared raw-count criterion; the substantive
   scale result (quadratic effective cost, exact recovery to n=20) is
   favorable. Both are above; do not cite one without the other.
2. Task complexity did NOT scale with n — same doubling task, more
   distractors. Harder rule families at large n are untested.
3. Charges are synthetic (distinct unit vectors per type). Real charge
   design for non-toy types is future work; the prune-factor growth depends
   on charges being discriminative.
4. 2b's compression schemas (S1/S2) were not re-tested at scale; library
   growth dynamics at n=20 are unmeasured.
5. Single run per experiment, fixed seeds (same standard as 2b).
6. S1 timing at n=14 skipped by RAM guard (box shared with training); the
   kill clause uses the exact combinatorial count, unaffected.
7. Timing numbers carry box-load variance (S2 at n=20: 9–12s across runs).
8. Multi-rule experiments assume independent substrates (disjoint); coupled
   rule systems (shared products, chains) are untested.
9. Per-substrate discovery in exp2c selects on train-cluster data with no
   per-substrate held-out; rigor comes from the joint held-out evaluation.

## Process notes

- Pre-declared kill criterion honored exactly as written; no post-hoc
  redefinition. The analysis section distinguishes what the criterion
  measured from what the method does — that distinction is the finding.
- Two data-generator fixes during the probe (documented above with
  before/after); discovery method untouched throughout.
- All assertions in the script passed (2b replication at n=6, closed-form
  counts vs brute force at every n, formula checks).
- Total wall 22s; stdlib only.
