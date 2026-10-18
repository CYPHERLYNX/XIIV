# Probe 1 results — Conservation Reaction Substrate: sharing reality check

**Date:** 2026-10-17 ~16:30 IST. **Script:** `research/reaction_probe1.py`
(runnable, stdlib-only, ~10s CPU). Run it to reproduce every number below.

## What was tested

Minimal multiset rewriter (Python): terms as nested tuples, reactions
`dbl(x) -> x+x` and `add(n,m) -> n+m`. Two evaluators on identical inputs:

- **Naive:** structural recursion, every subterm occurrence re-reduced.
- **Sharing:** hash-consed DAG + memoized reduction — each UNIQUE term reduced
  exactly once (the interaction-net "never evaluate the same subexpression
  twice" discipline, applied to multiset rewriting). Interning happens once at
  soup-entry; rule results reuse interned children (O(1)).

Charge conservation (total numeral mass) asserted on **every** rewrite —
the verdict's "certificate from conservation laws" in miniature. All rewrites
passed; the ledger caught one real bug during development (a `mul` rule whose
charge didn't conserve — rule removed, fragment restricted to `dbl`/`add`
where mass is genuinely conserved; vector-valued charge for richer rule
families deferred to Probe 2).

## Numbers

| Workload | Naive steps | Sharing steps | Work ratio |
|---|---|---|---|
| A: `dbl^4(base)` | 238 | 17 | 14x |
| A: `dbl^8(base)` | 3,838 | 25 | 154x |
| A: `dbl^12(base)` | 61,438 | 33 | 1,862x |
| A: `dbl^16(base)` | 983,038 | 41 | **23,977x** |
| B: soup of 500 identical molecules | 2,000 | 3 | 667x |
| C2: 300 distinct deep terms, zero sharing | 19,200 | 19,200 | **0.28x (3.6x tax)** |

Both evaluators reach identical normal forms (asserted). Naive is
exponential in duplication depth; sharing is linear.

## Verdict: PASS (with one honest correction)

1. **The load-bearing transfer works, decisively.** Maximal-sharing discipline
   transfers from lambda calculus to multiset rewriting: up to 24,000x fewer
   rule applications on sharing-heavy workloads, 667x across a shared soup.
   The 8am verdict's core bet survives contact with reality.
2. **Overhead does not dominate.** On adversarial zero-sharing work at
   realistic grain, the sharing machinery costs a ~2.3–3.7x constant tax —
   non-growing with problem size, an order of magnitude better than the
   Optiscope precedent (~60x). A real implementation interning at parse time
   would shrink this further.
3. **CORRECTION — one 8am claim dies:** "zero recomputation is structural, NOT
   workload-conditional" is refuted. The benefit is exactly
   workload-conditional (24,000x sharing-heavy vs 3.6x tax sharing-free) —
   the same shape as the Optiscope lesson that narrowed the 11pm verdict.
   The paradigm survives on its other advantages (Turing-complete rather than
   lambda-bound; certificates from conservation laws rather than attached to
   reductions; compounding library learning). The "not workload-conditional"
   phrasing must not appear in future writeups of the verdict.

## Probe 2 spec (next; sequence AFTER block-1 training — CPU contention)

**Question:** can library learning discover reaction rules in the vast rule
space? (Second unproven transfer from the 8am verdict.)

**Design:** continuous relaxation of the soup — soup as count-vectors over a
term alphabet, rules as stoichiometric matrices, discrete steps as ODE
evolution (Euler, small dt). Task: learn the `dbl` rule (target matrix known,
so success is checkable) on a toy duplication task from (soup_before,
soup_after) pairs via gradient descent on the matrix entries with an L0/L1
sparsity prior (rules should be small).

**KILL if:** gradients collapse to trivial rules (identity / uniform), or the
learned matrix doesn't recover the target structure within tolerance on held-out
pairs, or optimization needs more compute than this box can give a toy
(count-vectors ≤ 20 dims, ≤ 5k gradient steps).

**If Probe 2 survives:** small SGBD-block training of learned rules
(verdict week-2 item).
