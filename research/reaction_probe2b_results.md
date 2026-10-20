# Probe 2b results — Conservation Reaction Substrate: discrete rule discovery

**Date:** 2026-10-18 ~19:40 +04. **Script:** `research/reaction_probe2b.py`
(stdlib only, deterministic seed 20261018, ~0.1s CPU; run with `nice -n 10`).
Run it to reproduce every number below.

## What was tested

Redirect after Probe 2 KILLED gradient-based rule discovery (gain explosion
into a chaotic attractor; NaN divergence learning mass-action exponents).
This probe tests the honest redirect: DreamCoder-style **discrete
enumerative search + compression** — no gradients on rule matrices, no ODE
rollout, so both Probe-2 failure modes are sidestepped by construction.

Prior art (honest): DreamCoder (Ellis et al.) discovers library abstractions
via enumerative search over a DSL + compression. This probe transfers that
METHOD to reaction-rule discovery; no novelty claimed over DreamCoder.

Task: discover the catalyzed doubling rule `a + c -> 2a + c` (c = catalyst,
required but unconsumed) from 64 train / 32 held-out (soup_before,
soup_after) pairs, with inert distractors d1..d4 varying independently per
pair (same distractor discipline as Probe 2). Target rule known, so success
is checkable. Selection on train only (pre-declared tie-break: fewest atoms,
then lexicographic); the selected rule is evaluated ONCE on held-out.

SPEC CORRECTION: the probe spec said "at most 3 products", but its own
tripling example (`a + c -> 3a + c`) needs 4 product terms — internally
inconsistent. Grammar used uniformly: |reactants| <= 2, |products| <= 5.

## Numbers

| stage | candidates evaluated | train survivors | rule found (held-out exact) |
|---|---|---|---|
| raw enumeration | 12,936 | — | — |
| charge-pruned | 128 (101x pruning) | — | — |
| T1: c-catalyzed doubling | 128 | **1** | `a+c -> a+a+c` ✓ |
| T2: d1-catalyzed doubling | 128 | **1** | `a+d1 -> a+a+d1` ✓ |
| T3: c-catalyzed tripling | 128 | **1** | `a+c -> a+a+a+c` ✓ |
| compression S1 (seq anti-unification of T1,T2) | — | — | `(a,X0) -> (a,a,X0)` ✓ asserted |
| T4: d2-doubling **via S1** | **6** (vs 128 full) | **1** | `a+d2 -> a+a+d2` ✓ |
| compression S2 (numeric anti-unification of T1,T3) | — | — | `R={a:1,c:1}, P={a:N0,c:1}` ✓ asserted |
| T5: c-quadrupling **via S2** | **5** (vs 128 full) | **1** | `a+c -> a+a+a+a+c` ✓ |

Search-cost reduction from library reuse (the compounding step):
- T4: 6 schema instantiations vs 128 full-space = **21.3x reduction**
- T5: 5 schema instantiations vs 128 full-space = **25.6x reduction**

Distractor discipline audit (T1 reactive pairs): d1 0.46, d2 0.59, d3 0.52,
d4 0.28 — every distractor both present and absent, so latcher rules
(e.g. charge-conserving `a+d1 -> 2a+d1`) are killed behaviorally on pairs
lacking the distractor. Wall time 0.1s; single run, no tuning.

## Verdict: PASS

1. **Discrete discovery works where gradients structurally failed.** Exact
   rule recovery on held-out pairs for three task families, unique train
   survivors in all cases (no tie-break ambiguity, no selection on held-out).
2. **Compression finds the true common substructure, nothing spurious.**
   S1 = catalyst-general doubling `(a,X0)->(a,a,X0)` via Plotkin-style
   sequence anti-unification (asserted exact); S2 = c-catalyzed N-doubling
   via count-based numeric anti-unification (asserted exact). Both schemas
   regenerate their source rules (checked). No spurious substructure found.
3. **Reuse is measured, not theater.** Schema-guided search evaluates 5–6
   candidates instead of 128, with exact behavioral verification still
   applied (a wrong schema would fail loudly, not silently pass). This is
   the compounding mechanism — cost per task drops as the library grows —
   demonstrated at toy scale.

## What this does and does not imply — read carefully

1. **Established:** a discrete-search discovery engine exists for reaction
   rules at toy scale. The 8am verdict's learning half ("the system grows a
   library of certified reaction rules... cost per task decreases
   monotonically") now has a demonstrated mechanism — in the WEAKER,
   DreamCoder-standard discrete form, not the killed continuous form. The
   verdict stands on both legs again, with the learning leg explicitly
   scoped to discrete search.
2. **NOT established:** scalability. Alphabet 6, tiny grammar, 0.1s. Real
   rule spaces (larger alphabets, bigger bounds, noisy data) will need
   smarter search than brute-force enumeration; this probe is an existence
   proof, not a scaling claim. Any writeup must carry that scope.
3. **Charge-certificate accounting (honest):** q(a) = 0 is FORCED by the
   target rule form (q(a)+q(c) = 2q(a)+q(c) ⟹ q(a) = 0). The certificate
   therefore constrains catalyst/distractor conservation — where it does
   real work, pruning 12,936 → 128 (101x), killing every rule that
   creates/destroys c or d_i — but it cannot constrain the doubling
   structure itself; enumeration + behavioral data carry that part. The
   "charge conservation IS the certificate" claim survives, scoped: it
   certifies what flows are legal, not what the doubling factor is.

## Process notes

- Single run, fixed seed, no hyperparameter or design tuning after seeing
  results. All schema assertions were written before running.
- The only deviation from the probe spec is the |products| <= 5 bound
  (spec's own tripling example required 4; documented above).
- Not committed to git per probe instructions.
