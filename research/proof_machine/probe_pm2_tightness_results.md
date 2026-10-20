# Probe PM-2 Results: Tightness Loss Under Inexact Abstraction

**Date:** 2026-10-18
**Script:** `probe_pm2_tightness.py` (this dir)
**Verdict: KILL** — on the tightness-loss mechanism (spec §4.3) as specified and implemented.
The Phase-1 prototype PASS (exact domains) is untouched; what dies is the path from
toy exact domains to learned behavior under inexact abstraction.

---

## The kill gate (pre-registered before running)

The neural bridge's load-bearing untested mechanism: `L = L_task + λ·L_tight` is
supposed to train networks that are BOTH accurate AND tightly abstracted by a
FIXED inexact abstract transformer. Phase 1 never tested this — the README
documents L_tight as "vacuous for the exact domain — by construction" and notes
"the real tightness-loss research belongs to Phase 2+ with inexact domains."

Setup: f(x) = |x| on [−2,2]; fixed interval transformer |·|#([a,b]) = [0, max(|a|,|b|)]
(the standard, best-possible sound interval transformer for abs — not a strawman);
1→32→32→1 tanh MLP, Adam, 4000 steps, torch (project's existing dependency).

**KILL criterion:** no λ ∈ {0, 0.01, 0.1, 1.0} achieves BOTH test MSE < 1e-4
AND mean relative certificate width < 50%.
**PASS criterion:** some λ achieves both.

## Pre-registered predictions vs observations

**P1 — θ-independence (structural).** Certificates are `M#(α(x))`, a fixed function
of the input. They do not depend on network weights θ. Therefore no training loss
can tighten certificates; L_tight can only bias concrete outputs toward abstract
centers.

| λ | mean relative cert width |
|---|---|
| 0.00 | 100.0% |
| 0.01 | 100.0% |
| 0.10 | 100.0% |
| 1.00 | 100.0% |

Confirmed exactly. Training is provably incapable of tightening certificates under
a fixed transformer — the width column cannot move by construction, and the run
demonstrates it.

**P2 — bias conflict (structural + empirical).** For inexact domains the abstract
center (here |x|/2) ≠ the true value (|x|), so L_tight pulls the network toward the
WRONG function. Derived pointwise equilibrium:
M*(x) = |x|/2 for |x|<√λ; |x| − λ/(2|x|) for |x|≥√λ (bias diverges as |x|→0).

| λ | test MSE | max |err| | corr(learned, M*) |
|---|---|---|---|
| 0.00 | 8.47e-05 | 0.0482 | +0.9999 |
| 0.01 | 3.28e-04 | 0.0541 | +0.9999 |
| 0.10 | 3.94e-03 | 0.1017 | +0.9994 |
| 1.00 | 1.13e-01 | 0.5177 | +0.9891 |

Task accuracy degrades monotonically with λ: 3.9× worse at λ=0.01, 46× at λ=0.1,
**1334× at λ=1.0**. At λ=1 the network learns roughly HALF the target near the kink —
f(0.5)=0.276 (M* predicts 0.25, target is 0.5), f(1.0)=0.671 (M* 0.5, target 1.0) —
correlation +0.989 with the predicted wrong function. The tightness loss does not
merely fail to help; it actively corrupts the network, worst exactly where precision
matters most (near the kink, where the abstraction is loosest).

**KILL criterion: TRIGGERED.** At λ=0: MSE 8.47e-05 < 1e-4 ✓ but width 100% ✗.
At every λ>0: width 100% ✗ and accuracy strictly worse ✗. No λ achieves both.
There is no operating point where the mechanism does its advertised job.

## What exactly died, and what didn't

**Died:** mechanism (b)'s tightness loss as specified (§4.3) and implemented
(`tightness_loss_affine`). It is vacuous for exact domains (documented) and
counterproductive for inexact domains (shown here). The two failure modes cover
the entire space: there is no regime where it tightens certificates while
preserving accuracy, because (a) certificates are θ-independent under a fixed
transformer, so nothing can tighten them, and (b) the only θ-dependent effect
available — pulling outputs toward abstract centers — is harmful whenever the
abstraction is inexact.

**Did NOT die:** the Phase-1 prototype and its PASS verdict. Exact symbolic
domains, certified composition, the VRC harness — all untouched. The proof machine
remains a valid certified-composition substrate *for functions the designer can
abstract exactly*.

**The honest implication (scope-kill on the ASI claim):** the charitable retreat —
"restrict to exact domains only" — keeps the prototype alive but caps expressivity
at designer-understood functions. A network in this architecture is then a compiler
for functions the designer could already write an exact abstract transformer for.
That cannot learn genuinely novel behavior, which the ASI ambition requires. The
side track survives as a certified-composition substrate; it does not survive as a
path to ASI unless a replacement coupling is found.

## Honest caveats

1. Single (function, transformer) pair: |x| + canonical interval abs-transformer.
   P1 (θ-independence) is fully general for fixed transformers; P2's conflict needs
   only center ≠ target, true of any inexact transformer worth the name. A second
   pair would strengthen but not change the structural argument.
2. Single seed (identical init/data across λ — the comparison is fair; effect sizes
   are 46–1334×, far beyond seed noise).
3. The M*-shape match is approximate (optimization ≠ exact pointwise minimization),
   but the direction, monotonicity, and kink-region collapse all match prediction.
4. Runtime 7m37s CPU (contention with SGBD block 2 training); RAM negligible.

## Candidate replacements (untested, future work)

- **θ-dependent / differentiable abstraction** — spec's rejected option (c).
  Would break θ-independence, but "no existing system demonstrates differentiable
  proof construction at the reliability we need." The rejection stands until
  someone builds it.
- **Architecture–domain co-design** — choose (network family, abstract domain)
  pairs with exact transformers (the MOD Fourier-feature trick generalized).
  Honest cost: expressivity ceiling at designer-understood functions (scope-kill above).
- **Weaker coupling** — accept loose certificates and make interface checks
  robust to them (e.g., three-valued logic: verified / unknown / violated).
  Changes the guarantee from "every answer carries its proof" to "every answer
  carries its proof *or an explicit unknown*" — a thesis-level decision, not a
  probe-level fix.

## Bottom line

The neural bridge has no working coupling mechanism between learning and
certification beyond exact domains. The proof machine is a sound, tested,
certified-composition substrate for hand-abstractable functions — and currently
nothing more. Any further investment in this side track should target the coupling
problem first; scaling primitives, policies, or preconditions before that is
building on a missing foundation.
