# Noisy Inputs — §5.6 Task 3 Results (2026-10-23)

**Status: VALIDITY GATE PASS; two precision hypotheses FALSIFIED (documented below)**

Spec §5.6 task 3: input intervals instead of points — tests
abstract-interpretation precision under uncertainty, and was queued as the
step that "makes the tightness loss a real signal."

## What was built

- **`intervals.py`** — integer-interval abstract domain + fixed interval
  transformers for the 8 primitives + join (hull) + guard-interval decision
  (`then` / `else` / `straddle`).
- **`noise.py`** — `NoiseProofMachine`: neural nets execute on the nominal
  point x0 while the interval trace covers the true input set
  `[x0-r, x0+r]`; per-step neural↔interval agreement halts LOUD on
  divergence (VRC doctrine). IFZ under noise: decisive guards take one
  branch; straddling guards (0 ∈ G, G ≠ {0}) execute BOTH branches on the
  pre-guard interval and JOIN. Brute-force true-output-set oracles for
  validity checking.
- **`certificates.py`** — independent interval verifier: re-derives every
  step from its own rule table, re-decides every branch from the guard
  interval, re-derives joins (checks I1–I4 / J1–J5). Linear in |trace|.
- **`taskgen_noise.py`** — same distribution discipline as the point
  generators, plus noise-validity: every x in the input interval must keep
  every intermediate in [-1e6, 1e6] (rejected otherwise, not clipped).
- **`test_noise.py`** — 525 tests green (interval-transformer exactness vs
  brute force over 480 random intervals × 8 ops, MOD wrap corners, join
  soundness, 5 adversarial verifier rejections incl. fake-straddle/wrong-join,
  fault-injection divergence halt, nested noisy programs).
- **`killgate_noise.py`** — pre-registered sweep (gates written BEFORE
  running): depths {1,2,4,8,16,32,64,100} × radii {0,1,3,5}, 200 flat +
  120 conditional instances per cell (10,240 runs, 22.8s CPU).

## Validity gate: PASS

V(L, r) = **100.00% in all 32 cells**, flat and conditional, zero neural
divergences. Three sub-checks per instance: (a) independent verifier
accepts, (b) brute-force true output set ⊆ final interval, (c) neural
nominal ∈ final interval. **Soundness of the interval certificate holds at
every depth and every noise radius.** No kill triggers.

## Precision: two falsifications (the honest news)

**F1 — flat chains are NOT exact (my exactness claim was wrong).**
Pre-registered hypothesis P1 said the interval trace equals the true
reachable set on straight-line chains. Measured `exactW` (fraction of
instances where final width == true-set width): 1.000 at r=0, but 0.075 at
(L=100, r=1). Root cause, localized by the per-op tightness profile:
ADD/MUL/NEG looseness = **0.0000** (center-exact, as designed), MOD_7 =
0.265, MOD_13 = 0.180. The MOD wrap case returns the hull [0, m-1] while the
true image {0..b}∪{a..m-1} has gaps — a strict over-approximation I had
mischaracterized as "exact as an interval." The module docstring is
corrected (sound + endpoint-tight hull, exact only no-wrap / for affine
ops). Mitigating fact: flat-chain width inflation stays modest
(1.0–2.0× at all depths/radii) — sound AND usefully precise, just not exact.

**F2 — join precision loss COMPOUNDS across straddles.**
Pre-registered hypothesis P3 said join over-approximation stays localized.
Measured mean over-approx (hull width − true width): L16_r3: 186 (1
straddle) vs **9403** (≥2 straddles); L32_r5: 0 vs 1,194,288. The mechanism
is F1's: each MOD wrap plants hull gaps, subsequent MUL chains scale them,
and each join re-hulls. Joins compound through arithmetic — documented, not
hand-waved.

## Guard decisiveness evaporates under noise

Straddle fraction (conditional suite): 0.00 at r=0 for all L; 0.275 at
(L=4, r=1); **1.00 at (L≥16, r≥1)**. Once the guard input interval is wider
than the modulus, a MOD guard always straddles 0 → the machine executes
both branches and hulls, every time. Under noise, IFZERO-with-MOD-guards
degrades to "run both, take the hull": certificates stay valid but branch
precision collapses. This is a property of the guard family (narrow MOD
guards + wide input intervals), stated as such — not a claim about
branching in general. Guards that keep intervals narrow (affine chains,
decisive thresholds) are the Phase-2 design direction this points at.

## The tightness loss is now a real signal (as queued)

With point inputs L_tight was vacuous by construction (documented in
`primitives.py`). Under noise, per-op looseness
`|center(out) − op(center(in))| / width(out)` separates the ops cleanly:
affine ops 0.0000, MOD_7 0.265, MOD_13 0.180. The signal measures
**abstraction looseness, not network error** — no amount of training fixes
the MOD wrap hull; the fix is program-level (keep noisy values out of MOD,
or a relational domain). That distinction is the Phase-2 training
objective this task was queued to produce: train/choose programs whose
guard intervals stay decisive under the expected noise radius.

## Files

- `intervals.py`, `noise.py`, `taskgen_noise.py`, `test_noise.py`,
  `killgate_noise.py` (new); `certificates.py` += interval verifier
- `results/killgate_noise_20261023_012645.json`

## Next

- **CGAU** (today's Part-A pick, `research/daily_2026-10-23.md`):
  Certificate-Gated Adaptive Unmasking — proof-certificate-gated sampler
  for the masked-diffusion LM; queued invention candidate with stated
  falsifiers. Do NOT build uncritically from the sweep sketch; the
  inventive act stays with the main agent per the invention covenant.
- Proof-machine line: decisive-guard program synthesis (VRC generating
  programs whose guard intervals avoid straddle under noise); relational
  (non-hull) MOD abstraction as a precision project.
