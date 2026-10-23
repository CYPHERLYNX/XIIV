# Compositional Proof Machine — Phase-1 Kill-Gate Prototype

**Spec:** `../composition_operator_v1.md` (formalization, 2026-10-16).
**Built:** 2026-10-17 (daily model-improvement run). CPU-only, ~350K-param
transformer baseline + 8 small neural primitives. No GPU used.

## The bet under test

> Explicit certified composition maintains accuracy with composition depth,
> while implicit neural composition (transformer baseline) degrades.

The kill gate is pre-registered in `killgate.py` and honored literally:
PASS requires ALL of A_PM(100) ≥ 90%, a flat accuracy-vs-depth curve
(slope not significantly ≠ 0), and ≥ 40pp separation from the transformer at
depth 100. ANY of A_PM(100) < 70%, a significantly negative slope, or a
certificate failure KILLS the prototype iteration. A kill with a precise
diagnosis is a successful experiment.

## File map

| File | Role |
|---|---|
| `certificates.py` | Canonical symbolic expressions, fixed abstract-transformer rule table, JSON (de)serialization, the **independent verifier** |
| `primitives.py` | The 8 primitives as (M_i, M_i#) pairs: small torch MLPs + exact symbolic transformers; tightness-loss definition |
| `compose.py` | The ∘ operator: interface check, parallel concrete+symbolic execution, certificate construction |
| `taskgen.py` | Kill-gate task generator (depths × instances, rejection-sampled intermediates) |
| `train_primitives.py` | Supervised training of the 8 MLPs (L_task + λ·L_tight) |
| `baseline_transformer.py` | Decoder-only transformer baseline (~350K params, digit I/O), train + eval |
| `killgate.py` | Depth sweep {1,2,4,8,16,32,64,100} × 1000, pre-registered metrics |
| `checkpoints/` | Trained primitive weights + baseline checkpoint (git-ignored if large) |
| `results/` | Kill-gate result JSONs |

## Run

```bash
VENV=~/workspace/ai-tools/corsac/.venv/bin/python
$VENV train_primitives.py          # ~minutes, CPU
$VENV killgate.py --pm-only        # proof-machine depth sweep
$VENV baseline_transformer.py --train   # background; early-stops on val plateau
$VENV baseline_transformer.py --eval    # baseline depth sweep
```

## Documented design decisions (deviations from the spec are explicit)

1. **Intermediate bound [-1e6, 1e6].** Task instances are rejection-sampled so
   every intermediate stays in range. The claim under test is accuracy vs
   *depth*; unbounded intermediates would conflate it with numerical
   extrapolation (float64 exact only to 2^53). Both systems train/test on the
   same distribution. Survivor bias (fewer long MUL runs survive at depth 100)
   is reported with results.
2. **MOD primitives use Fourier input features** `[x/S, sin(2πx/m), cos(2πx/m)]`
   with a residue-class classifier. A vanilla ReLU MLP cannot place ~300k
   sharp periodic boundaries over [-1e6, 1e6]; this is a documented
   primitive-design choice (same features at train and test), not test gaming.
   Per-step exact-match is measured and reported honestly — if MOD fails to
   reach ~100%, the kill gate fires on the *neural bridge*, with the failure
   localized (certificates stay valid: V(L) = 100%).
3. **Tightness loss is vacuous for the exact domain — by construction.**
   With an exact symbolic transformer, center(M#(α(x))) is the exact value and
   width → 0; the term reduces to scaled MAE (λ = 0.01). Documented in
   `primitives.py` rather than hidden. The real tightness-loss research
   belongs to Phase 2+ with inexact domains.
4. **Baseline is ~350K params, ~5× the proof machine's neural total (~70K).**
   The spec's pre-registered worry is an *under*-parameterized baseline, so the
   baseline gets the parameter advantage. A win against a larger baseline is
   the stronger architectural signal.
5. **Phase-1 preconditions are all `true`.** The interface check is implemented
   and executed (loud, step-localized failure mode), but vacuously passes until
   Phase-2 primitives introduce nontrivial preconditions.

## Results

- **2026-10-17 — primitive per-step exact-match** (10k fresh samples each,
  inputs uniform in [-1e6, 1e6]):
  ADD_1/ADD_3/ADD_7/MUL_2/MUL_5/NEG: **100.00%** (linear models, exact least
  squares fit, max residual ~1e-9). MOD_7/MOD_13: **100.00%** (Fourier-feature
  MLP classifiers, Adam). Per-step error ~ 0: the depth sweep below isolates
  composition, not primitive learnability.
- **2026-10-17 — proof-machine depth sweep** (`results/killgate_20261017_013853.json`):
  A(L) = **100.00%** and V(L) = **100.00%** at every depth
  L in {1,2,4,8,16,32,64,100} (1000 fresh instances each, 8000 total).
  OLS slope b = 0.000000 (t = 0.00): perfectly flat curve.
  Certificate size grows linearly (steps = L), as the theory predicts.
  PM-only kill triggers: **none**.
- **2026-10-17 — transformer baseline** (`results/baseline_20261017_021923.json`):
  352,439 params, 10k steps / 36 min to loss plateau ~0.95, depths 1-16.
  A_TF: 57.8% → 26.0% → 10.9% → 7.9% → 10.1% → **1.3% → 1.1% → 1.5%**
  at L = 1,2,4,8,16,32,64,100. Note the degradation appears WITHIN the
  training distribution (58%→10% from L=1 to L=16): the compositionality-wall
  signature, not pure OOD failure.
- **VERDICT: PASS** (`results/verdict_2026-10-17.json`). All three
  pre-registered pass criteria hold: A_PM(100) = 100% ≥ 90%; slope b = 0
  (t = 0.00, flat); separation = **98.5pp** ≥ 40pp. Zero kill triggers.
  Caveats recorded in the verdict JSON: baseline is small and its in-dist
  accuracy modest — a larger/longer baseline might narrow the gap, but the
  within-distribution depth degradation is the architectural signal the gate
  was designed to detect. Synthetic arithmetic chains: a clean test of the
  composition-scaling claim, not a natural-language claim.
- **Next:** §5.6 robustness tasks (logic chains, conditional composition,
  noisy inputs) and Phase-2 composition policy — today's Part-A VRC proposal
  (SKILLFORGE-style verifier-routed harness over the ∘ interface check) is
  the queued candidate.

## Phase 2 — VRC: Verity-Routed Composition (2026-10-18)

**Files:** `vrc_harness.py` (harness), `test_vrc.py` (8 test groups, all green).
**Design:** yesterday's Part-A proposal (`research/daily_2026-10-17.md`), after SKILLFORGE
(arXiv:2608.29841) — an honest PORT of the verifier-routed-harness mechanism; the
invention surface is the oracle (the ∘ interface check, not a program verifier) and
rollback that preserves certificate validity by construction.

A deterministic harness (not an agent) synthesizes primitive sequences from target
postconditions `y = E(x0)`. Oracle = interface check + independent certificate
verifier + exact symbolic target equality. Failure categories route deterministically:
only `interface-fail` continues (REPAIRSTEP: taboo the (position, op), keep the
validated prefix, replan); `spec-rejected`, `target-mismatch`, `cert-invalid`,
`neural-divergence`, `repair-exhausted`, `budget-exhausted` all halt LOUD.
Termination: monotone taboo growth (≤ 64 positions × 8 ops) + hard cap of 20 iterations.

Key lemma (`realize()`): the spec language is strictly larger than the trace language —
`add(E, c<0)` never appears in a trace (proved: every ADD_k has k > 0, induction over
RULES), so it rewrites to the NEG sandwich; `mul(E, k<0)` rewrites to `neg(mul(E,-k))`;
constant targets are unreachable (no op discards x0) and rejected, not searched.
The planner adds a factoring peephole (`k*E2+c`, `k|c` → plan `E2+c/k` then `MUL_k`)
for shorter certificates.

**Results (2026-10-18):** 7/7 synthesis targets succeed in 1 iteration (depths 3–11;
every probe's certificate verifies, concrete == symbolic on all probes); 7/7
out-of-closure specs rejected loud with exact causes (factor-3 coefficient, bad
modulus, constant, nonlinear, unparsable); fault injection (zeroed MOD_7 weights) →
`neural-divergence` naming the op, no certificate shipped; simulated Phase-2
interface failure → REPAIRSTEP reroutes `[MUL_2, ADD_3]` → `[MUL_2, ADD_1, ADD_1, ADD_1]`
and the repaired composition still verifies. Layer-attribution order is deliberate:
concrete-vs-symbolic is checked BEFORE cert verification so execution failures
diagnose as `neural-divergence` (retrain the primitive), not as proof-layer bugs.

**Honest Phase-1 notes:** the interface leg is executed but vacuous (all P = true);
REPAIRSTEP's live path is exercised via a labeled simulated Phase-2 failure.
STRENGTHEN/WEAKEN are documented Phase-2 no-ops. No training, no GPU, KBs of RAM.

**Next:** §5.6 robustness tasks (logic chains, conditional composition, noisy inputs);
a learned composition policy (supervised/RL on certificate-validity reward, or
PRoSFI-style process rewards — see `research/daily_2026-10-18.md`) succeeds VRC's
constructive planner when the primitive library grows beyond the 8-op closure.

- **2026-10-19 — §5.6 task 2 DONE: proof-carrying conditionals (IFZERO).**
  `conditionals.py` (new): `IFZ(cond, then, else)` programs; guard decided on
  the NEURAL value, taken branch runs on the pre-guard value; preconditions
  non-vacuous for the first time (`P_then(c):=(c==0)`, `P_else(c):=(c!=0)`),
  so the `∀y.(Q_f → P_g)` interface check does real routing work; neural
  divergence at a guard halts LOUD (VRC doctrine at branch points).
  `certificates.py`: generalized `verify_trace` + branch-node checks B1–B4
  (guard re-derivation, neural/symbolic guard agreement, taken-consistency,
  branch sub-trace validity). `compose.py`: `run_program` (recursive),
  `symbolic_result_program`. `taskgen_conditional.py`: programs with
  taken-path cost == L exactly, all intermediates (incl. untaken branches)
  bound-checked in [-1e6, 1e6]. `test_conditional.py`: 33/33 green (branch
  execution, nesting, 5 forgery rejections, adversarial interface checks,
  fault-injected divergence halt). `vrc_harness.py`: `synthesize_conditional`
  plans cond/then/else specs via GENSEQ and verifies IFZ programs on probes.
- **2026-10-19 — conditional kill gate, PM-only** (`results/killgate_cond_20261019_025556.json`):
  depths {1,2,4,8,16,32,64,100} × 1000 fresh programs.
  A_CIF = **100.00%** at every depth; V_CIF = **100.00%** at every depth
  (all branch certificates verify); OLS slope b = 0.000000 (t = 0.00), flat;
  0 loud halts across 8000 instances (~44k branch decisions; frac_then
  ≈ 0.11–0.13, both outcomes well covered). PM-only kill triggers: **none** →
  **PASS (PM-only)**. Caveats: PM-only — the transformer-separation leg is
  queued (baseline needs retraining on the conditional program encoding);
  generator uses equal branch lengths (documented); guards are MOD_m
  (narrow but real conditionals). Per-step primitive exactness was already
  100% in Phase-1, so this sweep isolates the claim that *branching does not
  break compositional scaling*.
- **Next (updated 2026-10-23):** §5.6 task 3 DONE — see `noise_results.md`
  (validity gate PASS at all depths × radii; MOD-wrap hull falsified the
  flat-exactness claim; join over-approx compounds; guard decisiveness
  evaporates under noise for MOD guards; L_tight now a real per-op signal).
  Conditional transformer baseline (separation leg for the conditional kill
  gate) still queued. Today's Part-A pick queued as an invention candidate:
  **CGAU** (Certificate-Gated Adaptive Unmasking), `research/daily_2026-10-23.md`.

## Relation to the main research line

The 2026-10-16 11pm verdict named **Proof-Carrying Optimal Reduction**
(differentiable interaction-combinator substrate) as XIIV's main line, with
the Compositional Proof Machine as a side track under its own kill gate.
This prototype IS that side track's kill gate. A PASS keeps it alive as a
certified-composition substrate candidate; a KILL retires this iteration with
a written diagnosis. Either outcome is recorded, not hyped.
