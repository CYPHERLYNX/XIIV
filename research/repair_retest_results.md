# Fair combined retest — results (2026-10-22)

Pre-registered in research/repair_retest_probe.py before the run.
Operator: Verifier-Guided Repair (xiiv/repair.py) — diagnosis-targeted
repair with constraint-enforced refill, shared identically by all arms.
Probe: research/repair_retest_probe.py; raw: repair_retest_results.json;
runtime 90s.

## Part 1 — the GLOD allocation question, with a capable operator

n=200 paired drafts, family A (pilot draft intervention rate 0.80).
Intervention rates after repair (2 rounds, same budget, same refill):

| arm | rate | positions touched/trial |
|---|---|---|
| verifier (diagnosis) | **0.025** | 1.3 |
| confidence | 0.045 | 6.7 |
| guarantee-load | 0.620 | 5.2 |

McNemar load vs confidence: b01=0, b10=115, p≈0.0000.
**Pre-registered verdict: FAIL** — and decisively. Under the fair test
the earlier diagnosis demanded, load targeting is not merely neutral,
it is far WORSE than confidence targeting. Historical reference
(descriptive, cross-probe): free refinement in gate 2 scored
0.720–0.735 for every arm; the operator change alone collapses the
rates to 0.025–0.045 for the two viable targeting policies.

Why load fails, mechanistically: guarantee load measures constraint
*interaction density* around a position. Actual violations — a
forbidden token sitting somewhere, a missing required token — usually
live at low-interaction positions. Load spends the whole budget
re-rolling structurally complex but valid positions while the real
violation sits untouched. Confidence works because violations do
concentrate where the model was unsure, and enforced refill prevents
regeneration. Diagnosis beats both because it does not predict where
violations are — it computes them.

**Closure for Invention #1 (GLOD):** the signal is real (gate 1:
distinct from confidence, Spearman 0.123) but the payoff claim is
dead — tested under a weak operator (FAIL, 2026-10-21) and now under
a strong operator (FAIL, decisively). Do not revive allocation-by-
guarantee-load as a payoff mechanism. What survives and advances:
VGR itself — verifier diagnosis + enforced refill — which is the
guarantee-native architecture working as designed.

## Part 2 — the CSM routing question, with a capable operator

Ledger rebuilt identically (A_mixed 0.731, B_forbid_fav 0.808,
C_req_forbid 0.692, D_req_only 0.288). Phase B AUROC on fresh drafts:
ledger 0.684 vs raw confidence 0.474 (gap +0.210) — the certified
self-knowledge result REPLICATES (original: 0.721 vs 0.468).

Phase C' (VGR both arms, equal total rounds): control 0.015,
routed 0.015, zero discordant pairs.
**Pre-registered verdict: CEILING** — with control at 0.015 (< 0.05),
there is almost nothing left for routing to save; the question is
undecidable at this operator strength, not failed. Substantive
reading: when repair is near-perfect and cheap (1.3 positions/trial),
risk-based compute routing has no headroom. Routing may matter again
where repair is expensive — at 1B scale, not toy scale. ABSTAIN-WITH-
PROOF remains the Mirror Ledger's lead application, unchanged.

## What this establishes

1. The repair-competence bottleneck diagnosis (drawn independently
   from gate 2, CSM Phase C, and the GMMD pilot) was correct: change
   the operator and intervention rates fall ~30x on this testbed
   (0.80 drafts → 0.025 after VGR).
2. Diagnosis > prediction for repair targeting, by a wide margin,
   at ~5x less touched compute than the best predictive signal.
3. Two payoff claims are now honestly closed (GLOD allocation) or
   honestly ceiling-bound (CSM routing at toy scale). The surviving
   assets: VGR (new core machinery), the Mirror Ledger's certified
   self-knowledge (replicated), and the guarantee-native verifier as
   the system's center of gravity — exactly where the design put it.
