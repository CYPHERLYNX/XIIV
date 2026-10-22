# CSM prototype probe — results (2026-10-21)

Phase B: AUROC confidence=0.468, ledger=0.721, gap=+0.254, base failure rate=0.735

Phase C (equal compute, 400 rounds/arm): control=0.695, routed=0.690, McNemar p=1.0000 (b01=5, b10=4)

**Kill gate: FAIL** (PhaseB gap ok=True, routing ok=False)

## Interpretation (main agent, 2026-10-21)

**Overall gate: FAIL** — the pre-registered gate required BOTH halves; the
routing half did not pay. Reported as-is, per the gate-2 discipline.

**What PASSED, decisively — the self-knowledge core.** On held-out tasks,
the verifier-built ledger predicts the model's failures at AUROC 0.721,
while the model's own confidence signal scores 0.468 — *below chance*:
at this scale the model's readable confidence is anti-predictive. Gap
+0.254, five times the required margin. The certified ledger from Phase A
(A_mixed 0.904, C_req_forbid 0.788, B_forbid_fav 0.673, D_req_only 0.308)
is real, graded self-knowledge: the system knows what it cannot do,
with counts, from a judge it cannot influence. This is the literature's
central finding reproduced in our own stack — self-report fails, the
external statistician works — except here the statistician's output lives
inside the system as its self-model.

**What FAILED — routing payoff, and why.** Routed 0.690 vs control 0.695
(p = 1.0). This is the *same bottleneck* Invention #1 gate 2 hit, now
independently replicated with a far stronger allocation signal: at toy
scale the refinement operator is too weak — re-denoising with the same
model mostly regenerates the same violations, so no allocation policy
(static heuristic or certified ledger) can pay. The constraint is repair
competence, not allocation intelligence.

**Design consequences (recorded in DESIGN.md):**
1. ABSTAIN-WITH-PROOF becomes the lead application of the ledger — it
   needs no repair operator; a 0.721-AUROC certified self-model already
   supports honest certified refusal and proved ceilings.
2. ROUTE/ALLOCATE are deferred until refinement has teeth
   (verifier-guided resampling / enforcement during refinement, or a
   competent-scale model). Fair retest defined, not assumed.
3. The self-knowledge claim of CSM stands measured at toy scale; the
   payoff-via-routing claim is killed at toy scale.
