# 2026-11-06 research: RPD + OSCAR + LAVE + proof-carrying certs -> CCCA

## Findings

### RPD — Reliable Parallel Decoding in Masked Diffusion LMs (arXiv:2609.36452, He et al., UVA, Sep 2026)
Parallel commitments chosen by confidence are unreliable: "confidence
alone does not [determine reliability]". RPD selects candidates by a
layerwise signal and defers predictions that fail it. Training-free.
On LLaDA and Dream, math reasoning + code gen.
Diagnosis absorbed: commit reliability needs a SECOND signal beyond
single-pass top-1 confidence. Mechanism NOT ported: RPD's signal is
layerwise agreement, which is unavailable through XIIV's
`model(x) -> logits` sampler interface; CCCA's second signal is
cross-chain agreement, which needs only that interface. RPD *defers*
unreliable predictions; CCCA defers AND rolls back AND certifies what
remains.

### OSCAR — Orchestrated Self-verification and Cross-path Refinement (arXiv:2604.01624, Apr 2026)
N parallel denoising chains with randomized reveal orders; cross-chain
Shannon entropy localizes unreliable commitments; targeted remasking
conditioned on retrieved evidence. Training-free. LLaDA-8B / Dream-7B
on TriviaQA, HotpotQA, RAGTruth, CommonsenseQA.
Diagnosis absorbed: cross-chain disagreement localizes unreliable
commits. Mechanism NOT ported: OSCAR's response is remask-with-
evidence and its second key is a hallucination score; CCCA's response
is a commit gate with rollback, and its second key is the independent
certificate verifier.

### LAVE — Lookahead-then-Verify (arXiv:2602.00612, Feb 2026)
Constrained decoding for dLLMs under CFGs: a commit is safe only if
the remaining canvas stays completable; lookahead over the parallel
distributions verifies the proposed token preserves completability.
Diagnosis absorbed: completability of the remaining canvas matters,
not just the current canvas. Mechanism NOT ported: LAVE checks CFG
completability; CCCA checks certificate completability — when a run
ends incomplete, one diagnostic forward pass counts DEAD positions
(no certificate-safe candidate in the top-k), so the run reports WHY
it could not complete.

### Proof-Carrying Certificates for LLM Pipelines (arXiv:2605.16407, Koomullil, May 2026)
Lean 4 trust-boundary architecture; three certificate families plus a
Maximal Certifiable Residue operator — turns abstention into the
maximum-weight certifiable residue with audit-logged dropped claims.
Thesis-adjacent (proof-carrying outputs), different axis: it verifies
the deterministic structured computations AROUND the model, not
generation itself.
Diagnosis absorbed: when everything cannot be certified, certify the
maximal residue and NAME the rest. CCCA's non-consensus promotions are
exactly that residue, itemized position by position in the ccca-v1
certificate.

### Also noted (watch tier, not adopted)
- DSL-LLaDA (arXiv:2606.01024): continuous denoising via Discrete
  Stochastic Localization; needs 1,000 steps of continued pretraining
  — not CPU-adoptable, training-time not decoding-time.
- Batch-Paced (arXiv:2610.17917): fixed-budget batch decoding, jointly
  ranks confidence across sequences — batch-scheduling, not a
  single-sequence commit rule; watch for the serving rig.
- Entropy-Valley (length-adaptive MT decoding): canvas-level entropy
  from all-mask passes as a length selector — a cheap global signal,
  watch for length selection in the sampler family.
- EoS-as-hidden-reasoning (arXiv:2603.05197): masked diffusion LMs use
  EoS-token representations as latent compute — relevant to the 1B's
  generation behaviour, not to today's decoding work.

## Fit assessment for XIIV
All four core findings fit the diffusion + guarantee-native thesis and
are training-free decoding strategies — implementable and verifiable on
this CPU-only box. None is adopted as a port:
- From RPD we take the *second-signal* diagnosis, but the signal is
  cross-chain agreement, not layerwise agreement.
- From OSCAR we take the *cross-chain disagreement* shape, but the
  response is consensus-gated commit + disagreement rollback + an
  independent certificate as second key, not remask-with-evidence.
- From LAVE we take the *completability* shape, but the checked
  property is certificate completability (dead positions), not CFG
  completability.
- From Koomullil we take the *maximal certifiable residue* shape: the
  certificate itemizes non-consensus promotions instead of hiding them.

## What was invented (xiiv/ccca.py, this commit)
CCCA — Cross-Chain Consensus Arbitration:
1. K parallel chains (default 3) denoise the same canvas with
   different deterministic reveal orders (confidence / left-to-right /
   right-to-left / seeded shuffle), each verifier-gated against its
   own working canvas.
2. CONSENSUS: a position committed by >= quorum chains (default 2)
   with the SAME token, and verifier-safe against the shared canvas,
   commits to the shared canvas. Two independent keys: cross-chain
   agreement AND the certificate gate.
3. DISAGREEMENT: chains proposing different tokens for one position
   are a detected unreliability — the position is NOT committed while
   disputed; every chain rolls it back to masked and resyncs.
4. STANDING PROPOSAL: a lone proposal waits for a second chain to
   confirm; after `confirm_rounds` rounds unconfirmed it PROMOTES as
   an explicitly non-consensus commit (verifier-gated, counted,
   itemized) — the liveness fallback.
5. The ccca-v1 certificate claims: final canvas verifier-clean, every
   consensus commit confirmed by >= quorum reveal orders, and the
   non-consensus residue listed position by position. Fail-closed:
   incomplete runs never claim certification; a diagnostic pass counts
   dead positions when a run ends incomplete.
6. Cost: K forward passes per round, counted honestly in the trace.

## Verification (CPU scripted stubs only, real-checkpoint untested)
9/9 tests in xiiv/tests/test_ccca.py green, plus the full sampler
suite (27 passed: ccca + cbrs + scv). Pre-registered falsifiers:
F1 consensus convergence (clean stub: consensus_rate 1.0, 0
disagreements, 8 forward passes over 3 rounds); F2 disagreement
deferral + rollback (fork stub: dispute at pos 4 round 2, 51 vs 50,
commit lands round 3 with post-rollback token 50); F3 convergence
after rollback (consensus_rate 1.0, 0 non-consensus); F4 final
soundness (completed => verify_canvas clean, recomputed); F5
fail-closed (repeater stub never completes, masks remain,
dead_positions_final >= 1); F6 determinism; F7 promotion honesty
(K=1: all 6 commits promoted, consensus_rate 0.0, positions itemized).
CLI: `sample.py generate --ccca` wired (config CCCAConfig, ccca-v1
certificate printed on completion).
