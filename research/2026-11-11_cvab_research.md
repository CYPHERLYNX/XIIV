# 2026-11-11 research: RSA parallel error + Batch-Paced + PV-Decoding + frequency-aware remasking + FEROM/PURL -> CVAB

## Findings

### Masked diffusion as random sequential adsorption (arXiv:2611.07845, Nov 5, 2026)
Exactly solvable model for parallel decoding: for any data, the error
of parallel generation is the entropy production of the generation
process relative to the reversed masking process; for Markov data the
process is random sequential adsorption with labelled gaps, solved
exactly; the longer the correlations, the more the error concentrates
at the start of generation; for the short-range correlations of
English text the optimal schedule is nearly uniform.
Diagnosis absorbed: commit width is a real decision with a measurable,
structured error, and the error is governed by a correlation length.
Mechanism NOT ported: the RSA schedule itself. CVAB reads the
correlation length off the certificate instead of estimating it from
data (see below).

### Batch-Paced — Fixed-Budget Batch Decoding in Discrete Diffusion LMs (arXiv:2610.17917)
Training-free method for fixed-budget dLLM batch decoding: jointly
ranks masked-token confidence across requests to distribute commits,
committing the same fixed number of tokens for every request at each
denoising pass; gains generally grow with batch size and persist
across decoding configurations (their numbers).
Diagnosis absorbed: width is a first-class decoding variable, and
fixing it per request wastes budget.
Mechanism NOT ported: fixed width per request, decided by
cross-request confidence ranking — a hyperparameter, never justified
for the batch being committed. CVAB makes width adaptive per round
and justified per batch by the certificate.

### PV-Decoding — Progressive Verification Decoding with Decode Gating for Diffusion LMs (arXiv:2610.24722)
Confidence-driven speculative decoding with progressive verification
for block diffusion: masked positions are partitioned into directly
decoded / speculative / unresolved tokens by two confidence
thresholds, and speculative tokens are verified with progressively
stronger policies; reduces sequential model forward passes by 8%–37%
with the largest observed accuracy drop at 1.3% (their numbers,
GSM8K/MBPP/HumanEval on three diffusion LMs).
Diagnosis absorbed: commit-then-verify beats commit-only, and
verification strength should scale with risk.
Mechanism NOT ported: the verifier is the model's own distributions
(progressively stronger policies) — self-referential. CVAB verifies
each batch with the independent no-repeat certificate instead.

### Frequency-aware remasking test-time scaling (arXiv:2611.10486, logged 2026-11-10)
Two-stage: fast base sampler for easy questions, then frequency-aware
remasking on hard ones (tracks how often each token's prediction
changes across denoising steps; commits frequently-changing tokens to
explore alternative reasoning paths). LLaDA-8B / Dream-7B: +3.73pp /
+7.33pp on MATH-500, coverage gains up to ~11pp on HumanEval (their
numbers). Inference-only, no retraining.
Diagnosis absorbed: prediction change across steps marks the decision
points. Not ported: it picks *which* positions to reconsider; CVAB
picks *how many* to commit per round. Complementary — a future run
can stack them.

### FEROM (arXiv:2611.10369) / PURL (arXiv:2611.11551) (logged 2026-11-10)
Reveal-order marginal RL and planned unmasking-and-revision joint
learning. Both need training/retraining — shelf, future phase only.
Diagnosis absorbed: reveal order is an endogenous variable worth
optimizing. CVAB's batch is an order over positions too (rank order
within a round), optimized greedily under a certificate instead of
by RL.

### DMax (arXiv:2604.08302) / DSL-LLaDA (arXiv:2606.01024)
Aggressive parallel decoding (self-revising mask-to-hybrid-embedding)
and continuous denoising (DSL + SDE sampler, delayed hard commit).
Both training-required (OPUT / continue-pretraining). Not drop-in;
shelf.

### Scout state (run 67, 06:02 IST today)
No new Wed Nov-11 arXiv batch posted at run time; keyword scan of
800 recent cs.CL/cs.LG/cs.AI entries found nothing new beyond run
65's diffusion cluster. Prior-art watch clean; moat thesis intact.

## The invention: CVAB — Certificate-Verified Adaptive Batching

Every sampler in the family commits exactly one position per round.
The literature above circles one open question — how many tokens to
commit per round — and answers it with a fixed number (Batch-Paced),
a confidence heuristic (PV-Decoding), or a training procedure
(FEROM/PURL/DMax). CVAB answers it with a proof.

The certificate's interaction radius IS the correlation length. The
no-repeat-n-gram certificate of order n has interaction radius n−1:
two commits can only interact — jointly complete a repeated n-gram —
if they are less than n apart, because a window of length n contains
two positions iff their distance is at most n−1. So a batch is
provably certificate-safe iff (a) every member passes the per-commit
gate against the pre-batch canvas, and (b) members are pairwise at
least n apart. No window then contains two new tokens, so each
window's check is exactly the single-commit check the gate already
ran. The model only ranks candidates; the certificate decides how
many may go together. Width adapts per round to canvas geometry (the
greedy maximal separated set among ranked candidates) instead of
being a fixed hyperparameter or a confidence heuristic.

What was built (`xiiv/cvab.py`, `xiiv/tests/test_cvab.py`, `--cvab`
in `xiiv/sample.py`):
- `interaction_radius(n) = n−1`, `min_separation(n) = n`.
- Per-round greedy maximal separated set in rank order; dropped
  candidates counted as shrinks and itemized in the certificate.
- Batch verification with the independent `verify_canvas` (defense
  in depth): admissibility plus a clean pre-batch canvas provably
  implies a clean simulated canvas, so a failed check is an alarm
  with a width-1 fallback and fail-closed stall.
- Preexisting repeat in the frozen prompt fails closed at round
  zero with zero forward passes.
- cvab-v1 certificate: per-round pre-round canvas, batch members,
  target/examined/admitted widths, shrinks, min separation,
  verified flag. `cvab_verify_certificate` replays the chain
  model-free (admissibility, masked-before, chain exactness,
  intermediate checks, completed soundness).

Measured (scripted models only, CPU, no real checkpoint — same
honest scope as the rest of the family):
- 16-mask stub, distinct position-stable proposals: cvab completes
  in 4 forward passes where vanilla needs 16, identical final
  canvas (F2).
- Adjacent-conflict stub: naive 4-wide batch would repeat the
  frozen 4-gram; cvab shrinks to width 1 per round (6 shrinks),
  finishes clean as `[10,11,12,13,10,11,12,99]` (F3).
- Full suite: 68/68 pass (60 pre-existing + 8 new).

What it is NOT: not a measurement on the 1B (still 0/5 generation,
unchanged), not a claim that batching improves quality on real text
— only that width can be a certificate-verified decision with a
measured forward-pass reduction on scripted stubs. The mechanism is
genuinely novel for the family: the first sampler to commit more
than one position per round, and the first to make batch size a
certificate decision rather than a hyperparameter.
