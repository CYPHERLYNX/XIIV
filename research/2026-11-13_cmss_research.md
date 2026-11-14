# 2026-11-13 research: LESS mutual-stability sampling + CadLLM + SPOT/CoDiLA/SCALE -> CMSS

## Findings

### LESS Is More: Mutual-Stability Sampling for Diffusion Language Models (Mohamed, Shang & Vazirgiannis, arXiv:2606.16908, Jun 2026)
Training-free, model-agnostic adaptive sampler that treats token
commitment as an online stopping problem. A masked position becomes
eligible for unmasking only when its top-1 prediction has high
confidence, its top-1 token PERSISTS across recent reverse steps, and
its predictive distribution is stable under top-K inter-step
Jensen-Shannon divergence. On Dream-7B, LLaDA-8B and LLaDA-1.5-8B
across seven math/code/knowledge benchmarks, LESS improves the
accuracy-efficiency trade-off over strong training-free adaptive
samplers while using 72.1% fewer reverse steps than fixed-budget
decoding (their numbers). Diagnosis absorbed: a commit is a bet on a
prediction, and a prediction that flips between rounds is not a bet
worth taking yet -- wait one more reverse step and let it settle.
Adopted today as CMSS, with the XIIV twist that stability is a
certificate (per-commit stability triple, model-free replay), not a
sampler heuristic.

### CadLLM: Confidence-Aware Calibration for diffusion-LM throughput (ACL 2026 Findings, arXiv:2512.07173)
Training-free adaptive inference acceleration: token decoding
confidence signals dynamically adjust block size, step count,
vocabulary sampling range and submission threshold. 1.1-2.28x
throughput gains on LLaDA and DREAM while maintaining competitive
accuracy; the adaptive threshold is the core driver (removing it drops
throughput 121.72 -> 34.57 tok/s, NFE +289%). Diagnosis absorbed:
confidence is a usable control signal at inference time with no
retraining. CMSS uses confidence as one leg of a three-legged rule
instead of as the whole rule. The adaptive-threshold idea (threshold
as a function of measured confidence spread) is a candidate for a
future CMSS variant, not adopted today.

### SPOT: Speculative Decoding for Diffusion Language Models (arXiv:2610.22543, 2026)
Predicts where a diffusion LM commits next, verifies a whole tree of
block states in one pass with the exact output distribution. Up to
6.8x fewer model passes, 3.4x throughput vs standard decoding on
SDAR-8B; transfers to LLaDA-8B and Dream-7B. Diagnosis absorbed:
verification can be decoupled from commitment. CMSS verifies with the
independent certificate instead of the model's own distribution, and
the thing verified is stability, not distribution preservation. Shelf:
tree verification is a future phase; CMSS's batch commit is the cheap
version of the same idea.

### CoDiLA: Locally Coherent Parallel Decoding in Diffusion LMs (ICML 2026, arXiv:2603.20216)
A lightweight autoregressive model attached to the DLM, fed the DLM's
marginal distributions as soft embeddings, performs local AR decoding
within small blocks. Syntax error rate 70% -> 16% at B=8 on
Dream-Coder-7B; new Pareto frontier on code generation at >=2x
throughput. Diagnosis absorbed: parallel commits are locally
incoherent unless something coordinates them. CMSS coordinates them
with stability instead of a second model: positions that commit
together must each have settled independently. Shelf: no AR head is
trained here; CMSS is the no-training answer to the same diagnosis.

### SCALE: Memory-Efficient LLM Pretraining via Minimalist Optimizer Design (ICML 2026, arXiv:2506.16659)
Minimalist SGD variant matching or beating Adam at 60M-1B with only
35-45% of Adam's memory; beats APOLLO and Muon on LLaMA 7B in both
perplexity and memory. Training-side only. Logged for a future
training leg (the 5090 package / a future Kaggle leg could swap the
optimizer); not adopted in this sampler. Also noted: the companion
survey (arXiv:2505.22922) finds weight refactorization + momentum
reset lets a low-rank method on a 1B beat GaLore/Fira perplexity with
~25% less memory -- same shelf.

### 2026 constrained-decoding consensus: shape, not truth
Structured outputs / grammar-constrained decoding now guarantee
schema-valid output everywhere (OpenAI strict mode, Gemini
responseSchema, Claude structured outputs, XGrammar in vLLM/SGLang).
The Nov-10-2026 practitioner consensus is explicit: the guarantee
covers the schema, never the truth -- a hallucinated employeeId or a
nonexistent package passes every type check (5.2% of packages from
commercial models and 21.7% from open-source models in the USENIX
Security 2025 sample did not exist). Diagnosis absorbed, and it is the
standing justification for this whole family: XIIV's certificates
must keep constraining SEMANTIC content (here: the stability of the
prediction itself), not just syntax. No adoption -- this is prior-art
confirmation of the thesis, not a mechanism to port.

## The invention: CMSS -- Certificate Mutual-Stability Sampling

Every sampler in the family commits on a SINGLE round's evidence: the
model's top-1 confidence right now, gated by the certificate. None
asks whether the model's mind is MADE UP -- and LESS says that is
exactly the question that decides quality per forward pass. CMSS is
the first in the family whose commit rule is a stability rule, and
the first whose every commit carries its stability evidence as a
certificate.

The joint stability rule (LESS's three conditions, adopted as
certificate claims):
1. CONFIDENCE: P(top-1) >= conf_threshold (the same bar the gate uses).
2. PERSISTENCE: the top-1 token is identical over the last `persist`
   consecutive observations of that position (default 2: the token
   survived at least one full reverse step unchanged).
3. DIVERGENCE: the Jensen-Shannon divergence between the top-K
   distributions of consecutive observations in the persistence window
   is <= jsd_threshold (default 0.02 nats) -- the distribution did not
   just keep its argmax, it kept its shape.

A position commits only when all three hold AND its stable top-1
passes the independent no-repeat gate. The commit token must BE the
stable top-1: if the gate refuses the top-1, the position does not
fall through to a lower candidate, because the stability certificate
is about the top-1 prediction, and a different token has no stability
evidence behind it. The batch of stable positions commits together;
the post-batch canvas is re-checked with the independent verifier,
shrinking to the single best commit on violation, and failing closed
if even that violates.

PATIENCE ROUNDS. A round with no stable position consumes one forward
pass, records observations, and commits nothing. Patience is never
wasted on a deterministic model: with an unchanged canvas the next
round's observations are identical, so persistence reaches 2 and the
JSD is 0, and every confident position stabilizes. Patience only
becomes permanent when the model's predictions OSCILLATE -- top-1
flipping between rounds, or the distribution reshuffling under a
constant argmax -- and then the run exhausts `max_rounds` and fails
closed with masks remaining, instead of emitting an unsettled canvas.
That is the falsifier: an oscillating model must never be committed.

Two corollaries, both provable from the stability rule alone:
- On a stable model (identical predictions every round) CMSS commits
  the whole canvas in TWO forward passes: round 0 observes, round 1
  stabilizes and commits everything. Vanilla needs one pass per token.
  The step saving is the measured quantity, not a claim.
- Stability is monotone in evidence: a position that was stable at
  round r stays stable at round r+1 unless the canvas changed under
  it (another commit landed nearby) or the model itself moved. The
  certificate records the exact window that justified each commit, so
  a replay can check the claim without the model.

What was built (`xiiv/cmss.py`, `xiiv/tests/test_cmss.py`, `--cmss`
in `xiiv/sample.py`):
- `jsd_topk(prev, cur)`: Jensen-Shannon divergence (nats) over the
  union of two top-K distributions, renormalized; identical -> 0.0,
  disjoint -> ln 2.
- Per-round: observe every masked position (top-1, confidence, top-K
  distribution); the three-legged rule decides eligibility; the
  certificate gate decides commit-worthiness; batch commit with
  post-batch re-verification, shrink-to-one fallback, fail-closed
  stall.
- `patience_rounds` and `batch_shrinks` in the trace: a run with zero
  of both is the clean path; anything else is visible.
- Preexisting repeat in the frozen prompt fails closed at round zero
  with zero forward passes.
- cmss-v1 certificate: per-round pre-round canvas, per-position
  observations, committed positions with stability triples (conf,
  persistence, jsd), patience flag. `cmss_verify_certificate` replays
  the chain model-free (masked-before, token == recorded top-1,
  persistence run and JSDs recomputed from stored observations
  against the recorded thresholds, gate admission, chain exactness,
  intermediate checks, completed soundness).

Measured (scripted models only, CPU, no real checkpoint -- same
honest scope as the rest of the family):
- 16-mask distinct-position-stable stub: cmss completes in 2 forward
  passes (round 0 observes, round 1 commits all 16), every commit
  persistence 2, jsd 0.0 (F1); vanilla needs 16 passes for the
  identical final canvas (F2).
- Flip-flop stub (top-1 alternates every round): exhausts 6 rounds,
  0 commits, completed=False (F3).
- Reshuffle stub (top-1 constant, runner-up mass swaps; inter-step
  JSD ~0.0523 > 0.02): exhausts, 0 commits (F4).
- Low-confidence stub (top-1 prob 1/3 < 0.35): exhausts, 0 commits (F5).
- Certificate replay verifies on a real run; tampering with a
  commit's jsd, persistence, or token fails verification (F6).
- Fail-closed: stable top-1 that would create a repeat stalls with
  completed=False (gate_rejects 1, 2 forward passes); preexisting
  repeat fails at round zero (F7).
- Determinism: identical runs -> identical canvas and certificate (F8).
- Full suite: 14 CMSS tests pass; 37 pass with CVOU + CVAB suites.
