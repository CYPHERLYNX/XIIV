# 2026-11-12 research: L2R unmasking order + RSA + FEROM/PURL -> CVOU

## Findings

### Left-to-right training boosts diffusion-LM reasoning (Tsinghua LeapLab, ICML 2026 Oral, arXiv:2601.15165, Nov 8, 2026)
Training LLaDA-8B-Instruct with strict left-to-right unmasking + GRPO
beats confidence-based unmasking on GSM8K (89.1% vs 86.9% SPG, matched
conditions) and MATH-500. Training-order insight for diffusion LMs;
relevant to CSC leg-3 thinking, but NOT blitz-adoptable as a training
change (BASC leg 2 v5 is done; changing unmasking order mid-leg is out
of scope). Shelf for training; the ORDER diagnosis is adopted here at
inference time instead.
Diagnosis absorbed: unmasking order is not a cosmetic choice — on
reasoning it beats confidence order by 2.2pp. The literature's order
is either confidence (heuristic) or strict L2R (fixed). Neither is
justified per-position by anything the sampler can check.

### Masked diffusion as random sequential adsorption (arXiv:2611.07845, Nov 5, 2026)
Exactly solvable model for parallel decoding: the error of parallel
generation is the entropy production of the generation process
relative to the reversed masking process; for Markov data the process
is random sequential adsorption with labelled gaps, solved exactly;
the longer the correlations, the more the error concentrates at the
start of generation. Diagnosis absorbed: commit order interacts with
correlation structure — early commits fix the context later commits
see. CVAB read the correlation length off the certificate (interaction
radius n-1); CVOU reads the ORDER off the same certificate.

### FEROM (arXiv:2611.10369) / PURL (arXiv:2611.11551) (logged 2026-11-10)
Reveal-order marginal RL and planned unmasking-and-revision joint
learning. Both need training/retraining — shelf, future phase only.
Diagnosis absorbed: reveal order is an endogenous variable worth
optimizing. CVOU optimizes it greedily under a certificate instead
of by RL: the order is whatever the certificate's closure rule
admits, and the model only ranks within the admissible set.

### PV-Decoding (arXiv:2610.24722) / SpecCTMC (arXiv:2611.08362) (shelf)
Commit-then-verify with self-referential verifiers; exact
distribution-preserving speculative sampling for masked diffusion
(1.73-4.38x fewer sequential evals, their numbers). Both verify with
the model's own distributions or preserve the model's distribution.
CVOU verifies with the independent no-repeat certificate instead, and
preserves the certificate rather than the distribution.

### Scout state (run 70, 06:02 IST today)
No new arXiv batch in window (Wed Nov-11 batch still latest, screened
run 68); Nov-12 batch not yet posted at run time. Qwen4-Max second
access door (Qwen Chat), still no benchmarks/card/params/price/
weights. MiniMax H3-Image weights (33.1B DiT, restrictive license).
Prior-art watch clean; moat thesis intact. Nothing adoptable for
BASC leg 2 / CSC leg 3. Blitz shelf unchanged.

## The invention: CVOU — Certificate-Verified Ordered Unmasking

Every sampler in the family lets the model choose WHERE to commit
next (confidence ranking) and, since CVAB, HOW MANY. None of them
constrains the ORDER — and the order is exactly what the Nov-8
Tsinghua result says matters for reasoning. CVOU is the first in the
family whose order is decided by the certificate, not by confidence
and not by a fixed rule.

The certificate defines a partial order over positions: the closure
frontier. A masked position p is order-admissible iff every position
in its left (n-1)-window [p-n+1, p-1] is already committed (frozen
prompt tokens count as committed). Only order-admissible positions
may be committed; among them the model ranks by safe-candidate
confidence and commits the best. One commit per round.

The payoff is PREFIX FINALITY — incremental certification. When the
rightmost position of any n-gram window commits, that window is
fully committed and its verdict is final: no later commit can land
inside it, because every other position in the window is already
committed. That much is true of ANY commit order. What the closure
rule adds is that windows close in left-to-right order, so after
each round the committed PREFIX [0..m] (m = leftmost masked position
minus one) is final — every n-gram fully inside it is decided
forever, and a verifier can check the output incrementally, round by
round, instead of waiting for the whole canvas. In vanilla
confidence order commits are scattered, so the certified prefix
stays 0 until the leftmost position happens to commit — typically
the last rounds — and nothing is certifiable before the run ends.
CVOU's certified prefix grows every round; vanilla's does not. The
beam/backtracking samplers (CGBS/CSBR) exist because branches must
be re-checked; CVOU sidesteps that cost for the common path by
making the committed prefix final as it grows.

Two corollaries, both provable from the admissibility rule alone:
- On a fully-masked canvas the admissible set is a singleton — the
  leftmost masked position — so CVOU IS strict left-to-right there.
  The certificate derives L2R; the Tsinghua empirical result (L2R
  beats confidence order on reasoning) is the measured shadow of
  this proof. CVOU is not "L2R plus a certificate": L2R is what the
  certificate's closure rule reduces to on a blank canvas.
- On a prompted canvas the admissible set is the set of segment
  heads — the leftmost masked position of each maximal masked run —
  so segments advance independently and the model ranks the heads.
  The order is per-segment L2R with model-ranked segment
  interleaving, still with per-window finality everywhere.
- Progress is guaranteed: the leftmost masked position is always
  admissible (everything left of it is committed), so the order rule
  alone can never stall the run; a stall means the gate refused
  every admissible candidate, and the run fails closed as usual.

What was built (`xiiv/cvou.py`, `xiiv/tests/test_cvou.py`, `--cvou`
in `xiiv/sample.py`):
- `order_admissible(canvas, pos, n, mask_id)`: the closure-frontier
  predicate. `segment_heads(canvas, n, mask_id)`: the admissible set.
- Per-round: rank admissible positions by best safe candidate
  (reusing `cvel._safe_candidates`); commit the best-ranked head.
  The gate still applies per commit, and the completed canvas is
  re-checked with the independent `verify_canvas` (defense in depth).
- `closed_windows(canvas, pos, n)` counts the windows a commit
  finalizes; the trace records `windows_closed` (cumulative) and
  `prefix_area` — the sum over rounds of the certified-prefix length
  after each round. On a fully-masked canvas CVOU's prefix grows by
  exactly one per round; vanilla's stays 0 until the leftmost
  position commits. That is the falsifier, not a claim.
- Preexisting repeat in the frozen prompt fails closed at round
  zero with zero forward passes.
- cvou-v1 certificate: per-round pre-round canvas, admissible set,
  chosen position, windows closed this round, certified prefix
  after the round, verified flag. `cvou_verify_certificate` replays
  the chain model-free (admissibility, masked-before, chain
  exactness, intermediate checks, prefix accounting, completed
  soundness).

Measured (scripted models only, CPU, no real checkpoint — same
honest scope as the rest of the family):
- 16-mask distinct-position-stable stub: cvou commits in strictly
  increasing position order (L2R), 16 forward passes, identical
  final canvas to vanilla (F1/F2).
- Adversarial right-first stub: vanilla's certified prefix is 0
  until the final round (prefix_area 8 on length 8); cvou's grows
  every round (prefix_area 36), and closes each window exactly
  once, in order (F3).
- Two-segment prompted stub: cvou interleaves segment heads by
  rank, still L2R within each segment (F4).
- Certificate replay verifies on a real run; tampering fails (F5).
- Fail-closed: no safe candidate at any admissible head stalls
  with completed=False; preexisting repeat fails at round zero (F6).
- Determinism: identical runs -> identical canvas and certificate (F7).

## Fit to the XIIV thesis
CVOU is guarantee-native in the strongest sense in the family so
far: it does not add a certificate check to an existing sampler, it
derives the sampler's order FROM the certificate. The order is a
proof obligation (prefix finality), not a hyperparameter. It stacks
with CVAB (width) and CGBS (beam) — a future run can batch within
the admissible set — but it ships alone, falsifiable, today.
