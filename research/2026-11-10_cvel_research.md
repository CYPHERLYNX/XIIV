# 2026-11-10 research: Entropy-Valley + LR-DLLM + RSA parallel error + LAVE + Flexibility Trap -> CVEL

## Findings

### Entropy-Valley — Length-Adaptive Decoding for Masked Diffusion MT (arXiv:2608.22274)
Training-free length selector for masked-diffusion machine translation:
scores candidate target canvases by mean predictive entropy from
all-mask forward passes and picks the canvas the backbone is most
prepared to fill. Recovers 64.9% / 65.3% / 33.0% of the COMET-22 gain
from reference target lengths on En→Zh / Zh→En / En→De, and ties or
leads a same-data LLaMA-3-8B AR model on En↔Zh.
Diagnosis absorbed: canvas length is a decision in its own right, and
the denoising-friendly length need not match the reference length.
Mechanism NOT ported: EV scores lengths by the model's *own* predictive
entropy — self-referential; a confidently-wrong model picks its own
most comfortable length. CVEL replaces the scorer with an independent
contract-satisfiability oracle (see below).

### LR-DLLM — Length Regularization for variable-length DLLM generation (arXiv:2602.07546)
Treats generation length as an explicit inference variable: probes
candidate lengths and greedily adjusts the span bidirectionally
(expansion *and* contraction) under a length-regularized criterion.
51.3% Pass@1 on HumanEval-Infilling under fully unknown lengths
(+13.4% vs DreamOn), 51.5% average on four-language McEval (+14.3%).
Diagnosis absorbed: there is an intrinsic length-induced bias in DLLM
confidence estimates, so naive confidence comparisons across lengths
are unreliable for choosing length. Mechanism NOT ported: LR-DLLM still
decides length from the model's own confidence (regularized, but the
same source). CVEL never consults the model distribution for the
length decision at all — the model only fills tokens; the oracle
decides length from the contract alone.

### Masked diffusion as random sequential adsorption (arXiv:2611.07845, Nov 5, 2026)
Exactly solvable model for parallel decoding: for any data, the error
of parallel generation is the entropy production of the generation
process relative to the reversed masking process; for Markov data the
process is random sequential adsorption with labelled gaps, solved
exactly; the longer the correlations, the more the error concentrates
at the start of generation; for the short-range correlations of
English text the optimal schedule is nearly uniform.
Diagnosis absorbed: parallel-commit error is a measurable, structured
quantity — not noise — and it concentrates where correlations are
long. CVEL commits one position per round (like the rest of the
family), so it adds no parallel error; the RSA result is the
theoretical backing for keeping the commit width at one while the
*length* decision is what gets searched. Watch-tier: a future sampler
could size its per-round commit batch from a correlation-length
estimate; not attempted today.

### LAVE — Lookahead-then-Verify, reliable constrained decoding for diffusion LMs under CFGs (arXiv:2602.00612, ISSTA Oct 2026)
dLLMs generate non-autoregressively, so intermediate outputs are often
incomplete prefixes and existing constrained decoders can strand a run
in a state no valid sentence can extend. LAVE uses the dLLM's parallel
per-position distributions to look ahead and verify that a proposed
token keeps a valid completion possible, preserving completability at
every step, at negligible overhead; Dream-v0-Instruct-7B syntax error
rates up to 23.8% are the motivating failure.
Diagnosis absorbed: the operative principle is *preservability* — a
step is safe only if the partial output can still be completed into a
valid sentence. CVEL applies preservability to the length axis instead
of the grammar axis: an expansion is justified only by a proof that
the current canvas *cannot* be completed into a contract-satisfying
sentence, and is kept only if completability is restored.
Mechanism NOT ported: LAVE verifies tokens against a CFG with the
model's own distributions; CVEL verifies *length* against an explicit
contract with a model-free oracle. Different axis, different verifier.

### The Flexibility Trap (ICML 2026 Outstanding Paper, arXiv:2601.15165)
Confidence-first decoding systematically routes around the real
decision points: in reasoning tasks the decision points are precisely
the high-entropy tokens, and filling the unambiguous surroundings first
pins the position down until only one branch still reads coherently.
Diagnosis absorbed: commit order and canvas length are orthogonal
decisions, and both are decisions — neither is a side effect of
confidence. CGBS searches the order; CVEL searches the length; the two
compose (a future run can stack them) and the certificate itemizes
each decision separately. No mechanism ported: the trap is a warning
about *which* positions to commit, CVEL is about *how many* positions
exist.

### Who Gets Unmasked? Sampler-Induced Bias in Diffusion LMs (arXiv:2610.24795)
Peeking samplers (commit the positions whose sampled tokens are most
probable, LLaDA's default) amplify majority options: they shift
gender-balanced occupations toward men and collapse opinion
distributions, more so under faster parallel decoding. Fork-First
Sampling decides the fork before anything else is written.
Diagnosis absorbed, honesty note for the whole sampler family: the
sampler changes the measurement, not just the speed; sampler
comparisons must be reported per-configuration, never as a bare model
property. CVEL inherits the note: its length decision is a sampler
choice and must be reported as such. No mechanism ported.

### CARVE (arXiv:2608.30922) — already logged 2026-11-09
Verified expansion via Jensen-Shannon divergence of the model's own
predictions. Still the closest prior art to CVEL; the contrast stands:
CARVE verifies expansions against the model's distribution, CVEL
verifies them against an independent contract oracle.

## What was built today: CVEL (xiiv/cvel.py)

Certificate-Verified Expansion Length. The first sampler in the family
that searches *canvas length* instead of commit order or trajectory:
it starts from a short canvas and grows it only on a proof of
unsatisfiability.

- LENGTH-SATISFIABILITY ORACLE, model-free. A contract is an explicit
  spec: required literal token spans that must appear contiguously, plus
  a minimum final length. The oracle answers, for a canvas, whether
  every span is still placeable (already present, or a window whose
  positions are all masked-or-matching) and whether the minimum length
  is met. A span is unplaceable iff the longest masked run is shorter
  than the span — provable from the window definition — so the deficit
  is exact: len(span) minus longest masked run.
- EXPANSION ONLY ON PROOF OF UNSATISFIABILITY. Before each commit
  round the oracle runs; if satisfiable, nothing happens (the run is
  exactly the vanilla greedy loop). If not, k = the exact deficit
  (capped by per-expansion and max-length budgets) masks are inserted
  at the end of the longest masked run — the only place that grows the
  binding run — and the oracle re-runs: the expansion is kept only if
  satisfiability is restored, else the run fails closed with
  length_failure=True. Committed tokens keep their relative order
  (insertion only); the certificate records insert position, k, reason,
  and the pre-expansion canvas, so the justification is replayable
  from the certificate alone.
- THE CERTIFICATE IS THE JUSTIFICATION, not a score. The cvel-v1
  certificate itemizes initial/final length, every expansion (round,
  position, k, reason, satisfiable-before/after, pre-expansion
  canvas), commits, rounds, forward passes, gate rejects, and
  length_failure. `cvel_verify_certificate` recomputes the oracle on
  each stored pre-expansion canvas and on the final canvas, plus the
  independent no-repeat re-check — no weights, no sampler state.
- Fail-closed like the rest of the family: a contract that can never
  be satisfied within max_length ends completed=False with masks
  remaining and length_failure=True, never a silently truncated canvas.

Pre-registered falsifiers in xiiv/tests/test_cvel.py: F1 no-expansion
equivalence (satisfiable-from-start contract -> cvel canvas == vanilla
canvas, expansions == []); F2 expansion rescues the short canvas
(vanilla completes but violates the contract, cvel expands by exactly
the deficit and the span appears); F3 every expansion justified
(recomputed oracle: unsatisfiable before, satisfiable after); F4 bound
(unsatisfiable-within-max_length contract -> completed=False,
length_failure=True, final length == max_length); F5 final soundness
(completed=True implies independent verify_canvas clean AND contract
satisfied); F6 determinism. KILL: F2 fails -> does not ship; F3/F5
fail -> not actually certificate-gated -> kill.

Honest scope: tested on scripted models only, not yet evaluated on a
real checkpoint — same disclaimer as the rest of the family. The 1B
does not generate (0/5 gates at every checkpoint); no sampler claim
changes that. The CLI path (--cvel) runs with no contract, so the
length check is inert there and the run is the vanilla greedy loop
with a cvel-v1 certificate; the contract-gated behavior is exercised
by the tests. Contraction is explicitly out of scope (deleting
committed tokens would break the certificate's order-preservation
claim); LR-DLLM's bidirectional adjustment is the cited contrast.

## Watch tier (not adopted)
- Research scout run 63 (Nov 9, 22:02 IST): no new arXiv batch then;
  weekend Nov 7-8 submissions land in the Tue Nov-10 announcement —
  this run re-screened the length/ordering literature above instead of
  duplicating the scout. GPT-6.2 Astra and Qwen 4 still unreleased as
  of Nov 9; Gemini 4 Carbon still restricted. No moat threat, nothing
  adoptable for BASC leg 2 / CSC leg 3 from this window.
- BASC leg 2 v5 verdict stands (Nov 8): loss 14.147 -> 12.491,
  generation 0/5 at every gate; leg 3 CSC staged from the step-12000
  checkpoint. Nothing in today's literature changes the leg-3 plan.
