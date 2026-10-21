# Invention #1 — Discriminating Experiment Results (2026-10-20)

**Invention #1:** proof-difficulty-adaptive diffusion sampling —
allocate denoising compute by predicted certificate-construction
difficulty. Prior-art verdict: NOVEL conditionally
(`invention01_prior_art.md`). **Kill condition:** if certificate
difficulty is token entropy in disguise, the invention collapses to
"CadLLM with extra steps."

**Experiment:** `research/invention01_discriminating_probe.py`.
Tiny XIIV model briefly trained on template-structured data (so its
confidence carries real signal). Over 1,760 masked positions across 40
randomized constraint sets, per position:
- **Confidence** = model top-1 probability.
- **Certificate difficulty** = GLOD guarantee load (constraint-
  interaction density: frozen proximity, required-count tightness,
  forbidden-phrase density, at_most pressure) — computed statically
  from constraint structure, without the model.

## Results

- **Spearman(confidence, guarantee load) = 0.123** — near-zero rank
  correlation.
- **Confident-but-hard quadrant: 10.0% of positions** (top-quartile
  confidence AND top-quartile load). Independence baseline 6.25%;
  redundancy would give ~0%. The quadrant is not just populated, it is
  *over*-populated.
- Hard-but-unconfident quadrant: 21.0%.

## Verdict: GATE 1 PASS — signals distinct

Certificate difficulty is NOT entropy in disguise. A confidence-based
compute allocator would systematically under-serve the 10% of
positions that are confident-but-certificate-hard — precisely the
positions where a guarantee can be lost cheaply-looking. The novelty
claim in the prior-art report stands.

## Honest caveats

- Guarantee load is static by construction, so some independence is
  structural; the live question was whether trained-model confidence
  would *end up* tracking load anyway (constraint-dense positions also
  being unpredictable). It does not (rho 0.123).
- Toy scale: tiny model, one constraint family, brief training.
- Gate 2 (not yet run): does allocating steps by predicted certificate
  difficulty actually reduce interventions / wall-clock on constrained
  generation vs confidence allocation? Gate 1 says the signal exists;
  gate 2 must show it pays.
