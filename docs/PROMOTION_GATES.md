# XIIV promotion gates (declared 2026-10-01, while training was at step ~2150/5000)

A checkpoint ships **iff** every gate passes. Gates are evaluated by
`scripts/check_gates.py`, which fails closed: any missing input is a
failure, exit code non-zero. Declared *before* training finished, so they
cannot be fit to the outcome.

## Gates

| # | gate | threshold | source | rationale |
|---|---|---|---|---|
| G1 | training completed | reached `total_steps` (5000), no non-finite abort | `checkpoints/train_log.jsonl` | a crashed run never ships |
| G2 | held-out masked-LM NLL bound | best ≤ **4.0** nats/tok | train log `val_nll` | step-500 measured 4.59, step-1500 3.82 and falling; 4.0 is a real gate (fails the step-500 checkpoint) with headroom |
| G3 | exact-length infill | satisfaction = **1.0** | `bench/constraints/results.json` | structural (fixed canvas) — a code-correctness gate |
| G4 | forbidden-word exclusion | satisfaction = **1.0** | constraint bench | structural (logit −inf) — code-correctness gate |
| G5 | anchor/lock preservation | satisfaction = **1.0** | constraint bench | structural (frozen mask) — code-correctness gate |
| G6 | required-word inclusion | satisfaction ≥ **0.95** | constraint bench | forcing mechanism; margin for pathological cases |
| G7 | CPU generation floor | ≥ **50** tok/s (128 tok / 16 steps, 2 threads) | `benchmarks/results_xiiv.json` | floor catches breakage; measured value expected in the hundreds |
| G8 | inference smoke | load → generate → infill → steer-round → constrain, first try, no errors | `check_gates.py` runs it | a top-lab researcher runs this on arrival |
| G9 | weight sanity | all exported weights finite, param count = 10,059,840 | `check_gates.py` | corrupt export never ships |

## Explicitly NOT gates (honest scope)

- **AR-perplexity comparison.** The masked-diffusion NLL bound and the AR
  next-token NLL are different objectives; ranking them is meaningless.
  Both are reported, never ranked.
- **Frontier-model parity.** XIIV is a 10M TinyStories model. It is not
  compared to frontier models on general benchmarks. Its claim is
  constraint-native generation at tiny scale — measured where it counts
  (constraint satisfaction, parallel-decode efficiency).

## If a gate fails

The checkpoint does not ship. The failure, the metric, and the decision go
in `RUN_LEDGER.md`; the fix becomes the next falsifiable experiment. No
gate is ever relaxed after the fact — a gate that needs moving was the
wrong gate, and changing it requires a new declared revision.
