# Logic Chains — §5.6 Task 1 Results (2026-10-20)

**Status: PASS** (full kill-gate incl. transformer-baseline separation)

## What was built

Boolean analogue of the Phase-1 arithmetic kill gate, testing the
compositional proof machine beyond arithmetic on **bit-vector programs**:

- **Domain:** 8-bit vectors (bytes 0–255), LSB-first.
- **8 primitives** (mirroring the 8-primitive Phase-1 library):
  `NOT`, `XOR_A5`, `XOR_3C`, `AND_F0`, `OR_0F`, `SHL_1`, `SHR_1`, `ROL_1`
  (constant-operand ops parallel Phase-1's `ADD_k` design).
- **Symbolic domain:** per-bit boolean expressions over input bits
  `b0..b7`, with canonical constructors (const-folding, double-negation
  elimination, idempotence, complement annihilation, xor parity
  cancellation, deterministic operand ordering). All 8 abstract
  transformers verified **exact by brute force over all 256 inputs**.
- **Neural primitives:** 8→32→32→8 ReLU MLPs on the bit vector (sigmoid
  outputs), trained on the full 256-value domain to **100% exact match**
  each. Documented choice (see `bool_primitives.py`): the finite domain
  lets per-step error be ~0 so the sweep isolates *composition*, exactly
  as Phase-1's lstsq fit did for affine ops.
- **Independent verifier** (`verify_bcertificate`): re-derives every
  bit-expression from its own rule table, checks chaining, and requires
  the final vector to evaluate to the claimed byte. Adversarial tests:
  tampered step / wrong result / unknown op / empty trace all rejected.

## Depth sweep (proof machine)

Depths {1, 2, 4, 8, 16, 32, 64, 100} × 1000 fresh instances:

| L | A_BOOL | V_BOOL |
|---|--------|--------|
| 1 | 100.00% | 100.00% |
| 2 | 100.00% | 100.00% |
| 4 | 100.00% | 100.00% |
| 8 | 100.00% | 100.00% |
| 16 | 100.00% | 100.00% |
| 32 | 100.00% | 100.00% |
| 64 | 100.00% | 100.00% |
| 100 | 100.00% | 100.00% |

- Slope b = 0.000000, t = 0.00 (crit 2.447) — perfectly flat.
- PM-only kill triggers: **none**. Verdict: **PASS**.

## Transformer baseline (same protocol as Phase 1)

- Architecture: identical decoder-only transformer (d=96, 3 layers,
  3 heads, ffn=384; ~400K params incl. larger byte vocab — still ~30×
  the proof machine's ~13K neural params, erring *against* the PM).
- Tokenization: byte-level (one token per byte value) + 8 op tokens.
- Training: depths 1–16, fresh random data, 10k steps, early stopping
  on validation exact-match (same protocol as `baseline_transformer.py`).
  Final model: best-validation checkpoint (13.2% mean over depths 1–16).

| L | A_TF |
|---|------|
| 1 | 32.80% |
| 2 | 12.90% |
| 4 | 7.20% |
| 8 | 6.90% |
| 16 | 7.20% |
| 32 | 1.30% |
| 64 | 0.90% |
| 100 | 1.00% |

Degradation slope b_TF = −0.0017 (collapses with depth, same wall as
Phase 1). Note the baseline is not crippled: a depth-1-only diagnostic
reaches 99.8% on single ops in 1500 steps — it learns the primitives
fine; *composition* is what kills it.

## Full kill-gate verdict (with baseline separation)

- A_BOOL(100) = 100.00% (≥90% ✓)
- Flat: slope b = 0.000000, t = 0.00 (not significant ✓)
- Separation: 100.00 − 1.00 = **99.0pp** (≥40pp ✓)
- V(L) = 100% at every depth (no certificate failure ✓)
- Kill triggers: **none**
- **VERDICT: PASS**

Results: `results/killgate_logic_20261020_034921.json`,
`results/baseline_logic_20261020_034833.json`.

## Interpretation (honest)

The claim under test was narrow: exact composition does not degrade with
depth, in a second domain beyond arithmetic. The boolean certificate
machinery (canonical boolean abstract interpretation + independent
re-deriving verifier) held at 100% validity across 8000 instances, and
accuracy stayed flat at 100% through depth 100. No hype beyond that:
program synthesis (inferring op sequences) remains Phase-2 work, and the
primitive nets memorize a finite domain by design.

## Files

- `logic.py` — boolean expressions, RULES_BOOL, verifier
- `bool_primitives.py` — BitNet, exact_fn, apply_concrete
- `train_bool_primitives.py` — full-domain training → checkpoints_logic/
- `taskgen_logic.py`, `compose_logic.py` — generator, LogicProofMachine
- `killgate_logic.py` — the sweep; `test_logic.py` — 28 tests
- `baseline_logic.py` — transformer baseline (reuses Phase-1 architecture)
- `results/killgate_logic_<stamp>.json`, `results/baseline_logic_<stamp>.json`
