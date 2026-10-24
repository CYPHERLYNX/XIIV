<p align="center">
  <img src="assets/banner.png" alt="XIIV" width="100%"/>
</p>

<h1 align="center">XIIV</h1>

<p align="center">
  A 1B-parameter masked diffusion language model with contract-verified generation.<br/>
  Outputs are checked against explicit constraints by an independent verifier before they are accepted.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/pytorch-2.x-ee4c2c" alt="PyTorch 2.x"/>
</p>

## Overview

XIIV is a language model built around masked discrete diffusion rather than autoregressive decoding. Generation starts from a fully masked sequence and resolves positions in parallel over a series of denoising steps, ordered by model confidence. Working on the full sequence at once makes three operations native to the model instead of bolted on: masked infilling, revision of already-generated spans, and verification of the complete output before it is returned.

Constraints are expressed as **contracts** — machine-checkable conditions supplied with the request (required or forbidden content, locked spans, structural rules). A separate verifier module evaluates the generated tokens against the contract; positions that violate it are re-masked and regenerated. The verifier is independent code: correctness is established by checking, not by the model's own confidence.

The 1B model was not initialised at full size. It descends from a single trained 133M checkpoint, grown stage by stage (133M → 500M → 1B) using function-preserving expansion with drift gating at each stage, and is trained block-wise with Seed-Grown Block Distillation (SGBD) — a low-resource training method developed for this project that trains each block against cached teacher activations, making full-gradient training of a 1B model feasible without a large GPU cluster.

<p align="center">
  <img src="assets/how-it-works.png" alt="Generation in XIIV: a masked sequence is resolved in parallel and verified against its contract" width="100%"/>
</p>

## Key features

- **Masked diffusion decoding** — parallel, confidence-ordered generation with controllable order (confidence, left-to-right, right-to-left, ends-first)
- **Contract-verified outputs** — declarative constraints compiled into generation and enforced by an independent verifier, with a per-output certificate
- **Masked infilling and span locking** — arbitrary positions can be frozen, edited, or regenerated in place
- **Grown, single-lineage scaling** — Net2Net-style width/depth expansion with drift gates; no stage is randomly re-initialised
- **Block-wise distillation training (SGBD)** — the 1B is trained in eight blocks against a cached teacher, on commodity and free-tier GPU hardware
- **Reproducible evaluation** — guarantee-focused benchmarks with published methodology; results are reported only from measured runs

## Architecture

| Component | Specification |
|---|---|
| Parameters | 1B (915,988,480 unique) |
| Layers | 36 |
| Hidden dimension | 1,280 |
| Attention heads | 20 |
| Attention | Bidirectional, full, RoPE |
| Feed-forward | SwiGLU, 3,584 |
| Vocabulary | 32,768 |
| Embeddings | Tied input/output |
| Context length | 128 tokens |
| Additional module | Contract encoder, cross-attention in every layer |

Full specification: [docs/MODEL_SPEC_1B.md](docs/MODEL_SPEC_1B.md) · Model card: [docs/MODEL_CARD_1B.md](docs/MODEL_CARD_1B.md)

## Installation

```bash
git clone https://github.com/CYPHERLYNX/XIIV.git
cd XIIV
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.x. GPU training dependencies are listed separately in `requirements-gpu.txt` / `requirements-train.txt`.

## Quick start

```python
from xiiv import XIIV

model = XIIV(weights="weights/xiiv.safetensors")

# Parallel generation from a prompt
text = model.generate(prompt="The history of the Roman Empire", steps=32)

# Masked infilling — [MASK] spans are resolved in context
filled = model.infill("The capital of France is [MASK].")
```

Lower-level sampling, including contract constraints and frozen spans, is exposed through `xiiv.diffusion.sample_trajectory`; contract definitions and verification live in `xiiv.contracts`.

## Training

Training proceeds in two parts:

1. **Growth.** Each scale stage is produced from the previous checkpoint by function-preserving expansion (width 1024 → 1280, depth 27 → 36, heads 16 → 20). A stage is accepted only if its output drift against the parent stays below a fixed gate. Protocol: [docs/SCALING_PROTOCOL.md](docs/SCALING_PROTOCOL.md).
2. **SGBD block training.** The 1B network is partitioned into eight blocks. Each block is trained for 1,000 steps against activations cached from the 133M teacher, using Kaggle T4 GPUs and local hardware. Checkpoints and loss histories are retained per block.

**Current training status**

| Stage | State |
|---|---|
| 133M teacher | Trained |
| 500M | Grown from 133M |
| 1B blocks 0–4 | Trained — 1,000 steps each (622.5M parameters) |
| 1B blocks 5–7 | Pending |
| Assembled 1B | Evaluation follows completion of block training |

Runbook: [docs/HEAVY_TRAINING_RUNBOOK.md](docs/HEAVY_TRAINING_RUNBOOK.md) · Data: [docs/DATASET_CARD.md](docs/DATASET_CARD.md)

## Evaluation

Evaluation focuses on the properties the architecture is designed to provide. Methodology: [docs/GUARANTEE_BENCHMARKS.md](docs/GUARANTEE_BENCHMARKS.md).

| Benchmark | Result | Scope |
|---|---|---|
| Locked-span fidelity | 100% | 133M model, n = 100 |
| Generation-order control | 100% | 133M model, n = 24 (confidence / L2R / R2L / ends-first) |
| Contract compliance rate | In progress | Measured on the assembled 1B |
| Intervention rate | In progress | Verifier interventions per generation, assembled 1B |
| Certificate validity | In progress | Share of outputs passing independent verification, assembled 1B |

Standard language-model benchmarks are tracked as before/after training measurements and are published with the same runs. No results are reported for configurations that have not been run.

## Project structure

```
xiiv/          Model, diffusion sampler, contracts and verifier, training
tests/         Test suite (sampler, contracts, growth, training, API)
scripts/       Growth, data preparation, and training utilities
docs/          Model cards, protocols, runbooks, technical report
research/      Experiment designs, pre-registered probes, and results
eval/          Evaluation harnesses
benchmarks/    Benchmark definitions and results
examples/      Usage examples
assets/        Figures used in this README
```

## Documentation

- [Technical report](docs/TECHNICAL_REPORT.md) — design and methodology in full
- [Model card — 1B](docs/MODEL_CARD_1B.md) — capabilities, training state, limitations
- [Scaling protocol](docs/SCALING_PROTOCOL.md) — stage-by-stage growth rules and gates
- [Guarantee benchmarks](docs/GUARANTEE_BENCHMARKS.md) — what is measured, and how
- [Training runbook](docs/HEAVY_TRAINING_RUNBOOK.md) — reproducing the training runs
- [Novelty and prior art](docs/NOVELTY.md) — prior-art review for the approach

## Development

```bash
python -m pytest tests/ -q
```

CI runs the test suite on every push (`.github/workflows/`).

## License

MIT — see [LICENSE](LICENSE).

## Author

Fahad Abdulla ([@CYPHERLYNX](https://github.com/CYPHERLYNX))
