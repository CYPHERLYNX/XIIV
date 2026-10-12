# Contributing to XIIV

We welcome contributions. Here's how to help.

## What we need

- **Benchmark results**: run `bench/` on your hardware, submit numbers with methodology
- **Training runs**: if you have GPU, train the 1B and share loss curves
- **Bug fixes**: especially in `xiiv/diffusion.py` and `xiiv/train.py`
- **Docs**: guides, translations, examples

## How to contribute

1. Fork the repo
2. Create a branch: `git checkout -b fix/my-fix`
3. Make your change with a test if applicable
4. Run `python -m pytest tests/ -x -q`
5. Submit a PR with a clear description

## Honesty rules (non-negotiable)

- Never claim a benchmark you didn't run. Mark unrun as NOT RUN.
- Never copy numbers from another model.
- Every PR with numeric claims must include: n, methodology, hardware, seed.
- We do not merge hype. Evidence only.

## Code style

- Black formatting (`black xiiv/`)
- Type hints on public APIs
- Docstrings on all public functions
