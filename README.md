<p align="center">
  <img src="assets/xiiv-profile.png" alt="XIIV" width="480">
</p>

# XIIV

**XIIV is a from-scratch language model that doesn't predict left-to-right.** Every generation begins as a blank stretch of `[MASK]` tokens, and the model iteratively resolves the most confident tokens into text, in parallel, over a handful of steps. Nothing is written in order; everything takes shape at once. Text infilling, editing, and constrained regeneration fall out of the architecture natively — no prompting tricks, no scaffolding.

```
[M][M][M][M][M][M][M][M]  →  Lily found a **rabbit** in the garden.
   pure noise                       order, from noise
```

Nearly every tiny open model is another autoregressive transformer. XIIV is the *mechanism* that's different: discrete diffusion (the family LLaDA proved at 8B scale) shrunk to a genuinely trainable, CPU-runnable size with a reproducible one-day recipe.

> **Status: experimental research model (v0.1, 10M).** It trains and runs end to end, its constraint machinery is measured at 100%, and its numbers are reported honestly — including the gate it failed. See [Benchmarks](#benchmarks) and [Limitations](#limitations). A 52.5M-parameter XIIV is currently training; this repo will be updated with its weights when it completes.

## What it is / isn't

- **It is:** a 10,059,840-parameter bidirectional transformer trained from scratch on CPU with the LLaDA-style masked-diffusion objective. Real custom architecture, real trained weights, honest benchmarks.
- **It isn't:** a distilled/fine-tuned checkpoint, a wrapper, or a competitor to large AR models on raw fluency. Diffusion optimizes a different bound than next-token prediction — the numbers below state the tradeoff plainly.

## Quickstart

```bash
git clone https://github.com/CYPHERLYNX/XIIV.git && cd XIIV
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# generate from noise
.venv/bin/python -m xiiv.sample generate --length 128 --steps 16 --seed 7

# infill masked spans (the party trick)
.venv/bin/python -m xiiv.sample infill --text "The <mask> sat on the <mask>."

# watch the de-masking happen step by step (local demo page)
.venv/bin/python -m xiiv.serve   # → http://localhost:8765/
```

Weights (`weights/xiiv.safetensors`), tokenizer (`weights/tokenizer.json`) and config ship in the repo — inference makes **zero network calls**, ever. (The repo is private, so cloning will ask for your GitHub credentials.)

### On a Windows laptop (tested path for RTX 4050 machines)

```powershell
git clone https://github.com/CYPHERLYNX/XIIV.git; cd XIIV
py -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # CPU torch is enough for inference

.\.venv\Scripts\python -m xiiv.sample generate --length 128 --steps 16 --seed 7
.\.venv\Scripts\python -m xiiv.sample infill --text "The <mask> sat on the <mask>."
.\.venv\Scripts\python -m xiiv.serve   # → http://localhost:8765/
```

To **train** on your NVIDIA GPU instead of CPU, run the one-click launcher (creates its own venv, installs CUDA torch, starts training with the GPU recipe):

```powershell
.\train_xiiv_gpu.ps1
```

Training is resumable — interrupt any time with Ctrl-C; the latest `ckpt_*.pt` is picked up automatically on relaunch.

## Benchmarks (all measured, 2 CPU threads, no GPU)

| metric | value |
|---|---|
| parameters (exact) | 10,059,840 |
| weights file (fp32 safetensors) | 38.4 MiB |
| held-out masked-LM NLL, best validation (upper bound, nats/token) | 3.4021 |
| training | 5,000 steps, 30.7M tokens seen, bf16 |
| constraint: exact output length | 1.000 |
| constraint: locked-region preservation | 1.000 |
| constraint: required-token inclusion | 1.000 |
| constraint: forbidden-token satisfaction | 1.000 |
| sampler test suites | 27/27 pass (19/19 intervention tests incl.) |
| promotion gates | **8/9 pass** |
| generation, 128 tokens / 16 denoise steps | 19.4 tok/s (6.6 s, 16 forward passes) |
| peak RSS at inference | 400.2 MB |

The NLL number is an **upper bound** from the masked objective — it is *not* comparable to autoregressive perplexity, and we don't present it as such. There is no same-size AR baseline in this repo; the claim is paradigm novelty at tiny scale, not a quality crown.

**On the failed gate (G7, throughput).** The promotion gate required ≥25 tok/s; the model measured **19.4 tok/s** on a CPU-contended benchmark box, so G7 is recorded as **FAILED**, not waived. Throughput scales with denoise steps (fewer steps = faster, rougher) and with a quiet machine. If you need raw speed at this size, an autoregressive model is the better tool — diffusion's advantage is parallelism and infilling, not tokens per second.

Sample outputs (seed 7, 16 steps) — uncurated, exactly as the model produced them:

```
 to yummy cabin. At. mom smiled and the spider and day on They promised to play more. Her but they can keep get safe.


 a time, there was a little girl named Mia. They were best friends and played together all day. They lit out on his little dog:.
Hi, a boy!" boy and asked Tom, "What are you doing doing?" asked Mia asked?" Daddy said to, "Yes, we can help you find your friends for you." They went to the friend's house and saw it at the dog. The became very gentle and happy him
 
And that
```

Infill (seed 3, 16 steps) — `The <mask> sat on the <mask>.` became:

```
The  map sat on the  Liz.
```

Rough, but the mechanism is visible: masked spans resolve in place, locked context untouched.

## How it works

**Training (forward process).** For each batch, sample a mask ratio *t* ∼ Uniform(0.15, 1.0), replace each token with `[MASK]` independently with probability *t*, and train the model to reconstruct the original tokens at masked positions only (mean cross-entropy over masked positions).

**Inference (reverse process — the whole field at once).** Start from all-`[MASK]` (or a prompt with `<mask>` spans for infilling): the sequence begins as pure noise, a blank field with no information. Each step: run the full sequence through the bidirectional transformer, pick the *k* most confident masked positions, sample them from their tempered distributions. Text crystallizes out of the blank field a few confident tokens at a time. No causal mask, no KV cache, no left-to-right order. An optional `remask_ratio` re-masks the least-confident tokens each step for iterative refinement.

**Constraints, enforced during denoising.** Required tokens are force-placed, forbidden tokens are masked to −inf in the logits, locked spans are never written to, and the output canvas is always exactly the requested length. Measured compliance is 1.000 on all four constraint benchmarks.

**Architecture.** 6-layer bidirectional transformer, d=320, 5 heads, SwiGLU FFN (864), RoPE positions, RMSNorm, tied input/output embeddings, byte-level BPE vocabulary of 8,192, context length 128. 10,059,840 parameters.

## Reproduce the training

```bash
# 1. data: TinyStories (CC-BY via roneneldan/TinyStories) → tokenized blocks
.venv/bin/python scripts/prepare_data.py --out-dir data \
    --train-chars 150000000 --valid-chars 5000000

# 2. train (5,000 steps on 2 CPU cores, bf16; resumable)
.venv/bin/python -m xiiv.train --data data --ckpt-dir checkpoints \
    --total-steps 5000 --batch-size 48 --bf16

# 3. evaluate (writes eval_results.json)
.venv/bin/python -m xiiv.evaluate --weights weights/xiiv.safetensors

# 4. export
.venv/bin/python scripts/export_weights.py --ckpt checkpoints/best.pt
```

The full recipe, hyperparams, data sizes and run ledger are in [`TRAINING.md`](TRAINING.md). Configs are frozen with hashes (`run_config.json`) so any run is exactly reproducible.

## What's next

**XIIV-52.5M** (d=576, 12 layers, 9 heads, FFN 1536 — 52,508,736 parameters, same objective and constraint machinery) is training now. When it finishes and passes the same gates, its weights and an updated model card land in this repo. The 10M stays as the CPU-friendly reference.

The repo also ships two forward-looking, clearly-labeled scaffolds: a reasoning API (`generate_with_scratchpad`, `self_refine`, `generate_long` in `xiiv/diffusion.py` — mechanics verified on the 10M weights, quality rough) and an untrained multimodal architecture stub (`xiiv/vision.py`). Neither is claimed as a capability.

## Limitations (read before citing this model)

- Trained on TinyStories (simple children's-story English). It writes simple stories; it does not do math, code, or factual Q&A reliably.
- Diffusion LMs need several full-sequence forward passes per generation — slower per token than AR at the same size (see the failed G7 above); the win is parallelism + infilling, not raw throughput.
- Small-data training: 30.7M tokens is tiny by LM standards. Expect repetition and limited vocabulary — the samples above are typical, not cherry-picked.
- Like any LM, it can reproduce phrases from its training data. Don't deploy it where memorization matters.

## Security

- Weights ship as **safetensors** (no pickle). Inference loads only local files.
- The CLI and demo server make **zero network requests**.
- `xiiv.serve` binds to `127.0.0.1` only and caps request bodies at 64KB.
- Training checkpoints (`*.pt`) use pickle — only load checkpoints you trained yourself.

## Project layout

```
xiiv/            # model, diffusion objective/sampler, tokenizer, CLI, eval, demo server
scripts/         # prepare_data.py, export_weights.py, check_gates.py, ...
tests/           # unit tests (no data needed)
benchmarks/      # constraint + throughput benchmarks
weights/         # shipped safetensors + tokenizer.json + config.json
demo/            # local trajectory-visualization page
train_xiiv_gpu.ps1  # one-click RTX training launcher (Windows)
```

MIT licensed. See [TRAINING.md](TRAINING.md) for the full training log and [MODEL_CARD.md](MODEL_CARD.md) for the model card.
