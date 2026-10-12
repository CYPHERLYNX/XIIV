# Heavy Training Runbook — XIIV-500M on real data

No children's stories. This is the big-league regime: web knowledge +
code + books, trained hard on the RTX 4050.

## 1. Data (build once, reuse forever)

| Corpus | Script | Target | Why |
|---|---|---|---|
| FineWeb-Edu | `scripts/prepare_fineweb.py --source fineweb-edu` | 100M tokens | general knowledge — the biggest lever |
| StarCoder | `scripts/prepare_fineweb.py --source starcoder` | 30M tokens | the model has never seen code; non-negotiable |
| Books | `scripts/prepare_books.py` (existing) | ~10M tokens | long-form fluency |

> **Code data needs one free step:** StarCoder is gated on HuggingFace.
> On your PC: `hf auth login` (free account, 2 minutes), then the
> starcoder source works. Without it, train on FineWeb + books first —
> FineWeb-Edu already contains tutorials and StackOverflow-style content.

```powershell
# on your PC (bandwidth + training live there):
python scripts/prepare_fineweb.py --source fineweb-edu --tokenizer weights/tokenizer_32k.json --out-dir data_fineweb --max-tokens 100000000
python scripts/prepare_fineweb.py --source starcoder   --tokenizer weights/tokenizer_32k.json --out-dir data_code    --max-tokens 30000000
```

Mix by concatenation for the training run (document the ratio in the run
log): FineWeb first, then code, then books — or interleave at the block
level for a uniform mix. Record which you did.

**Stories are out.** `data_32k` (TinyStories) is not in the mix. If a
future run wants story fluency back, it gets added as a small fraction,
never the base.

## 2. Training config (frozen)

- Start: `checkpoints_500m_gn/ckpt_8218.pt` (grown init, optimizer fresh)
- LR: **2.06e-4** → 2.06e-5, cosine (inverse-width rule: 3e-4 × 704/1024)
- Warmup: 1000 steps (new data distribution — longer than the 500 default)
- Batch: micro 2 × accum 2 = **512 tok/step** (unchanged — comparability)
- Steps: **195k** for 100M tokens ≈ 4–6 days on the 4050 (measure the
  real s/step in the first hour and recompute)
- Contract training stays on: `--contract-prob 0.4 --lock-prob 0.4`
  (guarantee-native is the thesis — it trains on real data too)
- Optimizer: 8-bit AdamW, grad-clip 1.0, weight-decay 0.05

```powershell
# launch (after the data dirs exist — the script mixes them into data_heavy/ itself):
.\train_xiiv_500m_heavy.ps1
```

## 3. What to watch (every day)

- **Train loss**: should fall steadily. Spikes = bad batch, not disaster.
- **Val NLL** (seeded now — comparable across steps): the only number
  that decides "better". If it rises 3 vals in a row, stop and inspect.
- **tok/s**: if it drops suddenly, something's wrong with the machine,
  not the model.
- **Non-finite streak**: the trainer auto-recovers (restores best.pt,
  halves LR). Two recoveries in a day = lower the LR manually.

## 4. Evals (honest numbers only)

After training, before any claim:
1. Val NLL on the held-out set (seeded).
2. Generation samples: fixed prompts + seeds, saved to a file.
3. Contract compliance: `scripts/eval_alignment.py` probes.
4. **Public benchmarks** (this is what lets people compare us with
   famous models): `lm-eval` is installed in the venv. **Critical:**
   stock lm-eval scores choices with autoregressive teacher-forcing,
   which is INVALID for a diffusion LM. Choices must be scored by
   diffusion NLL (mask the continuation, denoise, mean CE — the same
   math as `evaluate_nll_bound`, conditioned on the context). A custom
   `xiiv/lm_eval_wrapper.py` implementing lm-eval's LM interface with
   diffusion scoring is required before any benchmark number is quoted.
   Until that wrapper exists and is tested, report only 1–3.

## 5. What "good" looks like at 500M

- Val NLL well under 4.0 on web data (stories were 4.16; web is harder,
  so compare against the starting point, not the story number).
- Coherent multi-paragraph answers on general topics.
- Code that parses (not code that works — that needs 10x more code data).
- Contract guarantees holding at ~100% with low intervention.

## 6. What this will NOT do

It will not match Opus 4.6 on benchmarks. That takes ~10^8x this
compute. What it WILL do: produce the strongest open 500M anyone's
trained on this recipe, with a guarantee-native capability no other
500M has. That is the honest win — and it's a real one.
