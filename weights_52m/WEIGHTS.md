# XIIV-52.5M weights

`xiiv-52m.safetensors` (210 MB, fp32, safetensors) — the finished 52,508,736-param
model, exported 2026-10-02 from the final training checkpoint
(`checkpoints_50m/ckpt_5000.pt`, step 5,000).

- SHA256: `0a6ad8533740bc381e5efe76542c4ef7da73d1913bc486ecad43909c47d5b101`
- Load: `from xiiv import XIIV; m = XIIV("weights_52m/xiiv-52m.safetensors")`
  (resolves `config.json` + `tokenizer.json` from this directory)
- Reproduce the export: `python scripts/export_weights.py --ckpt
  checkpoints_50m/ckpt_5000.pt --out-dir weights_52m` then rename
  `xiiv.safetensors` → `xiiv-52m.safetensors`

## Why the binary is not in this commit

GitHub rejects plain-git files over 100 MB. The 210 MB weight file ships via
a GitHub Release asset (private repo → private asset) or Git LFS — whichever
upload lands first. Until then, the weights are fully reproducible from the
retained final checkpoint with the command above, and the measured numbers
below are from this exact file.

## Measured on this file (2026-10-02, AMD EPYC, 2 CPU threads)

- Held-out masked-NLL bound: **3.34 nats/token** (64 batches)
  — upper bound from the masked objective, NOT comparable to AR perplexity
- Training best: 3.3414 (step 4,500); final step-5,000 eval: 3.3473
- 10M vs 52.5M on the same objective: 3.4021 → 3.3414 (0.06 nats)
- Throughput: **20.0 tok/s** @ 16 denoise steps; 37.9 tok/s @ 8 steps
- Test suite: 31/31 pass (incl. 19/19 structural constraint tests)
