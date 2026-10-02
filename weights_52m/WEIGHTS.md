# XIIV-52.5M weights

`xiiv-52m-fp16.safetensors` (105 MB, fp16, safetensors) — the finished 52,508,736-param
model, exported 2026-10-03 from the best training checkpoint
(`checkpoints_50m/best.pt`, step 4,500, best val NLL 3.3414). fp16 vs fp32 max abs
diff on embeddings: 0.0153 (ordinary quantization noise); inference verified
(generate + infill) on this file.

- Load: `from xiiv import XIIV; m = XIIV("weights_52m/xiiv-52m-fp16.safetensors")`
  (resolves `config.json` + `tokenizer.json` from this directory)
- Get it: download from the [xiiv-52m-v0.1 release](https://github.com/CYPHERLYNX/XIIV/releases/tag/xiiv-52m-v0.1)
  and place it in this directory.
- Reproduce the export: `python scripts/export_weights.py --ckpt
  checkpoints_50m/best.pt --out-dir weights_52m --fp16` then rename
  `xiiv.safetensors` → `xiiv-52m-fp16.safetensors`

## Why the binary is not in this commit

GitHub rejects plain-git files over 100 MB. The 105 MB weight file ships as a
GitHub Release asset (private repo → private asset). Until downloaded, the weights
are fully reproducible from the retained best checkpoint with the command above,
and the measured numbers below are from this exact file.

## Measured on this file (2026-10-02, AMD EPYC, 2 CPU threads)

- Held-out masked-NLL bound: **3.34 nats/token** (64 batches)
  — upper bound from the masked objective, NOT comparable to AR perplexity
- Training best: 3.3414 (step 4,500); final step-5,000 eval: 3.3473
- 10M vs 52.5M on the same objective: 3.4021 → 3.3414 (0.06 nats)
- Throughput: **20.0 tok/s** @ 16 denoise steps; 37.9 tok/s @ 8 steps
- Test suite: 31/31 pass (incl. 19/19 structural constraint tests)
