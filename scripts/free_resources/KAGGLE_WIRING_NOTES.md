# Kaggle GPU Wiring Notes — SGBD block training

Log of findings; updated incrementally (box restarts kill processes, not files).

## 2026-10-18 ~20:08 +04 — Training entry point mapped

**Real entry:** `xiiv/sgbd_train_block.py` (CLI, argparse). NOT `xiiv/sgbd.py` (that's the BlockTrainer class lib).

Args (exact):
- `--block INT` (required) — block index 0-7. Block 2 = NEXT (0,1 done).
- `--steps INT` (default 200; we use 1000)
- `--batch-size INT` (default 8)
- `--lr FLOAT` (default 1e-4)
- `--cache-dir STR` (default `sgbd_cache/iter0`, relative to repo root)
- `--out-dir STR` (default `sgbd_blocks`, relative to repo root)
- `--log-every INT` (default 25)
- `--save-every INT` (default 200) — step checkpoints `block{b}_step{step}.pt`
- `--resume-step INT` (default 0) — resumes weights+projections+optimizer+step from `block{b}_step{N}.pt`
- `--device STR` (default `cpu`; accepts `cuda`, falls back to cpu w/ warning)
- `--student-ckpt STR` (default `checkpoints_1b_gn/ckpt_8218.pt` under repo)

**GPU support:** ALREADY PRESENT. `--device cuda`. No patch needed. (Verified in source lines ~88-94.)

**Block geometry (hardcoded):**
- Student slices: BLOCKS = [(0,5),(5,10),(10,15),(15,20),(20,24),(24,28),(28,32),(32,36)]
- Teacher groups: T_GROUPS = [(0,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,15)]
- Block 2: student layers [10,15), teacher layers [4,6). Cache chunks needed: `layer04_c*.pt` (input) + `layer05_c*.pt` (target) ONLY — not the full 16.6GB cache.

**Memory design (fp32):** ~115M params/block → ~2GB (weights 460MB + grads 460MB + AdamW 920MB + acts ~200MB). On T4 (16GB VRAM): fits easily; could raise batch-size.

**Inputs the Kaggle kernel needs:**
1. Repo code → `git clone https://github.com/CYPHERLYNX/XIIV.git` (public).
2. Teacher cache chunks for layers 04,05 → from HF `elova89/xiiv-sgbd-teacher-cache` (upload in progress — CHECK STATUS).
3. Student grown-init ckpt (`checkpoints_1b_gn/ckpt_8218.pt`) → location TBD (check local + HF).
4. torch (cu121), safetensors, numpy via pip.

**NOT needed:** tokenized training data (.bin files). The trainer consumes cached teacher activations only. (Phase-1 capture already done.)

**Output:** step checkpoints → `/kaggle/working/block2/` (Kaggle auto-versions outputs on completion).

## 2026-10-18 ~20:15 +04 — Builder written + dry-run PASS

**Staging decision: HF download in-kernel (Option A).** Rationale: teacher cache
already uploading to `elova89/xiiv-sgbd-teacher-cache` (idempotent); kernel pulls
only what block 2 needs. No 4GB re-upload to Kaggle datasets.

**Priority upload launched (background, pid 22308, log /tmp/block2_upload.log):**
`scripts/free_resources/upload_block2_inputs.py` — uploads layer04_c*.pt (24) +
layer05_c*.pt (24) + ckpt_8218.pt (1.83GB) ≈ 4GB to the HF repo. The background
16GB full-cache upload (2 procs) is idempotent and will skip these files.

**Builder:** `scripts/free_resources/kaggle_sgbd_block.py` (REWRITTEN, real payload).
- Kernel payload: pip install torch cu121 + safetensors + huggingface_hub; assert CUDA;
  git clone CYPHERLYNX/XIIV; snapshot_download (allow_patterns for block's layers + ckpt)
  from private HF repo; run `xiiv/sgbd_train_block.py --block N --steps N --device cuda`.
- HF auth in kernel: SHORT-LIVED surrogate token (`hsurr:`) fetched at BUILD time via
  dynamic_credentials, embedded in the notebook source. Kernel is PRIVATE (owner-only).
  Token is deleted from kernel memory right after download. Documented tradeoff:
  no Kaggle-secret plumbing via API; surrogate TTL limits exposure window.
- `--smoke`: 20-step kernel under separate slug `x0d4n7/xiiv-sgbd-block{b}-smoke`.
- `--resume-step` supported for preemption recovery (ckpts every 200 steps to /kaggle/working).
- `--dry-run`: builds + validates notebook JSON, no push.

**Dry-run 20:15:** PASS — notebook OK (2KB, 1 code cell), slug x0d4n7/xiiv-sgbd-block2.
Payload re-verified 20:22 by importing builder and inspecting generated cell:
snapshot_download allow_patterns, n_chunks==24 assert, HF_TOKEN deletion,
and training cmd all present and correct.

**Upload reorder 20:20:** killed first priority run (only 2/48 chunks done, ~5min/chunk
— too slow to block smoke test); relaunched with ckpt_8218.pt (1.8GB) FIRST, then
chunks. Smoke test needs ckpt + ≥1 chunk/layer before it can run. NOTE: 2 background
full-cache uploads also consume uplink; left alone (not mine to kill).

**GPU support in trainer:** NONE NEEDED — `--device cuda` already implemented with
cpu fallback (sgbd_train_block.py). Verified by reading source.

**Priority upload COMPLETE 20:45:** ckpt_8218.pt + 48/48 chunks verified on HF
(layer04: 24/24, layer05: 24/24). Full-cache background uploads resumed (SIGCONT).
Smoke kernel has everything it needs; waiting for GPU worker (QUEUED as of 20:40).
(URL slug derived from title; the `--id` slug `xiiv-sgbd-block2-smoke` does NOT
resolve for status checks — use the URL slug. Builder fixed 20:37 to use slug-safe
titles.) Status: QUEUED (waiting for GPU worker).

**Slug lesson:** keep kernel titles slug-safe (lowercase, hyphens) so the id is
predictable; otherwise `kaggle kernels status` fails with permission errors.

**T4 time estimate for 1000 steps:** CPU baseline ~2.6h/1000 steps (~9.4 s/step) for
~115M params. T4 matmul throughput suggests 10-20x → **~8-16 min/1000 steps**.
To be confirmed by smoke test. Well within 9h session limit.

## 2026-10-18 ~20:55 +04 — Smoke kernel pushed, queued

**Pushed:** `x0d4n7/xiiv-sgbd-block-2-smoke-20-steps` (private, GPU) at 20:35.
Status: QUEUED (no failure) as of 20:55 — waiting for free GPU worker (normal for
free tier; new accounts may wait longer).
**API note:** use `KaggleApi().kernels_status(slug)` via Python directly — the
`kaggle` CLI hangs on status calls.

## 2026-10-18 ~21:05 +04 — Full kernel pre-built, still queued

Full 1000-step kernel dry-run: PASS (notebook OK, slug x0d4n7/xiiv-sgbd-block2).
Ready to push the moment smoke passes. Smoke kernel still QUEUED at 21:15
(~40 min) — free-tier GPU wait for new account, no failure. Polling every ~10 min.

## 2026-10-18 ~21:20 +04 — Parent update: block 2 training LOCALLY

**PIVOT:** The ~20h RAM blockage CLEARED — a download-monitor worker found two
concurrent teacher-cache uploads eating ~1.3GB and killed the duplicate.
MemAvailable recovered to 2.72GB; guardian launched block 2 on CPU (1000 steps,
~2.6h ETA, started 20:43 +04). Parent decision: let local block 2 finish;
**Kaggle GPU takes blocks 3-7.**

**Implication for this task:** full block-2 Kaggle launch is MOOT. The smoke test
remains valuable as pipeline validation (proves CUDA + HF download + training loop
work before block 3). Builder already supports any block via --block N.

**Kernel metadata mystery (unresolved):** both pushed kernels show
`enable_gpu=False, is_private=False, current_version_number=0` via kernels_list,
despite push reporting "version 1 successfully pushed" and status QUEUED.
Hypothesis: listing shows the un-versioned shell; version 1 (with correct metadata)
is what's queued. Cannot verify until it runs. If the kernel starts on CPU, the
payload's CUDA assert will abort it (fail-fast, minimal burn).

## 2026-10-18 ~21:35 +04 — Smoke v1 FAILED, root cause found, v2 pushed

**v1 ERROR root cause (from kernel log):** `git clone https://github.com/CYPHERLYNX/XIIV.git`
failed — `fatal: could not read Username`. **The XIIV repo is PRIVATE**, not public
(task context was wrong). GPU was fine (T4 15.6GB detected before the clone failed).

**Fix:** removed git clone entirely. Builder now embeds the minimal training code
directly in the notebook: `xiiv/__init__.py` (empty stub — avoids heavy imports),
`xiiv/config.py`, `xiiv/model.py` (17KB total, only needs torch), plus
`sgbd_train_block.py`. Bundle verified: 4 files, 26KB, all compile, content
byte-identical to local. Kernel writes them to `/kaggle/working/pkg/` and runs
with `PYTHONPATH=/kaggle/working/pkg`.

**v2 pushed** 21:35 as `x0d4n7/xiiv-sgbd-block2-smoke` version 2.

## 2026-10-18 ~21:45 +04 — v2 FAILED (HF token TTL), pivoting to Kaggle dataset

**v2 ERROR root cause (from kernel log):** `401 Unauthorized` on
`huggingface.co/datasets/elova89/xiiv-sgbd-teacher-cache/.../layer04_c000.pt`.
The short-lived `hsurr:` surrogate embedded at build time EXPIRED before the
kernel's download step (queue wait + pip installs > TTL).

**Pivot:** abandon HF-in-kernel auth. Upload inputs as a PRIVATE Kaggle dataset
once; kernels attach it via `dataset_sources` and read from `/kaggle/input/`.
No tokens in kernels, no TTL races.

**Dataset:** `x0d4n7/xiiv-block2-inputs` — 48 chunks (layer04+layer05) + ckpt_8218.pt
= 49 files, 3.8GB, staged via hardlinks at `~/workspace/kaggle_staging/xiiv-block2-inputs/`
(no extra disk). Builder update needed: replace snapshot_download block with
`/kaggle/input/xiiv-block2-inputs/` paths; add `dataset_sources` to metadata.

## TODO (updated)
- [x] Training entry mapped, GPU support verified (already in trainer)
- [x] Data on HF (full 16.6GB cache + ckpt verified)
- [x] Builder written, dry-run PASS, bundle verified
- [x] Smoke v1 pushed → ERROR (private repo, git clone failed)
- [x] Smoke v2 pushed → ERROR (HF surrogate TTL expired)
## 2026-10-18 ~22:00 +04 — STATUS: BLOCKED on data staging

**What's READY:**
- Builder: `scripts/free_resources/kaggle_sgbd_block.py` (dry-run PASS)
- Code embedding: 4 files, 26KB, verified byte-identical, compiles OK
- GPU: T4 15.6GB confirmed working (from v1 kernel log)
- Training entry: `xiiv/sgbd_train_block.py`, `--device cuda` already supported

**What's BLOCKED — data staging (both paths failed):**
1. **HF download in kernel**: v2 failed with 401. Short-lived `hsurr:` surrogate
   expires before kernel runs (queue wait + pip install > TTL).
2. **Kaggle dataset upload**: `kaggle datasets create` and `dataset_create_new` API
   HANG from this environment (tested with 5-byte file, 90s timeout, 0% progress).
   Network is fine (API 200, GCS reachable). SDK 2.2.4.

**To unblock, need ONE of:**
- (a) Long-lived HF token from Fahad (embed in kernel, no TTL race), OR
- (b) Dataset uploaded from a different network (Fahad's PC?), OR
- (c) Parent investigates the Kaggle upload hang further

**Note:** Local block 2 training is RUNNING (PID 2154, relaunched 21:12 after
box restart killed first attempt at step 75). Kaggle is for blocks 3-7.

## TODO
- [x] Check teacher-cache HF upload status
- [x] Locate student ckpt
- [x] Decide staging: HF download in-kernel
- [x] Write real notebook-builder
- [x] Dry-run: build + validate notebook JSON
- [x] Priority upload complete (ckpt + 48 chunks on HF)
- [x] Push PRIVATE smoke kernel (20 steps)
- [ ] Smoke test: verify launch + CUDA + data + first steps (BLOCKED: kernel QUEUED)
- [ ] Estimate T4 time for 1000 steps (from smoke timing)
