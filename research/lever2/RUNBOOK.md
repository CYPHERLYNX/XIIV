# Lever-2 launch runbook — BASC training-step variant

**Authorization gate:** run ONLY if pretrain leg 1 completes (~25M tokens)
with NO generation gate triggered. Do NOT launch while any pretrain
kernel is running. Do NOT burn GPU quota speculatively.

**What lever 2 is:** leg 2 continues the FineWeb-Edu pretraining token
budget (next 25M tokens, sequential data order) with EXACTLY ONE change
vs leg 1 — the training step. Everything else (data, model, optimizer,
schedule, gates, harness) is identical. One-variable discipline.

The one change: `general_loss()` in the trainer becomes
`belief_anchored_loss` from `research/lever2/basc.py`
(L = L_mask + λ_self·L_self + λ_anchor·L_anchor, defaults
λ_self=1.0, λ_anchor=0.5, τ=1.0).

## Exact launch steps

1. Snapshot the leg-1 trainer and apply the single swap:
```bash
mkdir -p ~/workspace/xiiv-kaggle-gpu/lever2-jobs
cp ~/workspace/xiiv-kaggle-gpu/pretrain-jobs/pretrain_train.py \
   ~/workspace/xiiv-kaggle-gpu/lever2-jobs/lever2_train.py
```
In `lever2_train.py`, inside `main()` after `tok` is built, replace the
`general_loss` closure with:
```python
sys.path.insert(0, os.path.join(args.repo_dir, "research", "lever2"))
from basc import belief_anchored_loss
def general_loss(batch_ids):
    return belief_anchored_loss(expanded, batch_ids, tok.mask_id,
                                lambda_self=1.0, lambda_anchor=0.5)
```
Also change the `variant` string in the summary dict to
`"LEVER2-BASC, all 857,429,760 runnable params, FineWeb-Edu only"`.
Nothing else in the file changes.

2. Build the kernel dir (mirror `pretrain-jobs/build_pretrain_kernel.py`):
ship `lever2_train.py` + the repo (which now contains
`research/lever2/basc.py`); the data slug is unchanged
(`x0d4n7/xiiv-1b-pretrain-data`); resume from the leg-1 final
checkpoint:
```
--resume-weights <leg1-final xiiv_1b_pretrain_step<N>.safetensors>
--start-step <leg1 final step>   # sequential data order: exact resume
--lr <leg1 probe winner, skip re-probe>
```
The builder docstring's `--resume-dataset` flag does NOT exist in the
trainer — hand-edit the dataset slug into `kernel-metadata.json` as in
the session-2 runbook.

3. Push (one command):
```bash
source ~/workspace/tmp-xiiv-kaggle-venv/bin/activate
kaggle kernels push -p ~/workspace/xiiv-kaggle-gpu/lever2-kernel
kaggle kernels status x0d4n7/xiiv-1b-lever2-basc   # expect RUNNING
```

4. Gate-trigger conversion (already wired): on `GATE_TRIGGERED` the
kernel saves the full 284-tensor checkpoint + audit + SHA, then runs
`pretrain_harness.py` (100 prompts, spec v1) on GPU in-kernel.
Box-side after download:
```bash
cd ~/workspace/ai-tools/corsac/research/lever2
python audit_exact1b.py --weights <downloaded>.safetensors \
    --expect-sha <sha-from-pretrain_gates.json>
```
then read the harness JSON from the kernel output. No stitch needed —
the checkpoint is already the exact-1B layout.

## Kill criteria for lever 2

- If the first two generation gates (2k steps) show generation WORSE
  than the leg-1 final checkpoint on the 5 probes, stop: BASC is not
  the fix, do not burn the remaining token budget.
- Held-out loss down + generation still 0/5 at leg-2 end = fifth
  masked-loss negative; report as measured, do not invent lever 3
  from the same family.
