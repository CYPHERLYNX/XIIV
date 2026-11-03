# Lever-3 launch runbook — CSC (Cold-Start Consistency) training-step variant

**Authorization gate:** run ONLY if lever 2 (BASC) completes with NO
generation gate triggered (generation still 0/5 on the 5 probes at
leg-2 end). Do NOT launch while any other kernel is running. Do NOT
burn GPU quota speculatively. Lever 3 is the backup for the case BASC
fails — it attacks a DIFFERENT failure mode (the step-0 void, not
premature-commitment derailment), so a BASC negative does not
pre-empt it.

**What lever 3 is:** leg 3 continues the FineWeb-Edu pretraining token
budget (next 25M tokens, sequential data order) with EXACTLY ONE change
vs leg 2 — the training step. Everything else (data, model, optimizer,
schedule, gates, harness) is identical. One-variable discipline.

The one change: `general_loss()` in the trainer becomes
`cold_start_loss` from `research/lever3/csc.py`
(dual-objective: standard denoising batches + p_cold fraction of
100%-masked cold batches training the full joint in one shot;
defaults p_cold=0.15, lambda_cold=1.0).

CPU smoke test PASS (2026-10-30): tiny XIIVModel, planted bigram —
cold loss 68.43 -> 0.99, cold-shot pattern accuracy 0.000 -> 0.500.
Mechanism trains; step-0 prior measurably improves on synthetic data.
Scale behavior UNTESTED — that is what this leg measures.

## Exact launch steps

1. Snapshot the leg-2 trainer and apply the single swap:
```bash
mkdir -p ~/workspace/xiiv-kaggle-gpu/lever3-jobs
cp ~/workspace/xiiv-kaggle-gpu/lever2-jobs/lever2_train.py \
   ~/workspace/xiiv-kaggle-gpu/lever3-jobs/lever3_train.py
```
In `lever3_train.py`, inside `main()` after `tok` is built, replace the
`general_loss` closure with:
```python
sys.path.insert(0, os.path.join(args.repo_dir, "research", "lever3"))
from csc import cold_start_loss
def general_loss(batch_ids):
    return cold_start_loss(expanded, batch_ids, tok.mask_id,
                           p_cold=0.15, lambda_cold=1.0)
```
Also change the `variant` string in the summary dict to
`"LEVER3-CSC, all 857,429,760 runnable params, FineWeb-Edu only"`.
Nothing else in the file changes.

2. Build the kernel dir (mirror `pretrain-jobs/build_pretrain_kernel.py`):
ship `lever3_train.py` + the repo (which now contains
`research/lever3/csc.py`); the data slug is unchanged
(`x0d4n7/xiiv-1b-pretrain-data`); resume from the leg-2 final
checkpoint:
```
--resume-weights <leg2-final>.safetensors
--start-step <leg2 final step>   # sequential data order: exact resume
--lr <leg2 LR, skip re-probe>
```

3. Push (one command):
```bash
source ~/workspace/tmp-xiiv-kaggle-venv/bin/activate
kaggle kernels push -p ~/workspace/xiiv-kaggle-gpu/lever3-kernel
kaggle kernels status x0d4n7/xiiv-1b-lever3-csc   # expect RUNNING
```

4. Gate-trigger conversion (same wiring as legs 1-2): on
`GATE_TRIGGERED` the kernel saves the full 284-tensor checkpoint +
audit + SHA, then runs the 100-prompt harness on GPU in-kernel.

## Kill criteria for lever 3

- If the first two generation gates (2k steps) show generation WORSE
  than the leg-2 final checkpoint on the 5 probes, stop: CSC is not
  the fix either; do not burn the remaining token budget.
- Held-out loss down + generation still 0/5 at leg-3 end = sixth
  masked-loss negative; report as measured. Next step is the
  BASC+CSC COMBINED leg (both terms active — the two failure modes
  may interact), NOT a fourth new mechanism from the same family.
- p_cold tuning: if cold batches dominate and denoise loss REGRESSES
  vs leg 2 (held-out FineWeb up >5% at gate 1), drop p_cold to 0.08
  and relaunch — the cold objective must stay auxiliary.
