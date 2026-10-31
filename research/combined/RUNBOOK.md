# BASC+CSC combined-leg launch runbook

**Authorization gate:** run ONLY if lever 3 (CSC) completes with NO
generation gate triggered (generation still 0/5 on the probes at leg-3
end) AND the leg-3 verdict is "held-out loss down + generation still 0/5"
— i.e. the sixth masked-loss negative. Do NOT launch while any other
kernel is running. Do NOT burn GPU quota speculatively. The combined leg
is the prescribed next step after a leg-3 negative: the two failure modes
(premature-commitment derailment, step-0 void) may interact, and the
interaction is itself measurable. It is NOT a fourth new mechanism from
the same family — if the combined leg also moves loss without moving
generation, the next step is a different family of mechanism (sampler-side
commit policy, e.g. confidence-gated block-parallel unmasking in the
Clover-35B line — see research/clover35b_diffuspace_notes.md), not another
training-step variant.

**What the combined leg is:** continues the FineWeb-Edu pretraining token
budget (next 25M tokens, sequential data order) with EXACTLY ONE change
vs leg 3 — the training step. Everything else (data, model, optimizer,
schedule, gates, harness) is identical. One-variable discipline.

The one change: `general_loss()` in the trainer becomes `combined_loss`
from `research/combined/combined.py` — a step-mixture: with probability
p_cold the step is the CSC cold objective (l_cold on a 100%-masked
canvas), otherwise it is the BASC triple term
(L_mask + lambda_self·L_self + lambda_anchor·L_anchor). Defaults
p_cold=0.15, lambda_cold=1.0, lambda_self=1.0, lambda_anchor=0.5, tau=1.0.

CPU smoke test PASS (2026-10-31): tiny XIIVModel, planted bigram —
both branches sampled and both losses decrease; cold-shot pattern
accuracy improves (mechanism trains on synthetic data).
Scale behavior UNTESTED — that is what this leg measures.

## Exact launch steps

1. Snapshot the leg-3 trainer and apply the single swap:
```bash
mkdir -p ~/workspace/xiiv-kaggle-gpu/combined-jobs
cp ~/workspace/xiiv-kaggle-gpu/lever3-jobs/lever3_train.py \
   ~/workspace/xiiv-kaggle-gpu/combined-jobs/combined_train.py
```
In `combined_train.py`, inside `main()` after `tok` is built, replace the
`general_loss` closure with:
```python
sys.path.insert(0, os.path.join(args.repo_dir, "research", "combined"))
from combined import combined_loss
def general_loss(batch_ids):
    return combined_loss(expanded, batch_ids, tok.mask_id,
                         p_cold=0.15, lambda_cold=1.0,
                         lambda_self=1.0, lambda_anchor=0.5)
```
Also change the `variant` string in the summary dict to
`"LEVER23-COMBINED, all 857,429,760 runnable params, FineWeb-Edu only"`.
Nothing else in the file changes. (A pre-built copy lives at
`~/workspace/xiiv-kaggle-gpu/combined-jobs/combined_train.py` — verify its
diff against the leg-3 trainer is exactly this swap + variant string
before use, same one-variable check as legs 2/3.)

2. Build the kernel dir (mirror `pretrain-jobs/build_pretrain_kernel.py`):
ship `combined_train.py` + the repo (which now contains
`research/combined/combined.py`); the data slug is unchanged
(`x0d4n7/xiiv-1b-pretrain-data`); resume from the leg-3 final checkpoint:
```
--resume-weights <leg3-final>.safetensors
--start-step <leg3 final step>   # sequential data order: exact resume
--lr <leg3 LR, skip re-probe>
```

3. Push (one command):
```bash
source ~/workspace/tmp-xiiv-kaggle-venv/bin/activate
kaggle kernels push -p ~/workspace/xiiv-kaggle-gpu/combined-kernel
kaggle kernels status x0d4n7/xiiv-1b-lever23-combined   # expect RUNNING
```

4. Gate-trigger conversion (same wiring as legs 1-3): on
`GATE_TRIGGERED` the kernel saves the full 284-tensor checkpoint +
audit + SHA, then runs the 100-prompt harness on GPU in-kernel.

## Kill criteria for the combined leg

- If the first two generation gates (2k steps) show generation WORSE
  than the leg-3 final checkpoint on the 5 probes, stop: the interaction
  is harmful; do not burn the remaining token budget.
- Held-out loss down + generation still 0/5 at combined-leg end =
  seventh masked-loss negative; report as measured. Next step is a
  different family (sampler-side commit policy), NOT another
  training-step variant.
- Cold/dominated diagnostics: if cold batches dominate and denoise loss
  REGRESSES vs leg 3 (held-out FineWeb up >5% at gate 1), drop p_cold to
  0.08; if the anchor KL explodes (>2x leg-2 levels), halve
  lambda_anchor to 0.25.
