#!/usr/bin/env python3
"""SGBD training guardian — keeps the 8-block full training run moving on the 7GB box.

Standing context (2026-10-16): the user has NO local system available, so ALL
training happens on this box and SGBD block-by-block training is the primary
(and only) training path. This guardian is invoked by the xiiv-download-monitor
cron after its existing checks and enforces the resource rules:

1. NEVER start while the HF archive upload (archive_to_hf.py) is running —
   it holds ~3.4GB of the 7GB and a concurrent block run would OOM-kill both.
2. NEVER start while another sgbd_train_block.py is running (one block at a time).
3. Require MemAvailable >= 2.5GB, else skip quietly and try next run.
4. Find the first block b in 0..7 without block{b}_final.pt under sgbd_blocks/block{b}/.
5. Resume from the highest existing block{b}_step{N}.pt via --resume-step N (0 if none).
6. Target: 1000 steps per block (matches the last block0 run's target).
7. Launch detached with nohup; log to sgbd_guardian_block{b}.log.
8. Update sgbd_milestones.json block count (idempotent) every run so the
   hourly milestone watcher can notify at each 100M params trained.

Usage: sgbd_guardian.py [--dry-run]
Exit 0 = started a block, or nothing to do / preconditions unmet (both logged).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/hatch/workspace/ai-tools/corsac")
OUT_DIR = REPO / "sgbd_blocks"
MILESTONES = REPO / "sgbd_milestones.json"
GUARDIAN_LOG = REPO / "sgbd_guardian.log"
N_BLOCKS = 8
TARGET_STEPS = 1000
MIN_MEM_AVAILABLE_GB = 2.5

TRAIN_SCRIPT = REPO / "xiiv" / "sgbd_train_block.py"
VENV_PY = REPO / ".venv" / "bin" / "python"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(GUARDIAN_LOG, "a") as f:
        f.write(line + "\n")


def pgrep_count(pattern):
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        ).stdout.strip()
        return len([p for p in out.splitlines() if p.strip()])
    except Exception:
        return 0


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def block_final_exists(b):
    return (OUT_DIR / f"block{b}" / f"block{b}_final.pt").exists()


def highest_step_checkpoint(b):
    """Highest N such that sgbd_blocks/block{b}/block{b}_step{N}.pt exists."""
    d = OUT_DIR / f"block{b}"
    best = 0
    if d.is_dir():
        for p in d.glob(f"block{b}_step*.pt"):
            m = re.fullmatch(rf"block{b}_step(\d+)\.pt", p.name)
            if m:
                best = max(best, int(m.group(1)))
    return best


def update_milestones():
    done = sum(1 for b in range(N_BLOCKS) if block_final_exists(b))
    try:
        data = json.loads(MILESTONES.read_text())
    except Exception as e:
        log(f"milestones read failed: {e}; skipping update")
        return
    sgbd = data.setdefault("sgbd_blocks", {})
    if sgbd.get("blocks_completed") != done:
        sgbd["blocks_completed"] = done
        # Recompute the trained-params estimate from the block count, using the
        # file's own fields: estimate = initially_trained + done * params_per_block.
        # (2026-10-18 fix: update_milestones() bumped blocks_completed but never
        # recomputed the estimate, so the 300M milestone was almost missed.)
        seed = data.get("initially_trained", 133_000_000)
        per_block = sgbd.get("params_per_block", 97_900_000)
        data["current_trained_estimate"] = int(seed + done * per_block)
        data["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S+05:30")
        MILESTONES.write_text(json.dumps(data, indent=2))
        log(f"milestones updated: blocks_completed={done}/{N_BLOCKS}, "
            f"current_trained_estimate={data['current_trained_estimate']:,}")


def main():
    dry = "--dry-run" in sys.argv
    update_milestones()

    if pgrep_count("archive_to_hf\\.py") > 0:
        log("SKIP: HF archive upload is running (holds RAM); will train after it finishes")
        return 0
    if pgrep_count("sgbd_train_block\\.py") > 0:
        log("SKIP: a block training run is already in progress")
        return 0

    mem = mem_available_gb()
    if mem < MIN_MEM_AVAILABLE_GB:
        log(f"SKIP: only {mem:.2f}GB RAM available (need {MIN_MEM_AVAILABLE_GB}GB)")
        return 0

    target = next((b for b in range(N_BLOCKS) if not block_final_exists(b)), None)
    if target is None:
        log("DONE: all 8 blocks have final checkpoints — full SGBD run complete")
        return 0

    resume = highest_step_checkpoint(target)
    if resume >= TARGET_STEPS:
        log(f"block{target}: checkpoint at step {resume} >= target {TARGET_STEPS} "
            f"but no final.pt — re-running to regenerate final (resume from {resume})")

    cmd = [
        str(VENV_PY), str(TRAIN_SCRIPT),
        "--block", str(target),
        "--steps", str(TARGET_STEPS),
        "--resume-step", str(resume),
    ]
    block_log = REPO / f"sgbd_guardian_block{target}.log"
    log(f"START block{target}: resume_step={resume}, target={TARGET_STEPS} steps, "
        f"log={block_log.name} (dry_run={dry})")
    if dry:
        log(f"would run: {' '.join(cmd)}")
        return 0

    with open(block_log, "a") as lf:
        lf.write(f"\n===== guardian launch {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                 f"block{target} resume_step={resume} =====\n")
        lf.flush()
        subprocess.Popen(
            ["nohup"] + cmd,
            stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(REPO), start_new_session=True,
            stdin=subprocess.DEVNULL,
        )
    log(f"block{target} launched detached (PID unknown, check block log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
