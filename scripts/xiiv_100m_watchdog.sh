#!/usr/bin/env bash
# XIIV-100M-GN watchdog: relaunch the training loop if it's dead.
# Runs every 10 min via cron. The loop (scripts/run_100m_gn.sh) already
# resumes from the latest checkpoint and restarts train.py on crashes;
# this covers full box reboots, which kill the loop itself.
# Reversible: delete the cron job to stop it. Log: checkpoints_100m_gn/watchdog.log
set -euo pipefail
DIR="$HOME/workspace/ai-tools/corsac"
LOG="$DIR/checkpoints_100m_gn/watchdog.log"

# training process alive for this run? nothing to do.
if pgrep -f "xiiv.train.*checkpoints_100m_gn" >/dev/null 2>&1; then
  exit 0
fi
# launcher loop alive (possibly between restarts)? nothing to do.
if pgrep -f "[r]un_100m_gn.sh" >/dev/null 2>&1; then
  exit 0
fi
# run finished? never relaunch a completed run.
latest=$(ls -t "$DIR/checkpoints_100m_gn"/ckpt_*.pt 2>/dev/null | head -1 || true)
if [ -n "$latest" ]; then
  n=$(basename "$latest" | sed 's/ckpt_\([0-9]*\)\.pt/\1/')
  if [ "$n" -ge 40000 ]; then
    echo "$(date -u +%FT%TZ) watchdog: run complete at step $n, standing down" >> "$LOG"
    exit 0
  fi
fi

echo "$(date -u +%FT%TZ) watchdog: launcher dead, relaunching (latest ckpt: ${latest:-none})" >> "$LOG"
cd "$DIR"
setsid nohup bash scripts/run_100m_gn.sh >> /tmp/run_100m_gn.out 2>&1 < /dev/null &
echo "$(date -u +%FT%TZ) watchdog: relaunched launcher PID $!" >> "$LOG"
