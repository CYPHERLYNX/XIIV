#!/bin/bash
# Overnight pilot: constraint-aware fine-tune of XIIV-52.5M with locked spans.
# Restarts training after crashes/reboots; exits when 400 steps are done.
cd ~/workspace/ai-tools/corsac
TARGET=400
while true; do
    last=$(ls checkpoints_locked/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$last" ]; then
        step=$(echo "$last" | sed 's/.*ckpt_\([0-9]*\)\.pt/\1/')
        if [ "$step" -ge "$TARGET" ]; then
            echo "pilot complete at step $step" >> checkpoints_locked/pilot.log
            break
        fi
    fi
    .venv/bin/python -m xiiv.train \
        --data data --ckpt-dir checkpoints_locked \
        --d-model 576 --n-layers 12 --n-heads 9 --ffn-dim 1536 \
        --lr 1e-4 --lr-min 1e-5 --warmup-steps 50 --total-steps $TARGET \
        --lock-prob 0.4 --init-ckpt checkpoints_50m/best.pt \
        --val-every 200 --val-batches 8 --log-every 25 \
        --batch-size 8 --threads 2 \
        >> checkpoints_locked/pilot.log 2>&1
    echo "train exited (code $?) at $(date); restarting in 60s" >> checkpoints_locked/pilot.log
    sleep 60
done
