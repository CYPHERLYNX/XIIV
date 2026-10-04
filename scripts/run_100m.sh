#!/bin/bash
# XIIV-100M: from-scratch training on the full corpus (stories + books).
# 100.41M params (d=704, L=15, h=11, ffn=2048; tied embeddings),
# constraint-native from step 0 via --lock-prob 0.4.
# Batch 4: RAM safety on the 7GB box.
# Restarts after crashes; exits when TARGET steps are done.
cd ~/workspace/ai-tools/corsac
TARGET=40000   # 40000 x 512 tok = 20.48M tokens (same budget as the 52.5M base run)
while true; do
    last=$(ls checkpoints_100m/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$last" ]; then
        step=$(echo "$last" | sed 's/.*ckpt_\([0-9]*\)\.pt/\1/')
        if [ "$step" -ge "$TARGET" ]; then
            echo "XIIV-100M complete at step $step" >> checkpoints_100m/train.log
            break
        fi
    fi
    .venv/bin/python -m xiiv.train \
        --data data_all --ckpt-dir checkpoints_100m \
        --d-model 704 --n-layers 15 --n-heads 11 --ffn-dim 2048 \
        --lr 3e-4 --lr-min 3e-5 --warmup-steps 500 --total-steps $TARGET \
        --lock-prob 0.4 \
        --val-every 500 --val-batches 8 --log-every 25 \
        --ckpt-every-min 240 \
        --batch-size 4 --threads 2 \
        >> checkpoints_100m/train.log 2>&1
    echo "train exited (code $?) at $(date); pruning to 10 newest ckpts; restarting in 60s" >> checkpoints_100m/train.log
    ls checkpoints_100m/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | head -n -10 | xargs -r rm -f
    sleep 60
done
