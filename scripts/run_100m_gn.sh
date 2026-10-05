#!/bin/bash
# XIIV-100M guarantee-native: from-scratch contract-conditioned training.
# 100.41M params (d=704, L=15, h=11, ffn=2048; tied embeddings) + the
# contract-native architecture (contract encoder + per-block cross-attention).
# --contract-prob 0.4: 40% of batches train P(target | noisy, contract);
# --lock-prob 0.4 keeps the constraint-aware locked-span objective.
# Checkpoints every 10 min (the 4h interval is what the 13:32 IST reboot
# exploited to erase the entire dead run). Restarts after crashes; exits
# when TARGET steps are done. NOTE: a full BOX reboot still kills this
# loop -- pair with the 10-min watchdog cron for reboot recovery.
cd ~/workspace/ai-tools/corsac
TARGET=40000   # 40000 x 512 tok = 20.48M tokens (same budget as the 52.5M base run)
while true; do
    last=$(ls checkpoints_100m_gn/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$last" ]; then
        step=$(echo "$last" | sed 's/.*ckpt_\([0-9]*\)\.pt/\1/')
        if [ "$step" -ge "$TARGET" ]; then
            echo "XIIV-100M-GN complete at step $step" >> checkpoints_100m_gn/train.log
            break
        fi
    fi
    .venv/bin/python -m xiiv.train \
        --data data_all --ckpt-dir checkpoints_100m_gn \
        --tokenizer weights/tokenizer.json \
        --d-model 704 --n-layers 15 --n-heads 11 --ffn-dim 2048 \
        --lr 3e-4 --lr-min 3e-5 --warmup-steps 500 --total-steps $TARGET \
        --lock-prob 0.4 \
        --contract-enabled --contract-prob 0.4 \
        --contract-adversarial-frac 0.3 --contract-max-tokens 96 \
        --val-every 500 --val-batches 8 --log-every 25 \
        --ckpt-every-min 10 \
        --batch-size 4 --threads 2 \
        >> checkpoints_100m_gn/train.log 2>&1
    echo "train exited (code $?) at $(date); pruning to 10 newest ckpts; restarting in 60s" >> checkpoints_100m_gn/train.log
    ls checkpoints_100m_gn/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | head -n -10 | xargs -r rm -f
    sleep 60
done
