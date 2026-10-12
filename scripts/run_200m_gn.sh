#!/bin/bash
# XIIV-200M growth phase: NOT a from-scratch model. This continues the
# single lineage: the 133M checkpoint is grown (scripts/grow_checkpoint.py)
# to ~204.7M params (d=768, L=18, h=12, ffn=2048, vocab 32768) and training
# continues from the growth step with the same data stream and LR schedule.
#
# Growth event sequence (run once, at the gate verdict):
#   1. .venv/bin/python scripts/expand_vocab.py \
#        --old-tokenizer weights/tokenizer.json --old-data data_all \
#        --new-tokenizer weights/tokenizer_32k.json \
#        --vocab-map weights/vocab_map_8k_32k.json --out-data data_32k
#   2. .venv/bin/python scripts/grow_checkpoint.py \
#        --src-ckpt checkpoints_100m_gn/ckpt_<G>.pt \
#        --out-ckpt checkpoints_200m_gn/ckpt_<G>.pt \
#        --d-new 768 --layers-new 18 --heads-new 12 \
#        --vocab-size-new 32768 --vocab-map weights/vocab_map_8k_32k.json
#      (verifies parent-vs-grown drift before writing)
#   3. launch this loop; it resumes from the grown checkpoint.
#
# Checkpoints every 10 min. Restarts after crashes; exits when TARGET steps
# are done. NOTE: a full BOX reboot still kills this loop -- pair with a
# 10-min watchdog cron for reboot recovery (see scripts/xiiv_100m_watchdog.sh
# for the pattern; add a checkpoints_200m_gn twin at launch).
cd ~/workspace/ai-tools/corsac
TARGET=80000   # lineage continues: 40k steps as 133M + 40k steps as 200M
               # (each phase = 40k x 512 tok = 20.48M tokens)
while true; do
    last=$(ls checkpoints_200m_gn/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$last" ]; then
        step=$(echo "$last" | sed 's/.*ckpt_\([0-9]*\)\.pt/\1/')
        if [ "$step" -ge "$TARGET" ]; then
            echo "XIIV-200M-GN complete at step $step" >> checkpoints_200m_gn/train.log
            break
        fi
    fi
    # LR schedule CONTINUES from the growth step (same 3e-4 peak as the
    # 133M parent): growth approximately preserves the function, so
    # optimization dynamics continue -- no restart, no rescaling.
    .venv/bin/python -m xiiv.train \
        --data data_32k --ckpt-dir checkpoints_200m_gn \
        --tokenizer weights/tokenizer_32k.json \
        --d-model 768 --n-layers 18 --n-heads 12 --ffn-dim 2048 \
        --lr 3e-4 --lr-min 3e-5 --warmup-steps 500 --total-steps $TARGET \
        --lock-prob 0.4 \
        --contract-enabled --contract-prob 0.4 \
        --contract-adversarial-frac 0.3 --contract-max-tokens 96 \
        --val-every 500 --val-batches 8 --log-every 25 \
        --ckpt-every-min 10 \
        --batch-size 4 --threads 2 \
        >> checkpoints_200m_gn/train.log 2>&1
    echo "train exited (code $?) at $(date); pruning to 10 newest ckpts; restarting in 60s" >> checkpoints_200m_gn/train.log
    ls checkpoints_200m_gn/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | head -n -10 | xargs -r rm -f
    sleep 60
done
