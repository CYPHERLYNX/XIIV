#!/bin/bash
# Book-prose fine-tune: adult writing + locked spans.
# Inits from the locked-span pilot (keeps constraint-aware skill),
# trains on Gutenberg books with lock_prob=0.4.
# Restarts after crashes; exits when TARGET steps are done.
cd ~/workspace/ai-tools/corsac
TARGET=2000
while true; do
    last=$(ls checkpoints_books/ckpt_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$last" ]; then
        step=$(echo "$last" | sed 's/.*ckpt_\([0-9]*\)\.pt/\1/')
        if [ "$step" -ge "$TARGET" ]; then
            echo "books fine-tune complete at step $step" >> checkpoints_books/books.log
            break
        fi
    fi
    .venv/bin/python -m xiiv.train \
        --data data_books --ckpt-dir checkpoints_books \
        --d-model 576 --n-layers 12 --n-heads 9 --ffn-dim 1536 \
        --lr 1e-4 --lr-min 1e-5 --warmup-steps 50 --total-steps $TARGET \
        --lock-prob 0.4 --init-ckpt checkpoints_locked/ckpt_400.pt \
        --val-every 500 --val-batches 8 --log-every 25 \
        --batch-size 8 --threads 2 \
        >> checkpoints_books/books.log 2>&1
    echo "train exited (code $?) at $(date); restarting in 60s" >> checkpoints_books/books.log
    sleep 60
done
