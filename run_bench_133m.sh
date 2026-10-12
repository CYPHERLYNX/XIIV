#!/bin/bash
# XIIV-133M official benchmark run: ARC-Easy + ARC-Challenge (full), HellaSwag (200).
# Launched in background; logs to /tmp/bench_133m.log, results to benchmark_133m_*.json.
cd ~/workspace/ai-tools/corsac
V=.venv/bin/python
echo "=== ARC-Easy + ARC-Challenge (full) ==="
$V scripts/run_benchmarks.py --weights checkpoints_100m_gn/ckpt_8218.pt --tokenizer weights/tokenizer.json --tasks arc_easy,arc_challenge --device cpu --batch-size 1 --out-json benchmark_133m_arc.json
echo "=== HellaSwag (200) ==="
$V scripts/run_benchmarks.py --weights checkpoints_100m_gn/ckpt_8218.pt --tokenizer weights/tokenizer.json --tasks hellaswag --device cpu --batch-size 1 --limit 200 --out-json benchmark_133m_hellaswag.json
echo "ALL DONE"
