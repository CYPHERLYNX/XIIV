#!/bin/bash
# XIIV-133M official benchmark run — RESUMABLE split version.
# Why this exists: the all-in-one run_bench_133m.sh only writes its result
# JSONs at the very end of a ~2.5h run, and this box reboots/OOMs often, so
# six launches in a row all died with zero results banked (2026-10-09).
# This driver runs each task separately and SKIPS any task whose output JSON
# already exists, so every reboot loses at most the in-progress task.
# Output contract (unchanged — the comparison-table step reads these two):
#   benchmark_133m_arc.json       <- merged {arc_easy, arc_challenge} results
#   benchmark_133m_hellaswag.json <- hellaswag (200Q) results
cd ~/workspace/ai-tools/corsac
V=.venv/bin/python
W=checkpoints_100m_gn/ckpt_8218.pt
T=weights/tokenizer.json
echo "=== ARC-Easy (full) ==="
[ -s benchmark_133m_arc_easy.json ] || $V scripts/run_benchmarks.py --weights $W --tokenizer $T --tasks arc_easy --device cpu --batch-size 1 --out-json benchmark_133m_arc_easy.json
echo "=== ARC-Challenge (full) ==="
[ -s benchmark_133m_arc_challenge.json ] || $V scripts/run_benchmarks.py --weights $W --tokenizer $T --tasks arc_challenge --device cpu --batch-size 1 --out-json benchmark_133m_arc_challenge.json
if [ -s benchmark_133m_arc_easy.json ] && [ -s benchmark_133m_arc_challenge.json ] && [ ! -s benchmark_133m_arc.json ]; then
  echo "=== merging ARC parts -> benchmark_133m_arc.json ==="
  $V -c "import json; e=json.load(open('benchmark_133m_arc_easy.json')); c=json.load(open('benchmark_133m_arc_challenge.json')); json.dump({**e, **c}, open('benchmark_133m_arc.json','w'), indent=2)"
fi
echo "=== HellaSwag (200) ==="
[ -s benchmark_133m_hellaswag.json ] || $V scripts/run_benchmarks.py --weights $W --tokenizer $T --tasks hellaswag --device cpu --batch-size 1 --limit 200 --out-json benchmark_133m_hellaswag.json
echo "ALL DONE"
