#!/bin/bash
# XIIV frontier data build — sequential pipeline
# Runs each source in order, resumable. Logs to data_frontier/build.log
cd ~/workspace/ai-tools/corsac

run_source() {
    local src=$1 outdir=$2 maxtok=$3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE: starting $src -> $outdir (target ${maxtok} tok)" >> data_frontier/build.log
    .venv/bin/python scripts/build_frontier_data.py \
        --source "$src" \
        --tokenizer weights/tokenizer_32k.json \
        --out-dir "$outdir" \
        --max-tokens "$maxtok" \
        --resume 2>&1 | while IFS= read -r line; do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line" >> data_frontier/build.log
        done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE: finished $src" >> data_frontier/build.log
}

# Priority order per plan
run_source "fineweb-edu" "data_frontier/fineweb_edu" 3000000000
run_source "dclm"        "data_frontier/dclm"        2000000000
run_source "starcoder"   "data_frontier/code"        1000000000
run_source "finemath"    "data_frontier/math"         500000000

echo "[$(date '+%Y-%m-%d %H:%M:%S')] PIPELINE: all sources complete" >> data_frontier/build.log
