#!/bin/bash
# XIIV 24/7 Download Monitor
# Checks if the current dataset download is running, restarts if dead.
# Also handles box reboots by resuming the correct dataset in sequence.

CORSAC_DIR="$HOME/workspace/ai-tools/corsac"
LOG_DIR="/tmp"

# Dataset sequence (must match streaming_sgbd.py order)
# Format: "name:source:out_dir:max_tokens:log_file"
# Reasoning (openthoughts) prioritized: it's the ASI driver
SEQUENCE=(
  "fineweb_edu:fineweb-edu:data_frontier/fineweb_edu:10000000000:fineweb_build.log"
  "openthoughts:openthoughts:data_frontier/reasoning:1200000000:openthoughts_build.log"
  # stackv2 DEFERRED 2026-10-15: bigcode/the-stack-v2 is a gated HF dataset
  # requiring accepted terms + authenticated token (5th crash). Even with auth,
  # the content mapping is broken (SW heritage IDs, no content field).
  # Revisit with working auth + fixed retrieval; queue continues without it.
  # "stackv2:stackv2:data_frontier/code:10000000000:stackv2_build.log"
  "papers:pes2o:data_frontier/papers:10000000000:papers_build.log"
  # megamath DEFERRED 2026-10-16: LLM360/MegaMath was restructured upstream —
  # the 'text' column no longer exists (schema is now file_info/repo_info metadata
  # only), so the build crashed with CastError and wrote 0 tokens. Defer until a
  # usable math source/config is found; queue continues without it.
  # "megamath:megamath:data_frontier/megamath:10000000000:megamath_build.log"
  "dclm:dclm:data_frontier/dclm:10000000000:dclm_build.log"
  "stackexchange:stackexchange:data_frontier/qa:10000000000:stackexchange_build.log"
  "pg19:pg19:data_frontier/books:5000000000:pg19_build.log"
)

# Datasets whose corpus is fully consumed even though bin tokens sit below the
# sequence target (stream ended naturally). A DONE marker on one of these means
# "finished", never "top up". A DONE marker on any other dataset means finished
# only if bin tokens reached the sequence target (dclm: DONE at 1.99B of 10B
# from an older 2B-target run -> top-up is legitimate, corpus far from empty).
# A DONE marker with 0 bin tokens is a crash, never completion.
EXHAUSTED_NAMES=("fineweb_edu" "papers" "stackexchange" "pg19")
# stackexchange added 2026-10-17: two consecutive DONE markers (01:43:28 UTC
# dropped=7043, 02:37:06 UTC dropped=0) over the identical 6,331,083 docs --
# stream fully consumed at 2,131,150,720 tokens (21.3% of 10B target). Without
# this, the top-up rule relaunched the pass on every monitor run.
# pg19 added 2026-10-17: two consecutive DONE markers (03:36:58 UTC dropped=23545,
# 04:29:39 UTC dropped=0) over the identical 28,602 docs -- emozilla/pg19 mirror
# (23 parquet shards) fully consumed at 75,220,608 tokens (1.5% of 5B target).

is_exhausted() {
  local n="$1"
  for e in "${EXHAUSTED_NAMES[@]}"; do
    [ "$e" = "$n" ] && return 0
  done
  return 1
}

# Token ground truth for an out_dir. train.bin is primary, but a completed
# bin may be archived/deleted after DONE (disk management); the script writes
# meta.json (train_tokens) at completion and that record survives. Fall back
# to meta.json only when the bin is absent or empty, so a genuinely crashed
# run (no DONE marker, no meta) still reads 0.
dataset_tokens() {
  local dir="$1"
  local bin="$dir/train.bin"
  local tok=0
  if [ -f "$bin" ]; then
    tok=$(stat -c%s "$bin" 2>/dev/null | awk '{printf "%d", $1/2}')
    [ -z "$tok" ] && tok=0
  fi
  if [ "$tok" -eq 0 ] && [ -f "$dir/meta.json" ]; then
    meta_tok=$(python3 -c "import json,sys; print(json.load(open('$dir/meta.json')).get('train_tokens',0))" 2>/dev/null)
    [ -n "$meta_tok" ] && tok="$meta_tok"
  fi
  printf "%s" "$tok"
}

# Check if any build is running.
# NOTE: anchored to argv[0] (the python binary). A plain `pgrep -f "build_frontier_data"`
# also matches shell wrappers whose -c string merely mentions the script (e.g. the
# nohup launch wrapper, which lingers after starting the real process) and would make
# the monitor wrongly conclude a download is running after a reboot killed it.
if pgrep -f "^[^ ]*python[0-9.]* scripts/build_frontier_data\.py" > /dev/null; then
  echo "[monitor] Download running, OK"
  exit 0
fi

echo "[monitor] No download running, finding next dataset..."

# Find the first dataset that hasn't reached its target
# We check by looking at the log for completion or by checking token count
cd "$CORSAC_DIR" || exit 1

for entry in "${SEQUENCE[@]}"; do
  IFS=':' read -r name source out_dir max_tokens log_file <<< "$entry"
  log_path="$LOG_DIR/$log_file"
  
  # Check if this dataset's log shows completion
  # For now, simple heuristic: if log doesn't exist or doesn't show "complete",
  # and the out_dir has less than expected, this is our target
  
  # Completion ground truth: the persistent script log + train.bin tokens.
  # The /tmp log is wiped on every reboot and must NOT be trusted alone.
  persist_log="$CORSAC_DIR/data_frontier/build.log"
  target_tok=$max_tokens

  # 1) Persistent log DONE marker — completion ground truth, with two guards:
  #    (a) a DONE marker with 0 bin tokens is a crash, never completion
  #        (2026-10-16 megamath: CastError, 0 tokens) — do not skip;
  #    (b) a DONE marker below the sequence target is completion only for
  #        corpus-exhausted datasets (EXHAUSTED_NAMES) — others get a top-up
  #        (2026-10-16 dclm: DONE at 1.99B of 10B target, corpus has far more).
  if [ -f "$persist_log" ] && grep -q "DONE $source:" "$persist_log"; then
    # dataset_tokens(): train.bin, falling back to meta.json train_tokens
    # (a completed bin may be archived/deleted; the meta record survives).
    bin_tok=$(dataset_tokens "$CORSAC_DIR/$out_dir")
    if is_exhausted "$name"; then
      echo "[monitor] $name complete (corpus exhausted, DONE in log), skipping"
      continue
    fi
    if [ "$bin_tok" -ge "$target_tok" ]; then
      echo "[monitor] $name complete (DONE in log, bin $bin_tok >= $target_tok), skipping"
      continue
    fi
    # A DONE marker at >=99% of target is effectively complete: a top-up resume
    # exits instantly (train+valid already >= target), so re-launching would just
    # spin the queue forever (2026-10-16 openthoughts: 1195.2M of 1200M).
    pct=$(( bin_tok * 100 / target_tok ))
    if [ "$pct" -ge 99 ]; then
      echo "[monitor] $name complete (DONE in log, bin $bin_tok = $pct% of target), skipping"
      continue
    fi
    if [ "$bin_tok" -gt 0 ]; then
      echo "[monitor] $name DONE below target ($bin_tok < $target_tok), top-up run"
    else
      echo "[monitor] $name DONE marker but 0 bin tokens (crash), retrying"
    fi
  fi

  # 2) train.bin tokens already at/above target
  bin="$CORSAC_DIR/$out_dir/train.bin"
  if [ -f "$bin" ]; then
    bin_tok=$(stat -c%s "$bin" 2>/dev/null | awk '{printf "%d", $1/2}')
    if [ -n "$bin_tok" ] && [ "$bin_tok" -ge "$target_tok" ]; then
      echo "[monitor] $name complete (bin tokens $bin_tok >= $target_tok), skipping"
      continue
    fi
  fi

  # 3) Legacy /tmp log heuristic (unreliable after reboot; fallback only)
  current_tok=0
  if [ -f "$log_path" ]; then
    last_line=$(grep -o "tok=[0-9.]*M" "$log_path" | tail -1)
    if [ -n "$last_line" ]; then
      tok_m=$(echo "$last_line" | grep -o "[0-9.]*")
      current_tok=$(echo "$tok_m" | awk '{printf "%d", $1 * 1000000}' 2>/dev/null)
      [ -z "$current_tok" ] && current_tok=0
    fi
  fi
  
  if [ "$current_tok" -lt "$target_tok" ]; then
    echo "[monitor] Resuming $name ($current_tok / $target_tok tokens)"
    
    # Build the command
    cmd=".venv/bin/python scripts/build_frontier_data.py --source $source --tokenizer weights/tokenizer_32k.json --out-dir $out_dir --max-tokens $max_tokens --resume"
    
    # Stack v2 needs HF token
    if [ "$name" = "stackv2" ]; then
      # Token is in environment or skip
      echo "[monitor] Stack v2 needs HF_TOKEN, checking..."
    fi
    
    nohup $cmd > "$log_path" 2>&1 < /dev/null &
    echo "[monitor] Started $name (PID $!)"
    exit 0
  else
    echo "[monitor] $name complete ($current_tok >= $target_tok), skipping"
  fi
done

echo "[monitor] All datasets complete!"
