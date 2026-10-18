# ASTRA_CLEAN + Coreset-Anchored Streaming — integration brief
Prepared 2026-10-14 ~11:35 IST by the proactivity research run. For the main agent /
`xiiv-daily-model-improvement` cron. Fahad is away until ~2026-10-18; nothing here needs him.

## What exists (all committed locally, push blocked on GitHub re-sign-in)

| File | Lines | Status |
|---|---|---|
| `xiiv/streaming_dedup.py` | 166 | `StreamingDedupFilter.is_duplicate(text, domain)`; `memory_mb()` for budgeting. Unit smoke only |
| `xiiv/quality_filter.py` | 249 | fastText train/score helpers. **No model trained, no calibration data** |
| `xiiv/contamination.py` | 98 | `ContaminationFilter.is_contaminated` (13-gram). **No benchmark sets loaded** |
| `xiiv/domain_balancer.py` | 146 | Micro-DoReMi `DomainBalancer.update/get_domain_for_batch`. Unit smoke only |
| `xiiv/coreset.py` | 171 | `select_coreset` (gradient-based), save/load |
| `xiiv/fisher.py` | 204 | `compute_fisher_diagonal`, `ewc_loss`, save/load |
| `xiiv/streaming_sgbd.py` | 223 | Orchestrator skeleton; `STREAMING_PLAN` of 8 datasets; `python -m xiiv.streaming_sgbd --status` → 0/8 |
| `scripts/build_frontier_data.py` | 255 | Active builder. Per-doc: `_doc_ok` (char limits + in-memory exact-dup `seen`) → tokenize → buffer → flush 128-tok uint16 blocks. `--resume` via block count + `resume_state.json` doc-skip |

Nothing is wired together. The active FineWeb-Edu 10B build (~33% done, PID 13719)
uses NONE of the ASTRA_CLEAN stages.

## Exact integration points in `build_frontier_data.py` (per-doc loop, after `_doc_ok`)

Intended order per his spec: exact dedup → SimHash near-dedup → heuristics →
fastText → decontam → domain balancing → coreset selection.

1. **SimHash** — insert after `_doc_ok` passes: `if dedup.is_duplicate(text, domain=args.source): n_dropped+=1; continue`.
   Caveat: `seen` set is already in-memory-only (resets every reboot); SimHash state
   needs the same persistence treatment or it silently re-admits near-dups after resume.
   Budget with `memory_mb()` — box has ~256Mi available right now.
2. **Heuristics** — extend `_doc_ok` (currently only min/max chars) with structural
   checks; keep it cheap, it runs per doc.
3. **fastText** — `score_documents` is batch-oriented; the loop is per-doc streaming.
   Needs a per-doc predict wrapper or a batching layer. BLOCKED on: training data
   (reviewed positive/negative samples per domain — none collected) and calibration
   (threshold per domain — none measured).
4. **Decontam** — `is_contaminated(text)` per doc after fastText. BLOCKED on: real
   benchmark n-gram sets loaded via `add_benchmark_file` (none loaded). Measure
   per-doc 13-gram cost on this box before enabling (7GB RAM, CPU-only).
5. **DomainBalancer** — does NOT belong in the data builder; it works on
   `(domain, loss)` at batch time → integrate into the SGBD training loop inside
   `streaming_sgbd.py`, not `build_frontier_data.py`.
6. **Coreset + Fisher** — post-training per dataset (need trained blocks + data);
   `select_coreset` is gradient-based → runs after that dataset's SGBD training,
   before bulk discard. Fisher diagonal cost on CPU needs measuring first.

## Do NOT integrate mid-build

FineWeb-Edu is 33% through with the current filter set. Changing filters mid-dataset
splits the corpus into two distributions and invalidates resume comparability.
Earliest integration point: the START of the next dataset (stackv2), after a dry run.

## Dry-run protocol (his ASTRA_CLEAN spec, still pending)

1. Representative sample (e.g. 100K docs) through the full filter chain.
2. Measure precision/recall of each stage against human-reviewed labels.
3. A–F downstream experiments (his spec's controlled experiments) before full-scale.
4. Final measured decision report — no unverified "optimal" claims.
   None of this has been done. Do not present filter quality numbers until measured.

## Approved operational gate (standing)

After each dataset completes streaming+SGBD training: run cross-domain evals on ALL
previously trained domains. If any prior-domain score drops, adjust before moving on.
He approved this ('Ok keep working').

## Inconsistencies to fix before the orchestrator goes live (flagged, not changed)

- **Order mismatch**: `scripts/monitor_downloads.sh` SEQUENCE starts with `fineweb_edu`;
  `xiiv/streaming_sgbd.py` STREAMING_PLAN starts with `stackv2`. The monitor comment
  claims they "must match" — they don't. Current fineweb-first order is correct per his
  "top up existing first" order; align the orchestrator when it goes live.
- **Stale numbers** in `streaming_sgbd.py` docstring + memory-budget comment: "80B+ tokens"
  and "8 × 50M tokens" — current targets are 66B and 200M coresets
  (`DatasetConfig.coreset_tokens` default is already 200M since 707918e).

## Today's box health (verified 11:19–11:35 IST)

- FineWeb-Edu build healthy: PID 13719, CPU-active (~34%), ~3.3B/10B tokens (~33%),
  post-reboot skip-ahead over 11.63M docs. Unauthenticated HF (rate-limit warning in log;
  no token available — his are burn-on-sight; don't ask while he's away).
- Box rebooted 2× today (~07:53, ~10:53 IST); monitor cron resumed both times.
- `monitor_downloads.sh` FIXED today: (1) pgrep now anchors to the python argv[0] —
  previously matched lingering nohup wrapper shells and could false-positive "running"
  after a reboot (verified with a simulated wrapper); (2) `bc` → `awk` for token math —
  `bc` is not installed, so a dead-download with tok= lines in the log would have been
  misread as "complete" and SKIPPED to the next dataset (verified with synthetic log).
- Swap impossible: `swapon` → Operation not permitted (test file created and removed).
  No OOM evidence in current boot's dmesg. RAM is tight (~256Mi available, download at
  2.9GB RSS and growing) — the one-at-a-time policy + 30-min monitor remain the guard.
- Milestones: 133M trained, 0/8 blocks, none reached — watchers silent per schedule.
- GitHub push still blocked (auth); he was already asked twice; pending update queued.
