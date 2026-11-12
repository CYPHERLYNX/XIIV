# 1B BLITZ: Nov 5 → Nov 6 (one week — Fahad's correction 2026-10-30)

**Correction:** the December deadline is for the 4B. The 1B has ONE
MORE WEEK (Nov 6). This doc supersedes the Nov 5 → Dec 10 campaign
schedule for the 1B. The 4B keeps its Dec-24 spec
(research/expansion_4b_moe.md) as side work.

**Objective:** the strongest honest 1B achievable in a single quota
week, delivered Nov 6 for Fahad's verdict.

**Budget:** one Kaggle weekly window ≈ **30 T4-hours** (quota resets
~Nov 4-5). Measured throughput 1.475M tok/h → **~44M tokens** max —
~2.5x the current 17.7M.

**Standing rules:** one lever at a time; every leg gated on the 5
probes + 8 generation gates; kill criteria per leg; no cherry-picking.

## Leg 2 — BASC (Nov 5, ~12 h)
- Staged: `research/lever2/` + `RUNBOOK.md`. Fires the instant quota resets.
- Continues FineWeb-Edu from the leg-1 final checkpoint (step 8000),
  next 25M tokens sequential.
- **Kill:** first two gates worse than leg-1 final on 5 probes → stop,
  pivot immediately to CSC.
- **Gate triggered** → full harness in-kernel, package, report.

## Leg 3 — CSC or BASC-continues (Nov 5–6, ~12 h)
- If BASC moved generation: BASC continues, deeper.
- If BASC failed (0/5 at leg-2 end): CSC leg per
  `research/lever3/RUNBOOK.md` (p_cold=0.15) — the backup for the
  other failure mode, already staged.
- **Kill:** same two-gate rule vs leg-2 final.

## Nov 6 — deliver
- Best checkpoint → full 100-prompt harness (GPU, in-kernel).
- Measured scoreboard; model card with MEASURED vs UNMEASURED.
- Delivery package to Fahad (same opencode-zip pipeline).

## Hour ledger (30 h)
| Use | Hours |
|---|---|
| Leg 2 (BASC) | ~12 |
| Leg 3 (CSC or BASC-deep) | ~12 |
| Final harness + package | ~4 |
| Buffer (push/retry) | ~2 |

## Honest ceiling (one week)
- 44M tokens cannot make a 1B factually rival 100B-class frontier.
  The week's winnable fights: (1) generation gates move off 0/5 —
  the model starts TALKING; (2) best measured 1B-class standing on
  the same rig; (3) the guarantee-native axis, where frontier = 0.
- If NEITHER lever moves generation by Nov 6, the 1B verdict is
  delivered as measured-negative and the 4B plan re-scopes around
  whatever the week's data actually showed. No faked wins.
