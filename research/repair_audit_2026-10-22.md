# Repair audit — 2026-10-22 (Fahad: "Whatever failed make it done correct, nothing missed")

Consolidated audit of every failed, blocked, and unfinished item across
XIIV and Lacuna, split by who can unblock it. Status as of 2026-10-22
18:20 IST. Companion: research/scale_plan_5b.md (step-zero repair §1).

## A. Agent-side — fixed or in the repair queue (mine)

1. **Growth-script contract defects** (kv V-half truncation; layers
   27–35 missing contract tensors). Script FIXED + fail-loud assertions
   + regression tests, commit 00e0dba, test_grow 9/9. Remaining:
   re-issue the repaired 1B checkpoint (two-phase rerun from the 500M
   parent) — gated on a RAM window (peak ~1.2GB; box at ~0.5–0.7GB
   available), then strict-load + 36/36 coverage + drift report +
   model-card addendum; old artifact kept, labeled SUPERSEDED.
2. **GLOD payoff gate FAIL** (Invention #1 gate 2) + **CSM routing FAIL**
   (Phase C): shared root cause — refinement operator too weak to fix
   violations, so allocation/routing signals cannot pay. Repair:
   build ONE stronger verifier-guided repair operator (structural
   enforcement during refinement, shared by all arms), pre-register
   the fair combined retest, rerun. No payoff claim until it passes.
3. **Guarantee-MMD pilot FAIL**: novelty holds; mechanism unproven.
   Stronger testbed queued: stable mid-band base, per-input
   accepted/rejected pairs, competent-enough generator.
4. **NCM probes**: mechanism real, naive conditioning fails (probes 1–2
   documented). Fair follow-ups queued in order: masked-canvas
   coarse-head training → end-to-end soft/sampled conditioning →
   multi-code/residual concepts → visible-fraction sweep.
5. **CGMS port + partial-certificate gate**: not yet implemented.
   Queued: implement `partial_certificate_check` in xIiv/contracts.py
   (forbidden-present fail, cap-exceeded fail, required-count
   feasibility, phrase/order only when unavoidable), add CGMS
   three-gate + certificate fourth gate to xIiv/diffusion.py with a
   most-stable fallback, tests.
6. **Selective-looping prototype**: paper verified (arXiv:2610.30066);
   not yet prototyped on XIIV's stack. Queued against one SGBD block
   budget; banked in the 100x ledger only if reproduced.
7. **Stale records**: reconciled 2026-10-22 — Lacuna GOAL.md
   (v0.3.0/192 tests), XIIV training GOAL.md (1B/SGBD/5B state),
   Adaptive GOAL.md (5/8 blocks). XIIV tracked-goal description and
   micro-inventions tracked-goal description still carry superseded
   text in the tracker UI; workspace files are corrected.
8. **XIIV GitHub content sync**: snapshot push 2026-10-22 reached
   remote 4927ac3c (Oct 21–22 research batch). Content-level sync of
   older local commits (683c6e5, e36e413, debaa7c, 76698a9, 1b9c8dd)
   still to be verified file-by-file at final sync; local history is
   canonical on this box.
9. **Lacuna code-side verification**: Day-7 claims (192/192, sdist,
   isolated install, examples) to be re-verified directly in a RAM
   window before the final gate is called closed.

## B. User-only blockers (cannot be generated from the agent side)

1. **Kaggle Secrets HF_TOKEN** — ONE one-time entry (Kaggle →
   Settings → Secrets, read-only HF token for
   elova89/xiiv-sgbd-teacher-cache). Gates: SGBD blocks 5–7 and all
   ~49 Kaggle units of the 5B program. Builder already supports it
   (commit 60f2624, dry-run passed). Fallback remains a per-block
   token paste. This is the single highest-leverage user action.
2. **Lacuna distribution** — Fahad's choice: flip repo public vs
   supply a PyPI token (name `lacuna-llm` free at last check
   2026-10-22). Conflicts with his standing "private long-term" call
   unless he changes it. GitHub push of Lacuna under the standing
   rule also awaits his PAT paste.
3. **Stack v2 HF terms accept** — optional; the 5B plan does not
   depend on it.
4. **Free-compute signups** — optional unlocks priced in the plan:
   Lightning AI (phone), Modal (payment method on file, ~50 T4-h/mo
   estimated), Oracle Always Free (card). Each converts directly
   into training throughput.
5. **Real-hardware validation** (Lacuna README item, llama.cpp
   real-tree compile, real-weight shadow scoring) — needs a GPU box;
   partially coverable via Kaggle once B1 lands.

## C. Environment facts bearing on the queue

- Box RAM: ~0.3–0.7GB available of 7.7GB (hypervisor pressure);
  heavy local jobs (checkpoint re-issue, test suites, local block
  training) run only in RAM windows; guardian gate 2.5GB stands for
  local training launches.
- Disk: ~22GB free (workspace mount) — sufficient for the repaired
  checkpoint (~2GB) alongside the superseded artifact.
- Kaggle CLI reinstalled persistently (uv tool) after the 17:22
  reboot wipe; auth verified live 2026-10-22.
