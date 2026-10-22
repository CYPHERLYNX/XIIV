# Daily Research — 2026-10-22

## Part A: research

**Run 7** (04:35 +04, window since Oct 21 22:06; full file
`~/workspace/goals/xiiv-133m-training-run/hidden_files/research_scout_2026-10-22_run7.md`):
nothing met reporting criteria — SILENT per discipline. One substantive
watch item: **Recursive Flint** (Oct 21) — Socher's Recursive
Superintelligence released a 40B/2.5B-active Apache-2.0 MoE coding model
(74.2% SWE-bench Verified, company-reported) with the UNVERIFIED vendor
narrative that its "Eureka Machine" automated-research system built "nearly
every part" of the stack — no technical report on the loop, no independent
verification (the same 0%-verified pattern as Harness²'s documented 9
regressions). Third entrant on the verified-self-improvement axis (Prime
Agent, Harness², Eureka Machine); Forge's independent-verifier edge
unchanged. Transformer-lineage, no certificates/charges/substrate — no moat
threat, not adoptable. Prior-art watch CLEAN; Mercury 3 still no tech
report; Axionic still no artifacts. NCM (run 2) remains the top adoptable
diffusion-backbone candidate.

**Dedicated prior-art sweep for the queued pick** (owed from 2026-10-21):
Guarantee-MMD post-training — full file
`~/workspace/goals/xiiv-133m-training-run/hidden_files/research_scout_2026-10-22_mmd_priorart.md`.
**Verdict: HOLDS.** 10 items, all arXiv IDs verified. Every *pair* of the
three elements exists (MMD+diffusion-LM training: Yandex arXiv:2610.06648,
D-MMD arXiv:2603.20155; verifier+diffusion-LM training: Seed Diffusion
arXiv:2508.02193, dInfer arXiv:2510.08666), but the **triple combination —
MMD training objective + diffusion LM + target = verifier-accepted region —
has no prior art.** Open ground for the invention. Honest caveats on file:
StableDRL couldn't be verified to an arXiv ID; keyword searches may miss
non-English/paywalled work.

## Part B: implemented — Guarantee-MMD pilot

Built `research/invention_gmmd_probe.py`: post-train the tiny XIIV with
loss = CE + λ·MMD_RBF(feats(own drafts), accepted-bank), λ ∈ {0, 0.3, 1.0,
3.0}; n=200 paired held-out trials; pre-registered PASS = significant
McNemar win + diversity/CE guardrails.

**Result: PILOT FAIL** (full: `research/invention_gmmd_results.md`).
Rates 0.985 / 0.980 / 0.985 / 0.970, best McNemar p=0.45 — no significant
reduction. The MMD term was mechanically active (feature distance fell with
λ) but didn't move the discrete outcome. Diagnosis: the testbed's dominant
effect was drift — continued training (even the pure-CE control) moved the
pilot base rate 0.63 → ~0.985, because sharpening toward the most-frequent
tokens increases forbidden-token (2nd-most-frequent) violations; the tiny
model also memorized its 10 templates (train CE ~0.6 vs held-out CE ~3.6).
Same bottleneck as Invention #1 gate 2 and CSM Phase C: distribution-level
nudges don't fix generative competence under constraints.

**Standing: novelty HOLDS / mechanism unproven at toy scale.** The idea
stays in the candidate queue behind a stronger testbed (stable base model,
per-input accepted/rejected pairs) — same honest split as Invention #1.
