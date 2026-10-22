# XIIV Scale Plan — 1B → 5B (the active program)

**Status: ACTIVE PLAN** — confirmed by Fahad 2026-10-22 18:00 IST
("800B is ultimate goal, I mean for now to 5B"), with the training-level
clause of 18:02 IST ("increase its training as powerful level to 5B
too") and the free-compute directive of 18:03 IST ("use free resources
like Collab Kaggle other etc to increase every thing to peak").

- **Current program target: XIIV-5B** — a sparse/MoE 5B grown from the
  trained XIIV-1B, trained by the phased 5B-level training program in §4
  on the free-compute fleet in §5.
- **Ultimate destination: the 800B+ class** (standing directive
  2026-10-21). It is named here only as where the sparse shape leads:
  the 5B's expert-streaming architecture is the 800B architecture at
  smaller expert count. No 800B work is scheduled by this plan.
- This plan follows `docs/SCALING_PROTOCOL.md` throughout: frozen
  configs, inverse-width LR rule, identical eval gates per rung, a
  model card per rung, promotion only on val NLL **and** intervention
  rate improving, no cherry-picking. Token budgets are honestly labeled
  against Chinchilla references, never presented as compute-optimal
  when they are not.

**Number labels used everywhere below:** `MEASURED` = produced by a
run or audit on our stack · `ESTIMATED` = arithmetic from measured
anchors · `TARGET` = a planned figure to be replaced by measurement ·
`SOURCED` = an external quota/price from a cited public source.

---

## 0. Ground truth today (2026-10-22, all audited)

| Fact | Value | Label |
|---|---|---|
| Model | XIIV-1B, masked discrete-diffusion LM, d=1280, L=36, h=20, ffn=3584, vocab 32768, ctx 128 | MEASURED (checkpoint config) |
| Exact unique trainable params | **915,988,480** (serialized 957,931,520; tied embed counted once) | MEASURED (safetensors header audit) |
| Trained so far | **622.5M** — 133M seed + SGBD blocks 0–4 of 8 (1,000 steps each) | MEASURED (`sgbd_milestones.json`) |
| Remaining for the 1B | SGBD blocks 5, 6, 7 | — |
| Corpus on hand | **27,112,760,576** unique train tokens (§4.1) | MEASURED (repo manifests) |
| This box | ~7.7 GB RAM (frequently <0.5 GB available), 22 GB disk free, 2 cores | MEASURED |
| Proven training lane | SGBD block training (~115M-param units, ~2 GB/unit) on Kaggle T4 | MEASURED (blocks 3, 4) |

---

## 1. Step zero — repair the contract-attention mismatch (pre-5B, blocking)

No growth, distillation, or 5B work starts until this is repaired and
re-verified. The guarantee-native thesis is the product; scaling a
checkpoint whose contract path does not match the code would scale a
defect.

### 1.1 What the audit found (MEASURED, 2026-10-22)

Tensor-level audit of `checkpoints_1b_gn/ckpt_8218.safetensors`
(406 tensors) against current `xiiv/model.py`:

1. **Layers 0–26** carry contract cross-attention, but with
   `contract_attn.kv.weight` shape **[1280, 1280]**. Current code
   builds `kv = nn.Linear(d, 2d)` → **[2560, 1280]**. The checkpoint's
   kv can produce only one d-wide projection, not the K and V pair the
   code splits from a 2d output.
2. **Layers 27–35** (the 9 depth-grown identity layers) have **no
   contract-attention tensors at all** — no `contract_attn`, no
   `contract_norm`. Under current code with `contract_enabled=True`
   they should exist (zero-initialized identity values).
3. Exact size of the discrepancy: current-code instantiation of the
   1B config counts **1,019,230,720** unique params (meta-device,
   §2.1) vs the artifact's **915,988,480** — a delta of
   **103,242,240**, decomposed exactly as 27 layers × 1,638,400
   (the missing V half of kv on layers 0–26) + 9 layers × 6,556,160
   (the entire contract block on layers 27–35). The decomposition
   accounts for the delta to the parameter.

### 1.2 Root cause (MEASURED — code inspection, 2026-10-22)

The 1B was assembled by `scripts/grow_1b_streaming.py`, not by
`scripts/grow_checkpoint.py` (whose `_widen_kv` handles the 2d row dim
correctly). Two defects in the streaming script:

1. **kv truncation.** Its contract-widening loop applies the generic
   `_widen_rows(w, g)` to `kv.weight`, where `g` is a row map of
   length d_new (1280) built for d-sized tensors. Applied to the
   parent's [2048, 1024] kv (d_old=1024), the map silently keeps only
   1280 rows — the K half survives (row-replicated), **the V half is
   dropped**. No shape assertion caught it.
2. **Missing contract blocks on grown layers.** For new layers the
   script writes zeroed contract tensors only if `has_ca` — a check
   for `layers.0.contract_attn` keys in the working state dict —
   evaluates true. In the phase-2 run (layers 18–35) it evidently
   evaluated false (phase 1 had consumed those keys), so layers 27–35
   were written with no contract tensors. The intermediate artifact
   `ckpt_1b_phase1.pt` (contract on layers 0–17 only, kv already
   [1280, 1280]) corroborates the phased-assembly story.

Nothing is unrecoverable: the 500M parent's kv is intact at
[2048, 1024] (verified in `ckpt_1b_parent_needed.pt`), so the V half
can be re-widened exactly from the parent.

### 1.3 Repair path (R-A, adopted)

1. Fix `grow_1b_streaming.py`: widen kv by splitting it into K and V
   halves, widening each with the d-map, and restacking
   (→ [2560, 1280]) — the same operator as `grow_checkpoint.py`'s
   `_widen_kv`. Add shape assertions for every contract tensor
   (fail loud, never truncate silently). Add a regression test in
   `tests/test_grow.py` covering a [2d, d] tensor through the
   streaming widening path.
2. Re-issue the 1B checkpoint as a **repaired artifact**: parent
   layers' kv rebuilt by the corrected widening from the 500M parent;
   layers 27–35 receive the exact-identity contract tensors the script
   intended (zero q/kv/out, zero gate, ones `contract_norm`). New
   file hash, new model-card addendum recording the repair and the
   superseded artifact. The truncated artifact is kept, labeled
   SUPERSEDED, never silently overwritten.
3. Re-verify before anything else proceeds (§7, Step-0 gates):
   strict load under current code, contract coverage 36/36 layers,
   drift of the repaired model vs the truncated artifact on fixed
   inputs **reported** (nonzero drift is expected — the V path
   activates; parent gates are small, max |gate| ≈ 0.03 as recorded
   in `xiiv/sgbd_train_block.py`, and grown-layer gates are 0).
4. Resolve the SGBD exclusion honestly: `sgbd_train_block.py`
   currently excludes contract attention from block distillation
   (teacher capture ran with `contract_embeds=None`). That stays
   true for the 1B's remaining blocks (frozen protocol, comparability
   with blocks 0–4), and is recorded as a known limitation of the 1B
   rung: **the contract path of the 1B is grown, not SGBD-trained.**
   For the 5B, §4 makes contract-conditioned capture and training a
   T2 design requirement, with a gate-movement gate (§7) — the 5B
   may not claim a trained contract path it does not have.

**Rejected alternative (R-B):** change `model.py` to a K-only contract
attention matching [1280, 1280]. It invents new semantics, discards
the parent's trained V weights, and still leaves layers 27–35
without tensors. Recorded here so the decision is auditable.

---

## 2. Verified parameter counts

### 2.1 Dense ladder — exact meta-device counts (MEASURED)

Method: `XIIVModel(ModelConfig(...))` instantiated on the torch
`meta` device (no allocation), counted with
`count_parameters()`; `ModelConfig.param_count_estimate()` agrees to
the parameter in every row. All rows: vocab 32768, head_dim 64,
contract path included at current-code shapes (contract encoder
256/2/4/512, contract cross-attention on every layer).

| Rung | d | L | heads | ffn | Exact unique params | Peak LR (inverse-width from 3e-4 @ d=704) |
|---|---|---|---|---|---|---|
| 1B (current code) | 1280 | 36 | 20 | 3584 | **1,019,230,720** | 1.65e-4 |
| D1 | 1536 | 42 | 24 | 4301 | **1,685,569,280** | 1.38e-4 |
| D2 | 1792 | 44 | 28 | 5018 | **2,386,301,952** | 1.18e-4 |
| D3 | 2048 | 48 | 32 | 5734 | **3,379,129,600** | 1.03e-4 |
| D4 ("dense 5B") | 2624 | 44 | 41 | 7347 | **5,065,000,512** | 8.05e-5 |

(The artifact 1B is 915,988,480 until the §1 repair lands; the
repaired 1B is the 1,019,230,720 row. Headline naming stays XIIV-1B
either way.)

### 2.2 Sparse/MoE option — exact counts from audited checkpoint shapes (MEASURED arithmetic)

Construction: keep the d=1280 / L=36 core; replace each layer's
single SwiGLU MLP (3·d·ffn = 13,762,560 params; ×36 layers =
495,452,160) with E expert MLPs of the same shape. Non-MLP total
(checkpoint shapes) = 420,536,320.

| Variant | Total params | Active, top-1 | Active, top-2 |
|---|---|---|---|
| E=8, checkpoint shapes | **4,384,153,600** | 915,988,480 | 1,411,440,640 |
| E=10, checkpoint shapes | **5,375,057,920** | 915,988,480 | 1,411,440,640 |
| E=10, repaired base (§1 applied first) | **5,478,300,160** | 1,019,230,720 | 1,514,682,880 |

Excluded above: a router (Linear d→E per layer) adds 460,800 params
at E=10 — counted in the rung's model card when the router design is
frozen, immaterial to every decision here. The repaired-base row is
the operative one: **the 5B is built on the repaired 1B**, so plan
figures use total ≈ 5.478B, active ≈ 1.02B (top-1) / 1.51B (top-2).
Checkpoint-shape figures are kept for traceability to the audit.

### 2.3 The dense-5B rejection (arithmetic, ESTIMATED from MEASURED counts)

| Quantity | D4 dense 5.065B | Sparse E10 (repaired) |
|---|---|---|
| FP32 checkpoint (unique) | 20.26 GB | 21.91 GB |
| BF16/FP16 checkpoint | 10.13 GB | 10.96 GB |
| Full AdamW train state (w+g+m+v, FP32, 16 B/param) | **81.0 GB** | n/a — trained block/expert-wise (§4) |
| Chinchilla reference tokens (20/param; sparse: per *active* param) | 101.3B | 20.4B (top-1) / 30.3B (top-2) |
| Full-training FLOPs at that reference (6·N·D) | 3.08e21 | §4.3 |
| T4-hours at 15 TFLOPS effective | **~57,000 h** | §4.3 (program: ~600 h) |

D4 does not fit this box's disk safely in FP32 (20.26 GB vs 22 GB
free, before any other file), cannot be trained whole anywhere in
the fleet (81 GB state vs 15.6 GB T4 VRAM), and its Chinchilla-scale
training is ~95× the entire executable 5B program's GPU-hours.
**Dense 5B is rejected on arithmetic, not preference.** D2/D3 are
skipped for the same reason at smaller scale: they spend the fleet's
hours without testing the sparse hypothesis the program is built on.

---

## 3. The staged path (the plan)

```
Step 0  Repair (§1)                                    [blocking]
Rung 1  Finish the 1B: SGBD blocks 5–7                [Kaggle lane, §5]
Rung 2  Dense 1.686B (D1) validation rung              [branch — see below]
Rung 3  Sparse 5B (E10) on the repaired d1280/L36 core [main line]
```

**Rung 1 — finish the 1B.** Blocks 5–7 run exactly as blocks 0–4
(1,000 steps, batch 8, lr 1e-4, same cache) — the rung's protocol is
frozen; changing it mid-rung would void comparability. Then the
assembled repaired 1B passes the §7 gates and gets its model card.
Everything else in this plan gates on this rung.

**Rung 2 — D1 (1,685,569,280) as a validation rung.** Grown from the
trained repaired 1B by `grow_checkpoint.py` (drift gate < 0.6), then
SGBD-trained in 6 layer-slices of 7 layers (~271M trainable per
slice; FP32 train state ≈ 4.3 GB + activations — fits a T4's 15.6 GB,
does **not** fit this box, so Kaggle-only). Its job is to de-risk
three mechanisms at moderate cost before the sparse jump: (a) Net2Net
width+depth growth beyond 1B, (b) context-512 staging (§6), (c) the
T2 consolidation pilot (§4.2) on a dense model where block training
is already understood.
*Lineage note (explicit, auditable):* the 5B main line grows from
the trained **1B**, not from D1 — expert replication must start from
the MLPs whose distillation is complete and verified, and D1's wider
core would raise the 5B's active params (1.69B/2.52B) and every T2
FLOP figure with them. D1 is a same-ancestry branch artifact (grown,
never random-init), kept and model-carded; its *lessons* feed Rung 3.
If strict linear laddering is ever preferred, Appendix B prices the
E=5-on-D1 variant.

**Rung 3 — sparse 5B (E10, repaired base: 5,478,300,160 total).**
Expert construction follows the growth convention, no random init:
expert 1 of each layer **is** the trained 1B MLP; experts 2–10 are
replicas of it with the standard 1% symmetry-breaking noise
(`grow_checkpoint.py` convention), then specialized by T1 training
on distinct domain-balanced shards (§4.2) so replicas diverge by
data, not just by noise. The router is the one genuinely new
component: initialized to uniform routing (zero weights → uniform
softmax), a deterministic init recorded in the rung config — it
starts from "no opinion", not from noise. Top-1 vs top-2 routing is
chosen once, by measured val NLL + intervention rate at the Rung-3
gate (§7), and frozen in the rung config.

**Storage rules for all rungs** (disk is a hard constraint):
checkpoints are FP16/BF16 safetensors; at most **one** local copy of
any 5B-class checkpoint (~11 GB); canonical copies live on
HuggingFace private datasets (the proven archive lane, §5); per-unit
step checkpoints are pruned to final + last-step; no download starts
if it would take free disk under 5 GB.

---

## 4. Training at 5B level (Fahad's clause: the *training* is 5B-class too)

A 5B parameter count trained with a 1B-sized training program would
be a label, not a model. This section defines the training program
in three phases, with the honest gap stated in §4.3 — the phases we
can execute on the fleet, and the reference point they are measured
against.

**Two kinds of exposure, never conflated:**
- **Distillation exposure (SGBD):** a block/expert unit matches
  cached teacher activations (MSE) for N steps × 1,024 token
  positions. Cheap, memory-bounded, proven. It transfers and
  specializes knowledge; it is **not** token-level LM training and
  is never reported as tokens trained.
- **Consolidation exposure (token-level):** full LM loss (masked
  reconstruction + OSCD self-correction where applicable) over real
  corpus tokens, streamed block-wise via coreset-anchored streaming
  (`xiiv/streaming_sgbd.py`, `xiiv/coreset.py`, `xiiv/fisher.py` —
  EWC-protected, 30% progressive replay). This is the expensive,
  capability-building phase, budgeted in tokens and FLOPs below.

### 4.1 Corpus on hand (MEASURED — `data_frontier/*/meta.json` train tokens)

| Dataset | Train tokens | Source (manifest) |
|---|---|---|
| FineWeb-Edu | 3,308,079,744 | HuggingFaceFW/fineweb-edu:sample-10BT |
| Papers (peS2o) | 9,954,813,824 | scientifi-papers/scientific-papers:pes2o |
| DCLM | 9,950,636,800 | mlfoundations/dclm-baseline-1.0 |
| StackExchange | 2,131,150,720 | donfu/oa-stackexchange |
| OpenThoughts3 (reasoning) | 1,195,217,024 | open-thoughts/OpenThoughts3-1.2M |
| FineMath | 497,641,856 | HuggingFaceTB/finemath:finemath-4plus |
| PG-19 (books) | 75,220,608 | emozilla/pg19 |
| **Total unique** | **27,112,760,576** | |

Deferred, honestly: **Stack v2** — HF-gated, needs Fahad's terms
accept (user action, not nagged); **MegaMath** — manifest shows
13,359,261 docs seen, **13,359,261 dropped, 0 train tokens**
(upstream schema break, MEASURED); candidate replacement
`nvidia/Nemotron-CC-Math-v1` pending a license check before any use.

### 4.2 The three phases

**T1 — Distillation-complete (SGBD blocks & experts).**
- 1B blocks 5–7: 1,000 steps each (frozen 1B protocol).
- D1 slices: 6 × 1,000 steps.
- 5B expert units: the unit is one SGBD layer-slice × one
  expert-group (2 new experts per layer; 5 groups × 8 slices =
  **40 units**, ~138M trainable expert params + ~66M trunk slice per
  unit — FP32 train state ≈ 2.2–3.3 GB, fits T4 and, when RAM
  allows, this box). **2,000 steps per unit (TARGET)**, governed by
  a plateau stop-rule (stop early if the final-quartile relative
  loss improvement on a held-out activation slice is < 5%).
  *Justification for raising steps above the 1B's 1,000 (MEASURED):*
  blocks 3 and 4 were still improving in their final quartile at
  step 1,000 — block 3 loss 31.86 → 27.17 (−14.7%), block 4
  41.81 → 36.21 (−13.4%) across the last 250 steps
  (`sgbd_blocks/block{3,4}/*_final.pt` loss histories). The 1,000
  step budget did not plateau; new experts, which must also
  *specialize* away from their shared replica start, get double
  with a stop-rule rather than a blind number.
- Router + cascade phase: after all 40 units, one SGBD Phase-3
  cascade pass (re-capture composed activations, short re-train,
  router trained against balanced-load + task loss). Allowance
  ~10 GPU-h (ESTIMATED).
- Contract requirement (from §1.3): expert/teacher capture for the
  5B runs **with contract embeddings present** for a contract-
  conditioned slice of each unit's batches, so the contract path is
  inside the training signal, not bypassed by it.

**T2 — Consolidation on the ASTRA_CLEAN mix (token-level).**
Pipeline (all modules exist in `xiiv/`): exact-dedup → SimHash
near-dedup (`streaming_dedup.py`) → heuristics → fastText quality
filter (`quality_filter.py`) → 13-gram decontamination
(`contamination.py`) → Micro-DoReMi domain balancing
(`domain_balancer.py`) → coreset selection (200M gradient-selected
tokens per dataset, `coreset.py`) with 30% progressive replay and
EWC Fisher protection (`fisher.py`).
Phased token targets (cumulative; each phase gated, §7):
- **T2-pilot** (at Rung 2, dense D1): 100M tokens — proves the
  pipeline end-to-end and **measures** true T4 throughput, replacing
  every ESTIMATED figure below with a measured one before T2a starts.
- **T2a**: 250M tokens at the 5B. **T2b**: cumulative 1B tokens.
  **T2c**: cumulative ~3B tokens ≈ coresets (8 × 200M = 1.6B) +
  reasoning allocation 0.9B exposures + general/replay 0.5B.
- Reasoning replay (standing plan): OpenThoughts3 carries a 2.5×
  balancer weight throughout T2; its full 3-epoch program
  (3.59B exposures) completes only inside T2-full (§4.3) — stated
  here so the partial state at T2c is not misreported as complete.
- Context staging is *inside* T2, not an afterthought (§6): pilot
  introduces 512-token packing; T2b onward mixes 512/1024.

**T3 — Adaptive/Forge post-training.** Starts only after the last
5B block/unit completes (standing trigger, goal
`goal_8c042d59147c`): verifiable-reward RL, adversarial self-play,
Forge's verifier-gated harness loop. Budgeted in verified tasks and
wall-clock, not GPU-hours; the independent verifier is the judge,
unmodifiable by the loop (standing Forge safety rules).

### 4.3 FLOPs and GPU-hour arithmetic (ESTIMATED, anchored as noted)

Per-token training FLOPs ≈ 6 × active params (attention term < ~5%
at ctx ≤ 1024, ignored and noted). T4 planning throughput for
consolidation: **15 TFLOPS effective** (ESTIMATED — T4 FP16 peak is
~65 TFLOPS; 20–35% MFU for checkpointed block-wise training; the
T2-pilot replaces this with a MEASURED figure). Distillation units
are wall-clock-bound, not FLOP-bound (small batches, Python
overhead): planning figure **1 GPU-h per 1,000 steps**, anchored on
the MEASURED block-3 kernel — pushed 2026-10-19 06:28 UTC, COMPLETE
with verified checkpoint by ~07:14 UTC, i.e. ≤ ~1 h end-to-end
including queue, setup, and cache download.

| Phase | Work | GPU-h (ESTIMATED) |
|---|---|---|
| T1 | 1B blocks 5–7 (3 × 1,000 steps) | ~3 (range 1.5–6) |
| T1 | D1 slices (6 × 1,000 steps, ~271M units) | ~12 (range 9–15) |
| T1 | 5B expert units (40 × 2,000 steps) | ~80 (range 40–160) |
| T1 | Router + cascade allowance | ~10 |
| T2-pilot | 100M tokens @ D1 dense (6·1.686e9·1e8 = 1.01e18 FLOPs) | ~19 |
| T2a | 250M tokens @ sparse top-2 (2.12e18 FLOPs) | ~39 |
| T2b | +750M tokens (cumulative 1B) | ~118 |
| T2c | +2B tokens (cumulative 3B) | ~313 |
| **Program total through T2c** | | **~600 GPU-h (range ~450–950)** |

**The honest gap — T2-full.** One full pass over the 27.11B-token
corpus at sparse active size costs 1.49e20 FLOPs (top-1) to
2.30e20 FLOPs (top-2) ≈ **2,800–4,300 T4-hours** — roughly the
Chinchilla exposure for the *active* size (18.3B tokens @ top-1 /
30.3B @ top-2), and the corpus on hand is almost exactly that size.
T2-full is therefore the correct definition of "fully 5B-level
training" for this architecture — and it is **not schedulable** on
the free fleet in any reasonable window (§5: +15–24 months even on
the unlocked fleet). The program ships T1 + T2a–T2c + T3 as
"5B-class training, phased", with T2-full named, priced, and left
open — never silently redefined as done. Dense-5B Chinchilla for
scale: 101.3B tokens, 3.08e21 FLOPs, ~57,000 T4-h (§2.3).

### 4.4 Token budgets per rung (summary)

| Rung | Distillation exposure (positions, not LM tokens) | Token-level (LM) budget | Reference points |
|---|---|---|---|
| 1B finish | 3.07M (3 × 1,000 × 1,024) — MEASURED protocol | none new (frozen rung) | — |
| D1 validation | 6.14M (6 × 1,000 × 1,024) — TARGET | pilot 100M — TARGET | — |
| 5B sparse | 81.9M (40 × 2,000 × 1,024) — TARGET, stop-rule governed | cumulative 0.25B → 1B → 3B (T2a/b/c) — TARGET | Chinchilla-active 20.4B/30.3B; corpus ceiling 27.11B — MEASURED |

---

## 5. The free-compute fleet (Fahad's directive: every free resource to peak)

Every lane, its quota, its evidence label, and its assigned job.
**Banked** hours are only what is verified or sourced; upside is
never planned against.

| Lane | Quota | Evidence | Assigned job in the 5B program | Status |
|---|---|---|---|---|
| **Kaggle GPU (T4, ~15.6 GB VRAM)** | 30 GPU-h/week, 9 h sessions | **VERIFIED** — auth live again 2026-10-22 (CLI reinstalled via `uv tool` after the reboot wipe; kernels list OK); blocks 3–4 trained here | The primary lane: SGBD blocks 5–7, D1 slices, all 40 expert units, T2 consolidation | LIVE |
| **Kaggle Secrets mode** | — (not compute) | **VERIFIED in code** — block builder committed with Secrets support: the kernel fetches `HF_TOKEN` from Kaggle Secrets at runtime | Removes the per-block token paste for **every** Kaggle unit above: one one-time user action (add `HF_TOKEN` under Kaggle → Settings → Secrets) replaces ~49 individual pastes | Awaiting the one-time user action |
| **Kaggle TPU** | reports conflict: ~20 h/week to 30 h/week (secondary sources) | **UNVERIFIED** — no smoke kernel has ever run our code on it; our trainer is PyTorch/CUDA, TPU needs a torch_xla port | Pilot only: one smoke unit to verify quota + port cost. **Zero hours banked** until it returns a checkpoint | Assessment queued |
| **GitHub Actions CPU** | 2,000 min/month private repos (GitHub Free); unlimited for public repos | **SOURCED** (GitHub billing docs summaries, 2026-10) | CI + the CPU eval gates (§7), drift verification, data re-packing for ctx 512/1024 (§6), coreset scoring — ~33 CPU-h/month on standard 2-core runners; 500 MB artifact cap means checkpoints never transit Actions (HF is the store) | Available now (XIIV repo is private → the 2,000-min figure applies) |
| **HuggingFace Hub** | our current private datasets: ~29.9 GB corpora + ~16.6 GB teacher cache | **VERIFIED in-house** (uploaded, size-checked 2026-10) | The archive: corpora, teacher caches, per-rung checkpoints, 5B canonical copies (§3 storage rules) | LIVE |
| **Lightning AI** | ~7 GPU-h after phone verification; other reports 15 credits/mo or 22 GPU-h/mo | **UNVERIFIED range** (secondary sources disagree) | One-off burst: a teacher-capture pass or one T2-pilot unit. No recurring plan weight | User action: phone signup |
| **Modal** | $30/month Starter compute credit | **SOURCED** (modal.com pricing); T4 listed at $0.000164/s ≈ $0.59/h → **~50 T4-h/month (ESTIMATED conversion)**; A10G (~24 GB) ≈ $1.10/h → ~27 h/month for larger-memory units | T2 overflow/burst lane; A10G units for the widest consolidation slices | User action: payment method on file |
| **Oracle Cloud Always Free** | 2 OCPU + 12 GB RAM, always-on | **SOURCED** (in-house audit 2026-10-19; recheck at signup) | Always-on CPU worker: download/queue monitor, data re-packing, eval harness host — 12 GB RAM is more usable memory than this box reliably has | User action: card for signup |
| **Google Colab (free)** | usage-based, no guaranteed hours | **VERDICT: NOT A LANE** — interactive-only, no supported automation/API path from this box, sessions and quotas are not schedulable, and scripted automation carries ToS/account risk | Nothing is planned on Colab. If Fahad ever runs a notebook by hand there, its output is welcome — the program never depends on it | Excluded, deliberately |

### Fleet totals (GPU-hours/month)

| Fleet state | GPU-h/month | What changes |
|---|---|---|
| **Current (today)** | **~130** (Kaggle 30/wk × 4.33) + ~33 CPU-h (Actions) | — |
| + Kaggle Secrets unlock | ~130 (same hours) | Friction removed for all ~49 Kaggle units; no unit waits on a paste |
| + Lightning signup | ~130 + one-off ~7–22 (UNVERIFIED, not counted) | Burst only |
| + Modal unlock | **~180** | +~50 T4-h/month recurring |
| + Oracle unlock | ~180 GPU + always-on CPU | Pipeline/eval work leaves this box entirely |

**Program vs fleet:** the ~600 GPU-h program through T2c (§4.3)
takes ~4.6 months on Kaggle alone, **~3.3 months on the unlocked
fleet** (ranges: ~2.5–7 months across the estimate bands). T2-full
(+2,800–4,300 GPU-h) remains beyond the fleet — the ceiling is
raised by exactly two things: more free lanes of Modal's class, or
a step-change lane (a real GPU of Fahad's own). Both are named;
neither is assumed.

---

## 6. Context-growth staging (128 → 512 → 1024)

Context is a training program, not a config edit. The model builds
RoPE tables per sequence length at runtime, so longer contexts are
architecturally legal today — but the model has only ever *trained*
at 128, and untrained context is unclaimed context.

- **Data re-pack (training item):** the corpus is stored as
  128-token blocks (`meta.json` seq_len 128). Re-packing
  concatenates documents in their existing order with EOS
  separators into 512- and 1024-token blocks (document order
  preserved per Scaling Protocol rule 2; token counts identical,
  block counts lower), producing versioned datasets with their own
  `meta.json`. Job assigned to GitHub Actions CPU / Oracle (§5),
  never to GPU hours.
- **C1 → 512 (at Rung 2):** the T2-pilot runs ≥ 25% of its tokens
  at 512 (TARGET mix). Batch tokens/step held constant across
  stages (micro-batch shrinks as context grows; exact values frozen
  in the rung config after a T4 smoke profile).
- **C2 → 1024 (at Rung 3, T2b onward):** 512/1024 mix; RoPE theta
  stays 10000 at both stages unless the C1 gate shows degradation —
  a theta change is a config change and would be recorded as a new
  rung decision, not smuggled in.
- **Gates:** at each stage, val NLL at the native context and the
  intervention rate at the native context are the §7 gate metrics.
  Kill criterion: if 1024-context NLL regresses > 5% vs 512 at
  equal tokens, hold at 512 and record it — the 5B ships at its
  best *measured* context, not its largest attempted one.

---

## 7. Eval gates, promotion, kill criteria (per Scaling Protocol §§4–5, 8)

**Identical at every rung** (same code, same sets): val NLL every
500 steps (8 val batches, full curve kept); the intervention-rate
benchmark identical to the 52.5M baseline; sampler/lock-span
regression tests; contract v2 checks, brick certificates, and the
independent verifier re-run against the rung; final **and** best-val
numbers reported side by side; a model card per rung
(architecture, exact count, data version + hash, token budget, eval
numbers, code commit, checkpoint location, known limitations).

**Promotion rule (unchanged):** a rung promotes only if val NLL is
trending down or plateaued at/below the previous rung's trajectory
**and** the intervention rate improves on the previous rung. A rung
that only memorizes does not promote.

**Rung-specific additions:**
- *Step 0 (§1):* strict load under current code; contract coverage
  36/36 layers; repaired-vs-truncated drift reported on fixed
  inputs; the new shape-assertion regression test green.
- *Rung 1:* blocks 5–7 loss curves consistent with blocks 0–4
  (final-quartile behavior compared, not just final loss); the
  assembled 1B — not the blocks — is what gets gated and carded.
- *Rung 2:* growth drift ratio < 0.6 (pre-registered gate);
  T2-pilot must show val NLL improvement over the grown init at
  equal tokens, and returns the MEASURED T4 throughput that re-bases
  §4.3.
- *Rung 3:* expert load balance on an eval slice — max expert load
  ≤ 2× mean (TARGET bar), router entropy reported; top-1 vs top-2
  chosen by measured NLL + intervention and frozen; **contract-gate
  movement:** mean |gate| of the contract cross-attention must move
  measurably during T2 vs its grown values — if it does not, the
  5B's model card downgrades the contract path to "grown, not
  trained", exactly as the 1B's card does (§1.3).

**Kill criteria (any one stops the rung; the result is recorded,
not buried):** growth drift ratio ≥ 0.6 · intervention rate not
improving after T1 + T2a · expert collapse (load bar failed after
one router-rebalance retry) · T2-pilot shows no NLL gain (T2 design
is reassessed before any quota is burned at scale) · the §6 context
kill rule · any storage-rule breach (§3) — pause rather than delete
a verified checkpoint to make room.

---

## 8. What stays unchanged

- **The guarantee-native thesis** — contract-native compute,
  deterministic guarantee bricks, proof-carrying certificates, an
  independent verifier the model cannot modify, intervention rate
  as a first-class metric — at every rung, re-proven per rung
  (Protocol rule 8). "Every answer carries its proof."
- **Single lineage** — every weight in the 5B descends from the
  133M seed through growth and distillation. No random init
  anywhere; new capacity is replicated-then-specialized (§3), the
  router's uniform init is the single documented exception-that-
  isn't (a deterministic no-opinion start, recorded in the config).
- **The architecture family** — masked discrete diffusion,
  bidirectional attention, RoPE, SwiGLU, tied embeddings, vocab
  32768, contract encoder (256/2/4/512) unchanged. Vocab growth is
  explicitly *not* in this program: the corpus is already tokenized
  at 32k, and re-tokenization would invalidate every manifest in
  §4.1 for a benefit this ladder does not need.
- **Honesty rules** — the benchmark board
  (`research/benchmark_targets_2026-10-22.md`) stays unmeasured for
  XIIV until a validated checkpoint runs the real harness; the 100x
  ledger (`research/improvement_ledger_100x.md`) banks only measured
  factors. A 5B label earned by §§4–7's gates is the claim — nothing
  larger, nothing vaguer.

---

## 9. Blockers and open items (complete list)

1. **Blocks 5–7 need the Kaggle lane unblocked** — one one-time user
   action (HF_TOKEN in Kaggle Secrets, §5) now covers them and all
   ~49 future Kaggle units. Until then, Rung 1 cannot finish and
   everything downstream gates on it.
2. **T2-full has no funding lane in the fleet** — priced in §4.3 (§4.3) — named gap, ~2,800–
   4,300 GPU-h. The phased program is the deliverable; T2-full
   closes only with new lanes.
3. **Stack v2** (data) — HF terms accept is Fahad's action; the plan
   does not depend on it (27.11B on hand suffices through T2-full's
   definition).
4. **MegaMath replacement** — `nvidia/Nemotron-CC-Math-v1` needs a
   license check before any download; FineMath (0.50B) carries the
   math allocation meanwhile.
5. **Kaggle TPU** — unverified quota + unported code; pilot-only
   (§5). No plan figure uses it.
6. **Local disk/RAM** — §3 storage rules are mandatory, and local
   training runs only when the box's RAM gate (≥2.5 GB available,
   the guardian's standing rule) passes.

---

## Appendix A — count reproduction

```python
import torch
from xiiv.config import ModelConfig
from xiiv.model import XIIVModel

def count(d, L, h, f):
    cfg = ModelConfig(vocab_size=32768, d_model=d, n_layers=L,
                      n_heads=h, ffn_dim=f, tie_embeddings=True,
                      contract_enabled=True)  # contract dims = defaults
    with torch.device("meta"):
        return XIIVModel(cfg).count_parameters()

count(1280, 36, 20, 3584)   # 1,019,230,720  (current code; artifact: 915,988,480 — §1)
count(1536, 42, 24, 4301)   # 1,685,569,280  (D1)
count(1792, 44, 28, 5018)   # 2,386,301,952  (D2)
count(2048, 48, 32, 5734)   # 3,379,129,600  (D3)
count(2624, 44, 41, 7347)   # 5,065,000,512  (D4)
# Sparse (§2.2): non_mlp + E * 36 * 3 * 1280 * 3584, non_mlp from the
# audited safetensors header (checkpoint) or the meta count (repaired).
```

## Appendix B — variant: sparse on the D1 core (strict-linear-ladder alternative)

If the 5B were grown from D1 (d=1536, L=42, MLP/layer 19,819,008;
non-MLP 853,170,944) instead of the 1B core: E=5 → **5,015,162,624**
total, active 1,685,569,280 (top-1) / 2,517,967,616 (top-2);
E=6 → 5,847,560,960. Rejected for the main line (§3): active params
1.8–2.7× the chosen design raise every T2 FLOP and hour figure in
§4.3 proportionally, and expert replication would start from MLPs
whose distillation is one rung less mature. Recorded so the road
not taken is priced, not forgotten.
