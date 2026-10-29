# XIIV Exact-1B Expansion — Design and Training Plan

**Status: DESIGN ONLY. NOT TRAINED. NOT SHIPPED.**

## 0. The statement that must not be blurred

The current assembled file is **not** an exact-1B model.

- Current assembled file: `checkpoints_1b_sgbd/xiiv_1b_sgbd.safetensors`
- Current runnable parameters (unique, tied embeddings counted once):
  **773,418,240**
- Current contract parameters excluded from the assembled file, carried
  only in the headline reconciliation: **142,570,240**
- Current headline total: 773,418,240 + 142,570,240 = **915,988,480**

The exact-1B file does **not** exist until the expansion in this document
is trained, stitched, and audited to exactly **1,000,000,000** unique
parameters. Until that audit passes, the current file must continue to be
described as 915,988,480 — never as "1B" in the exact sense, and never
by rounding 915,988,480 up to 1,000,000,000.

No renaming, no re-labelling, and no change to the counting convention
is used anywhere in this plan to reach 1,000,000,000. The gap is closed
with real, trainable parameters that are used in the forward pass.

## 1. Baseline geometry (verified against the assembled file)

| Item | Value |
|---|---|
| d_model | 1280 |
| Layers | 36 |
| Heads | 20 (head dim 64) |
| FFN dimension (SwiGLU) | 3584 |
| Vocabulary | 32,768, tied embeddings |
| Context | 128 |
| Contract | off in the assembled file (`contract_enabled: false`) |
| Activation | SwiGLU in transformer blocks: `down(silu(gate(x)) * up(x))` |

### 1.1 One transformer block, exactly

Per block, all weights bias-free, RMSNorm weights counted:

- Attention QKV: 1280 × 3840 = **4,915,200**
- Attention output: 1280 × 1280 = **1,638,400**
- Attention subtotal: **6,553,600**
- SwiGLU gate: 1280 × 3584 = **4,587,520**
- SwiGLU up: 1280 × 3584 = **4,587,520**
- SwiGLU down: 3584 × 1280 = **4,587,520**
- FFN subtotal: **13,762,560**
- RMSNorms (attn_norm + mlp_norm): 2 × 1280 = **2,560**
- **Per block: 6,553,600 + 13,762,560 + 2,560 = 20,318,720**

### 1.2 Runnable reconciliation, current file

- Embeddings (tied, counted once): 32,768 × 1280 = **41,943,040**
- 36 blocks: 36 × 20,318,720 = **731,473,920**
- Final norm: **1,280**
- Runnable total: 41,943,040 + 731,473,920 + 1,280 = **773,418,240** ✓
- Plus excluded contract: **142,570,240**
- Headline total: **915,988,480** ✓
- Tensors in file: 1 (embed) + 36 × 7 (per block) + 1 (final norm) = **254** ✓
  (the tied `lm_head` is the same tensor object as the embedding and is
  not counted twice)

### 1.3 The gap, and why layers alone cannot close it

```
Target:                         1,000,000,000
Current headline total:           915,988,480
Gap to close:                      84,011,520
```

```
4 standard blocks:  4 × 20,318,720 =  81,274,880
Remainder after 4 blocks:         84,011,520 − 81,274,880 = 2,736,640
5 standard blocks:  5 × 20,318,720 = 101,593,600  (overshoots by 17,582,080)
```

So a pure layer addition cannot hit exactly 1,000,000,000: the gap is
not a multiple of the block size (84,011,520 mod 20,318,720 =
2,736,640). **Any** exact-1B design on this geometry must therefore
contain one deliberately-sized remainder module. That remainder is
stated openly below; it is not hidden, and it is not a counting trick.
The remainder module is a real, trainable, forward-pass module.

## 2. Preferred option — 4 new blocks + 1 Expansion MLP block

### 2.1 Architecture change

1. **SGBD Block 9: 4 new standard transformer blocks, layers 36–39.**
   Same `Block` definition as layers 0–35: same d_model, heads, FFN
   dimension, RMSNorms, RoPE, bidirectional attention, contract off.
   This extends the layer count from 36 to **40**.

2. **Expansion MLP block (E-block), placed after layer 39 and before
   the final norm.** A residual, norm-free, bias-free two-matrix MLP:

   ```
   y = x + W_down · SiLU(W_up · x)
   W_up:   1280 → 1069, bias-free
   W_down: 1069 → 1280, bias-free
   ```

   No norm and no bias are included in the E-block, deliberately:
   adding a 1280-parameter norm would make the hidden size non-integral
   (see §4, rejected options). The E-block operates directly on the
   residual stream; the preceding block's output is already a residual
   stream, so no new normalisation stage is required for it to be
   well-defined.

### 2.2 Exact arithmetic, preferred option

```
4 new blocks:   4 × 20,318,720 = 81,274,880
E-block up:     1280 × 1069    =  1,368,320
E-block down:   1069 × 1280    =  1,368,320
E-block total:                   2,736,640
Expansion total: 81,274,880 + 2,736,640 = 84,011,520
```

```
New runnable total:
  Current runnable            773,418,240
  + expansion                  84,011,520
  = new runnable              857,429,760

New runnable breakdown (check):
  Embeddings                   41,943,040
  40 blocks: 40 × 20,318,720  812,748,800
  Final norm                    1,280
  E-block                     2,736,640
  Total                       857,429,760  ✓

New headline total:
  New runnable                857,429,760
  + contract (unchanged)      142,570,240
  =                         1,000,000,000  ✓
```

Contract is **unchanged** in this expansion: the new file stays
contract-off, exactly like the assembled file. Re-attaching contract
would add per-block contract cross-attention parameters and would
require a full recount before any exact-1B claim; that decision is
out of scope for this expansion and must not be folded in silently.

Tensor count, new file: 1 (embed) + 40 × 7 (blocks) + 1 (final norm)
+ 2 (E-block) = **284 tensors** (was 254).

### 2.3 Why this is the preferred option

- **No tokenizer change.** Vocabulary, tokeniser, prompts, and all
  existing evaluation prompts are unchanged. No new tokens to invent,
  merge, or distribute.
- **No interface change.** Input and output shapes are unchanged
  (vocab 32,768, context 128, tied embeddings).
- **Every new parameter receives gradient on ordinary data.** The
  E-block sees every sequence, unlike new vocabulary embeddings,
  which would only train when their (rare) tokens occur.
- **Identity-preserving initialisation is available** (see §3.2), so
  the expanded model starts exactly where the assembled model is,
  and any regression after training is measurable against that
  baseline rather than confounded with a new random initialisation.

The honest cost, stated plainly: the E-block hidden size **1069** is
not a round number. It is the exact integer forced by the remainder
(2,736,640 ÷ 2560 = 1069). A round hidden size would not sum to
exactly 1,000,000,000. The document does not round it, rename it, or
pretend otherwise.

## 3. Training plan — preferred option (GPU only)

This box cannot train this expansion (7 GB RAM; a full 1B load plus
optimizer state does not fit, and local CPU training is the grinding
path this project has already rejected). All training below is on a
Kaggle or Colab GPU (Tesla T4 class, ~15.6 GB VRAM, as already
verified on the Kaggle lane), or on the user's own GPU machine.
No step in this plan is a local CPU training step.

### 3.1 Data

- The same public, licensed, documented training streams used for
  SGBD (FineWeb-Edu, Papers, DCLM, StackExchange, PG-19, OpenThoughts
  reasoning, and the other curated sources already on record), in
  their existing tokenised form. No new data source is introduced
  by this expansion, and the standing exclusion applies unchanged:
  no CIA / conspiracy-adjacent "hyper-intelligence program" research
  content is used as training data.
- Held-out split: fix a held-out set **before** training starts and
  freeze it. It is used for the baseline (§3.3), the Phase B gate
  (§3.4), and the consolidation gate (§3.5). It is never trained on.
- Teacher activations: the SGBD teacher-cache route is not required
  for this expansion, because the teacher here is the assembled model
  itself (§3.2). If a teacher-cache run is later chosen, it must be
  recorded as a separate decision with its own cost; it is not
  assumed in the budget below.

### 3.2 Initialisation — grown from the trained 1B, not random-only

**Step 0 — identity gate (must pass before any training step).**

- New blocks 36–39: copy the non-output weights from the top
  trained block (layer 35) — QKV, SwiGLU gate/up, and both RMSNorm
  weights (RMSNorm weights are ones by construction, so this is
  exact) — and **zero-initialise the output projections**:
  attention `out` and SwiGLU `down` are set to zero in each new block.
  With contract off, each new block then computes
  `x + 0 + 0 = x`: an exact identity.
- E-block: initialise `W_up` with a small truncated-normal
  (scale to be fixed by the short probe in §3.4, starting value
  0.02) and **zero-initialise `W_down`**. The E-block then computes
  `x + 0 = x`: an exact identity, regardless of `W_up`.
- All existing parameters (layers 0–35, embeddings, final norm)
  are copied bit-for-bit from the audited assembled file
  (SHA cf548ec4…2c8d1) and are **not** re-initialised.
- Gate: run the assembled model and the expanded model, at
  initialisation, on the same prompts, same seed, same steps.
  Outputs must match the assembled model (bit-identical in fp32,
  or within a stated fp tolerance if the GPU run uses a lower
  precision — the tolerance must be stated in the run record, not
  assumed). If they do not match, stop: the expansion code is
  wrong, and training must not start on top of it.

This zero-output initialisation follows the precedent already in
the codebase (the contract cross-attention output gate is
zero-initialised so that enabling it is initially a no-op).

### 3.3 Baseline (before training)

On the frozen held-out set, record for the assembled model:

- Masked-token reconstruction loss (the diffusion training objective).
- Generation on the fixed prompt set used for the assembled CPU
  baseline, with steps, seed, temperature, and top_p recorded, plus
  generation seconds on the GPU.

These numbers are the regression baseline. They are not published
as "exact-1B results" — they belong to the 915,988,480 model.

### 3.4 Phase B — train only the new parameters

- **Freeze** layers 0–35, embeddings, and final norm.
  Trainable parameters: exactly the expansion, **84,011,520**
  (8.4% of the new headline total).
- Objective: the same masked-token reconstruction (diffusion)
  objective as SGBD block training, on the training streams in §3.1.
- Optimiser and precision: AdamW on the trainable parameters only
  (optimizer state therefore scales with 84M parameters, not with
  the full model), model weights in the same precision as the
  assembled file for the stitch (fp32), with a lower-precision
  compute probe allowed only if the run record states the precision
  and the final stitch is re-verified in fp32. Gradient
  checkpointing is available in the model (`grad_checkpoint`) and
  should be enabled if activations threaten VRAM.
- Budget (starting point, not a proven optimum): 1000 steps, matching
  the SGBD per-block standard (each trained block was 1000/1000
  steps). Batch size is set by a VRAM probe on the GPU machine
  (largest batch at context 128 that fits with optimizer state and
  activations, then fixed for the run). Learning rate is set by a
  short probe (a few dozen steps at two candidate rates, comparing
  held-out loss movement); the starting candidate is an order of
  magnitude below a from-scratch rate, because the new blocks start
  at identity and the frozen trunk must not be shocked. The chosen
  rate, batch, and probe result go in the run record — they are not
  asserted here as tuned values.
- Gate to pass before Phase C: held-out loss strictly better than
  the §3.3 baseline, and generation on the fixed prompt set not
  regressed (no increase in repetition/glitch rate versus the
  baseline record, judged on the raw outputs, not on a summary).
  If the gate fails, do not proceed to consolidation: diagnose
  (learning rate, batch, data mix) and re-run Phase B from the
  identity checkpoint, or stop the expansion.

### 3.5 Phase C — consolidation (unfreeze-all, low rate, short)

Only after Phase B passes:

- Unfreeze all parameters. Train at a low rate (below the Phase B
  rate) for a short, fixed consolidation run, on the same streams
  with the same held-out set.
- Gate: held-out loss must not regress versus the Phase B
  checkpoint, and generation must not regress versus Phase B.
  If either regresses, **revert to the Phase B checkpoint** and ship
  that (subject to §3.6 audit). Consolidation is an improvement
  attempt, not a requirement, and a failed consolidation never
  overwrites a passing Phase B checkpoint.

### 3.6 Stitch and audit — the exact-1B claim is earned here

The stitch extends the existing stitch procedure
(`scripts/stitch_1b_sgbd.py` pattern) into a new script for this
expansion (e.g. `scripts/stitch_exact_1b.py`); the new script is
written before training, not improvised after it. The stitch:

1. Loads the frozen assembled parameters (layers 0–35, embeddings,
   final norm) from the audited assembled file and the trained
   Phase B / Phase C parameters (layers 36–39, E-block).
2. Writes a new config: `n_layers: 40`, plus an explicit
   `expansion_block` field describing the E-block (hidden 1069,
   SiLU, bias-free, norm-free, placed after layer 39); all other
   fields unchanged (vocab 32768, d_model 1280, heads 20, ffn_dim
   3584, seq_len 128, `tie_embeddings: true`,
   `contract_enabled: false`).
3. Writes the new `.safetensors` file under a **new filename**
   (the assembled file is never overwritten, and the new file is
   never given the assembled file's name).

The audit, all of which must pass, performed by a script independent
of the stitch script where feasible:

- [ ] Tensor count: **284** tensors, with the exact key list
  (embed, 40 × 7 block tensors, final norm, E-block up/down).
- [ ] Unique parameters (tied embeddings counted once):
  **857,429,760** runnable.
- [ ] Headline reconciliation: 857,429,760 + 142,570,240 (contract,
  unchanged, excluded) = **1,000,000,000** exactly.
- [ ] Per-tensor SHA match: every output tensor's SHA matches the
  corresponding training-checkpoint tensor; every frozen tensor's
  SHA matches the assembled file.
- [ ] Whole-file SHA256 recorded in the manifest.
- [ ] All tensors fp32 (or the single precision stated in the
  manifest, consistently), zero non-finite tensors.
- [ ] Load test via `xiiv/sample.py`'s `load_model` path: missing
  exactly `['lm_head.weight']` (tied), unexpected `[]`.
- [ ] Identity-regression check: the audited file, loaded on GPU,
  generates on the fixed prompt set, with raw output and generation
  seconds recorded. This is the first GPU generation of the
  exact-1B file; it is not back-filled from the assembled model.

Only when every box above passes may the new file be described as
"XIIV-1B (exact, 1,000,000,000 parameters)". A single failed box
means the file is not the exact-1B file, whatever its filename says.

### 3.7 Cost and schedule (estimates, not commitments)

- GPU memory budget (to be confirmed by the VRAM probe, not asserted
  as measured): model weights ~3.4 GB fp32 (runnable), AdamW state
  for 84M trainable parameters ~1.0 GB, gradients for trainable
  parameters ~0.3 GB, activations at context 128 with gradient
  checkpointing well under 2 GB. Total well within a 15.6 GB T4 for
  Phase B; Phase C (all-parameter optimizer state) is the tight
  phase and is the reason it is short and gated — if it does not
  fit, Phase B alone is the shippable result.
- Time: not estimated here. Step time on the target GPU is measured
  in the first Phase B probe run and only then extrapolated; no
  completion date is promised in this design document.
- This box: no training step, no stitch step that loads the full
  model in RAM alongside a training run. The stitch and audit are
  GPU-machine or high-RAM-machine jobs, or are run on this box only
  when a sustained RAM window is verified first.

## 4. Fallback option — 4 new blocks + vocabulary expansion

If the E-block is rejected (for example, if a future review requires
every block in the model to be a standard transformer block), the
fallback is:

- 4 new standard blocks, layers 36–39: **81,274,880**
- Vocabulary expansion: 32,768 → **34,906** tokens
  (Δvocab = 2,138; tied embeddings, so Δparams = 2,138 × 1280 =
  **2,736,640**)
- Expansion total: 81,274,880 + 2,736,640 = **84,011,520** ✓
- New runnable: 773,418,240 + 84,011,520 = **857,429,760** ✓
- Headline: 857,429,760 + 142,570,240 = **1,000,000,000** ✓
- Tensors: 254 + 28 = **282** (the embedding tensor grows in shape;
  it is still one tensor).

Why it is only the fallback:

- It changes the tokenizer and the input interface. Every prompt,
  evaluation, and downstream tool must be re-validated against the
  new vocabulary, and the 2,138 new tokens need a real purpose
  (the natural candidate is a proof / mathematics / code token set)
  plus training data in which they actually occur — otherwise
  their embeddings receive almost no gradient and the "expansion"
  is numerically exact but functionally hollow.
- It changes the output head shape (tied, so the same tensor), which
  touches every generation path, whereas the E-block touches only
  the residual stream after layer 39.
- The new vocabulary size **34,906** is, like 1069, a non-round
  number forced by exactness. It is stated here, not hidden.

Training for the fallback follows §3, with two additions: new
embeddings are initialised from the mean of semantically nearest
existing embeddings (not from a fresh random draw at full scale),
and the Phase B freeze set excludes the new embedding rows (they
are trainable) while freezing all old embedding rows.

## 5. Rejected options (shown so the choice is auditable)

- **Pure layer addition (4 or 5 blocks).** 4 blocks undershoot by
  2,736,640; 5 overshoot by 17,582,080. Fails exactness.
- **Widening FFN across all layers.** After 4 new blocks, the
  remainder requires Δffn = 2,736,640 ÷ (40 × 3840) = 17.82 — not
  an integer. After any other layer count, the same division is
  non-integral. Fails exactness.
- **SwiGLU-style bottleneck (gate/up/down).** Remainder requires
  hidden = 2,736,640 ÷ 3840 = 712.67 — not an integer. This is why
  the preferred E-block is a two-matrix SiLU MLP, not a SwiGLU
  block. A SwiGLU E-block cannot hit the remainder exactly.
- **Untying `lm_head`.** Adds 32,768 × 1280 = 41,943,040 in one step,
  which combined with any whole number of blocks overshoots or
  undershoots the gap and changes the tying convention the
  assembled file was audited under. Rejected.
- **Counting contract parameters as new capacity.** The contract
  figure (142,570,240) is a reconciliation figure for parameters
  excluded from the assembled file, not new runnable capacity.
  Re-attaching or re-counting it to reach 1,000,000,000 would be
  exactly the renaming trick this document forbids. Contract stays
  unchanged and excluded.

## 6. What this document does not claim

- The exact-1B file does not exist. No GPU generation, benchmark,
  or "beats frontier" result for it exists, and none is implied.
- The assembled 915,988,480 file's results (including its CPU
  baseline of 64 tokens in 38.22 s) belong to that file and are
  not transferable to the exact-1B file.
- No training step, learning rate, or time estimate in §3 is a
  measured result. Starting values are labelled as starting
  values; tuned values come from the probe runs described.
- The 9/10 gate, the frontier benchmark comparison, and the
  user's own system test all apply to the exact-1B file **after**
  the §3.6 audit passes, on that file's own runs — not to this
  design document.

## 7. Decision required before training starts

1. Approve the preferred option (§2) over the fallback (§4), or
   choose the fallback explicitly. The choice changes the config,
   the tokenizer, and the stitch script, and must be made before
   Phase B — not after training has started.
2. Confirm the GPU lane for Phase B (Kaggle T4, Colab T4, or the
   user's own GPU machine) and the held-out set. Training starts
   only after the §3.2 identity gate passes on that lane.
