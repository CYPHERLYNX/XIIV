# CNaRA — Contract-conditioned Noise-aware Rank Adaptation

**Date:** 2026-10-16
**Status:** Implemented in `xiiv/lora.py` (+ `cnara_auto` hook in `xiiv/diffusion.py`); 36/36 unit tests pass. Training-loop adoption pending (LoRA run currently paused per sequenced plan).
**Classification:** principled port (NaRA mechanism, cited) + novel-in-combination extension (contract conditioning). NOT a from-scratch invention — documented honestly as such.

## What it is

Standard LoRA learns a static low-rank update ΔW = (α/r)·B·A. In a diffusion
LM the input distribution and generation difficulty shift along the denoising
trajectory, so a noise-agnostic adapter is structurally misaligned. CNaRA makes
the update trajectory-aware AND contract-aware:

    ΔW(λ, c) = scaling · B · S(λ, c) · A,    S(λ, c) = I + η · core(λ, c)

where λ ∈ [0,1] is the batch noise level (mask ratio), c ∈ {0,1} is XIIV's
contract-conditioning flag (contract_ids present in the batch), and core(·) is
produced by a lightweight globally-shared hypernetwork:

    core(λ, c) = reshape( MLP( [Fourier(λ); Embed(c)] ) )     # r×r

Fourier(λ): 2K sinusoids (K=8, no learned params). Embed(c): learned 2×4
(zero-init). MLP: (2K+4)→64→64→r², tanh, final layer zero-init ⇒ S = I at init
⇒ step-0 behavior is EXACTLY static LoRA (safe rollout; also B is zero-init).

Cost: ~9.7K params at r=8 (hypernet) vs ~2.2M adapter params — negligible on
the CPU training loop (~4s/step). One hypernetwork per model, shared across
all wrapped layers (the NaRA design). The training loop sets the condition
once per step via `apply_cnara_condition(model, lam, contract)` or
automatically via `diffusion_loss(..., cnara_auto=True)`, which reads the
batch's own mask ratio and contract presence before the forward. Deployment
folding (`merge_cnara(model, lam, contract)`) fixes one condition — the
folded model is a plain static network.

## Prior art (what is adopted vs what is claimed)

**Adopted (cited, not invented):** NaRA — "Noise-Aware LoRA for
Parameter-Efficient Fine-Tuning of Diffusion LLMs" (Wang, Zhuang, Huang,
Zhang, arXiv:2605.29716, May 2026). Noise-conditioned core matrix via a
lightweight globally-shared hypernetwork; r=16 NaRA beat r=32 static LoRA on
GSM8K/AQuA. (Abstract verified against arXiv 2026-10-16.)

**Claimed novel-in-combination:** conditioning the hypernetwork on the
contract signal c alongside the noise level λ. No published work found
(2026-10-16 sweep) combines guarantee-native contract conditioning with
noise-aware PEFT — NaRA conditions on noise only. This turns XIIV's unique
standing feature (40% contract-conditioned training batches) into an
architectural advantage: the adapter's effective rank allocation shifts with
both where you are in the diffusion trajectory and whether the batch carries
a contract.

**Not claimed:** the hypernetwork mechanism itself, Fourier timestep
conditioning, or zero-init safe rollout — all standard / NaRA.

## Fit to the thesis

- Diffusion-native: yes — conditions on the diffusion noise level directly.
- Guarantee-native: yes — the contract signal becomes a first-class
  architectural input to the adapter, not just prompt text.
- Orthogonal to GCDD (decode-time veto) and GACD (unmasking schedule): CNaRA
  is training-time adapter architecture — it changes what the model learns,
  not how it samples. Stacks with both.

## Honest caveats (do not claim without measuring)

1. NaRA's gains were shown at 8B scale on GPU training. The gain at XIIV's
   1B scale on CPU is UNVERIFIED. Required experiment before any claim:
   train static LoRA vs CNaRA on the same data slice, compare masked-LM loss
   on a held-out set.
2. At true step 0 (B = 0) the hypernetwork receives exactly zero gradient —
   there is nothing to modulate yet. This is by construction, asserted in
   tests, not a bug.
3. `merge_cnara` folds at ONE condition; serving far from that condition
   loses the adaptation benefit. Pick the deployment condition deliberately.
4. The contract-conditioning extension is simple enough that independent
   invention elsewhere is plausible; the novelty claim is "no prior art
   found as of 2026-10-16", not "un-inventable by others".

## Files

- `xiiv/lora.py`: `fourier_features`, `CNARAHyperNet`, `CNARAController`,
  `CNARALinear`, `inject_cnara`, `apply_cnara_condition`, `merge_cnara`;
  `merge_lora` now refuses CNaRA models (points to `merge_cnara`).
- `xiiv/lora.py` also: fixed `merge_lora` fold math (was adding the
  transposed update — errored on non-square linears, silently wrong on
  square ones) and `inject_lora` now refuses double injection (was silently
  double-wrapping).
- `xiiv/diffusion.py`: `diffusion_loss(..., cnara_auto=False)` hook;
  `DenoisingState.from_snapshot` now initializes runtime-policy attributes
  (adaptive_decode etc.) to safe defaults — previously ANY pause/resume
  crashed in `step()` with AttributeError (pre-existing bug, test
  `test_stateful_pause_resume_identical` was failing on clean tree).
- `xiiv/test_cnara.py`: 36 tests, pure CPU, no weights.
