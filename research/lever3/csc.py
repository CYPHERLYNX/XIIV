"""Cold-Start Consistency (CSC) — lever-3 training-step variant.

Target failure mode: the step-0 void. At inference the sampler starts from
a 100%-masked canvas — a distribution the model essentially never trains on
(random per-token masking makes P(all 128 masked) ~ 0). The first unmasking
round therefore extrapolates from an untrained regime, commits garbage with
high confidence, and every later step conditions on that garbage. This is
the standing reading for why five levers cut held-out masked loss (-25% to
-66%) while free generation stayed 0/5 on factual/code probes: all five
optimized p(x_masked | x_observed) and none ever trained the joint p(x)
the sampler needs at step 0.

Prior-art position (honest, read before claiming anything):
- D2F (arXiv:2508.09192): block-wise AR-diffusion hybrid, progressive
  masking ratios per block, asymmetric distillation from a pretrained
  dLLM. Different: block-causal structure + teacher distillation; CSC is
  single-model, no teacher, no block structure.
- DSP (Asada & Miwa, EACL 2026): multi-step self-prediction — denoise the
  model's OWN intermediate outputs. Different: still conditional denoising
  p(x_m | x_o) on corrupted inputs; CSC trains the unconditional joint.
- ReMDM: training-free inference-time correction steps. Different: no
  training change at all.
- RCD (arXiv:2601.22954): recycles discarded low-confidence token info
  into the next step. Different: inference module, not a training
  objective.
- HiLo-RD (arXiv:2610.36901): cross-budget self-distillation. Different:
  teacher/student across compute budgets; CSC is single budget.
- Our own high-mask curriculum trial (FAILED, 0/5): still per-token
  conditional denoising with sparse context. CSC differs in the one way
  that matters: ZERO context, full-joint single-shot prediction, as an
  AUXILIARY to (never a replacement of) the denoising loss.
This is a documented candidate invention, NOT a proven world-first;
broader literature/patent review is still required before any such
claim.

Mechanism (one training step):
  With probability p_cold (default 0.15) the batch is a COLD batch:
    x_cold = all-mask canvas (every position = mask_id)
    logits = model(x_cold)
    L_cold = CE(logits, targets) over ALL positions
  Otherwise it is a standard denoising batch:
    (noisy, targets, mask) = mask_batch(x0, mask_id, t_min, t_max)
    L_denoise = CE(model(noisy)[mask], targets[mask])
  L = L_denoise + lambda_cold * L_cold   (only one term active per step;
  the cold term trains the joint prior, the denoise term keeps the
  conditional skill — dual-objective diffusion)

  At inference nothing changes structurally: step 0 now draws on a
  TRAINED joint prior instead of an untrained extrapolation, so the
  initial canvas is coherent enough for refinement steps to polish
  rather than garbage to compound.

Interface: drop-in replacement for the general_loss() closure in
pretrain-jobs/pretrain_train.py (same (loss, info) return contract as
xiiv.diffusion.diffusion_loss). One-variable discipline: swapping this
in changes ONLY the training step — data, model, optimizer, schedule,
and generation gates stay identical.

Swap (inside pretrain_train.py main(), after `tok` exists):
    sys.path.insert(0, os.path.join(args.repo_dir, "research", "lever3"))
    from csc import cold_start_loss
    def general_loss(batch_ids):
        return cold_start_loss(expanded, batch_ids, tok.mask_id)
"""

import torch
import torch.nn.functional as F

from xiiv.diffusion import mask_batch


def cold_start_loss(model, batch_ids: torch.Tensor, mask_id: int,
                    p_cold: float = 0.15, lambda_cold: float = 1.0,
                    t_min: float = 0.15, t_max: float = 0.90,
                    generator=None):
    """Dual-objective diffusion step: conditional denoising + cold-start joint.

    Returns (loss, info dict) matching the diffusion_loss contract.
    """
    device = batch_ids.device
    info = {}

    if generator is not None:
        u = torch.rand((), generator=generator, device=device).item()
    else:
        u = torch.rand(()).item()

    if u < p_cold:
        # COLD batch: 100% masked canvas, predict the full joint in one shot.
        x_cold = torch.full_like(batch_ids, mask_id)
        logits = model(x_cold)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               batch_ids.reshape(-1))
        info["cold"] = True
        info["l_cold"] = loss.detach()
        return lambda_cold * loss, info

    # Standard denoising batch (unchanged regime).
    noisy, targets, mask = mask_batch(batch_ids, mask_id, t_min, t_max,
                                      generator=generator)
    logits = model(noisy)
    loss = F.cross_entropy(logits[mask], targets[mask])
    info["cold"] = False
    info["l_denoise"] = loss.detach()
    return loss, info
