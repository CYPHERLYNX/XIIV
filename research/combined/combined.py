"""BASC + CSC combined leg — dual-failure-mode training step.

Attacks BOTH measured failure modes in one training step, by step-mixture:

  - Premature-commitment derailment (BASC, research/lever2/): on ordinary
    denoising steps the model fills masked positions from its OWN sampled
    predictive distribution and must both recover the targets from that
    corrupted canvas (l_self) and keep its corrupted-canvas beliefs anchored
    to its clean-canvas beliefs (l_anchor, KL).
  - The step-0 void (CSC, research/lever3/): with probability p_cold the
    step is a COLD step — a 100%-masked canvas whose full joint must be
    predicted in one shot (l_cold), training the unconditional joint prior
    the sampler draws on at inference step 0.

One step = exactly one branch, never both: the cold branch is the CSC cold
objective verbatim; the denoise branch is the BASC triple term verbatim.
No new mechanism is introduced here — this is the composition the lever-3
RUNBOOK prescribes as the next leg if legs 2 and 3 each show loss movement
without generation movement. One-variable discipline is preserved: swapping
this in as general_loss() changes ONLY the training step.

Prior-art position (honest, read before claiming anything):
- HiLo-RD (arXiv:2610.36901): cross-budget self-distillation across compute
  budgets. Different: no budget conditioning, no teacher/student split —
  single model, single budget, one step.
- OPSD (Zhao et al., 2026) / dOPSD (arXiv:2607.04428): on-policy
  self-distillation for diffusion LMs. Closest relatives of the BASC half:
  BASC differs by stochastic (temperature-sampled, not argmax) fills plus
  the KL belief-anchor term against the clean-canvas beliefs.
- T3D (Feb 2026): trajectory self-distillation + path-consistency
  regularization; explicitly names the train/inference masking mismatch
  that BASC's on-policy fills and CSC's cold batches both respond to.
  Different: T3D distills teacher ROLLOUTS (x0, xt pairs along generative
  trajectories); the combined leg never unrolls a trajectory — every term
  is a single-step objective.
- CDLM (arXiv:2605.00161, ICML 2026): Multi-Path Discrete Consistency —
  trains the denoiser to be path-invariant in expectation across stochastic
  posterior bridges, teacher-free. Adjacent goal (consistency of beliefs
  across corruption regimes) but a different mechanism: no cold-start
  joint term, no sampled-corruption recovery term.
- Clover-35B (DiffuSpace, 2026-10-28): confidence-gated block-parallel
  diffusion (commit at >=0.95 confidence), AR->diffusion conversion from
  Qwen3.6-35B-A3B. Inspiration for SAMPLER commit policy only; nothing of
  its mechanism is used here. Vendor claims unverified.
This is a documented candidate composition, NOT a proven world-first;
broader literature/patent review is still required before any such claim.

Interface: drop-in replacement for the general_loss() closure in the
Kaggle trainers (same (loss, info) return contract as
xiiv.diffusion.diffusion_loss). Swap (inside the trainer main(), after
`tok` exists):

    sys.path.insert(0, os.path.join(args.repo_dir, "research", "combined"))
    from combined import combined_loss
    def general_loss(batch_ids):
        return combined_loss(expanded, batch_ids, tok.mask_id,
                             p_cold=0.15, lambda_cold=1.0,
                             lambda_self=1.0, lambda_anchor=0.5)

Scale behavior UNTESTED — CPU smoke only (test_combined.py). What the leg
measures: whether the two failure modes interact (each lever alone moves
loss but not generation) or whether the generation gap needs a different
family of mechanism entirely.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO_ROOT, "lever2"),
           os.path.join(_REPO_ROOT, "lever3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

from basc import belief_anchored_loss


def combined_loss(model_fn, batch_ids: torch.Tensor, mask_id: int,
                  p_cold: float = 0.15, lambda_cold: float = 1.0,
                  t_min: float = 0.15, t_max: float = 0.90,
                  lambda_self: float = 1.0, lambda_anchor: float = 0.5,
                  tau: float = 1.0, generator=None):
    """Dual-failure-mode diffusion step: BASC denoise steps + CSC cold steps.

    Args:
        model_fn: callable mapping token ids -> logits over vocab
                  (e.g. the expanded exact-1B model).
        batch_ids: (B, L) clean token ids.
        mask_id: the [MASK] token id.
        p_cold: fraction of steps that are COLD steps (CSC cold branch).
        lambda_cold: weight of the cold joint term.
        t_min/t_max: uniform mask-ratio range for denoise steps (BASC).
        lambda_self/lambda_anchor: BASC term weights.
        tau: sampling temperature for BASC on-policy fills.
        generator: optional torch.Generator for reproducibility.

    Returns:
        (loss, info): scalar loss tensor and a dict. Cold steps carry
        {"cold": True, "l_cold": <detached>}; denoise steps carry the full
        BASC contract plus "cold": False.
    """
    if generator is not None:
        u = torch.rand((), generator=generator).item()
    else:
        u = torch.rand(()).item()

    if u < p_cold:
        # COLD step — verbatim the CSC cold branch: 100%-masked canvas,
        # predict the full joint in one shot.
        x_cold = torch.full_like(batch_ids, mask_id)
        logits = model_fn(x_cold)
        l_cold = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 batch_ids.reshape(-1))
        info = {"cold": True, "l_cold": l_cold.detach()}
        return lambda_cold * l_cold, info

    # DENOISE step — verbatim the BASC triple term.
    loss, info = belief_anchored_loss(model_fn, batch_ids, mask_id,
                                      t_min=t_min, t_max=t_max,
                                      lambda_self=lambda_self,
                                      lambda_anchor=lambda_anchor,
                                      tau=tau, generator=generator)
    info = dict(info)
    info["cold"] = False
    return loss, info
