"""Belief-Anchored Self-Correction (BASC) — lever-2 training-step variant.

Target failure mode: premature-commitment derailment. During iterative
unmasking the sampler commits an early token that is wrong, and every
later step conditions on that error, so the whole trajectory's beliefs
collapse around it. Teacher-forced masked loss never shows the model
this regime — training context is always clean or randomly masked —
which is the standing reading for why four levers cut held-out masked
loss (-31% to -66%) while free generation stayed 0/5 on factual/code
probes.

Prior-art position (honest, read before claiming anything):
- The research scout flagged arXiv:2610.36901 (HiLo-RD) as naming the
  same failure mode. HiLo-RD is inspiration / prior-art awareness ONLY:
  BASC copies none of its mechanism — no cross-budget self-distillation,
  no teacher/student across compute budgets, no budget-conditioned
  training. Single model, single budget, single training step.
- OPSD (Zhao et al., 2026) / dOPSD (arXiv:2607.04428): on-policy
  self-distillation for diffusion LMs — the closest published relatives
  of BASC's on-policy fills. BASC differs by stochastic
  temperature-sampled fills (not argmax/mode) plus the KL belief-anchor
  term tying corrupted-canvas beliefs to clean-canvas beliefs; OPSD/dOPSD
  match trajectories/representations without that anchor.
- T3D (Feb 2026): trajectory self-distillation + path-consistency
  regularization; explicitly names the train/inference masking mismatch
  (random masking in training vs non-random heuristics at inference)
  that BASC's on-policy fills respond to. Different: T3D distills the
  teacher's own generative ROLLOUTS; BASC never unrolls a trajectory.
- xiiv/onpolicy_distill.py (OSCD, in-repo) trains self-correction on
  inputs corrupted with the model's own ARGMAX predictions, for SGBD
  block distillation. BASC differs in two ways: (1) STOCHASTIC sampled
  fills instead of argmax — the realistic sampler regime, where
  derailment actually happens; (2) the KL belief-anchor term below,
  which is the new mechanism here.
This is a documented candidate invention, NOT a proven world-first;
broader literature/patent review is still required before any such
claim.

Mechanism (one training step):
  1. (noisy, targets, mask) = mask_batch(x0, mask_id, t_min, t_max)
     logits_m = model(noisy)
     L_mask = CE(logits_m[mask], targets[mask])            # standard
  2. x_p = noisy with masked positions filled by SAMPLES from
     softmax(logits_m[mask] / tau), detached (no_grad). This is the
     canvas after one realistic stochastic denoising step: the context
     now contains the model's own plausible-but-possibly-wrong commits.
     logits_p = model(x_p)
     L_self = CE(logits_p[mask], targets[mask])            # recoverability
  3. L_anchor = mean KL( softmax(logits_m[mask]).detach()
                         || softmax(logits_p[mask]) )
     The derailment signature is belief collapse: one wrong commit
     swings the whole predictive distribution. The anchor penalizes the
     swing directly — beliefs on the corrupted canvas must stay close to
     beliefs on the clean canvas. This term uses no target tokens; it is
     pure belief-consistency regularization.
  4. L = L_mask + lambda_self * L_self + lambda_anchor * L_anchor

Interface: drop-in replacement for the general_loss() closure in
pretrain-jobs/pretrain_train.py (same (loss, info) return contract as
xiiv.diffusion.diffusion_loss, same mask-ratio sampling). One-variable
discipline: swapping this in changes ONLY the training step — data,
model, optimizer, schedule, and generation gates stay identical.

Swap (inside pretrain_train.py main(), after `tok` exists):
    sys.path.insert(0, os.path.join(args.repo_dir, "research", "lever2"))
    from basc import belief_anchored_loss
    def general_loss(batch_ids):
        return belief_anchored_loss(expanded, batch_ids, tok.mask_id)
"""

import torch
import torch.nn.functional as F

from xiiv.diffusion import mask_batch


@torch.no_grad()
def sampled_on_policy_corrupt(noisy: torch.Tensor, logits_m: torch.Tensor,
                              mask: torch.Tensor, tau: float = 1.0,
                              generator=None) -> torch.Tensor:
    """Build the on-policy corrupted input x_p with STOCHASTIC fills.

    Each masked position is filled by a sample from
    softmax(logits_m[mask] / tau) — the model's own predictive
    distribution at that position, not its argmax. This matches the
    inference regime (temperature/top-p sampling), where derailment is
    caused by plausible-but-wrong sampled commits rather than by the
    single most-likely token.

    Runs under no_grad: fills are fixed context for the second forward
    pass, never a gradient path.
    """
    flat_logits = logits_m[mask].float() / tau
    probs = F.softmax(flat_logits, dim=-1)
    drawn = torch.multinomial(probs, num_samples=1,
                              generator=generator).squeeze(-1)
    x_p = noisy.clone()
    x_p[mask] = drawn.to(x_p.dtype)
    return x_p


def belief_anchored_loss(model_fn, x0: torch.Tensor, mask_id: int,
                         t_min: float = 0.15, t_max: float = 1.0,
                         lambda_self: float = 1.0,
                         lambda_anchor: float = 0.5,
                         tau: float = 1.0,
                         generator=None):
    """BASC loss for one training step (full-model pretraining variant).

    Args:
        model_fn: callable mapping token ids -> logits over vocab
                  (e.g. the expanded exact-1B model).
        x0: (B, L) clean token ids.
        mask_id: the [MASK] token id.
        t_min/t_max: uniform range for the mask ratio (same as the
                     standard diffusion loss).
        lambda_self: weight of the recoverability term.
        lambda_anchor: weight of the KL belief-anchor term.
        tau: sampling temperature for the on-policy fills.
        generator: optional torch.Generator for reproducibility.

    Returns:
        (loss, info): scalar loss tensor and a dict with mask_ratio,
        n_masked, l_mask, l_self, l_anchor. Always returns the tuple,
        even when no positions were masked (degenerate guard keeps the
        graph alive, mirroring xiiv.diffusion.diffusion_loss).
    """
    noisy, targets, mask = mask_batch(x0, mask_id, t_min, t_max, generator)
    logits_m = model_fn(noisy)
    n_masked = int(mask.sum().item())
    if n_masked == 0:
        zero = logits_m.sum() * 0.0
        return zero, {"mask_ratio": 0.0, "n_masked": 0,
                      "l_mask": 0.0, "l_self": 0.0, "l_anchor": 0.0}
    l_mask = F.cross_entropy(logits_m[mask], targets[mask])
    x_p = sampled_on_policy_corrupt(noisy, logits_m, mask, tau=tau,
                                    generator=generator)
    logits_p = model_fn(x_p)
    l_self = F.cross_entropy(logits_p[mask], targets[mask])
    # Belief anchor: corrupted-canvas beliefs must not swing away from
    # clean-canvas beliefs. Detach the clean side — it is the anchor,
    # not a gradient path.
    l_anchor = F.kl_div(
        F.log_softmax(logits_p[mask].float(), dim=-1),
        F.softmax(logits_m[mask].float().detach(), dim=-1),
        reduction="batchmean",
    )
    loss = l_mask + lambda_self * l_self + lambda_anchor * l_anchor
    return loss, {"mask_ratio": float(mask.float().mean().item()),
                  "n_masked": n_masked,
                  "l_mask": float(l_mask.detach().item()),
                  "l_self": float(l_self.detach().item()),
                  "l_anchor": float(l_anchor.detach().item())}
