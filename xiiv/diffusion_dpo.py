"""Diffusion-DPO for XIIV's masked-diffusion objective.

Vanilla DPO is derived for autoregressive log-probabilities. XIIV is a
masked diffusion LM (LLaDA-style absorbing state): there is no
left-to-right logprob to contrast. The professionally correct adaptation
is the Diffusion-DPO objective (Wallace et al., 2024): the denoising
loss upper-bounds NLL, so the *implicit reward* of a completion is the
policy's improvement over the reference model in denoising loss.

For a preference pair (prompt x, preferred y_w, dispreferred y_l):

    r_theta(y) = -(L_theta(y|x) - L_ref(y|x))      # lower loss = higher reward
    loss       = -log sigma(beta * (r_theta(y_w) - r_theta(y_l)))

Design notes (all deliberate):
- One corruption sample per completion, SHARED between policy and
  reference: this is the standard variance-reduction trick; without it
  the loss drowns in masking noise.
- Only response tokens are corrupted and scored. The prompt is
  conditioning (clean, frozen) — never a training target. Pass
  ``prefix_len`` = prompt length in tokens.
- The reference model must be frozen (no_grad) and is typically the SFT
  (or grown) checkpoint the policy starts from. Beta is the KL leash.
- Contract coupling (the XIIV-native extension): preference pairs where
  y_l violates a contract and y_w satisfies it teach the model to
  satisfy contracts *softly*; the sampler still enforces them *hard*.
  Shaping + enforcement, defense in depth.
"""

import torch
import torch.nn.functional as F

from .diffusion import mask_batch


def _masked_ce_per_sample(logits, targets, mask):
    """(B,) mean cross-entropy over masked positions, per sequence."""
    ce = F.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none"
    )  # (B, L)
    num = (ce * mask.float()).sum(dim=1)
    den = mask.float().sum(dim=1).clamp_min(1.0)
    return num / den


def per_sample_diffusion_loss(model, ids, mask_id, prefix_len=0,
                              t_min=0.15, t_max=1.0, generator=None,
                              _corruption=None):
    """(B,) denoising loss, one value per sequence.

    Args:
        _corruption: optional precomputed (noisy, targets, mask) triple —
            pass the same triple to policy and reference for variance
            reduction. If None, a fresh corruption is sampled.
    """
    if _corruption is None:
        noisy, targets, mask = mask_batch(ids, mask_id, t_min, t_max,
                                          generator=generator)
    else:
        noisy, targets, mask = _corruption
    if prefix_len > 0:
        # Prompt is conditioning: never corrupted, never scored.
        mask = mask.clone()
        mask[:, :prefix_len] = False
        noisy = noisy.clone()
        noisy[:, :prefix_len] = ids[:, :prefix_len]
    logits = model(noisy)
    return _masked_ce_per_sample(logits, targets, mask)


def diffusion_dpo_loss(model, ref_model, ids_w, ids_l, mask_id,
                       prefix_len_w=0, prefix_len_l=0,
                       beta=0.1, t_min=0.15, t_max=1.0, generator=None):
    """Diffusion-DPO loss for one batch of preference pairs.

    Args:
        model: policy (gradients on).
        ref_model: frozen reference (call under torch.no_grad()).
        ids_w / ids_l: (B, L) token ids of preferred / dispreferred
            completions, prompt included as prefix.
        prefix_len_w / prefix_len_l: prompt lengths (int or (B,) tensor).
        beta: KL leash to the reference (typical 0.05–0.5).

    Returns (loss, info dict with margin and per-side losses).
    """
    # One shared corruption per completion -> policy and reference see
    # exactly the same denoising problem.
    noisy_w, tgt_w, mask_w = mask_batch(ids_w, mask_id, t_min, t_max,
                                        generator=generator)
    noisy_l, tgt_l, mask_l = mask_batch(ids_l, mask_id, t_min, t_max,
                                        generator=generator)

    def _prep(prefix_len, noisy, mask, ids):
        if prefix_len:
            mask = mask.clone()
            noisy = noisy.clone()
            if isinstance(prefix_len, int):
                mask[:, :prefix_len] = False
                noisy[:, :prefix_len] = ids[:, :prefix_len]
            else:
                for b, pl in enumerate(prefix_len):
                    mask[b, :int(pl)] = False
                    noisy[b, :int(pl)] = ids[b, :int(pl)]
        return noisy, mask

    noisy_w, mask_w = _prep(prefix_len_w, noisy_w, mask_w, ids_w)
    noisy_l, mask_l = _prep(prefix_len_l, noisy_l, mask_l, ids_l)

    lw_theta = _masked_ce_per_sample(model(noisy_w), tgt_w, mask_w)
    ll_theta = _masked_ce_per_sample(model(noisy_l), tgt_l, mask_l)
    with torch.no_grad():
        lw_ref = _masked_ce_per_sample(ref_model(noisy_w), tgt_w, mask_w)
        ll_ref = _masked_ce_per_sample(ref_model(noisy_l), tgt_l, mask_l)

    # Implicit rewards: improvement over reference (lower loss = better).
    r_w = -(lw_theta - lw_ref)
    r_l = -(ll_theta - ll_ref)
    margin = beta * (r_w - r_l)
    loss = -F.logsigmoid(margin).mean()
    return loss, {
        "margin": margin.detach().mean().item(),
        "acc": (margin.detach() > 0).float().mean().item(),
        "lw_theta": lw_theta.detach().mean().item(),
        "ll_theta": ll_theta.detach().mean().item(),
    }
