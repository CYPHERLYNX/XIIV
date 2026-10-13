"""
On-Policy Self-Corrective Distillation (OSCD) for SGBD blocks
==============================================================
Closes the train/inference mismatch at the heart of masked-diffusion
training, inside Seed-Grown Block Distillation (SGBD) Phase 2.

The mismatch: conventional masked-diffusion training corrupts clean text
with RANDOM masks, but at inference the model denoises iteratively — the
unmasked context it sees comes from its OWN earlier predictions, not from
random corruption. Training never shows the model its own mistakes, so it
never learns to correct them.

The fix (adapted from On-Policy Uniform Training, reported 2026 for
DMax/LLaDA-2.0 masked diffusion LMs): build a second noisy input from the
model's own predictions and train a self-correction loss on it.

What is new here (potential XIIV invention — prior-art check 2026-10-13):
    OPUT was demonstrated as a fine-tuning objective for a whole
    masked-diffusion model. OSCD embeds on-policy self-correction INSIDE
    independent block distillation: each ~115M-parameter block of the
    grown 1B student trains on cached (input, target) teacher pairs with
      (a) standard masked reconstruction, and
      (b) self-correction on inputs corrupted with the block's OWN
          predictions (stop-gradient),
    on a CPU-only 7GB box with full gradients per block. The combination
    (grown-from-seed + independent full-gradient blocks + cached teacher
    activations + on-policy self-correction + masked discrete-diffusion
    LM + CPU-only training) has no known published prior art. This is a
    documented candidate invention, NOT a proven world-first — broader
    literature/patent review is still required before any such claim.

Method (token level; applies to the SGBD output block that owns the LM
head — earlier blocks keep MSE activation distillation in sgbd.py):
    1. x0: clean token ids (cached teacher targets).
    2. x_m: random-mask corruption of x0, mask ratio t ~ U(t_min, t_max).
    3. logits_m = block(x_m);  L_mask = CE(logits_m[mask], x0[mask]).
    4. x_p: copy of x_m with masked positions filled by the block's OWN
       predictions: x_p[mask] = argmax(logits_m[mask]) (detached).
       Clean positions stay clean — exactly the inference situation,
       where unmasked context came from earlier denoising steps.
    5. logits_p = block(x_p);  L_self = CE(logits_p[mask], x0[mask]).
    6. L = L_mask + lambda_self * L_self.

Why it helps block distillation specifically: cached teacher pairs make
the TARGETS stable, but the INPUT distribution is still random-mask
noise. The self-correction term forces the block to recover clean tokens
even when its context is its own imperfect predictions — the exact regime
of iterative denoising and of SGBD cascade refinement, where each cascade
re-captures targets from the composed student's own activations.

Verification status (honest): the mechanism is unit-tested on a toy
block on CPU (tests/test_onpolicy_distill.py). It does NOT yet have
measured gains on the real 916M model — that requires SGBD Phase 2 with
real teacher pairs, where block CE before/after is the metric to watch.
"""

import torch
import torch.nn.functional as F

from xiiv.diffusion import mask_batch


@torch.no_grad()
def on_policy_corrupt(x_m: torch.Tensor, logits_m: torch.Tensor,
                      mask: torch.Tensor, generator=None,
                      sample_predictions: bool = False) -> torch.Tensor:
    """Build the on-policy corrupted input x_p.

    Copies the randomly-masked input and fills every masked position with
    the block's OWN prediction (argmax, or a sample when
    ``sample_predictions`` is True). Clean positions are untouched, so x_p
    looks like an intermediate denoising canvas at inference time.

    Runs under no_grad: predictions are fixed targets for the
    self-correction pass, never a gradient path.
    """
    if sample_predictions:
        probs = F.softmax(logits_m, dim=-1)
        flat = probs.reshape(-1, probs.size(-1))
        drawn = torch.multinomial(flat, num_samples=1,
                                  generator=generator).squeeze(-1)
        preds = drawn.reshape(probs.shape[:-1])
    else:
        preds = logits_m.argmax(dim=-1)
    x_p = x_m.clone()
    x_p[mask] = preds[mask]
    return x_p


def self_corrective_distill_loss(block_fn, x0: torch.Tensor, mask_id: int,
                                 t_min: float = 0.15, t_max: float = 1.0,
                                 lambda_self: float = 1.0,
                                 sample_predictions: bool = False,
                                 generator=None):
    """OSCD loss for one SGBD block training step (token level).

    Args:
        block_fn: callable mapping token ids -> logits over vocab
                  (the SGBD output block incl. LM head).
        x0: (B, L) clean token ids — cached teacher targets.
        mask_id: the [MASK] token id.
        t_min/t_max: uniform range for the mask ratio.
        lambda_self: weight of the self-correction term.
        sample_predictions: sample on-policy fills instead of argmax.
        generator: optional torch.Generator for reproducibility.

    Returns:
        (loss, info): scalar loss tensor and a dict with mask_ratio,
        n_masked, l_mask, l_self. Always returns the tuple, even when no
        positions were masked (degenerate guard keeps the graph alive).
    """
    noisy, targets, mask = mask_batch(x0, mask_id, t_min, t_max, generator)
    logits_m = block_fn(noisy)
    n_masked = int(mask.sum().item())
    if n_masked == 0:
        zero = logits_m.sum() * 0.0
        return zero, {"mask_ratio": 0.0, "n_masked": 0,
                      "l_mask": 0.0, "l_self": 0.0}
    l_mask = F.cross_entropy(logits_m[mask], targets[mask])
    x_p = on_policy_corrupt(noisy, logits_m, mask, generator=generator,
                            sample_predictions=sample_predictions)
    logits_p = block_fn(x_p)
    l_self = F.cross_entropy(logits_p[mask], targets[mask])
    loss = l_mask + lambda_self * l_self
    return loss, {"mask_ratio": float(mask.float().mean().item()),
                  "n_masked": n_masked,
                  "l_mask": float(l_mask.detach().item()),
                  "l_self": float(l_self.detach().item())}
