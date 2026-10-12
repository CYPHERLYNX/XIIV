"""Defensible diffusion pseudo-likelihood scorer for multiple-choice evaluation.

PROBLEM WITH THE OLD SCORER (xiiv/lm_eval_wrapper.py):
  mask_batch() applied random masking across the FULL sequence (context +
  continuation) with t ~ Uniform(0.15, 1.0), then the context region was
  unmasked. This left the continuation's EFFECTIVE masking ratio uncontrolled:
  - For short continuations, zero tokens might be masked -> degenerate 0/0 score
  - The masking ratio varied randomly per sample, adding noise to rankings
  - Different choices got different effective masking, biasing comparisons

THIS SCORER:
  1. Masks ONLY continuation positions (context is never in the random mask pool)
  2. Uses leave-one-out: each continuation token is scored with ALL other
     tokens visible. This is the diffusion-native analogue of teacher forcing.
  3. Deterministic: no random masking samples, fully reproducible rankings.
  4. Handles edge cases: empty continuation, single-token continuation.

METHODOLOGY CAVEAT (inherent, not a bug):
  With bidirectional attention, scoring token i sees continuation tokens j>i,
  unlike autoregressive scoring (sees only j<i). Diffusion NLL is therefore
  NOT directly comparable to AR logprobs. Rankings WITHIN diffusion models
  are valid; cross-architecture comparisons are directional only.

  This is documented prominently in COMPARISON_TABLE.md.
"""
import torch
import torch.nn.functional as F


def score_continuation(model, context_ids, cont_ids, mask_id,
                       device="cpu"):
    """Diffusion pseudo-likelihood of continuation given context.

    Leave-one-out: for each continuation position i, mask only position i,
    compute NLL of the true token. Mean over positions. Lower = better.

    Returns (nll, n_scored). n_scored=0 only if continuation is empty.
    """
    if len(cont_ids) == 0:
        return 0.0, 0
    ctx = torch.tensor([context_ids], dtype=torch.long, device=device)
    cont = torch.tensor([cont_ids], dtype=torch.long, device=device)
    n_cont = len(cont_ids)
    full = torch.cat([ctx, cont], dim=1)
    ctx_len = len(context_ids)

    model.eval()
    nlls = []
    with torch.no_grad():
        for i in range(n_cont):
            pos = ctx_len + i
            noisy = full.clone()
            noisy[0, pos] = mask_id
            logits = model(noisy)  # (1, T, V)
            # NLL of the true token at the masked position
            nll = F.cross_entropy(
                logits[0, pos:pos+1], full[0, pos:pos+1])
            nlls.append(nll.item())
    mean_nll = sum(nlls) / len(nlls)
    return mean_nll, len(nlls)


def rank_choices(model, context_ids, choice_ids_list, mask_id, device="cpu"):
    """Rank multiple choices by diffusion NLL (ascending = more likely first).

    Returns list of (choice_idx, nll) sorted by nll ascending.
    Deterministic: no randomness involved.
    """
    scored = []
    for idx, cont_ids in enumerate(choice_ids_list):
        nll, n = score_continuation(model, context_ids, cont_ids,
                                    mask_id, device=device)
        scored.append((idx, nll))
    scored.sort(key=lambda x: x[1])
    return scored
