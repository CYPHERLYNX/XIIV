"""NCM probe 2 — attacking the concept-inference bottleneck.

Probe 1 (research/ncm_prototype_probe.py) found: coarse-to-fine concept
conditioning is net NEGATIVE with hard argmax concepts (0.650 vs flat
0.672) even though oracle concepts hit the 0.850 noise ceiling. The
bottleneck is concept inference quality (79% concept accuracy; one
wrong hard concept misleads its whole block).

This probe tests two inference-time fixes on the SAME task, data, model
class and training recipe as probe 1 (imported from the probe-1 module,
so templates/seeds/training are identical by construction):

  (a) ITERATIVE CONCEPT REFINEMENT — concepts and tokens co-evolve.
      Infer concepts, denoise a share of the canvas, re-infer concepts
      from the partially denoised canvas, repeat for K=3 rounds within
      the same total denoise-step budget.
  (b) SOFT CONCEPT EMBEDDINGS — condition on the coarse head's full
      probability distribution (probability-weighted concept embedding)
      instead of a hard argmax, so an uncertain concept contributes a
      blend instead of one possibly-wrong commitment.

Baselines re-run in-process under identical weights/eval seed: FLAT,
C2F-hard (probe 1's variant), C2F-oracle (upper bound). Primary budget:
16 denoise steps (probe 1's largest); 8 steps also reported.

RAM note: at run time MemAvailable was ~0.4GB (< the ~1GB shrink
threshold). Footprint was shrunk operationally — torch threads capped
at 2, models trained/evaluated sequentially and freed between phases —
while the comparability-critical config (data, d=128, 500 steps,
bs=64, eval n=256, 25% visible, eval seed 99) is IDENTICAL to probe 1.
"""
import gc
import importlib.util
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(2)

_spec = importlib.util.spec_from_file_location(
    "ncm_probe1", Path(__file__).with_name("ncm_prototype_probe.py"))
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)

LEN, BLOCK, NBLOCK = p1.LEN, p1.BLOCK, p1.NBLOCK
K_TEMPLATES, D, MASK = p1.K_TEMPLATES, p1.D, p1.MASK


def infer_concept_logits(m, x):
    """Coarse-head logits per block from the current canvas (no concept
    conditioning — the probe-1 inference path)."""
    _, h = m(x, torch.full((x.size(0), NBLOCK), K_TEMPLATES,
                           dtype=torch.long))
    hb = h.reshape(x.size(0), NBLOCK, BLOCK, D).mean(dim=2)
    return m.coarse_head(hb)  # (B, NBLOCK, K)


def forward_soft(m, x, probs):
    """Token logits conditioned on soft concept distributions:
    ce = sum_k p_k * concept_emb[k], broadcast per block."""
    h = m.emb(x) + m.pos.weight[: x.size(1)].unsqueeze(0)
    ce = probs @ m.concept_emb.weight[:K_TEMPLATES]  # (B, NBLOCK, D)
    h = h + ce.repeat_interleave(BLOCK, dim=1)
    h = m.trunk(h)
    return m.head(h)


def _unmask_steps(m, x, remaining, n_sub, cond_fn):
    """Run n_sub confidence-unmasking sub-steps. cond_fn(m, x) -> logits."""
    for _ in range(n_sub):
        if not remaining.any():
            break
        logits = cond_fn(m, x)
        probs = logits.softmax(-1)
        conf, pred = probs.max(-1)
        conf = torch.where(remaining, conf, torch.full_like(conf, -1.0))
        per_step = max(1, int(remaining.sum(dim=1).float().mean().item())
                       // max(1, n_sub))
        k = min(per_step, int(remaining.sum(dim=1).max().item()))
        if k <= 0:
            break
        idx = conf.topk(k, dim=1).indices
        x.scatter_(1, idx, pred.gather(1, idx))
        remaining.scatter_(1, idx, False)
    return x, remaining


@torch.no_grad()
def eval_variant(m, mode, n_steps, n=256, seed=99, visible_frac=0.25,
                 rounds=3):
    """mode in {flat, hard, oracle, iter, soft}. Returns
    (token_acc, concept_acc_first, concept_acc_final)."""
    gen = torch.Generator().manual_seed(seed)
    x_true, block_ids = p1.make_batch(n, gen)
    visible = torch.rand(n, LEN, generator=gen) < visible_frac
    x = torch.where(visible, x_true, torch.full_like(x_true, MASK))
    remaining = ~visible
    cacc_first = cacc_final = None

    if mode == "flat":
        x, remaining = _unmask_steps(m, x, remaining, n_steps,
                                     lambda mm, xx: mm(xx, None)[0])
    elif mode == "oracle":
        concepts = block_ids.clone()
        cacc_first = cacc_final = 1.0
        x, remaining = _unmask_steps(
            m, x, remaining, n_steps,
            lambda mm, xx: mm(xx, concepts)[0])
    elif mode == "hard":
        concepts = infer_concept_logits(m, x).argmax(-1)
        cacc_first = cacc_final = (concepts == block_ids).float().mean().item()
        x, remaining = _unmask_steps(
            m, x, remaining, n_steps,
            lambda mm, xx: mm(xx, concepts)[0])
    elif mode == "soft":
        probs = infer_concept_logits(m, x).softmax(-1)
        cacc_first = cacc_final = (
            probs.argmax(-1) == block_ids).float().mean().item()
        x, remaining = _unmask_steps(
            m, x, remaining, n_steps,
            lambda mm, xx: forward_soft(mm, xx, probs))
    elif mode == "iter":
        # K rounds: re-infer concepts from the current canvas, then spend
        # an equal share of the step budget denoising under them.
        base = n_steps // rounds
        shares = [base] * rounds
        shares[-1] += n_steps - base * rounds
        for r in range(rounds):
            concepts = infer_concept_logits(m, x).argmax(-1)
            acc_r = (concepts == block_ids).float().mean().item()
            if r == 0:
                cacc_first = acc_r
            cacc_final = acc_r
            x, remaining = _unmask_steps(
                m, x, remaining, shares[r],
                lambda mm, xx, c=concepts: mm(xx, c)[0])
            if not remaining.any():
                break
    else:
        raise ValueError(mode)

    if remaining.any():  # fill the rest in one shot
        if mode == "flat":
            logits = m(x, None)[0]
        elif mode == "soft":
            logits = forward_soft(m, x, probs)
        elif mode == "oracle":
            logits = m(x, block_ids)[0]
        else:  # hard / iter: condition on latest inferred concepts
            logits = m(x, infer_concept_logits(m, x).argmax(-1))[0]
        x = torch.where(remaining, logits.argmax(-1), x)
    acc = (x[~visible] == x_true[~visible]).float().mean().item()
    return acc, cacc_first, cacc_final


def main():
    t0 = time.time()
    print("training FLAT (probe-1 recipe) ...", flush=True)
    flat, flat_losses = p1.train_model(False)
    fa16, _, _ = eval_variant(flat, "flat", 16)
    fa8, _, _ = eval_variant(flat, "flat", 8)
    print(f"flat: acc@16={fa16:.3f} acc@8={fa8:.3f} "
          f"train loss(last50)={sum(flat_losses[-50:]) / 50:.3f}", flush=True)
    del flat
    gc.collect()

    print("training C2F (probe-1 recipe) ...", flush=True)
    c2f, c2f_losses = p1.train_model(True)
    print(f"c2f train loss(last50)={sum(c2f_losses[-50:]) / 50:.3f}",
          flush=True)
    rows = {}
    for mode in ("hard", "iter", "soft", "oracle"):
        a16, cf, cl = eval_variant(c2f, mode, 16)
        a8, _, _ = eval_variant(c2f, mode, 8)
        rows[mode] = (a16, a8, cf, cl)
        print(f"{mode:>7}: acc@16={a16:.3f} acc@8={a8:.3f} "
              f"concept_acc first={cf if cf is None else round(cf, 3)} "
              f"final={cl if cl is None else round(cl, 3)}", flush=True)
    print(f"{'flat':>7}: acc@16={fa16:.3f} acc@8={fa8:.3f}")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
