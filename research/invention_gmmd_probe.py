"""Guarantee-MMD post-training — PILOT (Part-B experiment, 2026-10-22).

Candidate (queued from the 2026-10-21 Part-A pick): post-train a diffusion LM
with an added MMD term that pulls the SAMPLER's output distribution — in
frozen-model feature space — toward the region where the EXTERNAL verifier
accepts (certificate-valid generations). Perplexity is the wrong objective for
XIIV; the verifier INTERVENTION RATE is. This makes the training objective
match the metric. A dedicated prior-art sweep (research_scout_2026-10-22_mmd_
priorart.md, running in parallel) adjudicates the novelty claim; this probe
tests the MECHANISM at toy scale regardless.

Design (CPU-only, toy scale, mirrors the gate-1/2 harness so the numbers are
comparable):
  - Same tiny XIIV (d=64, 2 layers), template data, 250 brief training steps,
    family-A constraints, 48-token canvas with frozen anchors every 12.
  - Phase 0: confirm the calibrated family on 30 FRESH drafts; base rate must
    land in [0.15, 0.85] or the run is INCONCLUSIVE (same band as gate 2).
  - Phase 1: freeze the brief-trained model; sample N_BANK drafts; label with
    the EXTERNAL verifier `check_constraints`; keep the ACCEPTED drafts'
    features as the reference bank. Feature = mean-pooled final_norm hidden
    states over non-frozen positions of the completed draft. Bank is
    standardized once (zero mean / unit variance per dim).
  - Phase 2: clone the brief-trained model into K arms; post-train each with
    loss = CE(masked LM) + lambda * MMD_RBF(feats(own drafts), bank).
    Per post-train step: draft with the CURRENT model (no_grad, non-
    differentiable sampling stays outside the graph), forward with grad,
    MMD against the fixed bank (median-heuristic bandwidth per step).
    The MMD gradient flows through the model's REPRESENTATION of its own
    samples: distribution-level self-alignment toward the accepted region.
    No structural enforcement anywhere — the verifier stays external.
  - Phase 3: held-out evaluation — N_EVAL=200 fresh-seed free drafts per arm
    (paired seeds across arms), intervention rate via check_constraints.
    Guardrails: diversity (unique-token ratio within 20% of control) and
    CE on a held-out template batch (within 2x of control) — MMD must not
    win by collapsing the generator.

Pre-registered verdict:
  PASS: some lambda>0 arm has intervention rate < control (lambda=0) with
        paired McNemar p < 0.05 AND both guardrails hold.
  FAIL: no significant reduction, or any guardrail broken -> "Guarantee-MMD
        does not reduce the intervention rate at toy scale; do NOT advance
        the mechanism claim without a stronger testbed."

Honesty notes:
  - The bank is marginal (accepted drafts only); there is no per-input
    pairing of valid/invalid completions. This is deliberate: MMD matches
    DISTRIBUTIONS, not pairs — that is the claim's distinctive content vs
    contrastive or RL-with-verifier-reward formulations (the prior-art
    sweep adjudicates whether this distinction is novel).
  - Toy scale: one constraint family, one bank, brief training. A PASS
    licenses a larger follow-up, not an invention filing.
"""
import copy
import math
import time

import torch
import torch.nn.functional as F

from xiiv.config import ModelConfig
from xiiv.diffusion import sample_trajectory, check_constraints
from xiiv.model import XIIVModel

VOCAB = 256
LEN = 48
MASK_ID = 254
EOS_ID = 255
DRAFT_STEPS = 16
N_PILOT = 30
N_BANK = 512
POST_STEPS = 300
POST_BS = 32
N_EVAL = 200
LAMBDAS = (0.0, 0.3, 1.0, 3.0)


def tiny_cfg():
    return ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2,
                       n_heads=4, ffn_dim=128, mask_id=MASK_ID,
                       eos_id=EOS_ID, seq_len=64)


def make_data(n, gen):
    templates = torch.randint(0, 200, (10, 8), generator=gen)
    bids = torch.randint(0, 10, (n, 6), generator=gen)
    x = templates[bids].reshape(n, LEN).clone()
    noise = torch.rand(n, LEN, generator=gen) < 0.1
    x = torch.where(noise, torch.randint(0, 200, (n, LEN), generator=gen), x)
    return x


def common_tokens(k=12):
    gen = torch.Generator().manual_seed(7)
    data = make_data(4096, gen)
    counts = torch.bincount(data.reshape(-1), minlength=VOCAB)
    return torch.argsort(counts, descending=True)[:k].tolist()


def calibrated_constraints(common):
    return {"required_ids": common[:1], "forbidden_ids": common[1:2],
            "at_most": {common[2]: 3}}


def make_canvas(gen):
    frozen = torch.zeros(LEN, dtype=torch.bool)
    frozen[::12] = True
    canvas = torch.full((LEN,), MASK_ID, dtype=torch.long)
    canvas[frozen] = torch.randint(0, 200, (int(frozen.sum()),),
                                   generator=gen)
    anchors = {int(p): int(canvas[p]) for p in torch.where(frozen)[0]}
    return canvas, frozen, anchors


def train_briefly(m, steps=250, bs=64):
    gen = torch.Generator().manual_seed(7)
    data = make_data(4096, gen)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(0, 4096, (bs,))
        x = data[idx]
        mask = torch.rand(bs, LEN) < 0.5
        xm = torch.where(mask, torch.full_like(x, MASK_ID), x)
        loss = F.cross_entropy(m(xm).reshape(-1, VOCAB), x.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def draft(model, canvas, frozen, seed):
    fz = frozen.clone()
    ids, conf = sample_trajectory(
        model, LEN, MASK_ID, EOS_ID, steps=DRAFT_STEPS,
        temperature=1.0, top_p=0.95, seed=seed,
        prompt_ids=canvas.clone(), frozen=fz,
        constraints=None, device="cpu")
    return ids, conf


def intervention(ids, constraints):
    res = check_constraints(ids, constraints, MASK_ID)
    return 0 if all(res.values()) else 1, res


def pooled_features(model, ids_batch, frozen_batch):
    """Mean-pooled final_norm hidden states over non-frozen positions.
    ids_batch: (B, T) long; frozen_batch: (B, T) bool."""
    captured = {}

    def hook(mod, inp, out):
        captured["h"] = out
    handle = model.final_norm.register_forward_hook(hook)
    try:
        model(ids_batch)
    finally:
        handle.remove()
    h = captured["h"]                      # (B, T, d), grad flows
    w = (~frozen_batch).float().unsqueeze(-1)
    return (h * w).sum(1) / w.sum(1).clamp_min(1.0)


def mmd_rbf(x, y):
    """Biased V-statistic MMD^2 with median-heuristic bandwidth."""
    z = torch.cat([x, y], 0)
    d2 = torch.cdist(z, z, p=2).pow(2)
    med = d2.median().clamp_min(1e-6)
    K = torch.exp(-d2 / (2.0 * med))
    nx, ny = x.shape[0], y.shape[0]
    Kxx = K[:nx, :nx].mean()
    Kyy = K[nx:, nx:].mean()
    Kxy = K[:nx, nx:].mean()
    return Kxx + Kyy - 2.0 * Kxy


def build_bank(model, common, n_bank):
    """Sample drafts with the frozen model; keep ACCEPTED drafts' features."""
    model.eval()
    feats, n_acc, n_rej = [], 0, 0
    gen = torch.Generator().manual_seed(777001)
    with torch.no_grad():
        for i in range(n_bank):
            c = calibrated_constraints(common)
            canvas, frozen, anchors = make_canvas(gen)
            full_c = {**c, "anchors": anchors}
            ids, _ = draft(model, canvas, frozen, seed=30000 + i)
            ok, _ = intervention(ids, full_c)
            if ok == 0:
                f = pooled_features(model, ids.unsqueeze(0),
                                    frozen.unsqueeze(0))
                feats.append(f.squeeze(0))
                n_acc += 1
            else:
                n_rej += 1
    bank = torch.stack(feats) if feats else torch.zeros(0, model.cfg.d_model)
    if bank.shape[0] >= 2:
        bank = (bank - bank.mean(0)) / bank.std(0).clamp_min(1e-6)
    print(f"bank: accepted={n_acc} rejected={n_rej} "
          f"accept_rate={n_acc / max(1, n_bank):.3f}", flush=True)
    return bank


def post_train(model, bank, lam, steps=POST_STEPS, bs=POST_BS, ce_steps=250):
    """loss = CE(masked LM on template data) + lam * MMD(feats(own drafts),
    bank). CE data = fresh template batches (same distribution as the brief
    training); drafts for the MMD term are generated by the current model."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    data_gen = torch.Generator().manual_seed(999 + int(lam * 1000))
    ce_data = make_data(max(4096, steps * bs), data_gen)
    draft_gen = torch.Generator().manual_seed(31337 + int(lam * 1000))
    bank_f = bank
    for s in range(steps):
        # CE half: masked LM on template data
        idx = torch.randint(0, ce_data.shape[0], (bs,))
        x = ce_data[idx]
        mask = torch.rand(bs, LEN) < 0.5
        xm = torch.where(mask, torch.full_like(x, MASK_ID), x)
        ce = F.cross_entropy(model(xm).reshape(-1, VOCAB), x.reshape(-1))
        # MMD half: features of the model's OWN fresh drafts vs the bank
        with torch.no_grad():
            ids_list, fz_list = [], []
            for b in range(bs):
                canvas, frozen, _ = make_canvas(draft_gen)
                ids, _ = draft(model, canvas, frozen,
                               seed=40000 + s * bs + b)
                ids_list.append(ids); fz_list.append(frozen)
            ids_b = torch.stack(ids_list)
            fz_b = torch.stack(fz_list)
        feats = pooled_features(model, ids_b, fz_b)
        mmd = mmd_rbf(feats, bank_f)
        loss = ce + lam * mmd
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 100 == 0:
            print(f"  lam={lam}: step {s + 1}/{steps} ce={ce.item():.3f} "
                  f"mmd={mmd.item():.4f}", flush=True)
    return model


def mcnemar(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0, 0.0
    chi2 = (abs(b01 - b10) - 1.0) ** 2 / n
    z = math.sqrt(chi2)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return p, chi2


@torch.no_grad()
def heldout_ce(model):
    gen = torch.Generator().manual_seed(5150)
    x = make_data(512, gen)
    mask = torch.rand(512, LEN) < 0.5
    xm = torch.where(mask, torch.full_like(x, MASK_ID), x)
    return F.cross_entropy(model(xm).reshape(-1, VOCAB),
                           x.reshape(-1)).item()


def main():
    t_all = time.perf_counter()
    torch.manual_seed(0)
    cfg = tiny_cfg()
    base = XIIVModel(cfg)
    base.init_weights()
    print("brief training base model...", flush=True)
    base = train_briefly(base)
    base.eval()

    # ---- Phase 0: confirm calibrated family on FRESH drafts ----
    common = common_tokens()
    pilot_gen = torch.Generator().manual_seed(20261022)
    fails = 0
    for i in range(N_PILOT):
        c = calibrated_constraints(common)
        canvas, frozen, anchors = make_canvas(pilot_gen)
        ids, _ = draft(base, canvas, frozen, seed=61000 + i)
        f, _ = intervention(ids, {**c, "anchors": anchors})
        fails += f
    base_rate = fails / N_PILOT
    print(f"pilot: base intervention rate = {base_rate:.2f} (n={N_PILOT})",
          flush=True)
    if not (0.15 <= base_rate <= 0.85):
        print("INCONCLUSIVE: base rate outside [0.15, 0.85]; cannot decide "
              "the mechanism at toy scale.")
        return

    # ---- Phase 1: accepted-set feature bank (frozen model) ----
    bank = build_bank(base, common, N_BANK)
    if bank.shape[0] < 30:
        print(f"INCONCLUSIVE: only {bank.shape[0]} accepted drafts; bank "
              "too thin for MMD.")
        return

    # ---- Phase 2: post-train arms (checkpointed: re-runs resume) ----
    import os
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints_gmmd")
    os.makedirs(ckpt_dir, exist_ok=True)

    def ckpt_path(lam):
        return os.path.join(ckpt_dir, f"arm_lam{lam}.pt")

    arms = {}
    for lam in LAMBDAS:
        if os.path.exists(ckpt_path(lam)):
            print(f"arm lam={lam}: loading checkpoint "
                  f"(post-training already done)", flush=True)
            m = copy.deepcopy(base)
            m.load_state_dict(torch.load(ckpt_path(lam),
                                         map_location="cpu"))
        else:
            print(f"post-training arm lam={lam}...", flush=True)
            m = copy.deepcopy(base)
            m = post_train(m, bank, lam)
            torch.save(m.state_dict(), ckpt_path(lam))
            print(f"  checkpoint saved: {ckpt_path(lam)}", flush=True)
        m.eval()
        arms[lam] = m
        import gc
        gc.collect()

    # ---- Phase 3: held-out paired evaluation ----
    gen = torch.Generator().manual_seed(808808)
    fails = {lam: 0 for lam in LAMBDAS}
    # discordant counts: (lam-arm vs control) per lam>0
    disc = {lam: [0, 0] for lam in LAMBDAS if lam > 0}
    uniq = {lam: 0 for lam in LAMBDAS}
    tot_tok = {lam: 0 for lam in LAMBDAS}
    for t in range(N_EVAL):
        c = calibrated_constraints(common)
        canvas, frozen, anchors = make_canvas(gen)
        full_c = {**c, "anchors": anchors}
        outs = {}
        for lam in LAMBDAS:
            ids, _ = draft(arms[lam], canvas, frozen, seed=70000 + t)
            outs[lam] = ids
            f, _ = intervention(ids, full_c)
            fails[lam] += f
            open_ids = ids[~frozen]
            uniq[lam] += len(torch.unique(open_ids))
            tot_tok[lam] += open_ids.numel()
        fc = intervention(outs[0.0], full_c)[0]
        for lam in LAMBDAS:
            if lam == 0.0:
                continue
            fl = intervention(outs[lam], full_c)[0]
            if fc and not fl:
                disc[lam][0] += 1
            elif fl and not fc:
                disc[lam][1] += 1
        if (t + 1) % 50 == 0:
            print(f"eval {t + 1}/{N_EVAL} " +
                  " ".join(f"lam={lam}:{fails[lam] / (t + 1):.3f}"
                           for lam in LAMBDAS), flush=True)

    n = N_EVAL
    rates = {lam: fails[lam] / n for lam in LAMBDAS}
    ce_hold = {lam: heldout_ce(arms[lam]) for lam in LAMBDAS}
    div = {lam: uniq[lam] / tot_tok[lam] for lam in LAMBDAS}
    print(f"\nintervention rates (n={n}): " +
          " ".join(f"lam={lam}:{rates[lam]:.3f}" for lam in LAMBDAS))
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        p, chi2 = mcnemar(disc[lam][0], disc[lam][1])
        print(f"McNemar lam={lam} vs control: b01={disc[lam][0]}, "
              f"b10={disc[lam][1]}, chi2={chi2:.2f}, p={p:.4f}")
    print("held-out CE: " +
          " ".join(f"lam={lam}:{ce_hold[lam]:.3f}" for lam in LAMBDAS))
    print("diversity (unique-token ratio): " +
          " ".join(f"lam={lam}:{div[lam]:.3f}" for lam in LAMBDAS))

    # ---- Pre-registered verdict ----
    passed = []
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        p, _ = mcnemar(disc[lam][0], disc[lam][1])
        guard_div = div[lam] >= 0.8 * div[0.0]
        guard_ce = ce_hold[lam] <= 2.0 * ce_hold[0.0]
        if p < 0.05 and rates[lam] < rates[0.0] and guard_div and guard_ce:
            passed.append(lam)
        print(f"lam={lam}: rate<{rates[0.0]:.3f}? {rates[lam] < rates[0.0]}, "
              f"p<0.05? {p < 0.05}, div-guard? {guard_div}, "
              f"ce-guard? {guard_ce}")
    if passed:
        print(f"VERDICT: PILOT PASS — lam in {passed} reduces the "
              "intervention rate vs pure-CE post-training with guardrails "
              "intact. Licenses a larger follow-up; NOT an invention "
              "filing (novelty pending the prior-art sweep).")
    else:
        print("VERDICT: PILOT FAIL — Guarantee-MMD does not reduce the "
              "intervention rate at toy scale; do NOT advance the "
              "mechanism claim without a stronger testbed.")
    print(f"total wall-clock: {(time.perf_counter() - t_all) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
