"""Guarantee-MMD — stronger testbed (2026-10-22), pre-registered.

The pilot (research/invention_gmmd_results.md) FAILED with three
documented confounds; this testbed removes each by construction:

(a) DRIFT: pilot post-training CE ran on raw template data, sharpening
    every arm (incl. control) from 0.63 to ~0.98 intervention — the CE
    term fought the constraint. HERE: all arms post-train with CE on
    the base model's own ACCEPTED drafts only (rejection-sampling
    fine-tune), so CE reinforces constraint-satisfying behavior.
    STABILITY GATE (registered): if the RSFT control's post-training
    intervention rate is >= 0.90, the testbed is INVALID (drift
    recurred) and no mechanism verdict is drawn.
(b) MARGINAL-ONLY TARGET: pilot pulled toward a global accepted bank.
    HERE: per-input pairs — the same canvas sampled twice, one
    accepted + one rejected draft — and a PAIRED arm pulls each
    rejected sample's representation toward its own pair-mate's.
(c) COMPETENCE/STABILITY OF SIGNAL: features standardized with the
    accepted bank's statistics on BOTH sides (the pilot standardized
    the bank only — a scale mismatch in its MMD, fixed here and
    recorded as a deliberate deviation).

Arms (identical init, steps, lr, data order; only the added term
differs):
  rsft          — CE on accepted drafts (control)
  mmd_marginal  — rsft + 1.0 * MMD(feats(rejected pool batch),
                    accepted bank)          [pilot-style target]
  mmd_paired    — rsft + 1.0 * MMD(feats(rejected batch),
                    feats(their paired accepted mates))
MMD gradient path: identical to the pilot's — features are the
CURRENT model's re-encoding of fixed sampled ids (the pilot sampled
fresh ids per step but the gradient path is the same; fixed pools
make the three arms exactly comparable and the run affordable).

Eval: n=200 paired-seed free drafts per arm (fresh canvases),
intervention rate, McNemar vs control. Guardrails: diversity
(unique-token ratio within 20% of control), held-out CE within 2x
of control. PASS (per MMD arm): rate < control with McNemar p < 0.05
and guardrails intact. Anything else: FAIL for that arm.
"""
import copy
import importlib.util
import json
import os
import sys
import time

REPO = os.path.expanduser("~/workspace/ai-tools/corsac")
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location(
    "gmmd", os.path.join(REPO, "research/invention_gmmd_probe.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

LEN, MASK_ID, VOCAB = M.LEN, M.MASK_ID, M.VOCAB
POST_STEPS = 150
POST_BS = 32
LAM = 1.0
N_PAIR_CANVASES = 220
N_EVAL = 200


def sample_pair(model, gen, seed):
    """Two drafts of the SAME canvas; return (acc, rej) ids or None."""
    canvas, frozen, anchors = M.make_canvas(gen)
    c = M.calibrated_constraints(M.common_tokens())
    full_c = {**c, "anchors": anchors}
    outs = []
    for j in range(2):
        ids, _ = M.draft(model, canvas, frozen, seed=seed + 101 * j)
        ok, _ = M.intervention(ids, full_c)
        outs.append((ids, ok))
    acc = [i for i, ok in outs if ok == 0]
    rej = [i for i, ok in outs if ok == 1]
    if acc and rej:
        return acc[0], rej[0], frozen
    return None


def collect_pools(model):
    gen = torch.Generator().manual_seed(777001)
    accepted, rejected, pairs, frozens = [], [], [], []
    for i in range(N_PAIR_CANVASES):
        r = sample_pair(model, gen, seed=30000 + 7 * i)
        if r is not None:
            a, j, fz = r
            pairs.append((a, j))
            frozens.append(fz)
    # unpaired accepted pool for RSFT breadth: single samples
    gen2 = torch.Generator().manual_seed(555)
    common = M.common_tokens()
    for i in range(256):
        canvas, frozen, anchors = M.make_canvas(gen2)
        c = M.calibrated_constraints(common)
        ids, _ = M.draft(model, canvas, frozen, seed=60000 + i)
        ok, _ = M.intervention(ids, {**c, "anchors": anchors})
        (accepted if ok == 0 else rejected).append(ids)
    pair_acc = torch.stack([p[0] for p in pairs])
    pair_rej = torch.stack([p[1] for p in pairs])
    pair_fz = torch.stack(frozens)
    print(f"pools: pairs={len(pairs)} accepted_extra={len(accepted)} "
          f"rejected_extra={len(rejected)}", flush=True)
    return pair_acc, pair_rej, pair_fz, torch.stack(accepted)


def feats_std(model, ids_b, fz_b, mean, std):
    f = M.pooled_features(model, ids_b, fz_b)
    return (f - mean) / std


def post_train(base, arm, pair_acc, pair_rej, pair_fz, acc_pool,
               bank, bank_mean, bank_std):
    model = copy.deepcopy(base)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(2026)
    n_pairs = pair_acc.shape[0]
    for s in range(POST_STEPS):
        # RSFT CE: masked LM on accepted drafts (pair mates + pool)
        src = torch.cat([pair_acc, acc_pool], 0)
        idx = torch.randint(0, src.shape[0], (POST_BS,), generator=gen)
        x = src[idx]
        mask = torch.rand(POST_BS, LEN) < 0.5
        xm = torch.where(mask, torch.full_like(x, MASK_ID), x)
        ce = F.cross_entropy(model(xm).reshape(-1, VOCAB),
                             x.reshape(-1))
        loss = ce
        if arm != "rsft":
            pidx = torch.randint(0, n_pairs, (POST_BS,), generator=gen)
            rf = feats_std(model, pair_rej[pidx], pair_fz[pidx],
                           bank_mean, bank_std)
            if arm == "mmd_marginal":
                loss = ce + LAM * M.mmd_rbf(rf, bank)
            else:  # paired: targets are the pair-mates' encodings,
                # computed through the CURRENT model (grad flows)
                af = feats_std(model, pair_acc[pidx], pair_fz[pidx],
                               bank_mean, bank_std)
                loss = ce + LAM * M.mmd_rbf(rf, af)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % 50 == 0:
            print(f"  {arm}: step {s + 1}/{POST_STEPS} "
                  f"ce={ce.item():.3f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, common):
    gen = torch.Generator().manual_seed(777)
    fails, toks = 0, []
    for i in range(N_EVAL):
        canvas, frozen, anchors = M.make_canvas(gen)
        c = M.calibrated_constraints(common)
        ids, _ = M.draft(model, canvas, frozen, seed=90000 + i)
        f, _ = M.intervention(ids, {**c, "anchors": anchors})
        fails += f
        toks.append(ids)
    div = torch.cat(toks).unique().numel() / VOCAB
    return fails / N_EVAL, fails, div


def main():
    t0 = time.time()
    torch.manual_seed(0)
    cfg = M.tiny_cfg()
    base = M.XIIVModel(cfg)
    base.init_weights()
    print("brief training base...", flush=True)
    base = M.train_briefly(base)
    base.eval()
    common = M.common_tokens()

    base_rate, _, _ = evaluate(base, common)
    print(f"base intervention rate (eval seeds): {base_rate:.3f}",
          flush=True)

    pair_acc, pair_rej, pair_fz, acc_pool = collect_pools(base)
    # Bank = standardized features of the PAIRED accepted mates,
    # encoded by the frozen base (statistics applied to both sides of
    # every MMD in the arms — the pilot's one-sided standardization
    # fixed, as registered in the docstring).
    with torch.no_grad():
        pf = M.pooled_features(base, pair_acc, pair_fz)
    bank_mean = pf.mean(0)
    bank_std = pf.std(0).clamp_min(1e-6)
    bank = (pf - bank_mean) / bank_std

    arms = {}
    results = {}
    for arm in ("rsft", "mmd_marginal", "mmd_paired"):
        m = post_train(base, arm, pair_acc, pair_rej, pair_fz, acc_pool,
                       bank, bank_mean, bank_std)
        m.eval()
        rate, fails, div = evaluate(m, common)
        ce = M.heldout_ce(m)
        arms[arm] = (m, fails)
        results[arm] = {"rate": rate, "fails": fails, "diversity": div,
                        "heldout_ce": ce}
        print(f"{arm}: rate={rate:.3f} diversity={div:.3f} "
              f"heldout_ce={ce:.3f}", flush=True)

    ctrl_fails = arms["rsft"][1]
    # recompute per-trial paired outcomes for McNemar: evaluate stored
    # fail counts are aggregates; rerun paired check via seeds is
    # deterministic — instead track discordants during a paired pass:
    verdicts = {}
    for arm in ("mmd_marginal", "mmd_paired"):
        m = arms[arm][0]
        gen = torch.Generator().manual_seed(777)
        b01 = b10 = 0
        for i in range(N_EVAL):
            canvas, frozen, anchors = M.make_canvas(gen)
            c = M.calibrated_constraints(common)
            full_c = {**c, "anchors": anchors}
            ids_a, _ = M.draft(m, canvas, frozen, seed=90000 + i)
            ids_c, _ = M.draft(arms["rsft"][0], canvas, frozen,
                               seed=90000 + i)
            fa = M.intervention(ids_a, full_c)[0]
            fc = M.intervention(ids_c, full_c)[0]
            if fc and not fa:
                b01 += 1
            elif fa and not fc:
                b10 += 1
        p, _ = M.mcnemar(b01, b10)
        r = results[arm]
        guard = (abs(r["diversity"] - results["rsft"]["diversity"])
                 <= 0.2 * results["rsft"]["diversity"]
                 and r["heldout_ce"] <= 2 * results["rsft"]["heldout_ce"])
        passed = (results["rsft"]["rate"] < 0.90 and r["rate"]
                  < results["rsft"]["rate"] and p < 0.05 and guard)
        verdicts[arm] = {"b01": b01, "b10": b10, "p": p,
                         "guardrails_ok": bool(guard),
                         "verdict": "PASS" if passed else "FAIL"}
        print(f"{arm} vs rsft: b01={b01} b10={b10} p={p:.4f} "
              f"guard={guard} -> {verdicts[arm]['verdict']}", flush=True)

    stable = results["rsft"]["rate"] < 0.90
    out = {"base_rate": base_rate, "arms": results, "verdicts": verdicts,
           "testbed_stable": bool(stable), "lambda": LAM,
           "post_steps": POST_STEPS, "seconds": time.time() - t0}
    with open(os.path.join(REPO, "research/gmmd_testbed2_results.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE stable={stable} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
