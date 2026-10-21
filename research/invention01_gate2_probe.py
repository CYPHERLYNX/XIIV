"""Invention #1 — GATE 2: does the allocation pay?

Invention #1: proof-difficulty-adaptive diffusion sampling — allocate
denoising compute per position by PREDICTED CERTIFICATE-CONSTRUCTION
DIFFICULTY (not token confidence). Prior art: research/invention01_prior_art.md
(verdict: NOVEL conditionally). Gate 1 (PASS, 2026-10-20): the signal is
genuinely distinct from confidence (Spearman 0.123; confident-but-hard
quadrant 10.0% vs 6.25% independence).

Gate 2 (this experiment, pre-registered): does allocating refinement steps
by certificate difficulty actually REDUCE THE INTERVENTION RATE vs
confidence-based allocation, at EQUAL compute budget?

Design (paired, CPU-only, toy scale):
  - Same tiny XIIV model + template data + brief training recipe as gate 1,
    so model confidence carries real signal.
  - Per trial: fixed distribution-aware constraint family (family A:
    required=[most-frequent token], forbidden=[2nd most-frequent],
    at_most={3rd: 3}) with randomized frozen anchors; a FREE draft (no
    structural enforcement in the sampler — the model generates, an
    EXTERNAL certificate constructor `check_constraints` verifies).
    Violations = interventions, XIIV's native metric.
  - Three refinement arms, identical budget (2 rounds x k=15% remask),
    identical mechanics (re-denoise remasked positions, no structural
    enforcement — the verifier stays external). ONLY the targeting signal
    differs:
      random      — control: remask k random non-anchor positions
      confidence  — the CadLLM/SAID/CHLG-like baseline: remask k
                    lowest-confidence non-anchor positions
                    (mirrors xiiv.diffusion.self_refine's policy)
      load        — INVENTION #1: remask k highest-guarantee-load
                    non-anchor positions (static GLOD predictor, the same
                    signal gate 1 measured; predictor cost timed separately)
  - Paired design: all three arms refine the SAME draft per trial.
  - Primary metric: intervention rate = P(any check_constraints check fails).

Pre-registered verdict:
  PASS: load arm intervention rate < confidence arm, McNemar paired test
        p < 0.05, AND the win concentrates in trials whose drafts contain
        confident-but-hard positions (gate-1 quadrant stat, descriptive).
  FAIL: no significant difference, or confidence wins  -> "allocation by
        certificate difficulty does not pay at toy scale; do NOT advance
        the novelty claim on payoff grounds."

Honesty notes:
  - PROTOCOL AMENDMENT (documented, calibration still draft-only): the
    pre-registered knob (n_required in {3,4,5} over uniform-random tokens)
    produced base rate 1.00 at every level — random tokens are almost never
    generated AND almost never avoided, so the task was unmeasurable. The
    knob was changed to distribution-aware constraint families (tokens
    drawn from the training distribution's most-frequent list), calibrated
    on drafts only: family A {required=[t0], forbidden=[t1],
    at_most={t2:3}} measured base rate 0.70 over 40 drafts (failures:
    forbidden_absent 26, required_present 3, caps_met 1). The pilot below
    re-confirms the band on 30 FRESH drafts; arms still never run before
    calibration. Amendment preserves the anti-p-hacking property.
  - Static predictor: gate 2 tests the SIGNAL's payoff, not a learned
    difficulty head (Patch-Forcing-style head is future work; prior-art
    caveat 3's predictor-cost question is answered by timing the static
    predictor vs the refinement it directs).
  - Toy scale: one constraint family, brief training, 48-token canvas.
    A PASS here licenses a larger-scale follow-up, not a novelty claim.
"""
import time
import torch
import torch.nn.functional as F

from xiiv.config import ModelConfig
from xiiv.diffusion import (sample_trajectory, check_constraints,
                            guarantee_load_priority)
from xiiv.model import XIIVModel

VOCAB = 256
LEN = 48
MASK_ID = 254
EOS_ID = 255
DRAFT_STEPS = 16
REFINE_STEPS = 8
REFINE_ROUNDS = 2
REMASK_FRAC = 0.15
N_TRIALS = 200
N_PILOT = 30


def common_tokens(k=12):
    """Most-frequent tokens of the training distribution (deterministic:
    same seeded generator as train_briefly's data)."""
    gen = torch.Generator().manual_seed(7)
    data = make_data(4096, gen)
    counts = torch.bincount(data.reshape(-1), minlength=VOCAB)
    return torch.argsort(counts, descending=True)[:k].tolist()


def calibrated_constraints(common):
    """Family A: base intervention rate 0.70 measured over 40 drafts
    (diagnostic 2026-10-21)."""
    return {"required_ids": common[:1], "forbidden_ids": common[1:2],
            "at_most": {common[2]: 3}}


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


def train_briefly(m, steps=250, bs=64):
    gen = torch.Generator().manual_seed(7)
    data = make_data(4096, gen)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for s in range(steps):
        idx = torch.randint(0, 4096, (bs,))
        x = data[idx]
        mask = torch.rand(bs, LEN) < 0.5
        xm = torch.where(mask, torch.full_like(x, MASK_ID), x)
        logits = m(xm)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


def make_canvas(gen):
    """Canvas with pre-filled frozen anchors every 12 positions."""
    frozen = torch.zeros(LEN, dtype=torch.bool)
    frozen[::12] = True
    canvas = torch.full((LEN,), MASK_ID, dtype=torch.long)
    canvas[frozen] = torch.randint(0, 200, (int(frozen.sum()),),
                                   generator=gen)
    anchors = {int(p): int(canvas[p]) for p in torch.where(frozen)[0]}
    return canvas, frozen, anchors


@torch.no_grad()
def draft(model, canvas, frozen, seed):
    fz = frozen.clone()
    ids, conf = sample_trajectory(
        model, LEN, MASK_ID, EOS_ID, steps=DRAFT_STEPS,
        temperature=1.0, top_p=0.95, seed=seed,
        prompt_ids=canvas.clone(), frozen=fz,
        constraints=None, device="cpu")
    return ids, conf


@torch.no_grad()
def refine_round(model, ids, frozen, remask_pos, seed):
    noisy = ids.clone()
    noisy[remask_pos] = MASK_ID
    fz = torch.ones(LEN, dtype=torch.bool)
    fz[remask_pos] = False
    out, conf = sample_trajectory(
        model, LEN, MASK_ID, EOS_ID, steps=REFINE_STEPS,
        temperature=1.0, top_p=0.95, seed=seed,
        prompt_ids=noisy, frozen=fz, constraints=None, device="cpu")
    return out, conf


def select_positions(selector, ids, conf, load, frozen, k, gen):
    open_pos = torch.where(~frozen)[0]
    if selector == "random":
        perm = torch.randperm(len(open_pos), generator=gen)
        return open_pos[perm[:k]]
    if selector == "confidence":
        c = conf.clone()
        order = torch.argsort(c[open_pos])          # lowest confidence first
        return open_pos[order[:k]]
    if selector == "load":
        order = torch.argsort(load[open_pos], descending=True)
        return open_pos[order[:k]]
    raise ValueError(selector)


@torch.no_grad()
def run_arm(model, ids, conf, load, frozen, constraints, selector, seed):
    cur, cur_conf = ids.clone(), conf.clone()
    gen = torch.Generator().manual_seed(seed)
    k = max(1, int(REMASK_FRAC * int((~frozen).sum())))
    for r in range(REFINE_ROUNDS):
        remask = select_positions(selector, cur, cur_conf, load, frozen,
                                  k, gen)
        cur, cur_conf = refine_round(model, cur, frozen, remask,
                                     seed + 7919 * (r + 1))
    return cur


def intervention(ids, constraints):
    res = check_constraints(ids, constraints, MASK_ID)
    return 0 if all(res.values()) else 1, res


def mcnemar(b01, b10):
    """McNemar chi-square (with continuity correction) for paired binary
    outcomes. b01 = load-only failures... here: b01 = conf fails & load
    passes, b10 = load fails & conf passes."""
    n = b01 + b10
    if n == 0:
        return 1.0, 0.0
    chi2 = (abs(b01 - b10) - 1.0) ** 2 / n
    # chi-square(1) survival via the normal CDF relation
    import math
    z = math.sqrt(chi2)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return p, chi2


def main():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = XIIVModel(cfg)
    model.init_weights()
    model = train_briefly(model)
    model.eval()

    # ---- PILOT: confirm the calibrated family on fresh drafts only ----
    common = common_tokens()
    pilot_gen = torch.Generator().manual_seed(20261021)
    fails = 0
    for i in range(N_PILOT):
        c = calibrated_constraints(common)
        canvas, frozen, anchors = make_canvas(pilot_gen)
        full_c = {**c, "anchors": anchors}
        ids, _ = draft(model, canvas, frozen, seed=1000 + i)
        f, _ = intervention(ids, full_c)
        fails += f
    base = fails / N_PILOT
    print(f"pilot: base intervention rate = {base:.2f} (n={N_PILOT})",
          flush=True)
    if not (0.15 <= base <= 0.85):
        print("INCONCLUSIVE: base rate outside the measurable band "
              "[0.15, 0.85]; gate 2 cannot be decided at toy scale.")
        return

    # ---- MAIN: paired comparison ----
    gen = torch.Generator().manual_seed(424242)
    arms = ("random", "confidence", "load")
    fails = {a: 0 for a in arms}
    # paired discordant counts for load vs confidence
    b01 = b10 = 0   # b01: conf fails & load passes; b10: load fails & conf passes
    quad_high_wins, quad_low_wins = [], []
    t_predict, t_refine = 0.0, {a: 0.0 for a in arms}
    viol_breakdown = {a: {} for a in arms}

    for t in range(N_TRIALS):
        c = calibrated_constraints(common)
        canvas, frozen, anchors = make_canvas(gen)
        full_c = {**c, "anchors": anchors}
        ids0, conf0 = draft(model, canvas, frozen, seed=5000 + t)

        t0 = time.perf_counter()
        load = guarantee_load_priority(
            LEN, frozen, {}, {r: 1 for r in c["required_ids"]},
            c["at_most"], [[f] for f in c["forbidden_ids"]], device="cpu")
        t_predict += time.perf_counter() - t0

        # gate-1 quadrant stat on the draft (descriptive analysis only)
        masked = ~frozen
        cc, ll = conf0[masked].float(), load[masked]
        quad = ((cc >= cc.quantile(0.75)) &
                (ll >= ll.quantile(0.75))).float().mean().item()

        outs = {}
        for a in arms:
            t1 = time.perf_counter()
            outs[a] = run_arm(model, ids0, conf0, load, frozen, full_c, a,
                              seed=9000 + t)
            t_refine[a] += time.perf_counter() - t1
            f, res = intervention(outs[a], full_c)
            fails[a] += f
            for k, v in res.items():
                if not v:
                    viol_breakdown[a][k] = viol_breakdown[a].get(k, 0) + 1

        fc, fl = intervention(outs["confidence"], full_c)[0], \
            intervention(outs["load"], full_c)[0]
        if fc and not fl:
            b01 += 1
        elif fl and not fc:
            b10 += 1
        (quad_high_wins if quad >= 0.10 else quad_low_wins).append(fc - fl)

        if (t + 1) % 50 == 0:
            print(f"trial {t + 1}/{N_TRIALS} " +
                  " ".join(f"{a}={fails[a] / (t + 1):.3f}" for a in arms),
                  flush=True)

    n = N_TRIALS
    rates = {a: fails[a] / n for a in arms}
    p, chi2 = mcnemar(b01, b10)
    print(f"\nintervention rates (n={n}): " +
          " ".join(f"{a}={rates[a]:.3f}" for a in arms))
    print(f"McNemar load vs confidence: b01(conf-fail/load-pass)={b01}, "
          f"b10={b10}, chi2={chi2:.2f}, p={p:.4f}")
    hw = sum(quad_high_wins) / max(1, len(quad_high_wins))
    lw = sum(quad_low_wins) / max(1, len(quad_low_wins))
    print(f"mean (conf_fail - load_fail): quad-high trials={hw:.3f} "
          f"(n={len(quad_high_wins)}), quad-low={lw:.3f} "
          f"(n={len(quad_low_wins)})")
    print(f"predictor wall-clock: {t_predict * 1000 / n:.2f} ms/trial; " +
          "refine wall-clock per arm: " +
          ", ".join(f"{a}={t_refine[a] / n:.2f}s" for a in arms))
    print("violation breakdown:", {a: dict(v) for a, v in
                                   viol_breakdown.items()})

    if p < 0.05 and rates["load"] < rates["confidence"]:
        print("VERDICT: GATE 2 PASS — load-adaptive allocation reduces "
              "interventions vs confidence-adaptive at equal budget.")
    else:
        print("VERDICT: GATE 2 FAIL — allocation by certificate difficulty "
              "does not pay at toy scale; do NOT advance the novelty claim "
              "on payoff grounds.")


if __name__ == "__main__":
    main()
