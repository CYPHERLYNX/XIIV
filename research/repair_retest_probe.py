"""Fair combined retest: GLOD allocation + CSM routing, with TEETH.

Pre-registered 2026-10-22, before any run. Context: Invention #1 gate 2
(FAIL: load 0.720 vs conf 0.735, p=0.55) and CSM Phase C (FAIL: routed
0.690 vs control 0.695, p=1.0) shared one diagnosis — the refinement
operator was too weak to fix violations, so no targeting/routing signal
could pay. The documented fairer test: give every arm the SAME capable
repair operator and vary only the signal. The operator is
xiiv/repair.py's Verifier-Guided Repair (diagnosis-targeted by
default, enforced refill through the constraint-attached sampler).

PART 1 (GLOD question, family A, n=200 paired drafts, harness and
seeds identical to gate 2): three arms, identical budget (2 rounds,
k = 15% of open positions/round, 8 refill steps), identical enforced
refill; ONLY targeting differs:
  verifier   — diagnosed violating positions (VGR default)
  load       — highest guarantee-load positions
  confidence — lowest-confidence positions
Primary metric: intervention rate (any check_constraints failure).
Gate (unchanged from gate 2): PASS iff load < confidence with McNemar
p < 0.05. CEILING clause (registered now): if BOTH signal arms are
<= 0.02, the allocation comparison is undecidable at this operator
strength — verdict CEILING, not PASS/FAIL; the operator result is
then the finding. Historical free-refine rates (gate 2, same recipe:
load 0.720 / conf 0.735 / random 0.730) are a DESCRIPTIVE cross-probe
reference only, not a paired comparison.

PART 2 (CSM routing question): ledger rebuilt exactly as csm_probe
Phases A/B (n=200 + 200, same families/seeds). Phase C': control =
VGR (verifier targeting) with a uniform 2 rounds per trial; routed =
VGR with rounds allocated by the certified ledger rate, greedy-
corrected to the identical total. Gate (unchanged): PASS iff routed
< control with McNemar p < 0.05. CEILING clause: if control rate
< 0.05, routing is undecidable (nothing left to route against) —
verdict CEILING. Phase B AUROC gap is recomputed and reported; the
CSM self-knowledge claim already PASSED in 2026-10-21 and is not
re-gated here.
"""
import importlib.util
import json
import os
import sys
import time

REPO = os.path.expanduser("~/workspace/ai-tools/corsac")
sys.path.insert(0, REPO)

spec = importlib.util.spec_from_file_location(
    "gate2", os.path.join(REPO, "research/invention01_gate2_probe.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

spec2 = importlib.util.spec_from_file_location(
    "csm", os.path.join(REPO, "research/consciousness/csm_probe.py"))
C = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(C)

import torch  # noqa: E402
from xiiv.diffusion import guarantee_load_priority  # noqa: E402
from xiiv.repair import verifier_guided_repair  # noqa: E402

N_TRIALS = 200


def load_for(c, frozen):
    return guarantee_load_priority(
        G.LEN, frozen, {}, {r: 1 for r in c["required_ids"]},
        c["at_most"], [[f] for f in c["forbidden_ids"]], device="cpu")


def part1(model, common):
    c_base = G.calibrated_constraints(common)
    gen = torch.Generator().manual_seed(424242)
    arms = ("verifier", "load", "confidence")
    fails = {a: 0 for a in arms}
    touched = {a: 0 for a in arms}
    b01 = b10 = 0
    for t in range(N_TRIALS):
        canvas, frozen, anchors = G.make_canvas(gen)
        full_c = {**c_base, "anchors": anchors}
        ids0, conf0 = G.draft(model, canvas, frozen, seed=5000 + t)
        load = load_for(c_base, frozen)
        outs = {}
        for a in arms:
            out, rep = verifier_guided_repair(
                model, ids0, full_c, G.MASK_ID, G.EOS_ID, frozen=frozen,
                rounds=2, budget_frac=G.REMASK_FRAC, steps=G.REFINE_STEPS,
                seed=9000 + t, targeting=a, conf=conf0, load=load)
            outs[a] = out
            touched[a] += len(rep.positions_touched)
            f, _ = G.intervention(out, full_c)
            fails[a] += f
        fc = G.intervention(outs["confidence"], full_c)[0]
        fl = G.intervention(outs["load"], full_c)[0]
        if fc and not fl:
            b01 += 1
        elif fl and not fc:
            b10 += 1
        if (t + 1) % 50 == 0:
            print(f"part1 trial {t + 1}/{N_TRIALS} " +
                  " ".join(f"{a}={fails[a] / (t + 1):.3f}" for a in arms),
                  flush=True)
    rates = {a: fails[a] / N_TRIALS for a in arms}
    p, chi2 = G.mcnemar(b01, b10)
    if rates["load"] <= 0.02 and rates["confidence"] <= 0.02:
        verdict = "CEILING"
    elif p < 0.05 and rates["load"] < rates["confidence"]:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print(f"PART1 rates: {rates} McNemar(b01={b01}, b10={b10}) "
          f"p={p:.4f} -> {verdict}")
    print(f"PART1 mean positions touched: " +
          ", ".join(f"{a}={touched[a] / N_TRIALS:.1f}" for a in arms))
    return {"rates": rates, "mcnemar": {"b01": b01, "b10": b10, "p": p},
            "mean_touched": {a: touched[a] / N_TRIALS for a in arms},
            "verdict": verdict}


def part2(model, common):
    fams = C.families(common)
    names = list(fams.keys())
    recsA = C.run_trials(model, fams, seed0=1000)
    ledger = {}
    for fam in names:
        sub = [r for r in recsA if r["fam"] == fam]
        fails = sum(r["fail"] for r in sub)
        ledger[fam] = {"n": len(sub), "fails": fails,
                       "rate": (fails + 1) / (len(sub) + 2)}
    recsB = C.run_trials(model, fams, seed0=5000)
    labels = [r["fail"] for r in recsB]
    auc_conf = C.auroc([1.0 - r["conf"] for r in recsB], labels)
    auc_led = C.auroc([ledger[r["fam"]]["rate"] for r in recsB], labels)
    base_draft = sum(labels) / len(labels)
    print(f"PART2 ledger: " +
          ", ".join(f"{f}={ledger[f]['rate']:.3f}" for f in names))
    print(f"PART2 draft failure rate={base_draft:.3f} AUROC conf="
          f"{auc_conf:.3f} ledger={auc_led:.3f}")

    total_rounds = 2 * len(recsB)
    raw = [ledger[r["fam"]]["rate"] for r in recsB]
    alloc = [max(0, min(5, round(4 * x))) for x in raw]
    while sum(alloc) > total_rounds:
        i = max(range(len(alloc)), key=lambda j: alloc[j])
        alloc[i] -= 1
    while sum(alloc) < total_rounds:
        i = max(range(len(alloc)), key=lambda j: (raw[j], -alloc[j]))
        alloc[i] += 1
    assert sum(alloc) == total_rounds

    ctrl_fail, rout_fail = [], []
    for idx, r in enumerate(recsB):
        fam_c = fams[r["fam"]]
        out_c, _ = verifier_guided_repair(
            model, r["ids"], fam_c, G.MASK_ID, G.EOS_ID,
            frozen=r["frozen"], rounds=2, budget_frac=G.REMASK_FRAC,
            steps=G.REFINE_STEPS, seed=r["seed"] + 31337,
            targeting="verifier", conf=r["conf_vec"])
        ctrl_fail.append(G.intervention(out_c, fam_c)[0])
        out_r, _ = verifier_guided_repair(
            model, r["ids"], fam_c, G.MASK_ID, G.EOS_ID,
            frozen=r["frozen"], rounds=alloc[idx],
            budget_frac=G.REMASK_FRAC, steps=G.REFINE_STEPS,
            seed=r["seed"] + 31337, targeting="verifier",
            conf=r["conf_vec"])
        rout_fail.append(G.intervention(out_r, fam_c)[0])
    b01 = sum(1 for c, q in zip(ctrl_fail, rout_fail) if c == 1 and q == 0)
    b10 = sum(1 for c, q in zip(ctrl_fail, rout_fail) if c == 0 and q == 1)
    p, chi2 = G.mcnemar(b01, b10)
    rate_c = sum(ctrl_fail) / len(ctrl_fail)
    rate_r = sum(rout_fail) / len(rout_fail)
    if rate_c < 0.05:
        verdict = "CEILING"
    elif rate_r < rate_c and p < 0.05:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print(f"PART2 control={rate_c:.3f} routed={rate_r:.3f} "
          f"McNemar(b01={b01}, b10={b10}) p={p:.4f} -> {verdict}")
    return {"ledger": ledger, "draft_failure_rate": base_draft,
            "auroc_confidence": auc_conf, "auroc_ledger": auc_led,
            "control_rate": rate_c, "routed_rate": rate_r,
            "mcnemar": {"b01": b01, "b10": b10, "p": p},
            "verdict": verdict}


def main():
    t0 = time.time()
    torch.manual_seed(0)
    model = G.XIIVModel(G.tiny_cfg())
    model.init_weights()
    G.train_briefly(model)
    model.eval()
    common = G.common_tokens()

    # pilot band check on fresh drafts (as gate 2 registered)
    pilot_gen = torch.Generator().manual_seed(20261021)
    fails = 0
    for i in range(G.N_PILOT):
        canvas, frozen, anchors = G.make_canvas(pilot_gen)
        full_c = {**G.calibrated_constraints(common), "anchors": anchors}
        ids, _ = G.draft(model, canvas, frozen, seed=1000 + i)
        fails += G.intervention(ids, full_c)[0]
    base = fails / G.N_PILOT
    print(f"pilot draft intervention rate={base:.2f} (n={G.N_PILOT})",
          flush=True)
    if not (0.15 <= base <= 0.85):
        print("INCONCLUSIVE: draft band outside [0.15, 0.85].")
        return

    r1 = part1(model, common)
    r2 = part2(model, common)
    out = {"part1_glod": r1, "part2_csm_routing": r2,
           "pilot_draft_rate": base, "seconds": time.time() - t0,
           "historical_reference_descriptive": {
               "gate2_free_refine": {"load": 0.720, "confidence": 0.735,
                                     "random": 0.730},
               "csm_phaseC_free_refine": {"control": 0.695,
                                          "routed": 0.690}}}
    path = os.path.join(REPO, "research/repair_retest_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
