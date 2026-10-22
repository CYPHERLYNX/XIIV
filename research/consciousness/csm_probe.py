"""CSM (Certified Self-Model) — toy-scale prototype probe.

Design: research/consciousness/DESIGN.md (invention, 2026-10-21).
Question: does CERTIFIED self-knowledge — a per-domain failure ledger
written only by the external verifier — (a) predict the model's failures
better than the model's own readable confidence signal, and (b) pay when
used to route refinement compute at EQUAL total budget?

Reuses the Invention-#1 gate-2 harness verbatim (tiny XIIV d=64, template
data, brief training, free drafts, external check_constraints verifier).

Phases:
  A (ledger building, n=200): drafts across 4 constraint families;
     record verifier outcome + mean draft confidence per trial.
     Ledger = per-family failure rate (Laplace-smoothed), verifier-only.
  B (held-out, n=200 fresh): AUROC of failure prediction —
     (a) raw confidence (1 - mean_conf), (b) ledger family rate.
  C (paired, on Phase B drafts, EQUAL total refine rounds):
     control = uniform 2 rounds/trial, confidence-targeted;
     routed  = rounds allocated per-trial proportional to the ledger's
               certified family failure rate (greedy-corrected to the
               exact same total), confidence-targeted within a trial.
     Metric: intervention rate; McNemar paired test.

Pre-registered kill gate (DESIGN.md §5):
  PASS iff AUROC(ledger) - AUROC(confidence) >= 0.05 on Phase B
       AND routed interventions < control with McNemar p < 0.05.
  FAIL -> payoff claim dies at toy scale; reported as-is.

Declared confound: the ledger predictor is family-level statistics — an
external statistician by design. The claim tested is system-level:
certified, in-loop self-knowledge beats the model's own readable signal
and pays in routing. Not a claim of phenomenal consciousness.
"""
import importlib.util
import json
import math
import os
import sys
import time

REPO = os.path.expanduser("~/workspace/ai-tools/corsac")
sys.path.insert(0, REPO)

spec = importlib.util.spec_from_file_location(
    "gate2", os.path.join(REPO, "research/invention01_gate2_probe.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

import torch  # noqa: E402

N_PER_PHASE = 200


def families(common):
    t = common
    return {
        "A_mixed": {"required_ids": t[0:1], "forbidden_ids": t[1:2],
                    "at_most": {t[2]: 3}},
        "B_forbid_fav": {"required_ids": [], "forbidden_ids": t[0:1],
                         "at_most": {}},
        "C_req_forbid": {"required_ids": t[1:2], "forbidden_ids": t[3:4],
                         "at_most": {}},
        "D_req_only": {"required_ids": t[5:6], "forbidden_ids": [],
                       "at_most": {}},
    }


def auroc(scores, labels):
    """Rank-based AUROC with average ranks for ties. labels: 1 = failure."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [], 0
    srt = [p[0] for p in pairs]
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and srt[j + 1] == srt[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks.extend([avg] * (j - i + 1))
        i = j + 1
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (rank_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def run_trials(model, fams, seed0, n=N_PER_PHASE):
    """Draft n trials (families round-robin); return records."""
    names = list(fams.keys())
    recs = []
    for i in range(n):
        fam = names[i % len(names)]
        gen = torch.Generator().manual_seed(seed0 + i)
        canvas, frozen, _ = G.make_canvas(gen)
        ids, conf = G.draft(model, canvas, frozen, seed0 + i)
        fail, _ = G.intervention(ids, fams[fam])
        open_mask = ~frozen
        mean_conf = float(conf[open_mask].mean())
        recs.append({"fam": fam, "fail": fail, "conf": mean_conf,
                     "conf_vec": conf, "ids": ids, "frozen": frozen,
                     "seed": seed0 + i})
    return recs


def refine_with_rounds(model, ids, conf_vec, frozen, rounds, seed):
    cur, cur_conf = ids.clone(), conf_vec.clone()
    gen = torch.Generator().manual_seed(seed)
    k = max(1, int(G.REMASK_FRAC * int((~frozen).sum())))
    for r in range(rounds):
        remask = G.select_positions("confidence", cur, cur_conf, None,
                                    frozen, k, gen)
        cur, cur_conf = G.refine_round(model, cur, frozen, remask,
                                       seed + 7919 * (r + 1))
    return cur


def main():
    t0 = time.time()
    torch.manual_seed(0)
    model = G.XIIVModel(G.tiny_cfg())
    G.train_briefly(model)
    model.eval()
    common = G.common_tokens()
    fams = families(common)
    names = list(fams.keys())

    # ---- Phase A: build the ledger (verifier-written) ----
    recsA = run_trials(model, fams, seed0=1000)
    ledger = {}
    for fam in names:
        sub = [r for r in recsA if r["fam"] == fam]
        fails = sum(r["fail"] for r in sub)
        ledger[fam] = {"n": len(sub), "fails": fails,
                       "rate": (fails + 1) / (len(sub) + 2)}
    print("Phase A ledger (certified failure rates):")
    for fam in names:
        L = ledger[fam]
        print(f"  {fam}: {L['fails']}/{L['n']} rate={L['rate']:.3f}")

    # ---- Phase B: held-out self-knowledge ----
    recsB = run_trials(model, fams, seed0=5000)
    labels = [r["fail"] for r in recsB]
    auc_conf = auroc([1.0 - r["conf"] for r in recsB], labels)
    auc_led = auroc([ledger[r["fam"]]["rate"] for r in recsB], labels)
    base_B = sum(labels) / len(labels)
    print(f"Phase B: base failure rate={base_B:.3f}  "
          f"AUROC conf={auc_conf:.3f}  ledger={auc_led:.3f}  "
          f"gap={auc_led - auc_conf:+.3f}")

    # ---- Phase C: routing at equal total compute (paired on B drafts) --
    total_rounds = 2 * len(recsB)
    raw = [ledger[r["fam"]]["rate"] for r in recsB]
    alloc = [max(0, min(5, round(4 * x))) for x in raw]
    # greedy correction to the exact budget
    while sum(alloc) > total_rounds:
        i = max(range(len(alloc)), key=lambda j: alloc[j])
        alloc[i] -= 1
    while sum(alloc) < total_rounds:
        i = max(range(len(alloc)),
                key=lambda j: (raw[j], -alloc[j]))
        alloc[i] += 1
    assert sum(alloc) == total_rounds

    ctrl_fail = []
    rout_fail = []
    for idx, r in enumerate(recsB):
        out_c = refine_with_rounds(model, r["ids"], r["conf_vec"],
                                   r["frozen"], 2, r["seed"] + 31337)
        f_c, _ = G.intervention(out_c, fams[r["fam"]])
        out_r = refine_with_rounds(model, r["ids"], r["conf_vec"],
                                   r["frozen"], alloc[idx],
                                   r["seed"] + 31337)
        f_r, _ = G.intervention(out_r, fams[r["fam"]])
        ctrl_fail.append(f_c)
        rout_fail.append(f_r)
    b01 = sum(1 for c, q in zip(ctrl_fail, rout_fail) if c == 1 and q == 0)
    b10 = sum(1 for c, q in zip(ctrl_fail, rout_fail) if c == 0 and q == 1)
    p, chi2 = G.mcnemar(b01, b10)
    rate_c = sum(ctrl_fail) / len(ctrl_fail)
    rate_r = sum(rout_fail) / len(rout_fail)
    print(f"Phase C: control={rate_c:.3f} routed={rate_r:.3f} "
          f"(McNemar b01={b01} b10={b10} p={p:.4f})")

    gate1 = (auc_led - auc_conf) >= 0.05
    gate2 = (rate_r < rate_c) and (p < 0.05)
    verdict = "PASS" if (gate1 and gate2) else "FAIL"
    print(f"KILL GATE: phaseB_gap_ok={gate1} routing_ok={gate2} "
          f"-> {verdict}  ({time.time() - t0:.0f}s)")

    results = {
        "ledger": ledger,
        "phaseB": {"base_failure_rate": base_B, "auroc_confidence": auc_conf,
                   "auroc_ledger": auc_led, "gap": auc_led - auc_conf},
        "phaseC": {"control_rate": rate_c, "routed_rate": rate_r,
                   "b01": b01, "b10": b10, "mcnemar_p": p,
                   "total_rounds_each_arm": total_rounds},
        "gate": {"phaseB_ok": gate1, "routing_ok": gate2,
                 "verdict": verdict},
        "seconds": time.time() - t0,
    }
    out = os.path.join(REPO, "research/consciousness")
    with open(os.path.join(out, "csm_probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out, "csm_probe_results.md"), "w") as f:
        f.write("# CSM prototype probe — results (2026-10-21)\n\n")
        f.write(f"Phase B: AUROC confidence={auc_conf:.3f}, "
                f"ledger={auc_led:.3f}, gap={auc_led - auc_conf:+.3f}, "
                f"base failure rate={base_B:.3f}\n\n")
        f.write(f"Phase C (equal compute, {total_rounds} rounds/arm): "
                f"control={rate_c:.3f}, routed={rate_r:.3f}, "
                f"McNemar p={p:.4f} (b01={b01}, b10={b10})\n\n")
        f.write(f"**Kill gate: {verdict}** "
                f"(PhaseB gap ok={gate1}, routing ok={gate2})\n")
    print("results written.")


if __name__ == "__main__":
    main()
