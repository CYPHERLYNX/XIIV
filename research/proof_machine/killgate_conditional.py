"""
Conditional kill-gate evaluation (spec section 5.6, task 2). Pre-registered,
no peeking.

Depths:        L in {1, 2, 4, 8, 16, 32, 64, 100} (taken-path primitive count),
               1000 fresh conditional instances each.
Primary:       A_CIF(L) = fraction with output == expected (exact int match).
Secondary:     V_CIF(L) = fraction whose certificate mechanically verifies
               (branch nodes undergo B1-B4; expect 100% at all depths).

PASS (all must hold):
  1. A_CIF(100) >= 90%
  2. Flat curve: OLS fit A_CIF(L) = a - b*L; b not significantly != 0
     (two-sided t-test on slope, alpha = 0.05, df = n_depths - 2)
  3. V_CIF(L) = 100% at every depth (certificate construction sound for
     branch nodes)

KILL (any triggers):
  1. A_CIF(100) < 70%    -> absolute failure
  2. b significantly negative (p < 0.05) -> branching degrades like depth
  3. V_CIF(L) < 100% at any depth -> branch certificate construction broken

SEPARATION vs transformer baseline: NOT part of this gate. The Phase-1
baseline was trained on flat op sequences; a fair conditional comparison needs
the baseline retrained on the conditional program encoding (new token types
for IFZ structure) -- queued as follow-up work, documented in the results
JSON. The PM-only criteria above are the pre-registered gate for today.

A loud halt inside run_program (neural-divergence at a guard, interface
violation) is caught PER INSTANCE, printed with its diagnosis, and counted
as both an accuracy and a certificate failure -- it must fire the V<100%
kill trigger, not crash the sweep.

Usage:
  python killgate_conditional.py        # proof-machine conditional sweep
Results: results/killgate_cond_<stamp>.json
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskgen_conditional import gen_cbatch, branch_stats
from compose import ProofMachine
from certificates import verify_certificate
from conditionals import max_path_cost, prog_to_str

DEPTHS = [1, 2, 4, 8, 16, 32, 64, 100]
N_PER_DEPTH = 1000
T_CRIT_005_DF6 = 2.447  # two-sided t critical, alpha=0.05, df=6


def ols_slope_t(xs, ys):
    """OLS slope of y on x + t-statistic for H0: slope == 0."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, 0.0
    b = sxy / sxx
    a = my - b * mx
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    if ssr == 0:
        return b, 0.0  # zero residual variance: slope not significant
    se = math.sqrt(ssr / ((n - 2) * sxx))
    t = b / se if se > 0 else 0.0
    return b, t


def eval_pm_conditional(ckpt_dir):
    pm = ProofMachine(ckpt_dir)
    per_depth = {}
    for L in DEPTHS:
        batch = gen_cbatch(L, N_PER_DEPTH, seed=5000 + L)
        # sanity: every program really has max-path cost L
        assert all(max_path_cost(prog) == L for (_, prog, _, _) in batch)
        correct = 0
        cert_ok = 0
        loud_halts = 0
        t0 = time.time()
        for (x0, prog, expected, _) in batch:
            try:
                pred, cert = pm.run_program(x0, prog)
            except RuntimeError as ex:
                loud_halts += 1
                print(f"  LOUD HALT L={L} x0={x0}: {ex}", flush=True)
                continue
            if pred == expected:
                correct += 1
            ok, info = verify_certificate(cert)
            if ok:
                cert_ok += 1
            else:
                print(f"  CERT FAIL L={L}: {info} prog={prog_to_str(prog)}",
                      flush=True)
        dt = time.time() - t0
        per_depth[L] = {
            "accuracy": correct / N_PER_DEPTH,
            "cert_validity": cert_ok / N_PER_DEPTH,
            "loud_halts": loud_halts,
            "branch_stats": branch_stats(batch),
            "eval_seconds": dt,
        }
        bs = per_depth[L]["branch_stats"]
        print(f"L={L:3d}  A={per_depth[L]['accuracy']*100:6.2f}%  "
              f"V={per_depth[L]['cert_validity']*100:6.2f}%  "
              f"halts={loud_halts}  branches={bs['n_branches']}  "
              f"frac_then={bs['frac_then']:.3f}  ({dt:.1f}s)", flush=True)
    return per_depth


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(res_dir, exist_ok=True)

    print("== proof-machine CONDITIONAL depth sweep ==", flush=True)
    per_depth = eval_pm_conditional(ckpt)

    acc = [per_depth[L]["accuracy"] for L in DEPTHS]
    b, t = ols_slope_t(DEPTHS, acc)
    result = {
        "gate": "conditional-5.6-task2-pm-only",
        "depths": DEPTHS,
        "n_per_depth": N_PER_DEPTH,
        "depth_semantics": "taken-path primitive count (max-path; branches equal length)",
        "per_depth": per_depth,
        "slope_b": b,
        "slope_t": t,
        "t_crit_005": T_CRIT_005_DF6,
        "A_CIF_100": per_depth[100]["accuracy"],
        "V_CIF_100": per_depth[100]["cert_validity"],
        "baseline_separation": "QUEUED: transformer baseline needs retraining "
                               "on the conditional program encoding; not part "
                               "of this PM-only gate",
    }
    kills = []
    if per_depth[100]["accuracy"] < 0.70:
        kills.append("A_CIF(100) < 70%")
    if t < -T_CRIT_005_DF6:
        kills.append(f"slope significantly negative (t={t:.2f})")
    if any(per_depth[L]["cert_validity"] < 1.0 for L in DEPTHS):
        kills.append("V_CIF(L) < 100% at some depth")
    result["kill_triggers_pm_only"] = kills
    result["verdict_pm_only"] = (
        "KILL" if kills else
        ("PASS" if per_depth[100]["accuracy"] >= 0.90 and abs(t) < T_CRIT_005_DF6
         else "INCONCLUSIVE"))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(res_dir, f"killgate_cond_{stamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nslope b={b:.6f}, t={t:.2f} (crit {T_CRIT_005_DF6})", flush=True)
    print(f"A_CIF(100) = {per_depth[100]['accuracy']*100:.2f}%", flush=True)
    print(f"V_CIF all depths = "
          f"{min(per_depth[L]['cert_validity'] for L in DEPTHS)*100:.2f}%", flush=True)
    print(f"PM-only kill triggers: {kills if kills else 'none'}", flush=True)
    print(f"verdict (PM-only): {result['verdict_pm_only']}", flush=True)
    print(f"results -> {path}", flush=True)


if __name__ == "__main__":
    main()
