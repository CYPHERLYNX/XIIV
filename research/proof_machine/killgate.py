"""
Kill-gate evaluation (spec section 5.4). Pre-registered, no peeking.

Depths:        L in {1, 2, 4, 8, 16, 32, 64, 100}, 1000 fresh instances each.
Primary:       A(L) = fraction with output == expected (exact integer match).
Secondary:     V(L) = fraction of proof-machine outputs whose certificate
                     mechanically verifies (expect 100% at all depths).

PASS (all must hold):
  1. A_PM(100) >= 90%
  2. Flat curve: OLS fit A_PM(L) = a - b*L; b not significantly != 0
     (two-sided t-test on slope, alpha = 0.05, df = n_depths - 2)
  3. Separation: A_PM(100) - A_TF(100) >= 40 percentage points

KILL (any triggers):
  1. b significantly negative (p < 0.05) with |b| >= 50% of the transformer's
     degradation slope  -> proof machine degrades like a transformer
  2. A_PM(100) < 70%    -> absolute failure
  3. V(L) < 100% at any depth -> certificate construction broken

Usage:
  python killgate.py --pm-only        # proof machine sweep (needs checkpoints)
  python killgate.py --tf EVAL_JSON   # (not used directly; baseline eval is in
                                     #  baseline_transformer.py and merged here)
Results: results/killgate_<stamp>.json
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskgen import gen_batch
from compose import ProofMachine
from certificates import verify_certificate

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


def eval_proof_machine(ckpt_dir):
    pm = ProofMachine(ckpt_dir)
    per_depth = {}
    for L in DEPTHS:
        batch = gen_batch(L, N_PER_DEPTH, seed=1000 + L)
        correct = 0
        cert_ok = 0
        sizes = []
        t0 = time.time()
        for (x0, ops, expected, _) in batch:
            pred, cert = pm.run(x0, ops)
            if pred == expected:
                correct += 1
            ok, info = verify_certificate(cert)
            if ok:
                cert_ok += 1
            else:
                print(f"  CERT FAIL L={L}: {info}", flush=True)
            sizes.append(len(cert["steps"]))
        dt = time.time() - t0
        per_depth[L] = {
            "accuracy": correct / N_PER_DEPTH,
            "cert_validity": cert_ok / N_PER_DEPTH,
            "mean_cert_steps": sum(sizes) / len(sizes),
            "eval_seconds": dt,
        }
        print(f"L={L:3d}  A={per_depth[L]['accuracy']*100:6.2f}%  "
              f"V={per_depth[L]['cert_validity']*100:6.2f}%  "
              f"cert_steps={per_depth[L]['mean_cert_steps']:.0f}  ({dt:.1f}s)",
              flush=True)
    return per_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-only", action="store_true")
    args = ap.parse_args()

    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(res_dir, exist_ok=True)

    print("== proof-machine depth sweep ==", flush=True)
    per_depth = eval_proof_machine(ckpt)

    acc = [per_depth[L]["accuracy"] for L in DEPTHS]
    b, t = ols_slope_t(DEPTHS, acc)
    result = {
        "depths": DEPTHS,
        "n_per_depth": N_PER_DEPTH,
        "per_depth": per_depth,
        "slope_b": b,
        "slope_t": t,
        "t_crit_005": T_CRIT_005_DF6,
        "A_PM_100": per_depth[100]["accuracy"],
    }
    # Pre-registered kill checks that don't need the baseline:
    kills = []
    if per_depth[100]["accuracy"] < 0.70:
        kills.append("A_PM(100) < 70%")
    if t < -T_CRIT_005_DF6:
        kills.append(f"slope significantly negative (t={t:.2f})")
    if any(per_depth[L]["cert_validity"] < 1.0 for L in DEPTHS):
        kills.append("V(L) < 100% at some depth")
    result["kill_triggers_pm_only"] = kills

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(res_dir, f"killgate_{stamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nslope b={b:.6f}, t={t:.2f} (crit {T_CRIT_005_DF6})", flush=True)
    print(f"A_PM(100) = {per_depth[100]['accuracy']*100:.2f}%", flush=True)
    print(f"PM-only kill triggers: {kills if kills else 'none'}", flush=True)
    print(f"results -> {path}", flush=True)


if __name__ == "__main__":
    main()
