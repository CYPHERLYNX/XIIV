"""
Noisy-input kill-gate sweep (spec 5.6, task 3).

Pre-registered 2026-10-23, BEFORE running:

  VALIDITY GATE (must hold -- any failure is an implementation bug and kills
  this prototype iteration, mirroring the V(L)<100% rule of earlier gates):
    For every depth L in {1,2,4,8,16,32,64,100} and noise radius r in
    {0,1,3,5}, over N fresh instances:
      (a) the INDEPENDENT interval verifier accepts the certificate;
      (b) the TRUE output set (brute force over all x in [x0-r, x0+r])
          is contained in the final interval;
      (c) the neural nominal result lies in the final interval.
    Gate metric: V(L, r) = 100% required in all 32 cells, flat and
    conditional suites.

  PRECISION (reported, not gated -- the theory makes no precision promise):
    P1. flat chains: interval trace exactness -- measured final width vs the
        true reachable-set width (hypothesis: equal, i.e. zero precision loss
        from noise alone on straight-line chains).
    P2. conditional joins: over-approximation = width(hull) - width(true set)
        per straddle (hypothesis: > 0 sometimes -- the hull has gaps).
    P3. compounding: mean over-approx for programs with >= 2 straddles vs
        exactly 1 (hypothesis: does NOT compound -- joins are localized).
    P4. tightness profile: mean |neural_nominal - interval_center| / width
        per op (which ops are "loose" under noise -- the L_tight signal).

Usage: python killgate_noise.py [--n-flat N] [--n-cond N] [--seed S]
Writes results/killgate_noise_<stamp>.json.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from certificates import verify_noise_certificate
from intervals import width as iv_width
from noise import NoiseProofMachine, NoiseDivergence
from compose import ProofMachine
from taskgen_noise import gen_noisy_batch, gen_noisy_cbatch

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DEPTHS = [1, 2, 4, 8, 16, 32, 64, 100]
RADII = [0, 1, 3, 5]


def _op_point(op, x):
    """Exact integer semantics of op applied to round(x) (a point)."""
    from certificates import RULES as SR, eval_expr
    return eval_expr(SR[op](("const", int(round(x)))), 0)


def run_flat(npm, L, n, r, seed):
    batch = gen_noisy_batch(L, n, r, seed)
    valid = exact = 0
    widths, true_widths = [], []
    tight = {}  # op -> [sum_ratio, count]
    divergences = 0
    for x0, rr, ops in batch:
        try:
            x_nom, cert = npm.run_noisy(x0, rr, ops)
        except NoiseDivergence:
            divergences += 1
            continue
        ok, info = verify_noise_certificate(cert)
        true_set = NoiseProofMachine.true_output_set_flat(x0, rr, ops)
        flo, fhi = cert["final_interval"]
        contained = all(flo <= v <= fhi for v in true_set)
        nominal_in = flo <= x_nom <= fhi
        if ok and contained and nominal_in:
            valid += 1
        tw = max(true_set) - min(true_set)
        widths.append(fhi - flo)
        true_widths.append(tw)
        if fhi - flo == tw:
            exact += 1
        # tightness profile: per-op abstraction looseness under noise.
        # looseness(op) = |center(out) - op_point(center(in))| / width(out):
        # how far the interval's center drifts from what the op does to the
        # center point. Affine ops are center-exact (0); MOD after a wrap is
        # not -- this is the L_tight signal made real by noise.
        for s in cert["steps"]:
            op = s["op"]
            a, b = s["in"]
            c, d = s["out"]
            w = max(d - c, 1)
            center_in = (a + b) / 2
            pt = _op_point(op, center_in)
            loose = abs((c + d) / 2 - pt) / w
            acc = tight.setdefault(op, [0.0, 0])
            acc[0] += loose
            acc[1] += 1
    return {
        "n": n, "valid": valid, "validity": valid / n,
        "divergences": divergences,
        "exact_width_frac": exact / n,
        "mean_width": sum(widths) / n,
        "mean_true_width": sum(true_widths) / n,
        "max_width": max(widths),
        "tight": {op: v[0] / v[1] for op, v in tight.items()},
    }


def run_cond(npm, L, n, r, seed):
    batch = gen_noisy_cbatch(L, n, r, seed)
    valid = 0
    straddle_counts, overapprox = [], []
    multi_single = {"one": [], "multi": []}
    divergences = 0
    for x0, rr, prog in batch:
        try:
            x_nom, cert = npm.run_program_noisy(x0, rr, prog)
        except NoiseDivergence:
            divergences += 1
            continue
        ok, info = verify_noise_certificate(cert)
        true_set = NoiseProofMachine.true_output_set_program(x0, rr, prog)
        flo, fhi = cert["final_interval"]
        contained = all(flo <= v <= fhi for v in true_set)
        nominal_in = flo <= x_nom <= fhi
        if ok and contained and nominal_in:
            valid += 1
        n_str = cert["straddles"]
        straddle_counts.append(n_str)
        if n_str:
            oa = (fhi - flo) - (max(true_set) - min(true_set))
            overapprox.append(oa)
            key = "multi" if n_str >= 2 else "one"
            multi_single[key].append(oa)
    return {
        "n": n, "valid": valid, "validity": valid / n,
        "divergences": divergences,
        "mean_straddles": sum(straddle_counts) / n,
        "frac_with_straddle": sum(1 for c in straddle_counts if c) / n,
        "mean_overapprox": (sum(overapprox) / len(overapprox)
                            if overapprox else 0.0),
        "max_overapprox": max(overapprox) if overapprox else 0,
        "mean_oa_one": (sum(multi_single["one"]) / len(multi_single["one"])
                        if multi_single["one"] else 0.0),
        "mean_oa_multi": (sum(multi_single["multi"]) / len(multi_single["multi"])
                          if multi_single["multi"] else 0.0),
        "n_multi": len(multi_single["multi"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-flat", type=int, default=200)
    ap.add_argument("--n-cond", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    t0 = time.time()
    pm = ProofMachine(CKPT)
    npm = NoiseProofMachine(pm)

    flat, cond = {}, {}
    for L in DEPTHS:
        for r in RADII:
            seed = args.seed + 7919 * L + 104729 * r
            fr = run_flat(npm, L, args.n_flat, r, seed)
            cr = run_cond(npm, L, args.n_cond, r, seed + 1)
            flat[f"L{L}_r{r}"] = fr
            cond[f"L{L}_r{r}"] = cr
            print(f"L={L:3d} r={r}: flat V={fr['validity']:.4f} "
                  f"exactW={fr['exact_width_frac']:.4f} "
                  f"cond V={cr['validity']:.4f} "
                  f"straddle={cr['frac_with_straddle']:.3f} "
                  f"oa={cr['mean_overapprox']:.2f}", flush=True)

    all_flat_ok = all(v["validity"] == 1.0 and v["divergences"] == 0
                      for v in flat.values())
    all_cond_ok = all(v["validity"] == 1.0 and v["divergences"] == 0
                      for v in cond.values())
    verdict = "PASS" if (all_flat_ok and all_cond_ok) else "KILL"
    result = {
        "date": "2026-10-23",
        "depths": DEPTHS, "radii": RADII,
        "n_flat": args.n_flat, "n_cond": args.n_cond, "seed": args.seed,
        "flat": flat, "cond": cond,
        "gate": {"flat_all_valid": all_flat_ok, "cond_all_valid": all_cond_ok},
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS, f"killgate_noise_{stamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"VERDICT: {verdict} -> {path} ({result['elapsed_s']}s)")


if __name__ == "__main__":
    main()
