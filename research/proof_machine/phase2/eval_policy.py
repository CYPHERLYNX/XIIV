"""
Phase 2, step 1 — PRE-REGISTERED evaluation of the learned composition policy.

Metrics (fixed before training; no peeking, no post-hoc metric shopping):
  1. exact_match: predicted op sequence == VRC planner's op sequence.
  2. verified_success (the honest metric): the predicted sequence SOLVES the
     spec — executed through ProofMachine on 5 probes, with ALL of:
       (a) concrete == symbolic on every probe (neural execution faithful),
       (b) the independent certificate verifier accepts the certificate,
       (c) the final symbolic value equals the realized target at every probe
           (target-equality oracle — the anti-vacuity integrity check of
           Vempaty et al., arXiv:2611.05590: certificate validity ALONE is
           never enough, a policy could otherwise "verify" a trivial plan).
     A verified_success without exact_match is an alternative valid
     composition — genuine generalization beyond imitation, counted separately.
  3. invalid: empty predicted plan (no EOS-terminated op emitted).
  4. alt_valid: verified_success AND NOT exact_match.

Reported overall AND by planner-plan-length bucket {1-3, 4-6, 7-9, 10-12,
13-16}. The deterministic planner is 100% by construction on both metrics —
it is the reference trajectory, not a competitor; the learned policy is
judged on generalization, not on beating it.

Usage:
    python eval_policy.py [--ckpt ...] [--data ...] [--limit N]
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PM)

from certificates import eval_expr, verify_certificate  # noqa: E402
from compose import ProofMachine  # noqa: E402
from policy import Policy, encode_spec  # noqa: E402
from vrc_harness import parse_target, realize  # noqa: E402

torch.set_num_threads(4)

PROBES = (5, -3, 42, 1000, -1000)
BUCKETS = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 16)]


def bucket_of(plan_len):
    for lo, hi in BUCKETS:
        if lo <= plan_len <= hi:
            return f"{lo}-{hi}"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        HERE, "checkpoints", "policy_step1.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--pm-ckpt", default=os.path.join(PM, "checkpoints"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="step1",
                    help="tag in the output filename eval_<tag>_<stamp>.json")
    args = ap.parse_args()

    saved = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = Policy(d_model=saved["config"]["d_model"],
                   num_layers=saved["config"]["num_layers"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    pm = ProofMachine(args.pm_ckpt)

    with open(os.path.join(args.data, "test.jsonl")) as f:
        test = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        test = test[:args.limit]

    rows = []
    t0 = time.time()
    for i, item in enumerate(test):
        spec, gold = item["spec"], item["ops"]
        pred = model.greedy(encode_spec(spec))
        exact = pred == gold
        target = realize(parse_target(spec))
        verified, fail = False, None
        if not pred:
            fail = "empty-plan"
        else:
            try:
                ok = True
                for x0 in PROBES:
                    result, cert = pm.run(x0, pred)
                    sym = pm.symbolic_result(x0, pred)
                    if result != sym:
                        ok, fail = False, "neural-divergence"
                        break
                    vok, _ = verify_certificate(cert)
                    if not vok:
                        ok, fail = False, "cert-invalid"
                        break
                    if sym != eval_expr(target, x0):
                        ok, fail = False, "target-mismatch"
                        break
                verified = ok
            except Exception as ex:  # noqa: BLE001 — any execution failure is a fail
                fail = f"execution-error:{type(ex).__name__}"
        rows.append({"spec": spec, "gold": gold, "pred": pred,
                     "plan_len": len(gold), "exact": exact,
                     "verified": verified, "fail": fail})
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(test)} ... {time.time() - t0:.0f}s", flush=True)

    def summarize(sub):
        n = len(sub)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "exact_match": round(sum(r["exact"] for r in sub) / n, 4),
            "verified_success": round(sum(r["verified"] for r in sub) / n, 4),
            "alt_valid": round(sum(r["verified"] and not r["exact"]
                                   for r in sub) / n, 4),
            "invalid_empty": round(sum(r["fail"] == "empty-plan"
                                       for r in sub) / n, 4),
            "fail_breakdown": {k: sum(1 for r in sub if r["fail"] == k)
                               for k in sorted({r["fail"] for r in sub
                                                if r["fail"]})},
        }

    out = {
        "protocol": "pre-registered 2026-11-16 (eval_policy.py docstring); "
                    "planner reference = 100% exact + 100% verified by "
                    "construction; learned policy judged on generalization",
        "ckpt": args.ckpt, "n_test": len(test),
        "elapsed_s": round(time.time() - t0, 1),
        "overall": summarize(rows),
        "by_bucket": {f"{lo}-{hi}": summarize(
            [r for r in rows if lo <= r["plan_len"] <= hi])
            for lo, hi in BUCKETS},
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(HERE, "results", f"eval_{args.tag}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("overall", "by_bucket")}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
