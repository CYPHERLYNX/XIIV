"""One-command reproduction of all XIIV benchmark numbers.

Usage:
    python eval/run_all.py --ckpt checkpoints_1b_gn/ckpt_8218.pt --out eval/results/

Runs every benchmark in the README table and writes a JSON report with
methodology metadata (n, seed, date, hardware). This is how every number
in the README was produced — run it yourself to verify.
"""
import argparse
import datetime
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt file")
    ap.add_argument("--out", default="eval/results", help="output directory")
    ap.add_argument("--n", type=int, default=100, help="samples per benchmark")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = {
        "checkpoint": args.ckpt,
        "n": args.n,
        "seed": args.seed,
        "date": datetime.date.today().isoformat(),
        "platform": platform.platform(),
        "results": {},
    }

    print("=" * 60)
    print("XIIV benchmark reproduction")
    print(f"  checkpoint: {args.ckpt}")
    print(f"  n={args.n} seed={args.seed}")
    print("=" * 60)

    # Import here so --help works without torch
    from eval.guarantee_bench import (
        run_contract_compliance,
        run_intervention_rate,
        run_proof_verification,
        run_locked_span_fidelity,
        run_order_control,
    )

    benchmarks = [
        ("contract_compliance", run_contract_compliance),
        ("intervention_rate", run_intervention_rate),
        ("proof_verification", run_proof_verification),
        ("locked_span_fidelity", run_locked_span_fidelity),
        ("order_control", run_order_control),
    ]

    for name, fn in benchmarks:
        print(f"\n[{name}] running...")
        try:
            result = fn(args.ckpt, n=args.n, seed=args.seed)
            report["results"][name] = result
            print(f"[{name}] {result}")
        except Exception as e:
            report["results"][name] = {"error": str(e), "status": "FAILED"}
            print(f"[{name}] FAILED: {e}")

    out_path = os.path.join(args.out, f"repro_{datetime.date.today().isoformat()}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {out_path}")
    print("\nTo reproduce README numbers: compare report['results'] with the benchmark table.")


if __name__ == "__main__":
    main()
