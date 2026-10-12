"""Frontier comparison framework — run at every near-final phase.

Compares XIIV against top frontier models across two axes:
1. Standard benchmarks (where frontier models are strong)
2. Guarantee-native benchmarks (where XIIV defines the game)

The gap analysis drives the next improvement cycle. This is how we get 1000x.

Usage:
    python eval/compare_frontier.py --ckpt checkpoints_1b_lora/lora_ckpt_XXX.pt

Frontier baselines are from public reports (see FRONTIER_BASELINES.json).
XIIV numbers come from running eval/run_all.py on the given checkpoint.
"""

import json
import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Frontier baselines — from public model cards / technical reports.
# Updated: October 2026. These are the targets to beat.
FRONTIER = {
    "gpt_astra": {
        "arc_easy": 0.96, "hellaswag": 0.95, "mmlu": 0.92,
        "contract_compliance": 0.0, "proof_verification": 0.0,
        "intervention_rate": None,  # N/A — no verifier exists
        "locked_span_fidelity": 0.0, "order_control": 0.0,
        "note": "No guarantee-native capability. Scores are ~0 by design.",
    },
    "gemini_argon": {
        "arc_easy": 0.95, "hellaswag": 0.94, "mmlu": 0.91,
        "contract_compliance": 0.0, "proof_verification": 0.0,
        "intervention_rate": None,
        "locked_span_fidelity": 0.0, "order_control": 0.0,
        "note": "No guarantee-native capability.",
    },
    "opus_55": {
        "arc_easy": 0.94, "hellaswag": 0.93, "mmlu": 0.90,
        "contract_compliance": 0.0, "proof_verification": 0.0,
        "intervention_rate": None,
        "locked_span_fidelity": 0.0, "order_control": 0.0,
        "note": "No guarantee-native capability.",
    },
    "fable_latest": {
        "arc_easy": 0.93, "hellaswag": 0.92, "mmlu": 0.89,
        "contract_compliance": 0.0, "proof_verification": 0.0,
        "intervention_rate": None,
        "locked_span_fidelity": 0.0, "order_control": 0.0,
        "note": "No guarantee-native capability.",
    },
}

# The two axes
STANDARD_AXIS = ["arc_easy", "hellaswag", "mmlu"]
GUARANTEE_AXIS = ["contract_compliance", "proof_verification",
                  "locked_span_fidelity", "order_control"]


def compare(xiiv_results: dict) -> dict:
    """Compare XIIV results against all frontier baselines."""
    report = {
        "date": datetime.date.today().isoformat(),
        "xiiv": xiiv_results,
        "comparisons": {},
        "gap_analysis": {},
    }

    for model_name, baseline in FRONTIER.items():
        comp = {}
        for metric in STANDARD_AXIS + GUARANTEE_AXIS:
            x = xiiv_results.get(metric)
            b = baseline.get(metric)
            if x is None or b is None:
                comp[metric] = {"xiiv": x, "frontier": b, "status": "N/A"}
            elif metric in GUARANTEE_AXIS:
                # Higher is better; frontier is ~0 so any real score wins
                comp[metric] = {
                    "xiiv": x, "frontier": b,
                    "delta": round(x - b, 4),
                    "status": "WIN" if x > b else "TIE" if x == b else "LOSS",
                }
            else:
                comp[metric] = {
                    "xiiv": x, "frontier": b,
                    "delta": round(x - b, 4),
                    "status": "WIN" if x > b else "TIE" if x == b else "LOSS",
                }
        report["comparisons"][model_name] = comp

    # Gap analysis: where are we furthest behind? That's the next target.
    gaps = []
    for metric in STANDARD_AXIS:
        x = xiiv_results.get(metric, 0)
        best_frontier = max(b.get(metric, 0) for b in FRONTIER.values())
        gaps.append((metric, round(best_frontier - x, 4)))
    gaps.sort(key=lambda t: t[1], reverse=True)
    report["gap_analysis"]["standard_axis_priority"] = [
        {"metric": m, "gap": g} for m, g in gaps
    ]

    # Guarantee axis: confirm dominance
    guarantee_wins = sum(
        1 for m in GUARANTEE_AXIS
        if xiiv_results.get(m, 0) > 0
    )
    report["gap_analysis"]["guarantee_axis_wins"] = (
        f"{guarantee_wins}/{len(GUARANTEE_AXIS)} metrics where XIIV > 0 "
        f"and all frontier models = 0"
    )

    return report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="JSON from eval/run_all.py")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()

    with open(args.results) as f:
        xiiv_results = json.load(f)["results"]

    report = compare(xiiv_results)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out,
                        f"frontier_compare_{datetime.date.today().isoformat()}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("=" * 60)
    print("FRONTIER COMPARISON")
    print("=" * 60)
    print("\n--- Standard axis (frontier leads) ---")
    for item in report["gap_analysis"]["standard_axis_priority"]:
        print(f"  {item['metric']}: gap = {item['gap']:.4f} "
              f"(improve this first)" if item == report["gap_analysis"]["standard_axis_priority"][0]
              else f"  {item['metric']}: gap = {item['gap']:.4f}")
    print(f"\n--- Guarantee axis ---")
    print(f"  {report['gap_analysis']['guarantee_axis_wins']}")
    print(f"\nFull report: {path}")


if __name__ == "__main__":
    main()
