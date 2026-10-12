"""Run standard benchmarks on XIIV with diffusion-correct scoring.

Importing xiiv.lm_eval_wrapper registers the "xiiv_diffusion" model with
lm-eval; simple_evaluate runs the tasks. Scoring uses diffusion NLL
(see xiiv/lm_eval_wrapper.py) — the methodologically valid choice for a
masked-diffusion LM, NOT autoregressive teacher-forcing.

Usage (GPU box):
    python scripts/run_benchmarks.py --weights checkpoints_500m_gn/ckpt_8218.pt \\
        --tokenizer weights/tokenizer_32k.json --tasks arc_easy,hellaswag \\
        --device cuda --limit 200

--limit runs a subset for smoke-testing; omit it for full numbers.
Results print as a table and save to --out-json.
"""

import argparse
import json
import os
import sys

# Sandbox workaround: httpx (huggingface_hub, datasets) chokes parsing
# this environment's no_proxy entry "[::1]". Drop them before import.
for _v in ("no_proxy", "NO_PROXY"):
    os.environ.pop(_v, None)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import xiiv.lm_eval_wrapper  # noqa: F401  (registers the model)
from lm_eval import simple_evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tasks", default="arc_easy,hellaswag")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-json", default="benchmark_results.json")
    args = ap.parse_args()

    results = simple_evaluate(
        model="xiiv_diffusion",
        # device flows through simple_evaluate -> model constructor;
        # don't duplicate it in model_args.
        model_args={"weights": args.weights,
                    "tokenizer": args.tokenizer},
        tasks=args.tasks.split(","),
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    summary = {t: {k: v for k, v in d.items() if not k.endswith("_stderr")}
               for t, d in results["results"].items()}
    print(json.dumps(summary, indent=2))
    json.dump(results["results"], open(args.out_json, "w"), indent=2)
    print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
