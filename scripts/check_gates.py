"""Promotion-gate evaluation: fail closed (ecc_mle_workflow §4).

Reads measured artifacts and checks every gate in PROMOTION_GATES.md.
Any missing input is a FAILURE. Exit 0 iff all gates pass.

Usage:
    python scripts/check_gates.py --ckpt checkpoints/best.pt \
        --weights weights/xiiv.safetensors --tokenizer weights/tokenizer.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

EXPECTED_PARAMS = 10_059_840
GATES = []


def gate(name):
    def deco(fn):
        GATES.append((name, fn))
        return fn
    return deco


@gate("G1 train_completed")
def g1(ctx):
    recs = [json.loads(l) for l in open(ctx["train_log"])
            if l.strip().startswith("{")]
    steps = [r["step"] for r in recs if "step" in r and "loss" in r]
    aborted = any("aborted" in r for r in recs)
    ok = (not aborted) and steps and max(steps) >= ctx["total_steps"]
    return ok, f"max_step={max(steps) if steps else None} aborted={aborted}"


@gate("G2 val_nll_bound<=4.0")
def g2(ctx):
    recs = [json.loads(l) for l in open(ctx["train_log"]) if "val_nll" in l]
    best = min((r["val_nll"] for r in recs), default=None)
    ok = best is not None and best <= 4.0
    return ok, f"best_val_nll={best}"


def _cbench(ctx):
    p = "bench/constraints/results.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


@gate("G3 exact_length==1.0")
def g3(ctx):
    r = _cbench(ctx)
    v = r["xiiv"]["exact_length"]["satisfaction"] if r else None
    return v == 1.0, f"satisfaction={v}"


@gate("G4 forbidden_absent==1.0")
def g4(ctx):
    r = _cbench(ctx)
    v = r["xiiv"]["forbidden_words"]["satisfaction"] if r else None
    return v == 1.0, f"satisfaction={v}"


@gate("G5 anchors_intact==1.0")
def g5(ctx):
    r = _cbench(ctx)
    v = r["xiiv"]["edit_region_preservation"]["satisfaction"] if r else None
    return v == 1.0, f"satisfaction={v}"


@gate("G6 required_present>=0.95")
def g6(ctx):
    r = _cbench(ctx)
    v = r["xiiv"]["must_include_words"]["satisfaction"] if r else None
    ok = v is not None and v >= 0.95
    return ok, f"satisfaction={v}"


@gate("G7 cpu_tok_s>=50")
def g7(ctx):
    p = "benchmarks/results_xiiv.json"
    if not os.path.exists(p):
        return False, "results_xiiv.json missing"
    tps = json.load(open(p))["gen_128"]["tok_s"]
    return tps >= 50, f"tok_s={tps}"


@gate("G8 inference_smoke")
def g8(ctx):
    try:
        from xiiv.tokenizer import XIIVTokenizer
        from xiiv.sample import load_model
        from xiiv.diffusion import sample_trajectory, check_constraints
        torch.set_num_threads(2)
        tok = XIIVTokenizer(ctx["tokenizer"])
        model, cfg = load_model(ctx["weights"])
        # generate
        ids, _ = sample_trajectory(model, 64, cfg.mask_id, cfg.eos_id,
                                   steps=8, seed=1)
        assert len(ids) == 64 and (ids != cfg.mask_id).any()
        # infill
        pids, frozen = tok.encode_with_masks("The <mask> sat.")
        out, _ = sample_trajectory(model, len(pids), cfg.mask_id, cfg.eos_id,
                                   steps=8, seed=1, prompt_ids=pids,
                                   frozen=frozen)
        assert (out[frozen] == pids[frozen]).all()
        # constrain
        wid = tok.encode(" rabbit")
        out2, _ = sample_trajectory(
            model, 64, cfg.mask_id, cfg.eos_id, steps=8, seed=1,
            constraints={"forbidden_ids": wid})
        rep = check_constraints(out2, {"forbidden_ids": wid}, cfg.mask_id)
        assert rep["forbidden_absent"]
        return True, "generate+infill+constrain ok"
    except Exception as e:  # fail closed, with the reason
        return False, f"{type(e).__name__}: {e}"


@gate("G9 weights_finite_and_count")
def g9(ctx):
    from safetensors.torch import load_file
    state = load_file(ctx["weights"])
    n = sum(v.numel() for v in state.values())
    finite = all(torch.isfinite(v).all().item() for v in state.values())
    ok = finite and n == EXPECTED_PARAMS
    return ok, f"params={n} finite={finite}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--weights", default="weights/xiiv.safetensors")
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    ap.add_argument("--total-steps", type=int, default=5000)
    args = ap.parse_args()
    ctx = {"train_log": os.path.join(args.ckpt_dir, "train_log.jsonl"),
           "weights": args.weights, "tokenizer": args.tokenizer,
           "total_steps": args.total_steps}
    if not os.path.exists(ctx["train_log"]):
        print("FAIL: train log missing (fail closed)")
        sys.exit(2)
    results, failed = {}, []
    for name, fn in GATES:
        try:
            ok, detail = fn(ctx)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        results[name] = {"pass": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        if not ok:
            failed.append(name)
    json.dump(results, open("promotion_gate_results.json", "w"), indent=2)
    if failed:
        print(f"\nPROMOTION BLOCKED: {failed}")
        sys.exit(1)
    print("\nALL GATES PASS — checkpoint may ship")


if __name__ == "__main__":
    main()
