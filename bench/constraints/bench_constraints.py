"""Constraint-satisfaction benchmark for XIIV promotion gates G3-G6.

Measures structural guarantees over fixed seeds (non-cherry-picked):
  - exact_length:            generated length == requested length
  - forbidden_words:         forbidden token ids never appear
  - edit_region_preservation: frozen (locked) positions bit-identical after infill
  - must_include_words:      every required token id appears >= once

Writes bench/constraints/results.json with the schema scripts/check_gates.py
expects. Run only AFTER training completes (CPU-heavy).

Usage:
    python bench/constraints/bench_constraints.py \
        --weights weights/xiiv.safetensors --tokenizer weights/tokenizer.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from xiiv.tokenizer import XIIVTokenizer
from xiiv.sample import load_model
from xiiv.diffusion import sample_trajectory

N_TRIALS = 50
GEN_LENGTH = 64
STEPS = 16
# common single-token words (leading space = BPE word start)
WORDS = [" rabbit", " forest", " dragon", " happy", " quickly", " river"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/xiiv.safetensors")
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="bench/constraints/results.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tok = XIIVTokenizer(args.tokenizer)
    model, cfg = load_model(args.weights)
    model.eval()

    word_ids = []
    for w in WORDS:
        ids = tok.encode(w)
        assert len(ids) == 1, f"bench word {w!r} is not single-token: {ids}"
        word_ids.append(ids[0])

    def trial(fn):
        ok = 0
        for s in range(args.trials):
            if fn(s):
                ok += 1
        return ok / args.trials

    t0 = time.time()

    # G3: exact length
    def exact_length(s):
        ids, _ = sample_trajectory(model, GEN_LENGTH, cfg.mask_id,
                                   cfg.eos_id, steps=STEPS, seed=1000 + s)
        return len(ids) == GEN_LENGTH
    r_exact = trial(exact_length)

    # G4: forbidden words absent
    def forbidden(s):
        fw = word_ids[s % len(word_ids): s % len(word_ids) + 2]
        ids, _ = sample_trajectory(
            model, GEN_LENGTH, cfg.mask_id, cfg.eos_id, steps=STEPS,
            seed=2000 + s, constraints={"forbidden_ids": fw})
        return not any(bool((ids == t).any()) for t in fw)
    r_forbid = trial(forbidden)

    # G5: locked regions preserved bit-for-bit
    def preserved(s):
        text = "The <mask:8> ran through the <mask:6>."
        pids, frozen = tok.encode_with_masks(text)
        out, _ = sample_trajectory(
            model, len(pids), cfg.mask_id, cfg.eos_id, steps=STEPS,
            seed=3000 + s, prompt_ids=pids, frozen=frozen)
        return bool((out[frozen] == pids[frozen]).all())
    r_pres = trial(preserved)

    # G6: required words present
    def required(s):
        rw = [word_ids[s % len(word_ids)]]
        ids, _ = sample_trajectory(
            model, GEN_LENGTH, cfg.mask_id, cfg.eos_id, steps=STEPS,
            seed=4000 + s, constraints={"required_ids": rw})
        return bool((ids == rw[0]).any())
    r_req = trial(required)

    results = {
        "xiiv": {
            "exact_length": {"satisfaction": r_exact},
            "forbidden_words": {"satisfaction": r_forbid},
            "edit_region_preservation": {"satisfaction": r_pres},
            "must_include_words": {"satisfaction": r_req},
        },
        "meta": {
            "trials": args.trials,
            "seeds": f"1000..{1000 + args.trials - 1} etc (fixed, non-cherry-picked)",
            "gen_length": GEN_LENGTH,
            "steps": STEPS,
            "threads": args.threads,
            "elapsed_s": round(time.time() - t0, 1),
            "words": WORDS,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    for k, v in results["xiiv"].items():
        print(f"{k}: satisfaction={v['satisfaction']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
