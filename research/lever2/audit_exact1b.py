"""Box-side audit for a gate-triggered exact-1B checkpoint.

A triggered Kaggle checkpoint (xiiv_1b_pretrain_step<N>.safetensors) is
ALREADY the full exact-1B layout — pretrain_train.py's save_checkpoint
writes all 284 tensors (everything except tied lm_head.weight) with an
in-kernel audit. No stitch is needed. This script re-verifies the file
AFTER download to this box, without loading 3.4 GB at once (streams
tensor-by-tensor via safe_open):

  checks: 284 tensors, runnable 857,429,760, headline 1,000,000,000
          (runnable + 142,570,240 contract), all fp32, zero non-finite,
          whole-file SHA256.
  optional: --expect-sha <hex> compares against the SHA recorded in the
            kernel's pretrain_gates.json for that checkpoint.

Usage:
  python audit_exact1b.py --weights xiiv_1b_pretrain_step3000.safetensors
  python audit_exact1b.py --weights ... --expect-sha <sha-from-gates.json>

Exit 0 + "AUDIT PASS" only if every check passes. Anything else is a
FAIL — the file is not treated as the exact-1B delivery candidate.
"""
import argparse
import hashlib
import os
import sys

RUNNABLE_PARAMS = 857429760
CONTRACT_PARAMS = 142570240
HEADLINE_PARAMS = 1000000000
EXPECTED_TENSORS = 284


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--expect-sha", default=None)
    args = ap.parse_args()

    import numpy as np
    from safetensors import safe_open

    checks = {}
    with safe_open(args.weights, framework="pt") as f:
        keys = list(f.keys())
        checks["tensor_count_284"] = (len(keys) == EXPECTED_TENSORS, len(keys))
        total = 0
        all_fp32 = True
        all_finite = True
        for k in keys:
            t = f.get_tensor(k)
            total += t.numel()
            if str(t.dtype) not in ("torch.float32", "float32"):
                all_fp32 = False
            if not bool(np.isfinite(t.numpy()).all()):
                all_finite = False
    checks["runnable_857429760"] = (total == RUNNABLE_PARAMS, total)
    checks["headline_1000000000"] = (total + CONTRACT_PARAMS == HEADLINE_PARAMS,
                                     total + CONTRACT_PARAMS)
    checks["all_fp32"] = (all_fp32, all_fp32)
    checks["all_finite"] = (all_finite, all_finite)
    whole_sha = sha256_file(args.weights)
    print(f"whole_file_sha256={whole_sha}")
    if args.expect_sha:
        checks["sha_matches_gate_record"] = (
            whole_sha == args.expect_sha.lower(),
            whole_sha == args.expect_sha.lower())

    ok = True
    for name, (passed, detail) in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and passed
    print("AUDIT " + ("PASS" if ok else "FAIL") +
          f" :: {os.path.basename(args.weights)} "
          f"bytes={os.path.getsize(args.weights)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
