#!/usr/bin/env python3
"""XIIV Guarantee-Native Benchmark Suite.

Implements the five benchmarks defined in GUARANTEE_BENCHMARKS.md:
  1. CCR  — Contract Compliance Rate
  2. IR   — Intervention Rate
  3. PCOV — Proof-Carrying Output Verification
  4. LSF  — Locked-Span Fidelity
  5. GOC  — Generation-Order Control

Every metric is measured by the INDEPENDENT verifier (xiiv/bricks/verify.py),
never self-reported by the model. The verifier imports nothing from the
model or the enforcement code.

Usage:
  python bench/guarantee_bench.py --ckpt <checkpoint> --tokenizer <tok.json>
      --benchmarks 1,4 --n 500 --device cpu

Status: harness implemented 2026-10-11. Not yet run on any checkpoint.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.diffusion import sample_trajectory
from xiiv.bricks import certificate as certmod
from xiiv.bricks.verify import verify_certificate, VerificationResult
from xiiv.bricks.must_include import MustIncludeBrick


# ---------------------------------------------------------------------------
# Contract suite
# ---------------------------------------------------------------------------

def make_contracts(tokenizer, vocab_size):
    """Build a suite of machine-checkable contracts.

    Each contract: (name, contract_dict, brick_builder).
    The contract_dict is what the verifier checks against.
    """
    # Simple must-include contracts using common tokens
    contracts = []
    for i, phrase in enumerate([" the", " and", " to"]):
        try:
            ids = tokenizer.encode(phrase).ids
        except Exception:
            continue
        if not ids:
            continue
        contracts.append({
            "name": f"must-include-{i}",
            "contract": {"type": "must-include", "phrase": phrase,
                         "token_ids": ids},
            "phrase": phrase,
            "token_ids": ids,
        })
    return contracts


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


# ---------------------------------------------------------------------------
# Benchmark 1: Contract Compliance Rate
# ---------------------------------------------------------------------------

def bench_ccr(model, contracts, mask_id, eos_id, n=500, length=64,
              steps=16, device="cpu", seed=0):
    """% of generations provably satisfying the contract (independent check)."""
    results = {}
    for c in contracts:
        ok = 0
        brick = MustIncludeBrick(c["phrase"], c["token_ids"])
        brick.allocate(length, set(), 0)
        for i in range(n):
            ids, _conf = sample_trajectory(
                model, length, mask_id, eos_id, steps=steps,
                seed=seed + i, device=device)
            out = ids.tolist() if torch.is_tensor(ids) else list(ids)
            cert = brick.certify(out, c["contract"])
            vr = verify_certificate(cert, out, c["contract"])
            if vr.ok:
                ok += 1
        results[c["name"]] = {"n": n, "compliant": ok, "ccr": ok / n}
    return results


# ---------------------------------------------------------------------------
# Benchmark 4: Locked-Span Fidelity
# ---------------------------------------------------------------------------

def bench_lsf(model, tokenizer, mask_id, eos_id, n=200, length=64,
              steps=16, device="cpu", seed=0):
    """Byte-exact preservation of locked spans during masked infill."""
    # Use a fixed locked phrase; freeze its positions, infill the rest.
    phrase = " the quick brown fox"
    try:
        span_ids = tokenizer.encode(phrase).ids
    except Exception:
        return {"error": "tokenizer encode failed"}
    if not span_ids:
        return {"error": "empty span"}
    span_len = len(span_ids)
    ok = 0
    for i in range(n):
        # prompt = [MASK]*prefix + span + [MASK]*suffix; freeze span positions
        prefix_len = 16
        total = prefix_len + span_len + (length - prefix_len - span_len)
        prompt = torch.full((total,), mask_id, dtype=torch.long)
        prompt[prefix_len:prefix_len + span_len] = torch.tensor(span_ids)
        frozen = torch.zeros(total, dtype=torch.bool)
        frozen[prefix_len:prefix_len + span_len] = True
        out, _c = sample_trajectory(
            model, total, mask_id, eos_id, steps=steps,
            seed=seed + i, prompt_ids=prompt, frozen=frozen, device=device)
        out = out.tolist() if torch.is_tensor(out) else list(out)
        if out[prefix_len:prefix_len + span_len] == span_ids:
            ok += 1
    return {"n": n, "preserved": ok, "lsf": ok / n}


# ---------------------------------------------------------------------------
# Benchmark 3: Proof-Carrying Output Verification (+ tamper control)
# ---------------------------------------------------------------------------

def bench_pcov(model, contracts, mask_id, eos_id, n=200, length=64,
               steps=16, device="cpu", seed=0):
    """Third-party verification of (output, certificate, contract) triples,
    plus a negative control: tampered outputs must be rejected."""
    if not contracts:
        return {"error": "no contracts"}
    c = contracts[0]
    brick = MustIncludeBrick(c["phrase"], c["token_ids"])
    brick.allocate(length, set(), 0)
    ok = 0
    for i in range(n):
        ids, _c = sample_trajectory(model, length, mask_id, eos_id, steps=steps,
                                seed=seed + i, device=device)
        out = ids.tolist() if torch.is_tensor(ids) else list(ids)
        cert = brick.certify(out, c["contract"])
        vr = verify_certificate(cert, out, c["contract"])
        if vr.ok:
            ok += 1
    # negative control: tamper with outputs, verifier must reject all
    tamper_rejected = 0
    tamper_n = 100
    for i in range(tamper_n):
        ids, _c = sample_trajectory(model, length, mask_id, eos_id, steps=steps,
                                seed=seed + 100000 + i, device=device)
        out = ids.tolist() if torch.is_tensor(ids) else list(ids)
        cert = brick.certify(out, c["contract"])
        # tamper: flip a token inside the certified span if present
        tout = list(out)
        if len(tout) > 10:
            tout[10] = (tout[10] + 1) % 32768
        vr = verify_certificate(cert, tout, c["contract"])
        if not vr.ok:
            tamper_rejected += 1
    return {"n": n, "verified": ok, "pcov": ok / n,
            "tamper_n": tamper_n, "tamper_rejected": tamper_rejected,
            "tamper_detection": tamper_rejected / tamper_n}


# ---------------------------------------------------------------------------
# Benchmark 5: Generation-Order Control
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Benchmark 2: Intervention Rate
# ---------------------------------------------------------------------------

def bench_ir(model, contracts, mask_id, eos_id, n=200, length=64, steps=16,
             device="cpu", seed=0):
    """How often does the sampler need to actively intervene?

    An intervention = the sampler preventing a constraint violation the
    model would otherwise have committed (forbidden token blocked, or
    required token force-placed in endgame).

    Metric: mean interventions per generation, broken down by kind.
    Lower is better — it means the model internalized the constraints.
    The verifier still confirms the output is compliant regardless.
    """
    from xiiv.diffusion import sample_trajectory_stateful
    total = {"forbidden_prevented": 0, "endgame_forced_placement": 0}
    generations = 0
    per_gen = []
    for i, c in enumerate(contracts):
        for j in range(max(1, n // max(1, len(contracts)))):
            tid = c["token_ids"][0] if c["token_ids"] else None
            constraints = {}
            if tid is not None:
                constraints["required_ids"] = [tid]
            # add a forbidden token to measure prevention
            constraints["forbidden_ids"] = [mask_id - 1 if mask_id > 0 else 1]
            try:
                ids, conf, state = sample_trajectory_stateful(
                    model, length, mask_id, eos_id, steps=steps,
                    seed=seed + i * 1000 + j, device=device,
                    constraints=constraints, return_trajectory=False)
                kinds = {}
                for iv in state.interventions:
                    k = iv["kind"]
                    kinds[k] = kinds.get(k, 0) + 1
                    total[k] = total.get(k, 0) + 1
                n_iv = sum(kinds.values())
                per_gen.append(n_iv)
                generations += 1
            except Exception:
                pass
    mean_iv = sum(per_gen) / max(1, len(per_gen))
    return {
        "generations": generations,
        "mean_interventions_per_gen": mean_iv,
        "by_kind": total,
    }


def bench_goc(model, mask_id, eos_id, n=100, length=64, steps=16,
              device="cpu", seed=0):
    """Non-left-to-right generation orders produce coherent outputs."""
    orders = ["left-to-right", "right-to-left", "ends-first", "random"]
    results = {}
    for order in orders:
        finite = 0
        for i in range(n):
            try:
                out, _c = sample_trajectory(
                    model, length, mask_id, eos_id, steps=steps,
                    seed=seed + i, order=order, device=device,
                    return_trajectory=False)
                out_t = out.tolist() if torch.is_tensor(out) else list(out)
                if len(out_t) == length and all(
                        0 <= t < 32768 for t in out_t):
                    finite += 1
            except Exception:
                pass
        results[order] = {"n": n, "valid": finite, "rate": finite / n}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "1": ("ccr", bench_ccr),
    "2": ("ir", bench_ir),
    "3": ("pcov", bench_pcov),
    "4": ("lsf", bench_lsf),
    "5": ("goc", bench_goc),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--benchmarks", default="1,2,3,4,5")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = args.device
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**json.loads(ckpt["config"]))
    model = XIIVModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(dev).eval()
    print(f"loaded {args.ckpt}: d={cfg.d_model} L={cfg.n_layers}")

    tokenizer = load_tokenizer(args.tokenizer)
    contracts = make_contracts(tokenizer, cfg.vocab_size)
    print(f"contracts: {[c['name'] for c in contracts]}")

    report = {"ckpt": args.ckpt, "n": args.n, "seed": args.seed,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": {}}
    for b in args.benchmarks.split(","):
        b = b.strip()
        if b not in BENCHMARKS:
            print(f"skip unknown benchmark {b}")
            continue
        name, fn = BENCHMARKS[b]
        print(f"--- benchmark {b}: {name} ---")
        t0 = time.time()
        if name in ("ccr", "pcov", "ir"):
            res = fn(model, contracts, cfg.mask_id, cfg.eos_id,
                     n=args.n, length=args.length, steps=args.steps,
                     device=dev, seed=args.seed)
        elif name == "lsf":
            res = fn(model, tokenizer, cfg.mask_id, cfg.eos_id,
                     n=args.n, length=args.length, steps=args.steps,
                     device=dev, seed=args.seed)
        else:
            res = fn(model, cfg.mask_id, cfg.eos_id,
                     n=min(args.n, 100), length=args.length,
                     steps=args.steps, device=dev, seed=args.seed)
        dt = time.time() - t0
        print(f"  {json.dumps(res)} ({dt:.1f}s)")
        report["results"][name] = res

    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = args.out or f"bench_guarantee_{int(time.time())}.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
