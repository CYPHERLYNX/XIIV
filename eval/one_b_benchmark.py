"""1B-segment benchmark harness for XIIV — measured runs only.

What this is: a fixed-spec runner that puts the assembled XIIV 1B and,
later, one competitor at a time through the SAME prompt file
(eval/one_b_prompts.json) and writes raw outputs + seconds + scores to
eval/results/, one timestamped file per run, never overwriting a
previous run.

What this is NOT: a score. No competitor score exists in this file or
in eval/one_b_competitors.json; every competitor row is UNMEASURED
until a run file proves otherwise. Automated checks are rubric v0
(a reproducible screen); a category score is provisional until a
human rates the same outputs. Do not publish a "beat" claim from
automated checks alone.

Fixed spec (v1, do not drift without a new spec version):
    length 64, steps 16, seed 7, temperature 0.8, top_p 0.9,
    context canvas 128 (model seq_len), prompt must be < length and
    <= 24 tokens (so >= 40 tokens are generated).

RAM guard: a model load is refused when MemAvailable < 2.0 GB.
There is no --force. A failed OOM run is not a result.

Usage (from the repo root):
    # Validate prompts only — no model load, safe on this box:
    python3 eval/one_b_benchmark.py --dry-run
    # XIIV run (refused here while MemAvailable < 2 GB; run on GPU):
    python3 eval/one_b_benchmark.py --model xiiv \
        --weights checkpoints_1b_sgbd/xiiv_1b_sgbd.safetensors \
        --tokenizer weights/tokenizer_32k.json --device cuda
    # Competitor: only through a supplied adapter, never by invention:
    python3 eval/one_b_benchmark.py --model competitor \
        --competitor-adapter path/to/adapter.py --device cuda
    # Score an existing run file (pure, no model load):
    python3 eval/one_b_benchmark.py --score eval/results/one_b_....json

Competitor adapter contract (a Python file with one function):
    def generate(prompt: str, spec: dict) -> dict:
        # returns {"text": str, "token_ids": list[int],
        #          "generation_seconds": float}
        # spec carries length/steps/seed/temperature/top_p. For a causal
        # competitor, length means max_new_tokens and steps is recorded
        # but not applicable (adapter must say so in its return dict
        # under "note"). The adapter must use the same prompt verbatim.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(HERE, "one_b_prompts.json")
COMPETITORS_PATH = os.path.join(HERE, "one_b_competitors.json")
RESULTS_DIR = os.path.join(HERE, "results")

LENGTH = 64
STEPS = 16
SEED = 7
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_PROMPT_TOKENS = 24
MIN_RAM_AVAILABLE_GB = 2.0
EXPECTED_SHA_PREFIX = "cf548ec4"
EXPECTED_BYTES = 3093698487

CATEGORIES = [
    "factual_explanation",
    "general_knowledge_qa",
    "arithmetic_reasoning",
    "code_short",
    "instruction_following",
    "constraint_compliance",
    "coherence_no_glitch",
    "summarisation",
    "proof_carrying",
    "latency_reliability",
]
GATE_THRESHOLD = 9  # out of 10, in EVERY category; no averaging away a fail


def spec_dict(device):
    return {
        "length": LENGTH, "steps": STEPS, "seed": SEED,
        "temperature": TEMPERATURE, "top_p": TOP_P,
        "device": device, "spec_version": "1.0",
    }


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def check_ram_guard():
    """Return (ok, available_gb). Never load a model when not ok."""
    avail = mem_available_gb()
    return avail >= MIN_RAM_AVAILABLE_GB, avail


def word_repeat_ratio(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < 10:
        return 0.0
    return collections.Counter(words).most_common(1)[0][1] / len(words)


def check_prompt(item, generated_text):
    """Pure automated rubric v0. Returns (passed, failed_checks)."""
    failed = []
    for check in item.get("checks", []):
        kind = check["type"]
        if kind == "non_empty":
            if not generated_text.strip():
                failed.append("non_empty")
        elif kind == "contains":
            if check["value"].lower() not in generated_text.lower():
                failed.append(f"missing:{check['value']}")
        elif kind == "not_contains":
            if check["value"].lower() in generated_text.lower():
                failed.append(f"forbidden_present:{check['value']}")
        elif kind == "max_word_repeat_ratio":
            if word_repeat_ratio(generated_text) > float(check["value"]):
                failed.append("word_repeat_ratio")
        else:
            failed.append(f"unknown_check:{kind}")
    return (not failed), failed


def score_run(record):
    """Pure scoring of a run record. No model load. Returns gate report."""
    per_category = {}
    for item in record["items"]:
        cat = item["category"]
        per_category.setdefault(cat, {"passed": 0, "total": 0, "items": []})
        per_category[cat]["total"] += 1
        passed, failed = check_prompt(
            {"checks": item["checks"]}, item["generated_text"])
        if passed:
            per_category[cat]["passed"] += 1
        per_category[cat]["items"].append(
            {"id": item["id"], "passed": passed, "failed_checks": failed})
    categories = {}
    gate_failures = []
    for cat in CATEGORIES:
        entry = per_category.get(cat)
        if entry is None or entry["total"] != 10:
            categories[cat] = {"score_out_of_10": None,
                               "status": "INCOMPLETE",
                               "note": "Expected 10 items; run is incomplete."}
            gate_failures.append(cat)
            continue
        score = entry["passed"]
        status = "PASS" if score >= GATE_THRESHOLD else "FAIL"
        if status == "FAIL":
            gate_failures.append(cat)
        categories[cat] = {"score_out_of_10": score, "status": status,
                           "items": entry["items"]}
    # Speed is reported, not gated on an invented floor: a speed WIN
    # requires a side-by-side run on the same hardware (see spec).
    total_seconds = sum(i["generation_seconds"] for i in record["items"])
    total_tokens = sum(i["n_tokens_non_mask"] for i in record["items"])
    return {
        "gate": ("PASS" if not gate_failures else "FAIL"),
        "gate_rule": f"Every category >= {GATE_THRESHOLD}/10; "
                     "no averaging across categories.",
        "gate_failures": gate_failures,
        "categories": categories,
        "automated_only": True,
        "human_rating_required_before_publication": True,
        "timing": {
            "total_generation_seconds": round(total_seconds, 3),
            "total_tokens_non_mask": total_tokens,
            "mean_tokens_per_second": round(
                total_tokens / total_seconds, 3) if total_seconds else 0.0,
        },
    }


def unique_out_path(model, device):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    base = f"one_b_{model}_{device}_{stamp}.json"
    path = os.path.join(RESULTS_DIR, base)
    n = 2
    while os.path.exists(path):
        path = os.path.join(
            RESULTS_DIR, f"one_b_{model}_{device}_{stamp}_{n}.json")
        n += 1
    return path


def load_prompts():
    with open(PROMPTS_PATH) as f:
        data = json.load(f)
    prompts = data["prompts"]
    assert len(prompts) == 100, f"expected 100 prompts, got {len(prompts)}"
    counts = collections.Counter(p["category"] for p in prompts)
    for cat in CATEGORIES:
        assert counts[cat] == 10, f"{cat}: {counts[cat]} items, expected 10"
    return prompts


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("competitor_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(getattr(mod, "generate", None)), \
        "adapter must define generate(prompt, spec) -> dict"
    return mod


def run_xiiv(weights, tokenizer_path, device, prompts):
    ok, avail = check_ram_guard()
    if not ok:
        raise RuntimeError(
            f"RAM guard refused model load: MemAvailable {avail:.2f} GB "
            f"< {MIN_RAM_AVAILABLE_GB} GB. Run on GPU or a quieter box; "
            "do not record an OOM failure as a result.")
    import torch
    from xiiv.diffusion import sample_trajectory
    from xiiv.sample import load_model
    from xiiv.tokenizer import XIIVTokenizer

    tok = XIIVTokenizer(tokenizer_path)
    model, cfg = load_model(weights, device=device)
    assert tok.mask_id == cfg.mask_id and tok.eos_id == cfg.eos_id
    assert tok.vocab_size == cfg.vocab_size
    weights_sha = sha256_file(weights)
    if weights != "" and not weights_sha.startswith(EXPECTED_SHA_PREFIX):
        raise RuntimeError(
            f"weights SHA {weights_sha[:16]}... does not match the verified "
            f"assembled prefix {EXPECTED_SHA_PREFIX}... — refusing to run "
            "an unverified checkpoint as XIIV 1B.")
    items = []
    for item in prompts:
        prompt_ids = torch.tensor(tok.encode(item["prompt"]), dtype=torch.long)
        if len(prompt_ids) >= LENGTH:
            raise RuntimeError(
                f"prompt {item['id']} is {len(prompt_ids)} tokens, "
                f"not < LENGTH {LENGTH}")
        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            raise RuntimeError(
                f"prompt {item['id']} is {len(prompt_ids)} tokens, "
                f"over MAX_PROMPT_TOKENS {MAX_PROMPT_TOKENS}")
        t0 = time.time()
        ids, _conf = sample_trajectory(
            model, LENGTH, tok.mask_id, tok.eos_id, steps=STEPS,
            temperature=TEMPERATURE, top_p=TOP_P, seed=SEED,
            prompt_ids=prompt_ids, device=device)
        seconds = time.time() - t0
        full_text = tok.decode(ids.tolist())
        gen_ids = ids[len(prompt_ids):]
        gen_text = tok.decode(gen_ids.tolist())
        n_tok = int((gen_ids != tok.mask_id).sum())
        items.append({
            "id": item["id"], "category": item["category"],
            "prompt": item["prompt"], "checks": item["checks"],
            "prompt_token_count": int(len(prompt_ids)),
            "full_text": full_text, "generated_text": gen_text,
            "token_ids": [int(t) for t in ids.tolist()],
            "n_tokens_non_mask": n_tok,
            "generation_seconds": round(seconds, 3),
            "tokens_per_second": round(n_tok / seconds, 3) if seconds else 0.0,
        })
    return {
        "weights": {"path": weights, "bytes": os.path.getsize(weights),
                    "sha256": weights_sha},
        "config": {"d_model": cfg.d_model, "n_layers": cfg.n_layers,
                   "vocab_size": cfg.vocab_size,
                   "contract_enabled": cfg.contract_enabled,
                   "tie_embeddings": cfg.tie_embeddings,
                   "seq_len": cfg.seq_len},
        "items": items,
    }


def run_competitor(adapter_path, device, prompts):
    adapter = load_adapter(adapter_path)
    items = []
    for item in prompts:
        out = adapter.generate(item["prompt"], spec_dict(device))
        gen_text = out["text"]
        items.append({
            "id": item["id"], "category": item["category"],
            "prompt": item["prompt"], "checks": item["checks"],
            "prompt_token_count": None,
            "full_text": gen_text, "generated_text": gen_text,
            "token_ids": out.get("token_ids", []),
            "n_tokens_non_mask": len(out.get("token_ids", [])) or len(gen_text.split()),
            "generation_seconds": round(float(out["generation_seconds"]), 3),
            "tokens_per_second": out.get("tokens_per_second"),
            "adapter_note": out.get("note", ""),
        })
    return {"weights": None, "adapter": adapter_path, "items": items}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["xiiv", "competitor"], default="xiiv")
    ap.add_argument("--weights",
                    default="checkpoints_1b_sgbd/xiiv_1b_sgbd.safetensors")
    ap.add_argument("--tokenizer", default="weights/tokenizer_32k.json")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--competitor-adapter", default=None)
    ap.add_argument("--competitor-name", default=None,
                    help="Must match a model in one_b_competitors.json.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate prompts + spec only. No model load, "
                         "no scores, no run file.")
    ap.add_argument("--score", default=None,
                    help="Score an existing run file; no model load.")
    args = ap.parse_args()

    if args.score:
        with open(args.score) as f:
            record = json.load(f)
        report = score_run(record)
        print(json.dumps(report, indent=2))
        return

    prompts = load_prompts()
    if args.dry_run:
        # Token lengths need only the tokenizer, not the model.
        from xiiv.tokenizer import XIIVTokenizer
        tok = XIIVTokenizer(args.tokenizer)
        over = [(p["id"], len(tok.encode(p["prompt"])))
                for p in prompts
                if len(tok.encode(p["prompt"])) > MAX_PROMPT_TOKENS]
        print(json.dumps({
            "status": "DRY_RUN_OK" if not over else "DRY_RUN_FAIL",
            "prompts": len(prompts),
            "categories": {c: 10 for c in CATEGORIES},
            "spec": spec_dict(args.device),
            "over_long_prompts": over,
            "scores": "NONE — dry run produces no scores",
        }, indent=2))
        if over:
            raise SystemExit(1)
        return

    if args.model == "xiiv":
        body = run_xiiv(args.weights, args.tokenizer, args.device, prompts)
        model_label = "xiiv_1b"
    else:
        if not args.competitor_adapter or not args.competitor_name:
            raise SystemExit(
                "Competitor runs require --competitor-adapter and "
                "--competitor-name. Refusing to fabricate a competitor row: "
                "see one_b_competitors.json, where every row is UNMEASURED.")
        with open(COMPETITORS_PATH) as f:
            known = [c["model"] for c in json.load(f)["competitors"]]
        if args.competitor_name not in known:
            raise SystemExit(
                f"Unknown competitor {args.competitor_name!r}. "
                f"Known: {known}. Add a row to one_b_competitors.json "
                "first — do not invent a competitor at run time.")
        body = run_competitor(args.competitor_adapter, args.device, prompts)
        model_label = args.competitor_name

    record = {
        "harness": "eval/one_b_benchmark.py spec 1.0",
        "model": model_label,
        "spec": spec_dict(args.device),
        "disclaimer": ("Raw outputs + timing + automated rubric v0 only. "
                       "Automated category scores are provisional until a "
                       "human rates the same outputs. No 'beat' claim may "
                       "be published from this file alone."),
        **body,
    }
    record["gate"] = score_run(record)
    out = unique_out_path(model_label, args.device)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(out)
    print(json.dumps(record["gate"], indent=2))


if __name__ == "__main__":
    main()
