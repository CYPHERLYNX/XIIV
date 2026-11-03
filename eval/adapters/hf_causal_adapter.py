"""Hugging Face causal-LM competitor adapter for the 1B-segment benchmark.

Implements the adapter contract required by eval/one_b_benchmark.py:

    def generate(prompt: str, spec: dict) -> dict:
        # returns {"text": str, "token_ids": list[int],
        #          "generation_seconds": float}
        # optional: "note", "tokens_per_second"

Model selection: the XIIV_COMPETITOR environment variable, which MUST be
one of the model names in eval/one_b_competitors.json. Unknown or missing
names are a hard refusal — a competitor row can never be fabricated from
a wrong checkpoint or from memory of a vendor number.

Fixed-spec mapping (harness spec v1.0):
  - length=64 / prompt<=24 tokens  -> max_new_tokens = 64 - 24 = 40
    (for a causal competitor, `length` means max_new_tokens; `steps`
    is recorded for parity but is not applicable — stated in `note`).
  - temperature=0.8, top_p=0.9, seed=7 taken from spec verbatim.
  - The prompt is passed to the tokenizer verbatim, no chat template,
    no system prompt: base and instruct variants alike see the raw
    prompt string, exactly as XIIV does.

Runtime requirements (NOT needed on this box; needed where the run
happens — his RTX 5090 rig or Kaggle after the Nov 5 quota reset):
  pip install transformers torch

Gated repos (Llama, Gemma): the download is refused with the exact HF
error plus what to do (accept terms on huggingface.co, `hf auth login`).
A gated download is NOT a run.

Everything this adapter reports is measured in-process. It never fills
a score from a model card, a vendor blog, or memory.
"""

import json
import os
import sys
import time

COMPETITORS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "one_b_competitors.json",
)
# Harness spec v1.0: prompts are <= 24 tokens, canvas length 64, so the
# generated span is >= 40 tokens. Causal analogue of `length`.
PROMPT_TOKEN_BUDGET = 24


def known_models():
    with open(COMPETITORS_PATH) as f:
        data = json.load(f)
    return {c["model"]: c["hf_repo"] for c in data["competitors"]}


def _resolve_repo():
    name = os.environ.get("XIIV_COMPETITOR", "").strip()
    known = known_models()
    if not name:
        raise RuntimeError(
            "XIIV_COMPETITOR is not set. Set it to one of the models in "
            f"eval/one_b_competitors.json: {sorted(known)}. Refusing to "
            "guess a competitor."
        )
    if name not in known:
        raise RuntimeError(
            f"Unknown competitor {name!r}. Known: {sorted(known)}. "
            "Refusing to fabricate a competitor row."
        )
    return name, known[name]


def _load(repo, device):
    """Load (model, tokenizer). The only seam that touches HF/torch."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "This adapter needs `transformers` and `torch` installed. "
            "pip install transformers torch — it runs where the GPU is "
            "(his rig, or Kaggle after the Nov 5 quota reset), not on "
            "this RAM-constrained box."
        ) from e
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
    except Exception as e:
        raise RuntimeError(
            f"Could not load tokenizer for {repo}: {e}. If this is a gated "
            "repo (Llama/Gemma), accept the terms on huggingface.co and run "
            "`hf auth login` first. A gated download is not a run."
        ) from e
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=dtype, trust_remote_code=False
        ).to(device)
    except Exception as e:
        raise RuntimeError(
            f"Could not load weights for {repo}: {e}. If this is a gated "
            "repo (Llama/Gemma), accept the terms on huggingface.co and run "
            "`hf auth login` first. A gated download is not a run."
        ) from e
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, tok, n_params, torch


def _run(model, tok, torch_mod, prompt, spec, repo, n_params, device):
    device = spec.get("device", device)
    seed = int(spec.get("seed", 7))
    temperature = float(spec.get("temperature", 0.8))
    top_p = float(spec.get("top_p", 0.9))
    max_new = int(spec.get("length", 64)) - PROMPT_TOKEN_BUDGET

    torch_mod.manual_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch_mod.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=getattr(tok, "pad_token_id", None)
            or getattr(tok, "eos_token_id", None),
        )
    seconds = time.time() - t0

    gen = out[0][input_len:]
    gen_ids = gen.tolist() if hasattr(gen, "tolist") else [int(x) for x in gen]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    note = (
        f"causal autoregressive ({repo}): `steps` N/A for causal models "
        f"(recorded for parity only); length=max_new_tokens={max_new}; "
        f"prompt passed verbatim, no chat template; temp={temperature}, "
        f"top_p={top_p}, torch.manual_seed({seed}); params={n_params:,} "
        f"from loaded config; dtype/device={device}."
    )
    return {
        "text": text,
        "token_ids": [int(x) for x in gen_ids],
        "generation_seconds": float(seconds),
        "tokens_per_second": (len(gen_ids) / seconds) if seconds > 0 else 0.0,
        "note": note,
    }


_LOADED = {}


def generate(prompt, spec):
    """Adapter contract entry point: generate(prompt, spec) -> dict."""
    name, repo = _resolve_repo()
    device = spec.get("device", "cuda")
    key = (name, device)
    if key not in _LOADED:
        model, tok, n_params, torch_mod = _load(repo, device)
        _LOADED[key] = (model, tok, n_params, torch_mod)
    model, tok, n_params, torch_mod = _LOADED[key]
    out = _run(model, tok, torch_mod, prompt, spec, repo, n_params, device)
    out["competitor"] = name
    out["hf_repo"] = repo
    return out
