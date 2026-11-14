"""Re-runnable generation-gap evaluation for XIIV-1B checkpoints.

Reconstructs, in a durable repo location, the CPU measurements behind
``research/generation_gap_diagnosis_2026-11-13.md``. The original scripts
lived in /tmp (``xiiv_gen_gap_eval.py`` / ``xiiv_gen_gap_eval2.py``) and
were cleared on a restart, so the diagnosis numbers rested on the
diagnosis record alone. This script is the re-runnable replacement:
same procedure, same checkpoints, results written to a JSON file under
``eval/results/`` instead of /tmp.

Procedure (from the diagnosis record — see "Fix for the metric" and the
"Update — discriminating tests COMPLETE" section):
  * Standard loss: ``xiiv.diffusion.diffusion_loss`` with a FIXED mask
    ratio t = 0.175 (t_min == t_max), one generator seeded 1234 set ONCE
    outside the loop, N = 24 held sequences of 128 tokens from
    ``data_fineweb/valid.bin`` — identical sequences and masks at every
    checkpoint. The seed-once rule is the point: the Leg-3 gate reseeded
    1234 before EVERY sequence, so its first draw (0.029 < CSC's 0.15
    cold threshold) sent all 200 held sequences down the all-mask cold
    branch, and its "held-out loss" was 100% cold-start loss.
  * Cold loss: all-mask canvas, cross-entropy over all 128 positions,
    same 24 sequences. Reported as a SEPARATE number, never blended.
  * Fresh-mask cross-check: t ~ U(0.15, 0.90), N = 12, fresh generator.
    Its seed was not recorded in the diagnosis, so this run's absolute
    values are not expected to reproduce exactly — only the direction.
  * Masked next-token probe: mask ONLY the next position, full prompt
    as context; rank / probability of the expected token. Distinguishes
    "model does not know the fact" from "sampler cannot surface it".

Reference numbers from the prior record (NOT measured by this run —
listed here only so a re-run can be compared against them):
  standard loss: step-8000 5.728, step-12000 5.647, step-19600 5.402
  cold loss:     step-8000 7.747, step-12000 7.767, step-19600 7.742
  probe (step-19600): " Paris" rank 1307 (p 0.00006), " 4" rank 29,
  " 12" rank 101.

Resource note: bf16 load peaks at ~4.5 GB RSS and a forward pass
(batch 1 x 128) takes ~1.3 s on CPU, so a full three-checkpoint run
needs several GB of free RAM. Check the box first; do not run it while
another heavy job holds the RAM. ``--check`` validates paths, config,
tokenizer and data without loading any weights.

Usage (from the repo root):
    python3 eval/generation_gap_eval.py --check
    python3 eval/generation_gap_eval.py --out eval/results/generation_gap_2026-11-13.json
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from xiiv.diffusion import diffusion_loss
from xiiv.model import ModelConfig, XIIVModel
from xiiv.tokenizer import XIIVTokenizer

SEQ_LEN = 128
FIXED_SEED = 1234
T_FIXED = 0.175
N_FIXED = 24
N_FRESH = 12
FRESH_SEED = 1235  # not recorded in the diagnosis; direction only

# Same fixed prompts as the Kaggle gate (lever3_train.py FIXED_PROMPTS).
PROBE_PROMPTS = [
    ("The capital of France is", " Paris"),
    ("2 plus 2 equals", " 4"),
    ("3 times 4 equals", " 12"),
]

# Prior-record numbers, for comparison only — see module docstring.
REFERENCE_PRIOR_RECORD = {
    "standard_loss": {"step-8000": 5.728, "step-12000": 5.647, "step-19600": 5.402},
    "cold_loss": {"step-8000": 7.747, "step-12000": 7.767, "step-19600": 7.742},
    "probe_step-19600": {
        " Paris": {"rank": 1307, "p": 0.00006},
        " 4": {"rank": 29, "p": 0.0054},
        " 12": {"rank": 101, "p": 0.0012},
    },
    "source": "research/generation_gap_diagnosis_2026-11-13.md (record only; "
              "original /tmp scripts cleared — treat as reference, not ground truth)",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(REPO_ROOT))  # ~/workspace

DEFAULT_CHECKPOINTS = [
    os.path.join(WORKSPACE_ROOT, "xiiv-kaggle-gpu", "pretrain-out",
                 "xiiv-pretrain-out", "xiiv_1b_pretrain_step8000.safetensors"),
    os.path.join(WORKSPACE_ROOT, "xiiv-kaggle-gpu", "lever2-ckpt12000",
                 "xiiv-lever2-out", "xiiv_1b_pretrain_step12000.safetensors"),
    os.path.join(WORKSPACE_ROOT, "xiiv-kaggle-gpu", "lever3-out",
                 "xiiv-lever3-out", "xiiv_1b_pretrain_step19600.safetensors"),
]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def load_checkpoint_bf16(weights_path, config_path):
    """Load a safetensors checkpoint in bf16 on CPU.

    The training checkpoints ship WITHOUT a sibling config.json, so the
    standard ``xiiv.sample.load_model`` (which requires one) cannot be
    used directly — the config is passed explicitly instead.
    """
    from safetensors.torch import load_file

    cfg = ModelConfig.from_json(open(config_path).read())
    state = load_file(weights_path, device="cpu")
    model = XIIVModel(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys in {weights_path}: {unexpected}"
    assert set(missing) <= {"lm_head.weight"}, \
        f"missing keys in {weights_path}: {missing}"
    assert len(state) == 284, f"expected 284 tensors, got {len(state)}"
    model.to(torch.bfloat16).eval()
    return model, cfg


def held_sequences(valid_path, n):
    fw = np.memmap(valid_path, dtype=np.uint16, mode="r")
    assert fw.size >= n * SEQ_LEN, \
        f"valid.bin too short: {fw.size} tokens < {n * SEQ_LEN} needed"
    return [torch.tensor(np.asarray(fw[i * SEQ_LEN:(i + 1) * SEQ_LEN],
                                    dtype=np.int64))
            for i in range(n)]


def standard_loss(model, seqs, mask_id, t, n, seed):
    """Mean diffusion loss with one generator seeded once, outside the loop."""
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    with torch.no_grad():
        for ids in seqs[:n]:
            loss, _info = diffusion_loss(
                model, ids.unsqueeze(0), mask_id,
                t_min=t, t_max=t, generator=gen)
            total += float(loss.item())
    return total / n


def cold_loss(model, seqs, mask_id):
    """Mean all-mask (joint-prior) loss over the same sequences."""
    total = 0.0
    with torch.no_grad():
        for ids in seqs:
            x = torch.full_like(ids, mask_id).unsqueeze(0)
            logits = model(x)
            total += float(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids.reshape(-1)).item())
    return total / len(seqs)


def next_token_probe(model, tok, mask_id):
    """Rank / probability of the expected next token, prompt fully visible."""
    out = []
    with torch.no_grad():
        for prompt, expected in PROBE_PROMPTS:
            pids = tok.encode(prompt)
            exp = tok.encode(expected)
            assert len(exp) == 1, \
                f"expected {expected!r} is not a single token: {exp}"
            ids = torch.tensor(pids + [mask_id], dtype=torch.long).unsqueeze(0)
            logits = model(ids)[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            rank = int((logits > logits[exp[0]]).sum().item()) + 1
            out.append({
                "prompt": prompt,
                "expected": expected,
                "expected_id": int(exp[0]),
                "rank": rank,
                "vocab_size": int(logits.numel()),
                "p": float(probs[exp[0]].item()),
                "top_prediction": tok.decode([int(logits.argmax().item())]),
                "top_p": float(probs.max().item()),
            })
    return out


def evaluate_checkpoint(weights_path, config_path, tok, seqs, mask_id):
    label = os.path.basename(weights_path)
    t0 = time.time()
    print(f"[{label}] loading (bf16, CPU) ...", flush=True)
    model, cfg = load_checkpoint_bf16(weights_path, config_path)
    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "weights": {
            "path": weights_path,
            "bytes": os.path.getsize(weights_path),
            "sha256": sha256_file(weights_path),
        },
        "params": n_params,
        "standard_loss_fixed_t": standard_loss(
            model, seqs, mask_id, T_FIXED, N_FIXED, FIXED_SEED),
        "standard_loss_fresh_t": None,  # computed below (own generator)
        "cold_loss": cold_loss(model, seqs, mask_id),
        "probe": next_token_probe(model, tok, mask_id),
        "seconds": round(time.time() - t0, 1),
    }
    # Fresh-mask cross-check (t ~ U(0.15, 0.90)) needs its own generator;
    # diffusion_loss with t_min != t_max draws t per sequence.
    gen = torch.Generator().manual_seed(FRESH_SEED)
    total = 0.0
    with torch.no_grad():
        for ids in seqs[:N_FRESH]:
            loss, _ = diffusion_loss(model, ids.unsqueeze(0), mask_id,
                                     t_min=0.15, t_max=0.90, generator=gen)
            total += float(loss.item())
    result["standard_loss_fresh_t"] = total / N_FRESH
    print(f"[{label}] standard={result['standard_loss_fixed_t']:.3f} "
          f"cold={result['cold_loss']:.3f} "
          f"fresh={result['standard_loss_fresh_t']:.3f} "
          f"({result['seconds']}s)", flush=True)
    del model
    gc.collect()
    return result


def check_only(args, tok):
    problems = []
    for p in args.checkpoints:
        if not os.path.exists(p):
            problems.append(f"missing checkpoint: {p}")
    for p in (args.config, args.tokenizer, args.valid):
        if not os.path.exists(p):
            problems.append(f"missing file: {p}")
    cfg = ModelConfig.from_json(open(args.config).read())
    assert tok.mask_id == cfg.mask_id and tok.eos_id == cfg.eos_id, \
        "tokenizer/weights special-token mismatch"
    assert tok.vocab_size == cfg.vocab_size, \
        f"tokenizer vocab ({tok.vocab_size}) != config vocab ({cfg.vocab_size})"
    seqs = held_sequences(args.valid, N_FIXED)
    print("check OK:")
    print(f"  checkpoints: {len(args.checkpoints)} "
          f"({', '.join(os.path.basename(p) for p in args.checkpoints)})")
    print(f"  config: d_model={cfg.d_model} layers={cfg.n_layers} "
          f"vocab={cfg.vocab_size} mask_id={cfg.mask_id}")
    print(f"  tokenizer: vocab={tok.vocab_size} mask_id={tok.mask_id}")
    print(f"  held seqs: {len(seqs)} x {SEQ_LEN} tokens from {args.valid}")
    print(f"  procedure: fixed t={T_FIXED}, seed={FIXED_SEED} (once), "
          f"N={N_FIXED}; fresh t~U(0.15,0.90), N={N_FRESH}; cold + probe")
    if problems:
        print("problems: " + "; ".join(problems))
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS,
                    help="safetensors checkpoints to evaluate, in order")
    ap.add_argument("--config",
                    default=os.path.join(WORKSPACE_ROOT, "xiiv-kaggle-gpu",
                                         "exact1b-stitched-out",
                                         "xiiv-exact1b-stitched", "config.json"),
                    help="ModelConfig JSON (checkpoints ship without one)")
    ap.add_argument("--tokenizer",
                    default=os.path.join(REPO_ROOT, "weights",
                                         "tokenizer_32k.json"))
    ap.add_argument("--valid",
                    default=os.path.join(REPO_ROOT, "data_fineweb", "valid.bin"))
    ap.add_argument("--out",
                    default=os.path.join(REPO_ROOT, "eval", "results",
                                         "generation_gap_2026-11-13.json"))
    ap.add_argument("--check", action="store_true",
                    help="validate paths/config/tokenizer/data only; "
                         "do not load any weights")
    args = ap.parse_args()

    try:
        with open("/proc/meminfo") as f:
            mem = dict(line.split(":") for line in f.read().splitlines())
        avail_gb = int(mem["MemAvailable"].split()[0]) / 1e6
        print(f"MemAvailable: {avail_gb:.1f} GB "
              f"(bf16 load peaks ~4.5 GB; abort if this looks too low)",
              flush=True)
    except Exception:
        pass

    tok = XIIVTokenizer(args.tokenizer)
    if args.check:
        return check_only(args, tok)

    seqs = held_sequences(args.valid, N_FIXED)
    record = {
        "spec": {
            "procedure": "generation-gap diagnosis re-run",
            "source": "research/generation_gap_diagnosis_2026-11-13.md",
            "fixed_t": T_FIXED,
            "fixed_seed": FIXED_SEED,
            "n_fixed": N_FIXED,
            "n_fresh": N_FRESH,
            "fresh_seed": FRESH_SEED,
            "seq_len": SEQ_LEN,
            "dtype": "bfloat16",
            "device": "cpu",
        },
        "reference_prior_record": REFERENCE_PRIOR_RECORD,
        "checkpoints": {},
    }
    for path in args.checkpoints:
        label = os.path.basename(path).replace(
            "xiiv_1b_pretrain_", "").replace(".safetensors", "")
        record["checkpoints"][label] = evaluate_checkpoint(
            path, args.config, tok, seqs, tok.mask_id)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
