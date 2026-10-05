"""Intervention-rate benchmark: how often must the enforcer override the model?

This is the headline metric of guarantee-native training. For N eval cases
(prompt + required phrase):

  raw      -- the model generates freely: no brick, no contract conditioning.
              This is the "hope" baseline every frontier API ships.
  bricked  -- the same model behind a MustIncludeBrick (enforcement +
              proof certificate).

Metrics:
  raw_success_rate      fraction of raw outputs containing the phrase
  bricked_success_rate  fraction of bricked outputs containing it
                        (1.0 unless the brick itself is unsatisfiable)
  intervention_rate     fraction of cases where the brick CHANGED the
                        outcome (bricked ok, raw failed): the enforcer
                        had to intervene.

A contract-native model trained guarantee-natively should drive the
intervention rate DOWN versus the same architecture trained
unconditionally: the guarantee moves from the enforcer into the weights.

Usage:
    python scripts/measure_intervention.py --ckpt checkpoints/best.pt
        --tokenizer weights/tokenizer.json --cases 20
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xiiv import load_ckpt_trusted
from xiiv.bricks import MustIncludeBrick
from xiiv.bricks.verify import verify_certificate
from xiiv.config import ModelConfig
from xiiv.diffusion import DenoisingState
from xiiv.model import XIIVModel
from xiiv.tokenizer import XIIVTokenizer

# Story-domain phrases: the XIIV-52.5M weights were trained on TinyStories,
# so evaluation phrases must be things the model could plausibly emit.
PHRASES = [
    "once upon a time",
    "the little rabbit",
    "in the forest",
    "she smiled",
    "they played together",
    "the sun was shining",
    "he ran home",
    "a big bear",
    "the old tree",
    "they were happy",
    "one sunny day",
    "the small bird",
    "in the garden",
    "she laughed",
    "the dark night",
    "he was brave",
    "a tiny mouse",
    "the blue sky",
    "they danced",
    "it was fun",
]

PROMPTS = [
    "Lily found a door in",
    "The dog barked at",
    "Tom saw something under",
    "A girl named Mia loved",
    "The cat jumped over",
]


def has_phrase(ids, pids):
    n = len(pids)
    return any(ids[i:i + n] == pids for i in range(len(ids) - n + 1))


def clean_ids(tok, text):
    return [i for i in tok.encode(text)
            if i not in (tok.mask_id, tok.eos_id)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tok = XIIVTokenizer(args.tokenizer)
    ck = load_ckpt_trusted(args.ckpt)
    cfg = ModelConfig.from_json(ck["config"])
    model = XIIVModel(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    mask_id, eos_id = tok.mask_id, tok.eos_id
    results = []
    for i in range(args.cases):
        phrase = PHRASES[i % len(PHRASES)]
        prompt_text = PROMPTS[i % len(PROMPTS)]
        pids = clean_ids(tok, phrase)
        prompt_ids = torch.tensor(clean_ids(tok, prompt_text), dtype=torch.long)
        if not pids:
            continue

        # raw: hope baseline
        st = DenoisingState(length=args.seq_len, mask_id=mask_id,
                            eos_id=eos_id, steps=args.steps,
                            seed=args.seed + i, prompt_ids=prompt_ids,
                            constraints=None, vocab_size=cfg.vocab_size,
                            device="cpu")
        with torch.no_grad():
            while st.step(model):
                pass
        raw_ids = st.x.tolist()
        raw_ok = has_phrase(raw_ids, pids)

        # bricked: enforcement + proof
        brick = MustIncludeBrick(phrase, pids)
        st = DenoisingState(length=args.seq_len, mask_id=mask_id,
                            eos_id=eos_id, steps=args.steps,
                            seed=args.seed + i, prompt_ids=prompt_ids,
                            constraints=None, vocab_size=cfg.vocab_size,
                            device="cpu")
        brick.allocate(args.seq_len, occupied=set(),
                       prompt_len=len(prompt_ids))
        brick.enforce(st)
        with torch.no_grad():
            while st.step(model):
                pass
        bricked_ids = st.x.tolist()
        bricked_ok = has_phrase(bricked_ids, pids)
        cert = brick.certify(bricked_ids, {"require_phrase": phrase})
        verified = verify_certificate(cert, bricked_ids,
                                      {"require_phrase": phrase}).ok

        results.append({"phrase": phrase, "raw_ok": raw_ok,
                        "bricked_ok": bricked_ok, "verified": verified,
                        "intervened": (not raw_ok) and bricked_ok})

    n = len(results)
    raw_rate = sum(r["raw_ok"] for r in results) / n
    bricked_rate = sum(r["bricked_ok"] for r in results) / n
    interv = sum(r["intervened"] for r in results) / n
    verified_rate = sum(r["verified"] for r in results) / n
    print(f"cases={n} raw_success={raw_rate:.2f} "
          f"bricked_success={bricked_rate:.2f} verified={verified_rate:.2f} "
          f"INTERVENTION_RATE={interv:.2f}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"n": n, "raw_success_rate": raw_rate,
                       "bricked_success_rate": bricked_rate,
                       "verified_rate": verified_rate,
                       "intervention_rate": interv,
                       "cases": results}, f, indent=2)


if __name__ == "__main__":
    main()
