"""
Phase 2, step 1 — train the learned composition policy by supervised
imitation of the VRC planner (see policy.py for the recipe grounding).

Saves: checkpoints/policy_step1.pt (state_dict + config) and
results/train_log.json (per-epoch loss + teacher-forced val accuracy).
CPU-only. Deterministic given --seed.
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from policy import (  # noqa: E402
    Policy, SRC_PAD, TGT_PAD, count_params, encode_ops, encode_spec)

torch.set_num_threads(4)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def batchify(pairs, batch_size, rng, shuffle=True):
    idx = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [pairs[j] for j in idx[i:i + batch_size]]
        src = [encode_spec(p["spec"]) for p in chunk]
        tgt = [encode_ops(p["ops"]) for p in chunk]
        max_s = max(len(s) for s in src)
        max_t = max(len(t) for t in tgt)
        src_t = torch.full((len(chunk), max_s), SRC_PAD, dtype=torch.long)
        tgt_t = torch.full((len(chunk), max_t), TGT_PAD, dtype=torch.long)
        for r, (s, t) in enumerate(zip(src, tgt)):
            src_t[r, :len(s)] = torch.tensor(s)
            tgt_t[r, :len(t)] = torch.tensor(t)
        yield src_t, tgt_t


@torch.no_grad()
def teacher_forced_acc(model, pairs, batch_size=128):
    """Token accuracy (excluding PAD/BOS) under teacher forcing."""
    model.eval()
    correct = total = 0
    for src, tgt in batchify(pairs, batch_size, random.Random(0),
                             shuffle=False):
        logits = model(src, tgt[:, :-1],
                       src_pad_mask=(src == SRC_PAD),
                       tgt_pad_mask=(tgt[:, :-1] == TGT_PAD))
        pred = logits.argmax(-1)
        gold = tgt[:, 1:]
        mask = gold != TGT_PAD
        correct += int(((pred == gold) & mask).sum())
        total += int(mask.sum())
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train = load_jsonl(os.path.join(args.data, "train.jsonl"))
    val = train[-500:]
    train = train[:-500]
    print(f"train={len(train)} val={len(val)}", flush=True)

    model = Policy(d_model=args.d_model, num_layers=args.layers)
    print(f"params={count_params(model)}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=TGT_PAD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)

    log = {"config": vars(args), "params": count_params(model), "epochs": []}
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot_loss, nb = 0.0, 0
        for src, tgt in batchify(train, args.batch, rng):
            logits = model(src, tgt[:, :-1],
                           src_pad_mask=(src == SRC_PAD),
                           tgt_pad_mask=(tgt[:, :-1] == TGT_PAD))
            loss = loss_fn(logits.reshape(-1, logits.size(-1)),
                           tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss)
            nb += 1
        sched.step()
        vacc = teacher_forced_acc(model, val)
        row = {"epoch": ep + 1, "train_loss": tot_loss / max(nb, 1),
               "val_token_acc": vacc, "elapsed_s": round(time.time() - t0, 1)}
        log["epochs"].append(row)
        print(row, flush=True)

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "policy_step1.pt")
    torch.save({"state_dict": model.state_dict(),
                "config": {"d_model": args.d_model,
                           "num_layers": args.layers}}, ckpt)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved {ckpt} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
