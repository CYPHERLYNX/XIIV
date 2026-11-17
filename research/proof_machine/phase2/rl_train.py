"""
Phase 2, step 2 — GRPO training of the learned composition policy from the
supervised-warmup checkpoint (spec research/composition_operator_v1.md §6,
option 2: "RL with certificate-validity as a dense reward").

Method (GRPO, DeepSeekMath-style, no critic — fits the 669K-param policy
and this CPU box):
  - For each spec, sample G plans at temperature T from the current policy.
  - Reward each plan with the dense certificate-obligation reward
    (reward.probe_reward): fraction of the 5 probes whose full obligation
    is discharged (execute + certificate verifies + symbolic result ==
    realized target). The integrity audit is target equality: a
    verified-but-wrong plan is paid only for probes it happens to match
    (usually 0) — never for certificate validity alone.
  - Group-relative advantage: A = (r - group_mean) / (group_std + eps);
    groups with zero variance get zero advantage (no gradient).
  - Loss = -mean(A * log pi) + beta_kl * mean(log pi_ref - log pi),
    where pi_ref is the frozen step-1 warmup checkpoint (KL anchor against
    drift/collapse). No length term in the reward (arXiv:2605.30478:
    shaping can induce degeneration toward shorter completions).
  - Training specs are STRATIFIED toward compound tasks — bucket mix
    {4-6: 15%, 7-9: 25%, 10-12: 30%, 13-16: 30%} — per the
    decomposed-to-composed asymmetry (arXiv:2609.19465): RL on decomposed
    skills fails to transfer to compound tasks, while compound-task RL
    transfers back to atomic skills. Short plans (1-3) are held out of RL
    training entirely, so the guardrail below also tests that transfer.

PRE-REGISTERED evaluation (fixed before this run; judged by
eval_policy.py, greedy decode, held-out 1500 specs — the same protocol
as step 1):
  PRIMARY: verified_success improves vs the step-1 baseline (83.3%
  overall) AND the 13-16 bucket improves vs its step-1 value (41.4%).
  GUARDRAIL: the 1-3 and 4-6 buckets do not degrade by more than 5pp vs
  step 1 (compound -> atomic transfer check, per arXiv:2609.19465).
  If the primary fails, the RL stage is reported as a NEGATIVE lever —
  not hyped, not re-framed.

Audit: every --audit-every steps, 16 sampled rollouts are run through
the FULL neural + certificate path (reward.full_verify) and the
cert-valid fraction is logged — this checks the symbolic-reward
assumption that certificates are structural (Phase-1 V = 100%).

Saves: checkpoints/policy_step2.pt (state_dict + config) and
results/rl_log.json (config + per-step rows). CPU-only. Deterministic
given --seed.

Usage:
    python rl_train.py [--steps 150] [--batch 32] [--group 4]
                       [--lr 1e-5] [--beta-kl 0.02] [--temp 1.0]
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PM)

from compose import ProofMachine  # noqa: E402
from policy import (  # noqa: E402
    Policy, SRC_PAD, TGT_BOS, TGT_EOS, TGT_PAD, count_params, encode_ops,
    encode_spec)
from reward import full_verify, probe_reward  # noqa: E402

torch.set_num_threads(4)

BUCKETS = [(4, 6), (7, 9), (10, 12), (13, 16)]
BUCKET_MIX = {(4, 6): 0.15, (7, 9): 0.25, (10, 12): 0.30, (13, 16): 0.30}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def bucket_of(plan_len):
    for lo, hi in BUCKETS:
        if lo <= plan_len <= hi:
            return (lo, hi)
    return None


def stratified_batch(by_bucket, batch_size, rng):
    """Draw `batch_size` specs with the pre-registered compound-heavy mix."""
    out = []
    counts = {b: max(1, round(batch_size * w))
              for b, w in BUCKET_MIX.items()}
    # Fix rounding so the total is exactly batch_size.
    total = sum(counts.values())
    counts[BUCKETS[-1]] += batch_size - total
    for b, n in counts.items():
        pool = by_bucket[b]
        out.extend(rng.choices(pool, k=n))
    rng.shuffle(out)
    return out


def group_advantages(rewards, group):
    """GRPO group-relative advantages. rewards: flat list, group-major.
    Zero-variance groups get exactly zero advantage (no gradient)."""
    adv = []
    for i in range(0, len(rewards), group):
        g = rewards[i:i + group]
        mean = sum(g) / len(g)
        var = sum((r - mean) ** 2 for r in g) / len(g)
        std = math.sqrt(var)
        if std < 1e-6:
            adv.extend([0.0] * len(g))
        else:
            adv.extend([(r - mean) / (std + 1e-4) for r in g])
    return adv


def build_tensors(spec_ids_list, sampled_ids_list):
    """Pad specs and sampled op-id sequences into (src, tgt) tensors.
    tgt = BOS + ops + EOS (EOS only if the sample terminated)."""
    max_s = max(len(s) for s in spec_ids_list)
    max_t = max(len(t) for t in sampled_ids_list)
    src = torch.full((len(spec_ids_list), max_s), SRC_PAD, dtype=torch.long)
    tgt = torch.full((len(sampled_ids_list), max_t), TGT_PAD, dtype=torch.long)
    for r, (s, t) in enumerate(zip(spec_ids_list, sampled_ids_list)):
        src[r, :len(s)] = torch.tensor(s)
        tgt[r, :len(t)] = torch.tensor(t)
    return src, tgt


def seq_logprobs(model, src, tgt):
    """Teacher-forced log-probs of tgt[:, 1:] under the model.
    Returns (tok_logp, mask): mask excludes PAD (post-EOS) positions;
    EOS itself is included (the decision to stop is real)."""
    logits = model(src, tgt[:, :-1],
                   src_pad_mask=(src == SRC_PAD),
                   tgt_pad_mask=(tgt[:, :-1] == TGT_PAD))
    logp = F.log_softmax(logits.float(), dim=-1)
    gold = tgt[:, 1:]
    tok_logp = logp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    mask = (gold != TGT_PAD)
    return tok_logp, mask


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--init", default=os.path.join(
        HERE, "checkpoints", "policy_step1.pt"))
    ap.add_argument("--pm-ckpt", default=os.path.join(PM, "checkpoints"))
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--audit-every", type=int, default=50)
    ap.add_argument("--audit-n", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train = load_jsonl(os.path.join(args.data, "train.jsonl"))
    by_bucket = {b: [it for it in train if bucket_of(it["plan_len"]) == b]
                 for b in BUCKETS}
    print({f"{lo}-{hi}": len(v) for (lo, hi), v in by_bucket.items()},
          flush=True)

    saved = torch.load(args.init, map_location="cpu", weights_only=True)
    model = Policy(d_model=saved["config"]["d_model"],
                   num_layers=saved["config"]["num_layers"])
    model.load_state_dict(saved["state_dict"])
    print(f"init={args.init} params={count_params(model)}", flush=True)
    ref = Policy(d_model=saved["config"]["d_model"],
                 num_layers=saved["config"]["num_layers"])
    ref.load_state_dict(saved["state_dict"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    pm = ProofMachine(args.pm_ckpt)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = {"config": vars(args), "steps": []}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        specs = stratified_batch(by_bucket, args.batch, rng)
        spec_ids = [encode_spec(s["spec"]) for s in specs]
        sampled_ops = model.sample_batch(
            spec_ids, group=args.group, temperature=args.temp,
            seed=args.seed * 1000 + step)
        # Reward every rollout (symbolic path — cheap, exact).
        rewards = []
        for spec, ops in zip(
                [s["spec"] for s in specs for _ in range(args.group)],
                sampled_ops):
            r, _ = probe_reward(spec, ops)
            rewards.append(r)
        adv = group_advantages(rewards, args.group)

        # Teacher-forced log-probs of the sampled sequences.
        sampled_ids = []
        for ops in sampled_ops:
            ids = encode_ops(ops)
            sampled_ids.append(ids)
        src, tgt = build_tensors(
            [s for s in spec_ids for _ in range(args.group)], sampled_ids)
        model.train()
        tok_logp, mask = seq_logprobs(model, src, tgt)
        with torch.no_grad():
            ref_logp, _ = seq_logprobs(ref, src, tgt)
            ent_logits = model(src, tgt[:, :-1],
                               src_pad_mask=(src == SRC_PAD),
                               tgt_pad_mask=(tgt[:, :-1] == TGT_PAD)).float()
            ent_probs = torch.softmax(ent_logits, dim=-1)
            entropy = -(ent_probs * torch.log_softmax(ent_logits, dim=-1)
                        ).sum(-1)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        tok_count = mask.sum(1).clamp(min=1)
        pg = (-(adv_t.unsqueeze(1) * tok_logp * mask).sum(1)
              / tok_count).mean()
        kl = (((ref_logp - tok_logp) * mask).sum(1) / tok_count).mean()
        loss = pg + args.beta_kl * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Training signals (logged, never claimed as evaluation).
        solved = sum(1 for i in range(0, len(rewards), args.group)
                     if max(rewards[i:i + args.group]) >= 1.0 - 1e-9)
        row = {
            "step": step,
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "frac_groups_solved": round(solved / args.batch, 4),
            "frac_rollouts_perfect": round(
                sum(1 for r in rewards if r >= 1.0 - 1e-9) / len(rewards), 4),
            "pg_loss": round(float(pg.detach()), 4),
            "kl": round(float(kl.detach()), 4),
            "entropy": round(float((entropy * mask).sum()
                                   / mask.sum()), 4),
            "elapsed_s": round(time.time() - t0, 1),
        }
        # Full-path audit: neural execution + independent certificate.
        if step % args.audit_every == 0 or step == args.steps:
            audit = [full_verify(
                specs[i // args.group]["spec"], sampled_ops[i], pm)
                for i in range(0, min(args.audit_n, len(sampled_ops)))]
            row["audit_cert_valid_frac"] = round(
                sum(a["cert_valid_frac"] for a in audit) / len(audit), 4)
            row["audit_verified_frac"] = round(
                sum(a["verified_frac"] for a in audit) / len(audit), 4)
            row["audit_n"] = len(audit)
        log["steps"].append(row)
        if step % 10 == 0 or step == 1:
            print(row, flush=True)

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "policy_step2.pt")
    torch.save({"state_dict": model.state_dict(),
                "config": {"d_model": saved["config"]["d_model"],
                           "num_layers": saved["config"]["num_layers"]}},
               ckpt)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "rl_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved {ckpt} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
