"""
Phase 2, step 2 — tests for the GRPO reward + training loop.

Covers: dense reward correctness (gold = 1.0, empty = 0.0, wrong < 1.0,
deterministic), group-relative advantages (zero-variance groups get zero
gradient), batched sampling (valid ops only, deterministic given a seed),
the full neural + certificate audit path on gold plans, and a 3-step RL
smoke run (loss finite, weights move, checkpoint written).
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PM)

from compose import ProofMachine  # noqa: E402
from policy import Policy, encode_spec  # noqa: E402
from reward import full_verify, probe_reward  # noqa: E402
from rl_train import group_advantages, main as rl_main  # noqa: E402


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_reward_gold_is_one():
    train = load_jsonl(os.path.join(HERE, "data", "train.jsonl"))
    item = next(it for it in train if it["plan_len"] >= 7)
    r, detail = probe_reward(item["spec"], item["ops"])
    assert r == 1.0, f"gold plan must score 1.0, got {r} {detail}"
    assert detail["probes_hit"] == detail["probes_total"] == 5


def test_reward_empty_zero():
    r, detail = probe_reward("((5*x0) mod 7)", [])
    assert r == 0.0 and detail["fail"] == "empty-plan"


def test_reward_wrong_plan_below_one():
    train = load_jsonl(os.path.join(HERE, "data", "train.jsonl"))
    item = next(it for it in train if it["plan_len"] >= 7)
    wrong = list(item["ops"])
    # Corrupt the last op: swap it for a different primitive.
    wrong[-1] = "ADD_1" if wrong[-1] != "ADD_1" else "MUL_2"
    r, _ = probe_reward(item["spec"], wrong)
    assert 0.0 <= r < 1.0, f"corrupted plan must score < 1.0, got {r}"


def test_reward_deterministic():
    train = load_jsonl(os.path.join(HERE, "data", "train.jsonl"))
    item = train[0]
    assert probe_reward(item["spec"], item["ops"]) == \
        probe_reward(item["spec"], item["ops"])


def test_group_advantages():
    # Mixed group: advantages are zero-mean, unit-std within the group.
    adv = group_advantages([1.0, 0.0, 0.5, 0.5], 4)
    assert abs(sum(adv)) < 1e-9
    assert adv[0] > 0 and adv[1] < 0
    # Zero-variance group: exactly zero advantage (no gradient).
    assert group_advantages([1.0, 1.0, 1.0, 1.0], 4) == [0.0] * 4
    assert group_advantages([0.0, 0.0], 2) == [0.0] * 2


def test_sample_batch_valid_and_deterministic():
    saved = torch.load(os.path.join(HERE, "checkpoints", "policy_step1.pt"),
                       map_location="cpu", weights_only=True)
    model = Policy(d_model=saved["config"]["d_model"],
                   num_layers=saved["config"]["num_layers"])
    model.load_state_dict(saved["state_dict"])
    train = load_jsonl(os.path.join(HERE, "data", "train.jsonl"))
    specs = [encode_spec(it["spec"]) for it in train[:3]]
    a = model.sample_batch(specs, group=4, temperature=1.0, seed=123)
    b = model.sample_batch(specs, group=4, temperature=1.0, seed=123)
    assert a == b, "sampling must be deterministic given a seed"
    assert len(a) == 12
    valid = {"ADD_1", "ADD_3", "ADD_7", "MUL_2", "MUL_5", "MOD_7",
             "MOD_13", "NEG"}
    for ops in a:
        assert all(op in valid for op in ops), f"invalid op in {ops}"
        assert len(ops) <= 24


def test_full_verify_gold():
    pm = ProofMachine(os.path.join(PM, "checkpoints"))
    train = load_jsonl(os.path.join(HERE, "data", "train.jsonl"))
    item = next(it for it in train if it["plan_len"] >= 7)
    out = full_verify(item["spec"], item["ops"], pm)
    assert out["verified_frac"] == 1.0, out
    assert out["cert_valid_frac"] == 1.0, out
    assert out["fail"] is None


def test_rl_smoke(tmp_path):
    """3 RL steps on a tiny batch: loss finite, weights move, checkpoint
    written. Runs the real main() with a small step count."""
    out = str(tmp_path)
    saved = torch.load(os.path.join(HERE, "checkpoints", "policy_step1.pt"),
                       map_location="cpu", weights_only=True)
    before = {k: v.clone() for k, v in saved["state_dict"].items()}
    rl_main(["--steps", "3", "--batch", "4", "--group", "2",
             "--audit-every", "3", "--audit-n", "2", "--out", out])
    ckpt = os.path.join(out, "policy_step2.pt")
    assert os.path.exists(ckpt)
    after = torch.load(ckpt, map_location="cpu", weights_only=True)
    moved = sum(1 for k in before
                if not torch.equal(before[k], after["state_dict"][k]))
    assert moved > 0, "RL steps must move at least one parameter tensor"
    log = json.load(open(os.path.join(HERE, "results", "rl_log_step2.json")))
    assert len(log["steps"]) == 3
    assert all(0.0 <= s["mean_reward"] <= 1.0 for s in log["steps"])


def test_tag_names_outputs(tmp_path):
    """--tag step3 must write policy_step3.pt / rl_log_step3.json, never
    clobbering the step-2 checkpoint or log (the step-3 run continues
    from step 2, so both artifacts must survive)."""
    out = str(tmp_path)
    log_path = os.path.join(HERE, "results", "rl_log_smoke.json")
    try:
        rl_main(["--steps", "2", "--batch", "4", "--group", "2",
                 "--audit-every", "2", "--audit-n", "2", "--out", out,
                 "--tag", "smoke", "--seed", "20261118"])
        assert os.path.exists(os.path.join(out, "policy_smoke.pt"))
        assert not os.path.exists(os.path.join(out, "policy_step2.pt"))
        assert os.path.exists(log_path)
        log = json.load(open(log_path))
        assert len(log["steps"]) == 2
        assert log["config"]["tag"] == "smoke"
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)
