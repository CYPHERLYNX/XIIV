"""
Tests for Phase 2 step 1 (learned composition policy, supervised warmup).

Plain asserts, repo convention. Run:
    ~/workspace/ai-tools/corsac/.venv/bin/python test_phase2.py

CPU-only, no training (dataset gen is small and deterministic; the model
test uses a random-init net and checks shapes/termination only).
"""
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PM)

from certificates import canon  # noqa: E402
from gen_dataset import gen, label_expr, rand_expr  # noqa: E402
from policy import (  # noqa: E402
    OPS, Policy, count_params, decode_ops, encode_ops, encode_spec)


def test_dataset_deterministic():
    rng1, rng2 = random.Random(7), random.Random(7)
    seen1, seen2 = set(), set()
    p1, _ = gen(50, rng1, seen1)
    p2, _ = gen(50, rng2, seen2)
    assert p1 == p2, "same seed must give the same dataset"
    assert len(p1) == 50 and len({p["spec"] for p in p1}) == 50
    for p in p1:
        assert 1 <= p["plan_len"] <= 16
        assert p["ops"] and all(o in OPS for o in p["ops"])
        assert len(p["ops"]) == p["plan_len"]


def test_label_rejects_out_of_closure():
    # coefficient 3 is outside the {2,5} factor closure -> planner rejects
    from certificates import mul, var
    item, reason = label_expr(mul(var(), 3))
    assert item is None and reason == "plan-rejected", reason
    # constant target is unreachable -> rejected
    from certificates import const
    item, reason = label_expr(const(5))
    assert item is None, "constant target must be rejected"


def test_canon_parse_roundtrip():
    """Regression test for the 2026-11-16 canon/mod precedence fix:
    parse(canon(e)) must realize to exactly realize(e) for every generated
    expr — the spec string the policy sees must be unambiguous."""
    from vrc_harness import parse_target, realize
    rng = random.Random(4242)
    n = mism = 0
    for _ in range(300):
        e = rand_expr(rng)
        r1, r2 = realize(e), realize(parse_target(canon(e)))
        n += 1
        if r1 != r2:
            mism += 1
    assert mism == 0, f"{mism}/{n} canon roundtrips mismatch"


def test_spec_roundtrip_on_random_exprs():
    rng = random.Random(123)
    checked = 0
    for _ in range(200):
        item, _ = label_expr(rand_expr(rng))
        if item is not None:
            # encode_spec must accept every canon string the generator emits
            ids = encode_spec(item["spec"])
            assert ids[0] == 1 and ids[-1] == 2  # BOS ... EOS
            checked += 1
    assert checked > 100, f"too few labelable exprs: {checked}"


def test_tokenizer_roundtrip():
    ops = ["ADD_3", "MUL_2", "MOD_7", "ADD_1"]
    assert decode_ops(encode_ops(ops)) == ops
    assert decode_ops([1, 2]) == []  # BOS, EOS -> empty


def test_model_shapes_and_greedy_terminates():
    model = Policy(d_model=64, num_layers=1)
    assert count_params(model) < 200_000, count_params(model)
    src = [encode_spec("((x0+3)*2 mod 7)+1"), encode_spec("x0+1")]
    maxlen = max(len(s) for s in src)
    src_p = torch.full((2, maxlen), 0, dtype=torch.long)
    for r, s in enumerate(src):
        src_p[r, :len(s)] = torch.tensor(s)
    tgt = [encode_ops(["ADD_3", "MUL_2"]), encode_ops(["ADD_1"])]
    max_t = max(len(t) for t in tgt)
    tgt_p = torch.full((2, max_t), 0, dtype=torch.long)
    for r, t in enumerate(tgt):
        tgt_p[r, :len(t)] = torch.tensor(t)
    logits = model(src_p, tgt_p[:, :-1],
                   src_pad_mask=(src_p == 0),
                   tgt_pad_mask=(tgt_p[:, :-1] == 0))
    assert logits.shape == (2, tgt_p.size(1) - 1, 11), logits.shape
    assert torch.isfinite(logits).all()
    # greedy decode terminates and only emits vocabulary ops
    pred = model.greedy(encode_spec("((x0+3)*2 mod 7)+1"), max_len=24)
    assert isinstance(pred, list) and len(pred) <= 24
    assert all(o in OPS for o in pred)


def test_gold_plan_verifies():
    """The eval's verified_success oracle accepts the planner's own plan —
    if this fails, the oracle (not the policy) is broken."""
    from certificates import eval_expr, verify_certificate
    from compose import ProofMachine
    from vrc_harness import parse_target, realize
    pm = ProofMachine(os.path.join(PM, "checkpoints"))
    rng = random.Random(99)
    item = None
    for _ in range(50):
        item, _ = label_expr(rand_expr(rng))
        if item is not None:
            break
    assert item is not None, "no labelable expr in 50 tries"
    target = realize(parse_target(item["spec"]))
    for x0 in (5, -3, 42, 1000, -1000):
        result, cert = pm.run(x0, item["ops"])
        assert result == pm.symbolic_result(x0, item["ops"])
        vok, _ = verify_certificate(cert)
        assert vok
        assert pm.symbolic_result(x0, item["ops"]) == eval_expr(target, x0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} PHASE-2 TESTS GREEN")
