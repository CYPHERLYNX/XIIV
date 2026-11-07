"""Tests for xiiv/cwtc.py -- Certificate-Weighted Trace Credit.

Falsifiers (pre-registered in the module docstring):
- F1 credit convergence: stubborn stub (stable top-1 prob ~0.29, below
  the 0.5 confidence threshold) -> cwtc completes with every commit
  credit-driven; the confidence-only vanilla policy cannot complete.
- F2 gate on credit: a token certificate-unsafe at step start earns
  zero credit that step; the blocked repeat token never reaches the
  credit threshold and never commits; the run completes via the safe
  runner-up with a clean final canvas.
- F3 decay: a token endorsed on step 0 then abandoned never commits;
  its ledger credit stays below threshold and the final canvas carries
  the replacement token.
- F4 final soundness: completed=True implies the independent
  verify_canvas(final) is clean (recomputed).
- F5 determinism: identical runs -> identical canvas and commit log.
- F6 budget: forward_passes <= max_steps and at most ``budget``
  commits in any step.
"""

import math

import torch

from xiiv.cwtc import CWTCConfig, cwtc_sample, cwtc_certificate, POLICIES
from xiiv.scv import verify_canvas

MASK = 0


class ScriptedModel:
    """Canned logits: table maps pos -> [(tok, logit), ...]."""

    def __init__(self, length, vocab, table):
        self.length = length
        self.vocab = vocab
        self.table = table

    def __call__(self, x):
        logits = torch.full((1, self.length, self.vocab), -1e9)
        for p, cands in self.table.items():
            for tok, lg in cands:
                logits[0, p, tok] = lg
        return logits


def stubborn_table(length=8, vocab=128):
    """Every position: stable top-1 prob ~0.289 (below the 0.5
    confidence bar), three near-equal competitors. Distinct tokens, so
    no certificate issue anywhere."""
    table = {}
    for p in range(length):
        table[p] = [
            (10 + p, 0.0),
            (100 + p, -0.2),
            (110 + p, -0.2),
            (120 + p, -0.2),
        ]
    return table


def test_f1_credit_convergence():
    cfg = CWTCConfig()
    model = ScriptedModel(8, 128, stubborn_table())
    ids, trace, cert = cwtc_sample(model, 8, MASK, config=cfg, policy="cwtc")
    assert trace.completed, "cwtc must complete on the stubborn stub"
    assert trace.final_no_repeat
    assert trace.commits == 8
    assert trace.credit_driven_commits == 8, (
        f"every commit must be credit-driven (prob 0.29 < 0.5), got "
        f"{trace.credit_driven_commits} credit-driven of {trace.commits}")
    assert trace.confidence_commits == 0
    assert all(e["credit_at_commit"] >= cfg.credit_threshold
               for e in trace.commit_log)
    assert all(e["steps_endorsed"] >= 4 for e in trace.commit_log)
    assert cert["version"] == "cwtc-v1"
    assert cert["completed"] is True

    # Same stub, confidence-only policy: must NOT complete.
    model2 = ScriptedModel(8, 128, stubborn_table())
    _, vtrace, _ = cwtc_sample(model2, 8, MASK, config=cfg, policy="vanilla")
    assert not vtrace.completed
    assert vtrace.commits == 0
    assert vtrace.credit_driven_commits == 0


def test_f2_gate_on_credit():
    # Frozen prefix [10,11,12,13] + [10,11,12]: position 7's top-1 (13)
    # would complete the repeated 4-gram [10,11,12,13], so it is unsafe
    # from step 0. The safe runner-up (27) must carry the position.
    frozen = {0: 10, 1: 11, 2: 12, 3: 13, 4: 10, 5: 11, 6: 12}
    table = {7: [(13, 10.0), (27, 9.5)]}
    cfg = CWTCConfig()
    model = ScriptedModel(8, 128, table)
    ids, trace, _ = cwtc_sample(model, 8, MASK, config=cfg, policy="cwtc",
                                frozen=frozen)
    assert trace.completed
    assert trace.final_no_repeat
    assert ids.tolist() == [10, 11, 12, 13, 10, 11, 12, 27]
    assert trace.gate_rejects >= 1
    pos7 = [e for e in trace.commit_log if e["pos"] == 7]
    assert len(pos7) == 1
    assert pos7[0]["tok"] == 27
    assert pos7[0]["credit_driven"] is True
    # The blocked token 13 earned exactly zero credit at every step:
    # it is never the leading safe candidate, so no ledger entry exists.
    assert "7:13" not in trace.final_ledger
    assert all(e["tok"] != 13 or e["pos"] != 7 for e in trace.commit_log)


class FlipModel:
    """Step 0 proposes token A (20) at position 0; every later step
    proposes token B (21). Both at prob ~0.289, so both need credit.
    Positions 1-3 are fixed high-confidence tokens."""

    def __init__(self, length=4, vocab=128):
        self.length = length
        self.vocab = vocab
        self.calls = 0

    def __call__(self, x):
        logits = torch.full((1, self.length, self.vocab), -1e9)
        for p, tok in ((1, 30), (2, 31), (3, 32)):
            logits[0, p, tok] = 10.0
        lead = 20 if self.calls == 0 else 21
        logits[0, 0, lead] = 0.0
        for tok in (90, 91, 92):
            logits[0, 0, tok] = -0.2
        self.calls += 1
        return logits


def test_f3_decay():
    cfg = CWTCConfig()
    model = FlipModel()
    ids, trace, _ = cwtc_sample(model, 4, MASK, config=cfg, policy="cwtc")
    assert trace.completed
    assert ids.tolist() == [21, 30, 31, 32]
    assert all(e["tok"] != 20 for e in trace.commit_log), (
        "abandoned token A must never commit on stale credit")
    pos0 = [e for e in trace.commit_log if e["pos"] == 0]
    assert len(pos0) == 1 and pos0[0]["tok"] == 21
    assert pos0[0]["credit_driven"] is True
    # A's credit decayed every step after its single endorsement and
    # stayed far below the threshold.
    assert trace.final_ledger.get("0:20", 0.0) < cfg.credit_threshold
    assert trace.final_ledger.get("0:20", 0.0) < 0.25


def test_f4_final_soundness():
    cfg = CWTCConfig()
    model = ScriptedModel(8, 128, stubborn_table())
    ids, trace, _ = cwtc_sample(model, 8, MASK, config=cfg, policy="cwtc")
    assert trace.completed
    check = verify_canvas(ids.tolist(), MASK, cfg.n)
    assert check["final_no_repeat"] is True
    assert trace.final_no_repeat is True


def test_f5_determinism():
    cfg = CWTCConfig()
    ids1, t1, _ = cwtc_sample(ScriptedModel(8, 128, stubborn_table()),
                             8, MASK, config=cfg, policy="cwtc")
    ids2, t2, _ = cwtc_sample(ScriptedModel(8, 128, stubborn_table()),
                             8, MASK, config=cfg, policy="cwtc")
    assert ids1.tolist() == ids2.tolist()
    assert t1.commit_log == t2.commit_log
    assert t1.to_dict() == t2.to_dict()


def test_f6_budget():
    cfg = CWTCConfig(budget=4, max_steps=64)
    model = ScriptedModel(8, 128, stubborn_table())
    _, trace, _ = cwtc_sample(model, 8, MASK, config=cfg, policy="cwtc")
    assert trace.forward_passes <= cfg.max_steps
    per_step = {}
    for e in trace.commit_log:
        per_step[e["step"]] = per_step.get(e["step"], 0) + 1
    assert all(v <= cfg.budget for v in per_step.values())
    assert trace.forward_passes == 6, (
        f"stubborn stub should take 6 passes (credit crosses 1.0 at "
        f"step 4, 8 positions at budget 4), got {trace.forward_passes}")


def test_policies_known():
    assert set(POLICIES) == {"vanilla", "cwtc"}
    try:
        cwtc_sample(ScriptedModel(8, 128, stubborn_table()), 8, MASK,
                    policy="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown policy must raise ValueError")


def test_credit_math_spot_check():
    # One step of endorsement at prob p, then decay: ledger after k
    # further steps with no re-endorsement is p * decay**k.
    p = 1.0 / (1.0 + 3.0 * math.exp(-0.2))
    assert abs(p - 0.2893) < 1e-3
    assert p * 0.9 ** 5 < 0.25  # the F3 stale-credit bound
