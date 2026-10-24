"""Tests for xiiv.cgau (Certificate-Gated Adaptive Unmasking).

All tests are CPU-only, deterministic (fixed seeds), and run against small
scripted stub models -- no GPU, no real checkpoints.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiiv.cgau import (  # noqa: E402
    CGAUConfig,
    _culprit_positions,
    _tight_need,
    cgau_sample,
)
from xiiv.diffusion import check_constraints  # noqa: E402

VOCAB = 16
MASK_ID = 16
V = VOCAB + 1
LENGTH = 12


def _logits_for(plan):
    """Build a (L, V) logit matrix from {pos: [(tok, logit), ...]}."""
    base = torch.full((LENGTH, V), -1e9)
    for pos, pairs in plan.items():
        for tok, lv in pairs:
            base[pos, tok] = lv
    return base


class ScriptedStub(torch.nn.Module):
    """Returns a pre-scripted logit matrix per forward call index."""

    def __init__(self, scripts):
        super().__init__()
        self.scripts = scripts
        self.calls = 0

    def forward(self, x):
        idx = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        return self.scripts[idx].unsqueeze(0).expand(x.shape[0], -1, -1)


def _contract(**kw):
    c = {
        "anchors": {},
        "forbidden_ids": [],
        "required_ids": [],
        "required_counts": {},
        "at_most": {},
        "forbidden_phrases": [],
        "order_pairs": [],
    }
    c.update(kw)
    return c


def _benign_plan(required=()):
    plan = {}
    for pos in range(LENGTH):
        pairs = [(5, 3.0), (6, 2.0), (7, 1.0), (8, 0.5)]
        for t in required:
            pairs.append((t, 2.5))
        plan[pos] = pairs
    return plan


def test_gate_never_commits_forbidden():
    contract = _contract(forbidden_ids=[9], required_counts={5: 1})
    # stub confidently proposes the forbidden token everywhere
    plan = {pos: [(9, 5.0), (5, 1.0), (6, 0.5)] for pos in range(LENGTH)}
    for policy in ("gated", "cgau"):
        model = ScriptedStub([_logits_for(plan)] * 40)
        ids, tr = cgau_sample(model, length=LENGTH, mask_id=MASK_ID,
                              constraints=contract,
                              config=CGAUConfig(budget=2), seed=0,
                              policy=policy)
        assert 9 not in ids.tolist(), f"{policy} committed a forbidden token"
        assert tr.final_valid, f"{policy} should finish valid"


def test_vanilla_commits_violations():
    # documents the failure mode gating exists for: forbidden tokens are
    # logit-suppressed for every policy (shared sampler behavior), but
    # vanilla has no certificate gate, so it blows through at-most caps.
    contract = _contract(at_most={6: 1}, required_counts={5: 1})
    plan = {pos: [(6, 5.0), (5, 1.0)] for pos in range(LENGTH)}
    model = ScriptedStub([_logits_for(plan)] * 40)
    ids, tr = cgau_sample(model, length=LENGTH, mask_id=MASK_ID,
                          constraints=contract, config=CGAUConfig(budget=2),
                          seed=0, policy="vanilla")
    assert tr.completed
    assert not tr.final_valid
    assert ids.tolist().count(6) > 1


def test_determinism():
    contract = _contract(forbidden_ids=[9], required_counts={5: 2},
                         at_most={6: 1})
    plan = {pos: [(9, 4.0), (5, 3.0), (6, 2.0), (7, 1.0)]
            for pos in range(LENGTH)}
    outs = []
    for _ in range(2):
        model = ScriptedStub([_logits_for(plan)] * 40)
        ids, tr = cgau_sample(model, length=LENGTH, mask_id=MASK_ID,
                              constraints=contract,
                              config=CGAUConfig(budget=2), seed=123,
                              policy="cgau")
        outs.append((ids.tolist(), tr.to_dict()))
    assert outs[0][0] == outs[1][0]
    assert outs[0][1] == outs[1][1]


def test_early_exit_fires_on_cooperative_stub():
    contract = _contract(required_counts={5: 2})
    model = ScriptedStub([_logits_for(_benign_plan(required=(5,)))] * 40)
    ids, tr = cgau_sample(model, length=LENGTH, mask_id=MASK_ID,
                          constraints=contract,
                          config=CGAUConfig(budget=1, tau_margin=0.1),
                          seed=0, policy="cgau")
    assert tr.early_exits > 0, "early exit should fire on proof-stable spots"
    assert tr.final_valid


def test_tier2_backtrack_fires_on_persistent_cap_violation():
    # Last-resort backtracking: the stub proposes ONLY the capped token, so
    # the top-k scan can never find an alternative; on persistent
    # canvas-guilty stall tier-2 re-masks the holder. (The run cannot
    # complete -- the stub proposes nothing else -- so this tests the
    # mechanism firing + the cap invariant, not completion.)
    contract = _contract(at_most={6: 1})
    plan = {pos: [(6, 5.0)] for pos in range(LENGTH)}
    model = ScriptedStub([_logits_for(plan)] * 30)
    ids, tr = cgau_sample(model, length=LENGTH, mask_id=MASK_ID,
                          constraints=contract,
                          config=CGAUConfig(budget=2, taboo_cooldown=1,
                                            max_steps=20),
                          seed=0, policy="cgau")
    assert tr.remasks > 0, "tier-2 backtrack should fire on persistent cap hit"
    assert ids.tolist().count(6) <= 1, "cap must hold despite backtracking"


def test_anchors_never_remasked():
    contract = _contract(anchors={3: 11}, forbidden_ids=[9],
                         required_counts={5: 1})
    plan = {pos: [(9, 5.0), (5, 2.0), (6, 1.0)] for pos in range(LENGTH)}
    model = ScriptedStub([_logits_for(plan)] * 60)
    ids, tr = cgau_sample(model, length=LENGTH, mask_id=MASK_ID,
                          constraints=contract,
                          config=CGAUConfig(budget=2), seed=0, policy="cgau")
    assert int(ids[3]) == 11, "anchor must survive backtracking"
    assert tr.final_valid


def test_slot_reservation_forces_required_tokens():
    # tight contract: 2 slots, 2 required tokens; the stub prefers junk
    # (7, 8) but gives the required tokens small nonzero mass, so the
    # tight-regime ban on non-needed tokens forces their placement.
    contract = _contract(required_counts={5: 1, 6: 1})
    plan = {pos: [(7, 3.0), (8, 2.0), (5, 0.5), (6, 0.5)]
            for pos in range(LENGTH)}
    for policy in ("gated", "cgau"):
        model = ScriptedStub([_logits_for(plan)] * 40)
        ids, tr = cgau_sample(model, length=2, mask_id=MASK_ID,
                              constraints=contract,
                              config=CGAUConfig(budget=2), seed=0,
                              policy=policy)
        assert tr.completed, f"{policy}: tight slots must complete"
        assert tr.final_valid, f"{policy}: required tokens must be placed"
        assert set(ids.tolist()) == {5, 6}


def test_rollback_fires_on_scripted_stall_and_recovers():
    # Call 0 commits R@0 (cache established; every other position has no
    # viable logit and is skipped). Calls 1-3 propose nothing anywhere --
    # a pure 3-step stall -> rollback fires. Call 4+ cooperates -> valid.
    R, R2, B = 5, 6, 8
    contract = _contract(required_counts={R: 1, R2: 1})
    commit_r = _logits_for({0: [(R, 5.0)]})
    # "dead": every position predicts only the mask token -> banned ->
    # no candidates anywhere -> a pure stall. (All -1e9 would softmax to
    # uniform, which is committable -- not a stall.)
    dead = _logits_for({p: [(MASK_ID, 5.0)] for p in range(LENGTH)})
    phase_c = _logits_for({2: [(R2, 5.0)],
                           **{p: [(B, 3.0)] for p in range(LENGTH) if p != 2}})
    scripts = [commit_r, dead, dead, dead, phase_c, phase_c]
    model = ScriptedStub(scripts)
    cfg = CGAUConfig(budget=2, max_steps=30, stall_budget=3, taboo_cooldown=2)
    ids, tr = cgau_sample(model, length=LENGTH, mask_id=MASK_ID,
                          constraints=contract, config=cfg, seed=0,
                          policy="cgau")
    assert tr.rollbacks >= 1, "rollback must fire on a 3-step total stall"
    assert tr.final_valid, "run must recover to a valid canvas after rollback"
    assert int(ids[0]) == R, "cached commit must survive the rollback"
    rep = check_constraints(ids, contract, MASK_ID)
    assert all(rep.values())


def test_culprit_positions_parsing():
    x = torch.tensor([3, 7, 7, MASK_ID])
    committed = {0: (3, 1, 0.9), 1: (7, 2, 0.8), 2: (7, 3, 0.4)}
    # canvas-guilty reasons name committed holders
    assert set(_culprit_positions("cap_exceeded:7", x, committed, {})) == {1, 2}
    assert _culprit_positions("forbidden_present:3", x, committed, {}) == [0]
    # anchors are never blamed
    assert _culprit_positions("cap_exceeded:7", x, committed, {1: 7}) == [2]
    # non-canvas-guilty reasons -> no backtrack (taboo tier only)
    assert _culprit_positions("order_violated", x, committed, {}) == []
    assert _culprit_positions("required_infeasible:missing=2>slots=1",
                              x, committed, {}) == []


def test_tight_need():
    c = _contract(required_counts={5: 2})
    # one 5 missing, one slot left -> tight
    assert _tight_need([5, MASK_ID], c, {1}) == {5}
    # one 5 missing but two slots -> not tight
    assert _tight_need([5, MASK_ID, MASK_ID], c, {1, 2}) is None
    assert _tight_need([5, 5, MASK_ID], c, {2}) is None  # nothing missing
    assert _tight_need([MASK_ID] * 3, _contract(), {0, 1, 2}) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
