"""Tests for xiiv.dcmr (Decoupled Certificate-Margin Remasking).

All tests are CPU-only, deterministic (fixed seeds), and run against small
scripted stub models -- no GPU, no real checkpoints.

Pre-registered falsifiers:
- F1 certificate soundness: final canvas has no repeated n-gram (order n).
- F2 diversity: distinct-n-gram ratio strictly above vanilla on the
  repeater model.
- F3 structural pass budget: forward_passes <= 2 * steps.
- KILL: F1 failure -> the module does not ship.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiiv.dcmr import (  # noqa: E402
    DCMRConfig,
    dcmr_sample,
    find_repeated_ngrams,
)

VOCAB = 32
MASK_ID = 32
V = VOCAB + 1
LENGTH = 16
REPEAT = 5


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


def _repeater_plan():
    """Every position: top-1 = REPEAT, second-best = position-unique."""
    plan = {}
    for pos in range(LENGTH):
        plan[pos] = [(REPEAT, 5.0), (10 + pos, 2.0), (20, 1.0), (21, 0.5)]
    return plan


def _cfg(**kw):
    c = DCMRConfig(n=4, topk=4, budget=4, remask_budget=4, max_steps=64,
                   temperature=1.0, conf_floor=0.02, taboo_cooldown=4,
                   stall_budget=3)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_find_repeated_ngrams_unit():
    x = [1, 2, 3, 1, 2, 3, 9]
    rep = find_repeated_ngrams(x, 3, MASK_ID)
    assert rep == {(1, 2, 3): [0, 3]}
    # masks act as separators: no complete window repeats
    x2 = [1, 2, MASK_ID, 1, 2, 3]
    assert find_repeated_ngrams(x2, 3, MASK_ID) == {}
    # n larger than the canvas: vacuous
    assert find_repeated_ngrams([1, 2], 4, MASK_ID) == {}


def test_vanilla_repeats_sanity():
    """The test setup really does create the problem DCMR solves."""
    stub = ScriptedStub([_logits_for(_repeater_plan())])
    ids, trace = dcmr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                             config=_cfg(), seed=0, policy="vanilla")
    assert trace.completed
    assert not trace.final_no_repeat
    assert find_repeated_ngrams(ids.tolist(), 4, MASK_ID)
    assert trace.distinct_n_ratio < 0.2


def test_f1_certificate_soundness_on_repeater():
    stub = ScriptedStub([_logits_for(_repeater_plan())])
    ids, trace = dcmr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                             config=_cfg(), seed=0, policy="dcmr")
    assert trace.completed, "dcmr must complete on the repeater model"
    assert trace.final_no_repeat, "F1: repeated n-gram in final canvas"
    assert find_repeated_ngrams(ids.tolist(), 4, MASK_ID) == {}
    # the gate actually fired (not a vacuous pass)
    assert trace.gate_rejects > 0


def test_f2_diversity_beats_vanilla():
    cfg = _cfg()
    rep = ScriptedStub([_logits_for(_repeater_plan())])
    _, t_dcmr = dcmr_sample(rep, length=LENGTH, mask_id=MASK_ID,
                            config=cfg, seed=0, policy="dcmr")
    van = ScriptedStub([_logits_for(_repeater_plan())])
    _, t_van = dcmr_sample(van, length=LENGTH, mask_id=MASK_ID,
                           config=cfg, seed=0, policy="vanilla")
    assert (t_dcmr.distinct_n_ratio > t_van.distinct_n_ratio), (
        f"F2: dcmr {t_dcmr.distinct_n_ratio:.3f} not above vanilla "
        f"{t_van.distinct_n_ratio:.3f}")


def test_f3_structural_pass_budget():
    stub = ScriptedStub([_logits_for(_repeater_plan())])
    _, trace = dcmr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                           config=_cfg(), seed=0, policy="dcmr")
    assert trace.forward_passes <= 2 * trace.steps, (
        f"F3: {trace.forward_passes} passes in {trace.steps} steps")


def test_determinism():
    plan = _logits_for(_repeater_plan())
    ids1, _ = dcmr_sample(ScriptedStub([plan]), length=LENGTH,
                          mask_id=MASK_ID, config=_cfg(), seed=7,
                          policy="dcmr")
    ids2, _ = dcmr_sample(ScriptedStub([plan]), length=LENGTH,
                          mask_id=MASK_ID, config=_cfg(), seed=7,
                          policy="dcmr")
    assert torch.equal(ids1, ids2)


def test_frozen_never_remasked():
    frozen = {0: 3, 1: 4, 15: 6}
    stub = ScriptedStub([_logits_for(_repeater_plan())])
    ids, trace = dcmr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                             config=_cfg(), seed=0, policy="dcmr",
                             frozen=frozen)
    for p, t in frozen.items():
        assert int(ids[p]) == t
    for ev, detail in trace.events:
        assert not (ev == "remask" and detail["pos"] in frozen)


def test_fail_closed_on_unsatisfiable():
    """Only the repeat token has positive probability: the certificate
    cannot be satisfied, so the run must stall out, not emit repeats."""
    plan = {pos: [(REPEAT, 5.0)] for pos in range(LENGTH)}
    stub = ScriptedStub([_logits_for(plan)])
    ids, trace = dcmr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                             config=_cfg(), seed=0, policy="dcmr")
    assert not trace.completed
    # whatever got committed still honors the certificate
    assert find_repeated_ngrams(ids.tolist(), 4, MASK_ID) == {}


def test_regen_pass_fires_and_terminates():
    """Flat distribution -> every commit is under conf_floor ->
    decoupled regeneration must fire and the run must terminate."""
    flat = torch.zeros(LENGTH, V)  # uniform after softmax
    stub = ScriptedStub([flat])
    cfg = _cfg(conf_floor=0.05)  # 1/33 < 0.05
    ids, trace = dcmr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                             config=cfg, seed=0, policy="dcmr")
    assert trace.regen_passes > 0, "decoupled regen pass never fired"
    assert trace.completed
    assert trace.final_no_repeat
    assert trace.forward_passes <= 2 * trace.steps


def test_preexisting_repeat_recorded_not_remasked():
    """A repeating prompt is the caller's problem: recorded, and the
    sampler still protects its own commits."""
    frozen = {0: 7, 1: 7, 2: 7, 3: 7,
              4: 7, 5: 7, 6: 7, 7: 7}  # prompt repeats (7,7,7,7) twice
    stub = ScriptedStub([_logits_for(_repeater_plan())])
    _, trace = dcmr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                           config=_cfg(), seed=0, policy="dcmr",
                           frozen=frozen)
    assert trace.preexisting_repeat
    assert trace.completed  # fail-closed assert must not fire
