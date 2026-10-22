"""Tests for Verifier-Guided Repair (xiiv/repair.py): diagnosis must
name the exact violating positions, and the enforced refill must fix
them without touching frozen positions — the operator the failed
GLOD/CSM/GMMD tests all pointed at.
"""
import torch

from xiiv.diffusion import check_constraints
from xiiv.repair import diagnose, verifier_guided_repair

MASK_ID, EOS_ID, LEN, VOCAB = 254, 255, 16, 256
CONS = {"required_ids": [10], "forbidden_ids": [200], "at_most": {50: 1}}


class _StubModel(torch.nn.Module):
    """Every position predicts token 7 with high confidence."""

    def __init__(self, length=LEN, vocab=VOCAB, fav=7):
        super().__init__()
        logits = torch.full((length, vocab), -5.0)
        logits[:, fav] = 5.0
        self.register_buffer("logits", logits)

    def forward(self, x, **kw):
        return self.logits[:x.shape[1]].unsqueeze(0)


def _draft():
    ids = torch.full((LEN,), 7, dtype=torch.long)
    ids[3] = 200          # forbidden
    ids[5] = 50           # cap pair (cap is 1)
    ids[6] = 50
    return ids            # required 10 missing entirely


def test_diagnose_names_positions():
    d = diagnose(_draft(), CONS, MASK_ID)
    assert d["forbidden_positions"] == [3]
    assert d["cap_excess_positions"] == {50: [5, 6]}
    assert d["missing_required"] == {10: 1}
    assert not all(d["check"].values())


def test_diagnose_phrase_span():
    cons = {"forbidden_phrases": [([8, 9], 0)]}
    ids = torch.tensor([7, 8, 9, 7] + [7] * 12, dtype=torch.long)
    d = diagnose(ids, cons, MASK_ID)
    assert d["phrase_excess_spans"] == [(1, [8, 9])]


def test_vgr_repairs_all_violation_classes():
    out, rep = verifier_guided_repair(_StubModel(), _draft(), CONS,
                                      MASK_ID, EOS_ID, seed=0)
    assert rep.success, rep.after
    assert all(check_constraints(out, CONS, MASK_ID).values())
    assert int(out[3]) != 200
    assert int((out == 50).sum()) <= 1
    assert bool((out == 10).any())
    assert rep.rounds_used <= 2


def test_vgr_never_touches_frozen():
    ids = _draft()
    frozen = torch.zeros(LEN, dtype=torch.bool)
    frozen[3] = True      # the forbidden token sits on a frozen slot
    out, rep = verifier_guided_repair(_StubModel(), ids, CONS,
                                      MASK_ID, EOS_ID, frozen=frozen,
                                      seed=0)
    assert int(out[3]) == 200          # untouched
    assert not rep.success             # honestly reported unrepairable
    assert 3 not in rep.positions_touched


def test_vgr_confidence_targeting_can_repair():
    ids = _draft()
    conf = torch.full((LEN,), 0.9)
    conf[3] = 0.05                     # violation is the least confident
    conf[5] = 0.06
    out, rep = verifier_guided_repair(
        _StubModel(), ids, CONS, MASK_ID, EOS_ID, seed=0,
        targeting="confidence", conf=conf)
    assert int(out[3]) != 200
    assert int((out == 50).sum()) <= 1
