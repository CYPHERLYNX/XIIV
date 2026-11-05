"""Tests for xiiv/cbrs.py -- Certificate-Backed Repair Scheduling.

Falsifiers (pre-registered in the module docstring):
- F1 repair convergence: echo stub -> repair fixes every fault within
  budget; completed=True, final_no_repeat, rejects > 0.
- F2 locality: clean tail positions outside every repair zone are never
  remasked (draft tokens survive byte-identical); a clean position
  inside the zone keeps its draft token when top-1 stays
  certificate-safe.
- F3 no-regress: final canvas clean AND every position outside the
  cumulative repair zone equals its draft token.
- F4 final soundness: completed=True implies the independent
  verify_canvas(final) is clean.
- F5 fail-closed: the unfixable stub never completes; masks remain; the
  certificate claim is False.
- F6 determinism: identical runs -> identical repair zones and commit
  order.
"""

import torch

from xiiv.cbrs import (CBRSConfig, cbrs_sample, cbrs_certificate,
                       POLICIES)
from xiiv.scv import verify_canvas

MASK = 0


class ScriptedModel:
    """Canned logits: table maps pos -> [(tok, logit), ...] (rank order)."""

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


def _echo():
    """F1 stub: model top-1 always echoes token 7 (plants a repeated
    trigram in the draft); per-position unique safe alternates."""
    frozen = {0: 1, 1: 2}
    table = {p: [(7, 10.0), (30 + p, 9.0)] for p in range(2, 8)}
    return ScriptedModel(8, 64, table), frozen


def _localized():
    """F2/F3 stub: exactly one repeated trigram (3,4,5) at starts 2 and
    5; clean distinct tail 9..11 far from the fault + halo."""
    frozen = {0: 1, 1: 2}
    table = {
        2: [(3, 10.0), (32, 9.0)],
        3: [(4, 10.0), (33, 9.0)],
        4: [(5, 10.0), (34, 9.0)],
        5: [(3, 10.0), (35, 9.0)],
        6: [(4, 10.0), (36, 9.0)],
        7: [(5, 10.0), (37, 9.0)],
        8: [(20, 10.0), (38, 9.0)],
        9: [(21, 10.0), (39, 9.0)],
        10: [(22, 10.0), (40, 9.0)],
        11: [(23, 10.0), (41, 9.0)],
    }
    return ScriptedModel(12, 64, table), frozen


def _unfixable():
    """F5 stub: the model only ever predicts the echoing token 7."""
    table = {p: [(7, 10.0)] for p in range(7)}
    return ScriptedModel(7, 64, table), {}


def test_policies_known():
    assert POLICIES == ("vanilla", "cbrs")


def test_f1_repair_convergence():
    model, frozen = _echo()
    cfg = CBRSConfig(n=3, frontier=8, draft_rounds=4, draft_depth=1,
                     repair_depth=3, repair_rounds=4, halo=1)
    x, trace, cert = cbrs_sample(model, 8, MASK, config=cfg,
                                 frozen=frozen, policy="cbrs")
    assert trace.n_fault_windows_draft == 4, trace.n_fault_windows_draft
    assert trace.completed, "repair must fix the echo fault"
    assert trace.final_no_repeat
    assert trace.rejects >= 1, "the top-1 echo must be verifier-rejected"
    assert trace.repair_rounds >= 1
    assert trace.forward_passes == 2, trace.forward_passes  # 1+1
    assert cert["cert"] == "cbrs-v1"
    assert cert["completed"] is True
    assert cert["n_positions_repaired"] == 6  # positions 2..7
    assert x[0, 0].item() == 1 and x[0, 1].item() == 2  # frozen intact


def test_f2_locality():
    model, frozen = _localized()
    cfg = CBRSConfig(n=3, frontier=12, draft_rounds=4, draft_depth=1,
                     repair_depth=2, repair_rounds=4, halo=1)
    x, trace, _ = cbrs_sample(model, 12, MASK, config=cfg,
                              frozen=frozen, policy="cbrs")
    assert trace.completed
    draft = trace.draft_snapshot
    assert draft == [1, 2, 3, 4, 5, 3, 4, 5, 20, 21, 22, 23], draft
    final = x[0].tolist()
    # Clean tail outside every repair zone survives byte-identical.
    for p in (9, 10, 11):
        assert final[p] == draft[p] == 20 + (p - 8), (p, final[p])
    # Clean position inside the zone keeps its draft token (no
    # gratuitous rewrite when top-1 stays certificate-safe).
    assert final[8] == draft[8] == 20, (final[8], draft[8])
    # Cumulative zone is exactly the fault envelope {2..8}.
    cumulative = set()
    for z in trace.repair_zones:
        cumulative.update(z)
    assert cumulative == {2, 3, 4, 5, 6, 7, 8}, cumulative
    assert abs(trace.locality - 0.3) < 1e-9, trace.locality  # 3/10 untouched


def test_f3_no_regress():
    model, frozen = _localized()
    cfg = CBRSConfig(n=3, frontier=12, draft_rounds=4, draft_depth=1,
                     repair_depth=2, repair_rounds=4, halo=1)
    x, trace, _ = cbrs_sample(model, 12, MASK, config=cfg,
                              frozen=frozen, policy="cbrs")
    assert trace.completed
    final = x[0].tolist()
    check = verify_canvas(final, MASK, 3)
    assert check["final_no_repeat"], "final canvas must verify clean"
    cumulative = set()
    for z in trace.repair_zones:
        cumulative.update(z)
    for p in range(12):
        if p not in cumulative and p not in frozen:
            assert final[p] == trace.draft_snapshot[p], (
                p, "position outside the repair envelope changed")


def test_f4_final_soundness_independent():
    model, frozen = _echo()
    cfg = CBRSConfig(n=3, frontier=8, draft_rounds=4, draft_depth=1,
                     repair_depth=3, repair_rounds=4, halo=1)
    x, trace, cert = cbrs_sample(model, 8, MASK, config=cfg,
                                 frozen=frozen, policy="cbrs")
    assert trace.completed
    check = verify_canvas(x[0].tolist(), MASK, 3)
    assert check["final_no_repeat"]
    assert cert["final_no_repeat"] is True


def test_f5_fail_closed():
    model, frozen = _unfixable()
    cfg = CBRSConfig(n=3, frontier=8, draft_rounds=2, draft_depth=1,
                     repair_depth=3, repair_rounds=4, halo=1)
    x, trace, cert = cbrs_sample(model, 7, MASK, config=cfg,
                                 frozen=frozen, policy="cbrs")
    assert not trace.completed
    assert cert["completed"] is False
    assert cert["cert"] == "cbrs-v1"
    assert MASK in x[0].tolist(), "unfixable position must stay masked"
    # 1 draft pass + 1 repair pass; the second repair round finds no
    # repairable zone (masks remain but the canvas is otherwise clean)
    # and stops fail-closed.
    assert trace.forward_passes == 2, trace.forward_passes
    assert trace.repair_rounds == 1, trace.repair_rounds


def test_f6_determinism():
    model, frozen = _localized()
    cfg = CBRSConfig(n=3, frontier=12, draft_rounds=4, draft_depth=1,
                     repair_depth=2, repair_rounds=4, halo=1)
    _, t1, _ = cbrs_sample(model, 12, MASK, config=cfg,
                           frozen=frozen, policy="cbrs")
    _, t2, _ = cbrs_sample(model, 12, MASK, config=cfg,
                           frozen=frozen, policy="cbrs")
    assert t1.repair_zones == t2.repair_zones
    assert t1.repair_commit_order == t2.repair_commit_order
    assert t1.draft_commit_order == t2.draft_commit_order


def test_certificate_fail_closed_without_completion():
    model, frozen = _unfixable()
    cfg = CBRSConfig(n=3, frontier=8, draft_rounds=2, draft_depth=1,
                     repair_depth=3, repair_rounds=2, halo=1)
    _, trace, _ = cbrs_sample(model, 7, MASK, config=cfg,
                              frozen=frozen, policy="cbrs")
    cert = cbrs_certificate(trace)
    assert cert["cert"] == "cbrs-v1"
    assert cert["completed"] is False
    assert trace.forward_passes == 2, trace.forward_passes
