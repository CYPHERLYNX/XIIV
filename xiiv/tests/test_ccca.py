"""Tests for xiiv/ccca.py -- Cross-Chain Consensus Arbitration.

Falsifiers (pre-registered in the module docstring):
- F1 consensus convergence: clean stub -> completed, consensus_rate
  1.0, no disagreements, final_no_repeat.
- F2 disagreement deferral + rollback: fork stub -> the disputed
  position is never committed while disputed; n_disagreements >= 1,
  n_rollbacks >= 1; its shared-canvas commit lands strictly after the
  disagreement round, with the post-rollback token.
- F3 convergence after rollback: fork stub completes with
  consensus_rate 1.0 and zero non-consensus commits.
- F4 final soundness: completed=True implies the independent
  verify_canvas(final) is clean (recomputed).
- F5 fail-closed: the repeater stub never completes; masks remain; the
  certificate claim is False; the diagnostic pass reports dead
  positions.
- F6 determinism: identical runs -> identical shared canvas, commit
  order, and disagreement events.
- F7 promotion honesty: K=1 (nothing can ever confirm) -> every commit
  goes through promotion; completed with consensus_rate 0.0 and the
  certificate itemizing every promoted position.
"""

import torch

from xiiv.ccca import (CCCAConfig, ccca_sample, ccca_certificate,
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


class ForkModel:
    """Reveal-order-sensitive stub: the token proposed at position 4
    depends on whether its left neighbour (position 3) is already
    committed in the chain's working canvas. Chains that commit 3
    first propose 50; chains that reach 4 first (neighbour still
    masked) propose 51. All other positions have fixed, distinct
    tokens. Top-1-probability gaps order the confidence chain as
    4 > 3 > 6 > 7 > 5 > 2."""

    def __init__(self, length=8, vocab=128):
        self.length = length
        self.vocab = vocab

    def __call__(self, x):
        logits = torch.full((1, self.length, self.vocab), -1e9)
        rows = {
            2: [(20, 10.0), (21, 9.8)],
            3: [(30, 10.0), (31, 8.0)],
            5: [(55, 10.0), (56, 9.5)],
            6: [(60, 10.0), (61, 8.5)],
            7: [(70, 10.0), (71, 9.0)],
        }
        for p, cands in rows.items():
            for tok, lg in cands:
                logits[0, p, tok] = lg
        if int(x[0, 3].item()) != MASK:
            logits[0, 4, 50] = 10.0
            logits[0, 4, 51] = 7.0
        else:
            logits[0, 4, 51] = 10.0
            logits[0, 4, 50] = 7.0
        return logits


def _clean():
    """F1/F7 stub: fixed distinct tokens per position, no repeats
    possible, order-independent (all chains see identical logits)."""
    frozen = {0: 1, 1: 2}
    table = {p: [(30 + p, 10.0), (40 + p, 9.0)] for p in range(2, 8)}
    return ScriptedModel(8, 64, table), frozen


def _unfixable():
    """F5 stub: the model only ever predicts the echoing token 7."""
    table = {p: [(7, 10.0)] for p in range(7)}
    return ScriptedModel(7, 64, table), {}


def _cfg(**over):
    kw = dict(n=3, frontier=2, draft_depth=2, chains=3, quorum=2,
              confirm_rounds=3, max_rounds=16, stall_budget=3)
    kw.update(over)
    return CCCAConfig(**kw)


def test_policies_known():
    assert POLICIES == ("vanilla", "ccca")


def test_f1_consensus_convergence():
    model, frozen = _clean()
    x, trace, cert = ccca_sample(model, 8, MASK, config=_cfg(),
                                 frozen=frozen, policy="ccca")
    assert trace.completed, "clean stub must complete by consensus"
    assert trace.final_no_repeat
    assert trace.consensus_rate == 1.0, trace.consensus_rate
    assert trace.nonconsensus_commits == 0
    assert trace.disagreements == 0
    assert trace.rollbacks == 0
    assert trace.commits == 6
    assert trace.rounds == 3, trace.rounds
    # 3 + 3 + 2 passes: the rtl chain's canvas is fully synced after
    # round 2, so it runs no pass in round 3.
    assert trace.forward_passes == 8, trace.forward_passes
    assert cert["cert"] == "ccca-v1"
    assert cert["completed"] is True
    assert cert["consensus_rate"] == 1.0
    assert cert["nonconsensus_positions"] == []
    assert x[0].tolist() == [1, 2, 32, 33, 34, 35, 36, 37], x[0].tolist()


def test_f2_disagreement_deferral_and_rollback():
    model = ForkModel()
    frozen = {0: 1, 1: 2}
    x, trace, _ = ccca_sample(model, 8, MASK, config=_cfg(),
                              frozen=frozen, policy="ccca")
    assert trace.completed
    assert trace.disagreements >= 1, "a dispute must be detected"
    assert trace.rollbacks >= 1, "a dispute must roll back"
    # The dispute is at position 4, round 2 (0-indexed round 1), between
    # the pre-context token 51 and the post-context token 50.
    assert (1, 4, [50, 51]) in trace.disagreement_order, \
        trace.disagreement_order
    # Position 4's shared-canvas commit lands strictly AFTER the
    # disagreement round, with the post-rollback token 50 -- never the
    # disputed 51, never in the dispute round.
    commits_4 = [(r, t, c) for (r, p, t, c) in trace.commit_order if p == 4]
    assert len(commits_4) == 1, commits_4
    rnd_4, tok_4, cons_4 = commits_4[0]
    assert rnd_4 > 1, "commit must follow the disagreement round"
    assert tok_4 == 50, "post-rollback consensus token"
    assert cons_4 is True
    assert x[0, 4].item() == 50


def test_f3_convergence_after_rollback():
    model = ForkModel()
    frozen = {0: 1, 1: 2}
    x, trace, cert = ccca_sample(model, 8, MASK, config=_cfg(),
                                 frozen=frozen, policy="ccca")
    assert trace.completed, "denser context must resolve the dispute"
    assert trace.consensus_rate == 1.0, trace.consensus_rate
    assert trace.nonconsensus_commits == 0
    assert trace.forward_passes == 9, trace.forward_passes  # 3 x 3 rounds
    assert cert["n_disagreements"] == 1
    assert cert["n_rollbacks"] == 1


def test_f4_final_soundness():
    for model, frozen in (_clean(), (ForkModel(), {0: 1, 1: 2})):
        x, trace, _ = ccca_sample(model, 8, MASK, config=_cfg(),
                                  frozen=frozen, policy="ccca")
        assert trace.completed
        check = verify_canvas(x[0].tolist(), MASK, 3)
        assert check["final_no_repeat"], "completed => verifier-clean"


def test_f5_fail_closed():
    model, frozen = _unfixable()
    x, trace, cert = ccca_sample(model, 7, MASK,
                                 config=_cfg(confirm_rounds=3),
                                 frozen=frozen, policy="ccca")
    assert not trace.completed, "repeater must never complete"
    assert cert["completed"] is False
    assert MASK in x[0].tolist(), "masks must remain"
    assert trace.dead_positions_final >= 1, \
        "diagnostic pass must name dead positions"
    assert trace.final_no_repeat or True  # no claim either way


def test_f6_determinism():
    model = ForkModel()
    frozen = {0: 1, 1: 2}
    r1 = ccca_sample(model, 8, MASK, config=_cfg(),
                     frozen=frozen, policy="ccca")
    r2 = ccca_sample(model, 8, MASK, config=_cfg(),
                     frozen=frozen, policy="ccca")
    assert r1[0].tolist() == r2[0].tolist()
    assert r1[1].commit_order == r2[1].commit_order
    assert r1[1].disagreement_order == r2[1].disagreement_order
    assert r1[2] == r2[2]


def test_f7_promotion_honesty():
    model, frozen = _clean()
    cfg = _cfg(chains=1, quorum=2, confirm_rounds=2, max_rounds=16)
    x, trace, cert = ccca_sample(model, 8, MASK, config=cfg,
                                 frozen=frozen, policy="ccca")
    assert trace.completed, "promotion must provide liveness"
    assert trace.consensus_rate == 0.0, trace.consensus_rate
    assert trace.consensus_commits == 0
    assert trace.nonconsensus_commits == 6, trace.nonconsensus_commits
    assert trace.nonconsensus_positions == [2, 3, 4, 5, 6, 7]
    assert cert["nonconsensus_positions"] == [2, 3, 4, 5, 6, 7]
    assert cert["n_nonconsensus_commits"] == 6
    assert x[0].tolist() == [1, 2, 32, 33, 34, 35, 36, 37]


def test_vanilla_baseline():
    model, frozen = _clean()
    x, trace, cert = ccca_sample(model, 8, MASK, config=_cfg(),
                                 frozen=frozen, policy="vanilla")
    assert trace.completed
    assert trace.consensus_rate == 0.0, "vanilla makes no consensus claim"
    assert trace.commits == 6
    assert x[0].tolist() == [1, 2, 32, 33, 34, 35, 36, 37]
