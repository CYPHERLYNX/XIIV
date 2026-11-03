"""Tests for xiiv.ptcs (Planned-Trajectory Certificate Scheduler).

All tests are CPU-only, deterministic (fixed seeds), and run against small
scripted stub models -- no GPU, no real checkpoints.

Pre-registered falsifiers:
- F1 trivial-first bias: on the trivial-dominant stub, PTCS commits every
  informative position before any trivial one; vanilla TPP commits trivial
  first. KILL: PTCS's trivial-first rate >= vanilla's.
- F2 planner dividend: under a tight step budget, PTCS finishes in fewer
  forward passes than fixed-budget vanilla on the same stub.
- F3 settle soundness: on the stable stub, settles > 0, unsets == 0,
  est_query_skips > 0.
- F4 settle defense-in-depth: a repeat injected around a settled position
  (bypassing the gate, unit-testing the un-settle path) MUST un-settle,
  remask, and taboo the offender.
- F5 certificate soundness: repeater stub -> completed, no repeated n-gram.
- KILL: F1 or F5 fails -> the module does not ship.
"""

import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiiv.ptcs import (  # noqa: E402
    PTCSConfig,
    informativeness,
    plan_budget,
    ptcs_certificate,
    ptcs_sample,
    verify_positions,
)
from xiiv.dcmr import find_repeated_ngrams  # noqa: E402

VOCAB = 64
MASK_ID = 64
V = VOCAB + 1
LENGTH = 16
REPEAT = 20
TRIVIAL_IDS = set(range(1, 9))          # 8 distinct "trivial" tokens
INFO_IDS = list(range(10, 18))          # 8 distinct informative tokens

VENV_PY = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), ".venv", "bin", "python")


def _logits_for(plan, L=LENGTH):
    """Build a (L, V) logit matrix from {pos: [(tok, logit), ...]}."""
    base = torch.full((L, V), -1e9)
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


def _cfg(**kw):
    c = PTCSConfig(n=4, budget_max=8, trivial_weight=1.0, commit_bar=0.5,
                   hold_floor=0.05, commit_floor=0.5, settle_margin=1.0,
                   settle_streak=2, verify_every=1, gate_candidates=8,
                   taboo_cooldown=4, max_steps=64, stall_budget=3)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _trivial_dominant_plan(L=LENGTH):
    """Info positions 0-7 (p~0.82); trivial positions 8-15 (p~0.88)."""
    plan = {}
    for i in range(8):
        plan[i] = [(INFO_IDS[i], 2.0), (60, 0.5)]
    for j in range(8):
        plan[8 + j] = [(1 + j, 2.5), (60, 0.5)]
    return plan


def _trivial_dominant_run(policy, **kw):
    cfg = _cfg(**kw)
    stub = ScriptedStub([_logits_for(_trivial_dominant_plan())])
    return ptcs_sample(stub, length=LENGTH, mask_id=MASK_ID, config=cfg,
                       seed=0, policy=policy, trivial_ids=TRIVIAL_IDS)


def _repeater_plan(L=LENGTH):
    """Every position: top-1 = REPEAT, then a distinct fallback."""
    plan = {}
    for i in range(L):
        plan[i] = [(REPEAT, 2.0), (40 + i, 1.0), (30 + (i % 5), 0.0)]
    return plan


def _stable_plan(L=24):
    """All positions: distinct clean tokens, steady confidence."""
    plan = {}
    for i in range(L):
        plan[i] = [(1 + i, 2.0), (60, 0.5)]
    return plan


# --- unit: informativeness -------------------------------------------------

def test_informativeness_discounts_trivial_early_and_decays():
    early = informativeness(0.88, 3, TRIVIAL_IDS, step=1, max_steps=64,
                            trivial_weight=1.0)
    late = informativeness(0.88, 3, TRIVIAL_IDS, step=63, max_steps=64,
                           trivial_weight=1.0)
    info = informativeness(0.82, 10, TRIVIAL_IDS, step=1, max_steps=64,
                           trivial_weight=1.0)
    assert early < info < late, (early, info, late)
    # Non-trivial tokens are never discounted.
    assert informativeness(0.82, 10, TRIVIAL_IDS, 1, 64, 1.0) == 0.82


def test_plan_budget_expand_contract_base():
    cfg = _cfg()
    # Confident cluster -> expand to budget_max.
    assert plan_budget(16, 60, [0.9] * 16, cfg) == 8
    # Nothing worth committing -> contract to 1.
    assert plan_budget(16, 60, [0.01] * 16, cfg) == 1
    # Middling -> base budget finishes on time.
    assert plan_budget(16, 4, [0.3] * 16, cfg) == 4
    assert plan_budget(0, 4, [], cfg) == 0


# --- F1: trivial-first bias --------------------------------------------------

def test_f1_trivial_first_bias_planned_away():
    _, ptcs_trace, _ = _trivial_dominant_run("ptcs")
    _, vanilla_trace, _ = _trivial_dominant_run("vanilla")
    assert ptcs_trace.completed and vanilla_trace.completed
    ptcs_first8 = set(ptcs_trace.commit_order[:8])
    ptcs_last8 = set(ptcs_trace.commit_order[8:])
    vanilla_first8 = set(vanilla_trace.commit_order[:8])
    # PTCS: every informative position commits before any trivial one.
    assert ptcs_first8 == set(range(8)), ptcs_trace.commit_order
    assert ptcs_last8 == set(range(8, 16)), ptcs_trace.commit_order
    # Vanilla TPP: trivial-first (higher raw probability wins).
    assert vanilla_first8 == set(range(8, 16)), vanilla_trace.commit_order
    ptcs_trivial_first_rate = sum(1 for p in ptcs_trace.commit_order[:8]
                                  if p >= 8) / 8
    vanilla_trivial_first_rate = sum(1 for p in vanilla_trace.commit_order[:8]
                                     if p >= 8) / 8
    # KILL condition, phrased positively:
    assert ptcs_trivial_first_rate < vanilla_trivial_first_rate


# --- F2: planner dividend ----------------------------------------------------

def test_f2_planner_uses_fewer_passes_than_fixed_budget():
    _, ptcs_trace, _ = _trivial_dominant_run("ptcs")
    _, vanilla_trace, _ = _trivial_dominant_run("vanilla")
    assert ptcs_trace.completed and vanilla_trace.completed
    assert ptcs_trace.forward_passes < vanilla_trace.forward_passes, (
        ptcs_trace.forward_passes, vanilla_trace.forward_passes)
    # The planner record shows the expansion then the contraction.
    ks = [s["k"] for s in ptcs_trace.plan]
    assert ks[0] == 8 and all(k == 1 for k in ks[1:]), ks


# --- F3: settle soundness -----------------------------------------------------

def test_f3_settle_soundness_on_stable_stub():
    cfg = _cfg(settle_streak=2, verify_every=1)
    stub = ScriptedStub([_logits_for(_stable_plan(), L=24)])
    _, trace, _ = ptcs_sample(stub, length=24, mask_id=MASK_ID, config=cfg,
                              seed=0, policy="ptcs")
    assert trace.completed and trace.final_no_repeat
    assert trace.settles > 0, trace.to_dict()
    assert trace.unsets == 0, trace.to_dict()
    assert trace.est_query_skips > 0, trace.to_dict()
    assert len(trace.settled_final) == trace.settles


# --- F4: settle defense-in-depth (unit injection) -------------------------------

def test_f4_injected_rot_unsettles_and_remasks():
    cfg = _cfg()
    L = 8
    x_list = [11, 12, 13, 14, MASK_ID, MASK_ID, MASK_ID, MASK_ID]
    settled = {0, 1, 2, 3}
    streak = {0: 1, 1: 1, 2: 1, 3: 1}
    taboo = {}
    # Bypass the gate: inject a repeat around the settled positions,
    # simulating a non-local certificate break the commit gate missed.
    x_list[4:8] = [11, 12, 13, 14]
    committed = list(range(8))
    n_settled, n_unset, remasked = verify_positions(
        x_list, committed, frozenset(), settled, streak, taboo,
        MASK_ID, cfg, trace=None)
    # Minimal repair is the correct semantics: remasking position 0 breaks
    # window [0..3], which dissolves the only repeat -- positions 1-3 then
    # legitimately re-verify clean and stay settled. Over-unsettling all
    # four would discard proof the verifier still endorses.
    assert n_settled == 0
    assert n_unset == 1, (n_unset, remasked)
    assert remasked == [0]
    assert x_list[0] == MASK_ID
    assert settled == {1, 2, 3}
    # The offender is tabooed with the configured cooldown.
    assert taboo[0][11] == cfg.taboo_cooldown
    # The canvas no longer contains a repeat: the unsettled position is a
    # mask now, and the injected window stands alone (single occurrence).
    assert find_repeated_ngrams(x_list, cfg.n, MASK_ID) == {}


# --- F5: certificate soundness --------------------------------------------------

def test_f5_certificate_soundness_on_repeater():
    stub = ScriptedStub([_logits_for(_repeater_plan())])
    ids, trace, _ = ptcs_sample(stub, length=LENGTH, mask_id=MASK_ID,
                                config=_cfg(), seed=0, policy="ptcs")
    assert trace.completed, trace.to_dict()
    assert trace.final_no_repeat, trace.to_dict()
    assert int((ids == MASK_ID).sum()) == 0
    assert not find_repeated_ngrams(ids.tolist(), 4, MASK_ID)
    # The gate did real work on this stub.
    assert trace.gate_rejects > 0


# --- structural: pass accounting -----------------------------------------------

def test_pass_accounting():
    _, trace, _ = _trivial_dominant_run("ptcs")
    assert trace.forward_passes == trace.commit_passes + trace.regen_passes
    # Verifier rounds read the canvas directly: zero model passes. One
    # round per main-loop step plus the final verification round.
    assert trace.verify_rounds == trace.steps + 1 > 0


# --- certificate replay ----------------------------------------------------------

def test_certificate_replays_model_free():
    stub = ScriptedStub([_logits_for(_repeater_plan())])
    ids, trace, cert = ptcs_sample(stub, length=LENGTH, mask_id=MASK_ID,
                                   config=_cfg(), seed=0, policy="ptcs")
    assert cert["certificate"] == "ptcs-v1"
    assert cert["final_no_repeat"] == (
        not find_repeated_ngrams(cert["tokens"], 4, MASK_ID))
    assert cert["tokens"] == [int(t) for t in ids.tolist()]
    assert len(cert["commit_order"]) == trace.commits
    assert cert["n_settles"] == trace.settles
    assert cert["est_query_skips"] == trace.est_query_skips
    assert len(cert["plan"]) == trace.steps


# --- fail-closed: preexisting repeat is exempt, not hidden --------------------------

def test_preexisting_repeat_exempt_not_failed_closed():
    frozen = {i: t for i, t in
              enumerate([11, 12, 13, 14, 11, 12, 13, 14])}
    plan = {8 + i: [(30 + i, 2.0), (60, 0.5)] for i in range(8)}
    stub = ScriptedStub([_logits_for(plan)])
    _, trace, _ = ptcs_sample(stub, length=LENGTH, mask_id=MASK_ID,
                              config=_cfg(), seed=0, policy="ptcs",
                              frozen=frozen)
    assert trace.preexisting_repeat
    assert trace.completed  # exempt from fail-closed...
    assert not trace.final_no_repeat  # ...but honestly reported


# --- trivial tokens are never stranded ----------------------------------------------

def test_trivial_tokens_eventually_commit():
    ids, trace, _ = _trivial_dominant_run("ptcs")
    assert trace.completed
    assert int((ids == MASK_ID).sum()) == 0
    assert len(trace.commit_order) == LENGTH


# --- determinism ---------------------------------------------------------------------

def test_determinism():
    r1 = _trivial_dominant_run("ptcs")
    r2 = _trivial_dominant_run("ptcs")
    assert r1[0].tolist() == r2[0].tolist()
    assert r1[1].commit_order == r2[1].commit_order


# --- CLI wiring ------------------------------------------------------------------------

def test_cli_ptcs_flag_wired():
    # --help path: no model load, just the parser.
    out = subprocess.run(
        [VENV_PY, "-m", "xiiv.sample", "generate", "--help"],
        capture_output=True, text=True, cwd=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."),
        timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert "--ptcs" in out.stdout
    # Mutual exclusion with the other samplers, before any model load.
    out2 = subprocess.run(
        [VENV_PY, "-m", "xiiv.sample", "generate", "--length", "8",
         "--ptcs", "--cdr"],
        capture_output=True, text=True, cwd=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."),
        timeout=120)
    assert out2.returncode != 0
    assert "alternative samplers" in out2.stderr
