"""Tests for xiiv.cdr (Certificate-Drift Remasking).

All tests are CPU-only, deterministic (fixed seeds), and run against small
scripted stub models -- no GPU, no real checkpoints.

Pre-registered falsifiers:
- F1 certificate soundness: final canvas has no repeated n-gram (order n)
  on the repeater model, signal="certificate".
- F2 synthesis gate: on the deceptive stub, signal="certificate" repairs
  via drift (drift_remasks > 0, completed, no repeats) while
  signal="confidence" sees no drift (drift_remasks == 0) and fails closed
  (not completed, repeats remain). KILL: if certificate mode does not
  beat confidence mode here, the synthesis claim dies.
- F3 pass budget (structural): certificate -> probe_passes == 0;
  confidence -> probe_passes <= (steps + 3) * k;
  forward_passes == commit_passes + probe_passes + regen_passes.
- F4 TPP order: commits are position-by-own-max-prob; each position keeps
  its own top token.
- KILL: F1 or F2 fails -> the module does not ship.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiiv.cdr import (  # noqa: E402
    CDRConfig,
    cert_margin,
    cdr_certificate,
    cdr_sample,
    find_repeated_ngrams,
)

VOCAB = 32
MASK_ID = 32
V = VOCAB + 1
LENGTH = 16
REPEAT = 5
LIAR = 7
FIX = 8
TOK_X = 7


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
    c = CDRConfig(n=4, budget=4, check_every=1, probe_partitions=2,
                  tau=0.3, conf_floor=0.02, cert_floor=0.5,
                  taboo_cooldown=4, max_steps=64, stall_budget=3)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _repeater_plan(L=LENGTH):
    """Every position: top-1 = REPEAT, second-best = position-unique."""
    return {pos: [(REPEAT, 5.0), (10 + pos, 2.0), (30, 1.0)]
            for pos in range(L)}


def _stable_plan(L=8):
    """Honest model: position-unique tops, high confidence, never flips."""
    return {pos: [(10 + pos, 5.0), (20, 1.0)] for pos in range(L)}


def _liar_scripts():
    """Committed liar: pos 5 = LIAR at high confidence on the commit pass,
    then the model flips against LIAR on every later forward."""
    plan0 = {p: [(10 + p, 5.0), (20, 1.0)] for p in range(8)}
    plan0[5] = [(LIAR, 5.0), (FIX, 2.0), (20, 1.0)]
    plan1 = {p: [(10 + p, 5.0), (20, 1.0)] for p in range(8)}
    plan1[5] = [(FIX, 5.0), (LIAR, -2.0), (20, 1.0)]
    return [_logits_for(plan0, 8), _logits_for(plan1, 8)]


def test_cert_margin_unit():
    # (5,5,5,5) occurs at starts 0,1,2 -> positions 0..5 participate
    x = [REPEAT] * 6 + [9, 1]
    for p in range(6):
        assert cert_margin(p, x, MASK_ID, 4) == 0.0
    # uninvolved committed positions -> 1.0
    assert cert_margin(6, x, MASK_ID, 4) == 1.0
    assert cert_margin(7, x, MASK_ID, 4) == 1.0
    # a single 4-window is NOT a repeat (needs >= 2 occurrences)
    x1 = [REPEAT, REPEAT, REPEAT, REPEAT, 9, 1, 2, 3]
    for p in range(4):
        assert cert_margin(p, x1, MASK_ID, 4) == 1.0
    # masks act as separators: no complete window -> all clean
    x2 = [REPEAT, REPEAT, MASK_ID, REPEAT, REPEAT, 9, 1, 2]
    for p in (0, 1, 3, 4):
        assert cert_margin(p, x2, MASK_ID, 4) == 1.0


def test_t1_confidence_catches_committed_liar():
    """F-port check: signal='confidence' (faithful CoDR) detects the flip
    and repairs the liar position."""
    stub = ScriptedStub(_liar_scripts())
    ids, trace, _cert = cdr_sample(stub, length=8, mask_id=MASK_ID,
                                   config=_cfg(budget=8), seed=0,
                                   policy="cdr", signal="confidence")
    assert trace.completed
    assert trace.final_no_repeat
    assert trace.drift_remasks >= 1
    remask_events = [e for e in trace.events if e[0] == "remask"]
    liar_hits = [e for e in remask_events
                 if e[1]["pos"] == 5 and e[1]["rule"] == "drift"]
    assert liar_hits, "liar position 5 must be drift-remasked"
    assert liar_hits[0][1]["drift"] > 0.5
    assert int(ids[5]) == FIX, "liar token must be corrected on regen"


def test_t1_certificate_sees_no_drift_on_liar():
    """Honest negative control: the liar flip creates no repeat, so the
    certificate signal must NOT remask (no false drift)."""
    stub = ScriptedStub(_liar_scripts())
    ids, trace, _cert = cdr_sample(stub, length=8, mask_id=MASK_ID,
                                   config=_cfg(budget=8), seed=0,
                                   policy="cdr", signal="certificate")
    assert trace.completed
    assert trace.final_no_repeat
    assert trace.remasks == 0, \
        f"certificate signal must not invent drift: {trace.remasks} remasks"


def test_f2_synthesis_gate_deceptive_stub():
    """THE synthesis falsifier. The model stays confident on REPEAT
    forever: confidence drift is blind, certificate-margin drift is not."""
    cfg = _cfg()
    plan = _repeater_plan()

    cert_stub = ScriptedStub([_logits_for(plan)])
    ids_c, t_c, _ = cdr_sample(cert_stub, length=LENGTH, mask_id=MASK_ID,
                               config=cfg, seed=0, policy="cdr",
                               signal="certificate")
    conf_stub = ScriptedStub([_logits_for(plan)])
    ids_f, t_f, _ = cdr_sample(conf_stub, length=LENGTH, mask_id=MASK_ID,
                               config=cfg, seed=0, policy="cdr",
                               signal="confidence")

    # Certificate mode: drift detected, canvas repaired, certified.
    assert t_c.drift_remasks > 0, \
        "F2: certificate signal must detect the rotting proof"
    assert t_c.completed, "F2: certificate mode must complete"
    assert t_c.final_no_repeat, "F2: certificate mode must end repeat-free"
    assert find_repeated_ngrams(ids_c.tolist(), 4, MASK_ID) == {}

    # Confidence mode: no drift visible -> no remask -> fail closed.
    assert t_f.drift_remasks == 0, \
        f"F2: confidence signal must be blind here, saw {t_f.drift_remasks}"
    assert not t_f.completed, \
        "F2: confidence mode must fail closed, not emit a breaching canvas"
    assert find_repeated_ngrams(ids_f.tolist(), 4, MASK_ID) != {}, \
        "F2: the setup must really leave repeats under confidence mode"


def test_f1_certificate_soundness_on_repeater():
    stub = ScriptedStub([_logits_for(_repeater_plan())])
    ids, trace, _cert = cdr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                                   config=_cfg(), seed=0, policy="cdr",
                                   signal="certificate")
    assert trace.completed, "cdr must complete on the repeater model"
    assert trace.final_no_repeat, "F1: repeated n-gram in final canvas"
    assert find_repeated_ngrams(ids.tolist(), 4, MASK_ID) == {}
    assert trace.remasks > 0, "F1 must not pass vacuously (no repair done)"


def test_t3_honest_stable_no_false_remasks():
    """An honest, stable model must pass through CDR untouched."""
    for signal in ("certificate", "confidence"):
        stub = ScriptedStub([_logits_for(_stable_plan())])
        ids, trace, _cert = cdr_sample(stub, length=8, mask_id=MASK_ID,
                                       config=_cfg(budget=8), seed=0,
                                       policy="cdr", signal=signal)
        assert trace.completed, signal
        assert trace.final_no_repeat, signal
        assert trace.remasks == 0, \
            f"{signal}: false remasks on a stable model"
        assert trace.check_cycles >= 1, signal


def test_f3_pass_budget_structural():
    # Certificate signal: verifier reads need no forward passes.
    stub = ScriptedStub([_logits_for(_stable_plan())])
    _, t, _ = cdr_sample(stub, length=8, mask_id=MASK_ID,
                         config=_cfg(budget=8), seed=0, policy="cdr",
                         signal="certificate")
    assert t.probe_passes == 0, \
        "F3: certificate drift must cost zero probe forward passes"
    assert t.forward_passes == t.commit_passes + t.regen_passes, \
        "F3: pass accounting must balance exactly"

    # Confidence signal: k probes per check cycle, bounded.
    stub2 = ScriptedStub([_logits_for(_stable_plan())])
    _, t2, _ = cdr_sample(stub2, length=8, mask_id=MASK_ID,
                          config=_cfg(budget=8), seed=0, policy="cdr",
                          signal="confidence")
    k = 2
    assert t2.probe_passes <= (t2.steps + 3) * k, \
        f"F3: probe budget blown: {t2.probe_passes}"
    assert (t2.forward_passes
            == t2.commit_passes + t2.probe_passes + t2.regen_passes), \
        "F3: pass accounting must balance exactly"


def test_f4_tpp_position_first_selection():
    """Position 0 owns the global top token; every other position owns a
    weaker unique top. TPP commits position 0 first AND lets every other
    position keep its own top (no global-proposal collapse)."""
    plan = {0: [(TOK_X, 5.0), (30, 1.0)]}
    for p in range(1, LENGTH):
        plan[p] = [(10 + p, 4.0), (30, 1.0)]
    stub = ScriptedStub([_logits_for(plan)])
    ids, trace, _cert = cdr_sample(stub, length=LENGTH, mask_id=MASK_ID,
                                   config=_cfg(), seed=0, policy="cdr",
                                   signal="certificate")
    assert trace.completed
    # position-first: the highest own-max position commits first ...
    assert trace.commit_order[0] == 0
    assert int(ids[0]) == TOK_X
    # ... and no other position is forced onto the global token
    assert all(int(ids[p]) != TOK_X for p in range(1, LENGTH))
    # each position kept its own top token
    for p in range(1, LENGTH):
        assert int(ids[p]) == 10 + p
    assert len(set(int(t) for t in ids.tolist())) >= LENGTH - 1


def test_vanilla_policy_baseline_completes():
    stub = ScriptedStub([_logits_for(_stable_plan())])
    ids, trace, _cert = cdr_sample(stub, length=8, mask_id=MASK_ID,
                                   config=_cfg(budget=8), seed=0,
                                   policy="vanilla", signal="certificate")
    assert trace.completed
    assert trace.check_cycles == 0


def test_cdr_certificate_shape():
    stub = ScriptedStub([_logits_for(_stable_plan())])
    ids, trace, cert = cdr_sample(stub, length=8, mask_id=MASK_ID,
                                  config=_cfg(budget=8), seed=0,
                                  policy="cdr", signal="certificate")
    for key in ("certificate", "policy", "signal", "n", "tokens",
                "commit_order", "remasks", "completed", "final_no_repeat",
                "replay_needs_model"):
        assert key in cert, f"missing certificate key: {key}"
    assert cert["certificate"] == "cdr-v1"
    assert cert["replay_needs_model"] is False
    assert len(cert["tokens"]) == 8
    assert cert["final_no_repeat"] is True

    stub2 = ScriptedStub([_logits_for(_stable_plan())])
    _, _, cert2 = cdr_sample(stub2, length=8, mask_id=MASK_ID,
                             config=_cfg(budget=8), seed=0, policy="cdr",
                             signal="confidence")
    assert cert2["replay_needs_model"] is True


def test_unknown_policy_and_signal_rejected():
    stub = ScriptedStub([_logits_for(_stable_plan())])
    try:
        cdr_sample(stub, length=8, mask_id=MASK_ID, policy="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown policy must raise")
    try:
        cdr_sample(stub, length=8, mask_id=MASK_ID, signal="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown signal must raise")
