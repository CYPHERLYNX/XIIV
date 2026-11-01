"""Tests for xiiv.cgec + xiiv.cert_verify (Certificate-Guided Early Commit).

All tests are CPU-only, deterministic (fixed seeds), and run against
small scripted stub models -- no GPU, no real checkpoints.

Pre-registered falsifiers:
- F1 certificate soundness: scripted valid-expression model -> completed,
  final canvas parses as a complete expression, independent verifier says
  valid.
- F2 gate: scripted saboteur (top-1 always grammar-invalid) -> still
  completes via lower-ranked candidates, gate_rejects > 0, final canvas
  parses complete.
- F3 early commit: confident scripted model -> early_commits > 0 and
  forward_passes strictly below the vanilla policy on the same plan.
- F4 verifier: tampered certificates -> verify returns valid=False with
  a named failure; untampered -> valid.
- F5 grammar unit: PDA accepts/rejects a fixed battery of expressions.
- KILL: F1 or F4 fails -> the module does not ship.
"""

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiiv.cgec import (  # noqa: E402
    CGECConfig,
    arithmetic_alphabet_32k,
    cgec_sample,
    is_feasible,
)
from xiiv.cert_verify import verify_certificate  # noqa: E402

# Synthetic domain: token ids -> arith-v1 symbol classes.
ALPHABET = {
    1: "digit", 2: "plus", 3: "digit", 4: "mul", 5: "lparen",
    6: "rparen", 7: "digit", 8: "minus", 9: "div",
}
VOCAB = 12
MASK_ID = 11
V = VOCAB + 1
LENGTH = 7  # odd: arith-v1 requirement

# "(1+2)*3" -> [lparen, digit, plus, digit, rparen, mul, digit]
TARGET = [5, 1, 2, 7, 6, 4, 3]


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


def _cfg(**kw):
    c = CGECConfig(block_size=32, tau=0.95, gamma=0.10, topk=4,
                   max_steps=128, stall_budget=3)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _run(policy, plan, **kw):
    stub = ScriptedStub([_logits_for(plan)])
    return cgec_sample(stub, length=LENGTH, mask_id=MASK_ID,
                       alphabet=ALPHABET, config=_cfg(**kw), seed=0,
                       policy=policy)


def _valid_plan():
    return {p: [(t, 5.0)] for p, t in enumerate(TARGET)}


# ---------------------------------------------------------------- F5: PDA

# Token-id aliases for the synthetic alphabet.
_D, _PLUS, _MUL, _LP, _RP, _MINUS, _DIV = 1, 2, 4, 5, 6, 8, 9


def test_f5_grammar_battery():
    D, P, M, LP, RP, PL, MI, DV = _D, _PLUS, _MUL, _LP, _RP, _MINUS, 9, 3  # ids
    valid = [
        [D],                    # "3"
        [D, 2, D],              # "3+4"
        [LP, D, RP],            # "(3)"
        [5, 1, 2, 7, 6, 4, 3],  # "(1+2)*3"
        [D, 2, D, 4, D, 8, D, 9, D],  # "3+4*5-6/7"
        [LP, LP, D, RP, 2, D, RP],    # "((3)+4)"
    ]
    invalid = [
        [],
        [D, 2],            # "3+"
        [2, D],            # "+3"
        [D, 2, 2, D],      # "3++4"
        [LP, D],           # "(3"
        [D, RP],           # "3)"
        [LP, RP],          # "()"
        [D, D],            # "34"
        [D, 2, D, 2],      # "3+4+"
        [2, 2, 2, 2, 2, 2, 2],  # all operators
        [RP, D],           # ")3"
    ]
    for seq in valid:
        assert is_feasible(seq, ALPHABET, MASK_ID), f"should accept {seq}"
    for seq in invalid:
        assert not is_feasible(seq, ALPHABET, MASK_ID), \
            f"should reject {seq}"


def test_f5_masks_are_wildcards():
    # "[MASK]([MASK]+[MASK])" is completable; ")..." is not.
    assert is_feasible([MASK_ID, _LP, MASK_ID, _PLUS, MASK_ID, _RP, MASK_ID],
                       ALPHABET, MASK_ID)
    assert not is_feasible([_RP, MASK_ID, MASK_ID], ALPHABET, MASK_ID)


# ---------------------------------------------------------------- F1: soundness

def test_f1_certificate_soundness():
    ids, trace, cert = _run("cgec", _valid_plan())
    assert trace.completed, "cgec must complete on the valid plan"
    assert trace.grammar_valid
    assert ids.tolist() == TARGET
    assert cert["completed"] and cert["grammar_valid"]
    verdict = verify_certificate(cert)
    assert verdict["valid"], f"verifier rejected a good cert: {verdict}"
    assert trace.min_gap is not None and trace.min_gap > 0.99


# ---------------------------------------------------------------- F2: gate

def _saboteur_plan():
    """Top-1 is always grammar-invalid; second-best completes "(1+2)*3"."""
    return {
        0: [(6, 5.0), (5, 4.0)],  # ')' invalid first; '(' ok
        1: [(2, 5.0), (1, 4.0)],  # '+' after '(' invalid; digit ok
        2: [(1, 5.0), (2, 4.0)],  # digit after digit invalid; '+' ok
        3: [(2, 5.0), (7, 4.0)],  # '+' after '+' invalid; digit ok
        4: [(1, 5.0), (6, 4.0)],  # digit after digit invalid; ')' ok
        5: [(6, 5.0), (4, 4.0)],  # ')' at depth 0 invalid; '*' ok
        6: [(2, 5.0), (3, 4.0)],  # '+' after '*' invalid; digit ok
    }


def test_f2_gate_defeats_saboteur():
    ids, trace, cert = _run("cgec", _saboteur_plan())
    assert trace.completed, "gate must route around the saboteur"
    assert trace.gate_rejects >= 6, \
        f"expected >=6 gate rejects, got {trace.gate_rejects}"
    assert ids.tolist() == TARGET, f"got {ids.tolist()}"
    assert trace.grammar_valid
    verdict = verify_certificate(cert)
    assert verdict["valid"], f"verifier rejected: {verdict}"


# ---------------------------------------------------------------- F3: early commit

def _confident_plan():
    # top-1 prob ~0.88 (< tau 0.95, so no parallel fill), gap ~0.76 (>= gamma)
    plan = {}
    for p, t in enumerate(TARGET):
        other = 1 if t != 1 else 2
        plan[p] = [(t, 4.0), (other, 2.0)]
    return plan


def test_f3_early_commit_saves_passes():
    ids_c, trace_c, cert_c = _run("cgec", _confident_plan())
    ids_v, trace_v, _cert_v = _run("vanilla", _confident_plan())
    assert trace_c.completed and trace_v.completed
    assert trace_c.early_commits > 0, "all-in should have fired"
    assert trace_c.forward_passes < trace_v.forward_passes, (
        f"cgec {trace_c.forward_passes} vs vanilla {trace_v.forward_passes}")
    assert trace_v.forward_passes == LENGTH  # one commit per step
    assert ids_c.tolist() == ids_v.tolist() == TARGET
    assert verify_certificate(cert_c)["valid"]


def test_vanilla_baseline_is_uncertified():
    # Sanity: the vanilla policy really is the dumb baseline (no gate).
    ids, trace, cert = _run("vanilla", _saboteur_plan())
    assert trace.completed
    assert ids.tolist() != TARGET  # vanilla eats the saboteur's top-1s


# ---------------------------------------------------------------- F4: verifier

def test_f4_verifier_accepts_good_cert():
    _, _, cert = _run("cgec", _valid_plan())
    verdict = verify_certificate(cert)
    assert verdict == {"valid": True, "failures": []}


def test_f4_verifier_rejects_token_tamper():
    _, _, cert = _run("cgec", _valid_plan())
    bad = copy.deepcopy(cert)
    bad["tokens"][1], bad["tokens"][2] = bad["tokens"][2], bad["tokens"][1]
    verdict = verify_certificate(bad)
    assert not verdict["valid"]
    assert any("parse" in f or "prefix" in f for f in verdict["failures"])


def test_f4_verifier_rejects_symbol_tamper():
    _, _, cert = _run("cgec", _valid_plan())
    bad = copy.deepcopy(cert)
    bad["commits"][1]["symbol"] = "plus"  # was "digit"
    verdict = verify_certificate(bad)
    assert not verdict["valid"]
    assert any("mismatch" in f for f in verdict["failures"])


def test_f4_verifier_rejects_gap_tamper():
    _, _, cert = _run("cgec", _valid_plan())
    bad = copy.deepcopy(cert)
    bad["min_gap"] = 0.0
    verdict = verify_certificate(bad)
    assert not verdict["valid"]
    assert any("min_gap" in f for f in verdict["failures"])


def test_f4_verifier_rejects_dropped_commit():
    _, _, cert = _run("cgec", _valid_plan())
    bad = copy.deepcopy(cert)
    bad["commits"] = bad["commits"][:-1]
    verdict = verify_certificate(bad)
    assert not verdict["valid"]
    assert any("n_commits" in f for f in verdict["failures"])


# ---------------------------------------------------------------- guards

def test_even_length_fail_closed():
    stub = ScriptedStub([torch.full((6, V), -1e9)])
    ids, trace, cert = cgec_sample(stub, length=6, mask_id=MASK_ID,
                                   alphabet=ALPHABET, config=_cfg(), seed=0)
    assert not trace.completed
    assert not cert["completed"]
    assert any(e[0] == "bad_length" for e in trace.events)


def test_frozen_outside_alphabet_rejected():
    stub = ScriptedStub([_logits_for(_valid_plan())])
    with pytest.raises(ValueError, match="outside the .* alphabet"):
        cgec_sample(stub, length=LENGTH, mask_id=MASK_ID,
                    alphabet=ALPHABET, config=_cfg(), seed=0,
                    frozen={0: 99})


def test_infeasible_frozen_fail_closed():
    # Frozen ')' at position 0: no completion exists -> fail fast.
    stub = ScriptedStub([_logits_for(_valid_plan())])
    ids, trace, cert = cgec_sample(stub, length=LENGTH, mask_id=MASK_ID,
                                   alphabet=ALPHABET, config=_cfg(), seed=0,
                                   frozen={0: 6})
    assert not trace.completed
    assert any(e[0] == "infeasible_start" for e in trace.events)


TOKENIZER = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "weights", "tokenizer_32k.json")


@pytest.mark.skipif(not os.path.exists(TOKENIZER),
                    reason="weights/tokenizer_32k.json not present")
def test_real_tokenizer_alphabet():
    """The 32k-tokenizer alphabet helper: measured mapping, '*' absent."""
    alpha = arithmetic_alphabet_32k(TOKENIZER)
    assert len(alpha) == 16  # 10 digits + 6 operator/paren keys
    classes = set(alpha.values())
    assert classes == {"digit", "plus", "minus", "mul", "div",
                       "lparen", "rparen"}
    # 'x' (id 79) is the multiply key: '*' has no token in this vocab.
    assert alpha[79] == "mul"
