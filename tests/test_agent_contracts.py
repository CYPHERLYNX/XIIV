"""Tests for agent behavior contracts (application of the guarantee core)."""
import os

import pytest

from xiiv import XIIV
from xiiv.agents import ContractError, compile_contract, generate_plan
from xiiv.tokenizer import XIIVTokenizer

WEIGHTS_10M = os.path.join(os.path.dirname(__file__), "..", "weights",
                           "xiiv.safetensors")
TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "weights_52m",
                         "tokenizer.json")


@pytest.fixture(scope="module")
def tok():
    return XIIVTokenizer(TOKENIZER)


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(WEIGHTS_10M):
        pytest.skip("10M weights not present")
    return XIIV(WEIGHTS_10M)


def test_compile_rejects_multi_token_terms(tok):
    with pytest.raises(ContractError):
        compile_contract(tok, must_do=["verify"])  # 3 tokens
    with pytest.raises(ContractError):
        compile_contract(tok, never_do=["delete"])  # 3 tokens


def test_compile_accepts_single_token_terms(tok):
    _, c = compile_contract(tok, policy="Ask first.",
                            must_do=["ask", "step"], never_do=["cut"])
    assert c["required_ids"] and c["forbidden_ids"]


def test_compile_rejects_contradiction(tok):
    with pytest.raises(ContractError):
        compile_contract(tok, must_do=["ask"], never_do=["ask"])


def test_plan_enforces_contract(model):
    out = generate_plan(model, "Handle the request file.",
                        policy="Ask first. Record each step.",
                        must_do=["ask", "step"], never_do=["cut"],
                        steps=4, seed=7)
    assert out["guaranteed"] is True
    assert out["check"]["required_present"] is True
    assert out["check"]["forbidden_absent"] is True
    assert out["check"]["anchors_intact"] is True
    assert out["plan"].startswith("Ask first. Record each step.")
    low = out["plan"].lower()
    assert "ask" in low and "step" in low and "cut" not in low


def test_plan_baseline_is_unenforced(model):
    out = generate_plan(model, "Handle the request file.",
                        policy="Ask first.", must_do=["ask"],
                        never_do=["cut"], steps=4, seed=7, enforce=False)
    assert out["guaranteed"] is False
    assert isinstance(out["plan"], str)
