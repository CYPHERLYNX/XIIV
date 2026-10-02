"""Tests for the XIIV persona: deterministic identity answers + chat()."""
import os

import pytest

from xiiv import XIIV
from xiiv.api import (
    IDENTITY_RESPONSE,
    PERSONA,
    is_identity_question,
)

WEIGHTS_10M = os.path.join(os.path.dirname(__file__), "..", "weights",
                           "xiiv.safetensors")


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(WEIGHTS_10M):
        pytest.skip("10M weights not present")
    return XIIV(WEIGHTS_10M)


# --- identity matcher: no model needed -------------------------------------

@pytest.mark.parametrize("q", [
    "who are you",
    "Who are you?",
    "  WHAT ARE YOU  ",
    "who made you",
    "who created you",
    "who developed you",
    "who built you",
    "who trained you",
    "tell me about yourself",
    "introduce yourself",
])
def test_identity_questions_detected(q):
    assert is_identity_question(q)


@pytest.mark.parametrize("q", [
    "what is the capital of France",
    "who are they",          # not asking about the model
    "how are you today",
    "write a poem about masks",
])
def test_non_identity_questions_not_detected(q):
    assert not is_identity_question(q)


def test_identity_response_names_x0d4n():
    assert "X0D4N" in IDENTITY_RESPONSE
    assert "anonymous developer" in IDENTITY_RESPONSE


def test_persona_prompt_is_original_character():
    # The persona steers generation; it must not claim to be another model.
    assert "XIIV" in PERSONA
    for other in ("GPT", "Claude", "Astra", "Opus"):
        assert other not in PERSONA


# --- chat(): needs a loaded model ------------------------------------------

def test_chat_identity_is_deterministic(model):
    r1 = model.chat("who are you", seed=1)
    r2 = model.chat("Who are you?!", seed=99)
    assert r1 == IDENTITY_RESPONSE
    assert r2 == IDENTITY_RESPONSE


def test_chat_non_identity_generates(model):
    out = model.chat("The sky is", length=16, steps=4, seed=7)
    assert isinstance(out, str) and len(out) > 0
    assert out != IDENTITY_RESPONSE
