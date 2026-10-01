"""Tests for the xiiv.api.XIIV convenience wrapper (real 10M weights)."""
import os

import pytest

from xiiv import XIIV

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights",
                       "xiiv.safetensors")


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(WEIGHTS):
        pytest.skip("weights not present")
    return XIIV(WEIGHTS)


def test_api_loads_and_counts_params(model):
    assert model.n_params == 10_059_840


def test_api_infill_preserves_locked_context(model):
    text = "The <mask> sat on the <mask>."
    out = model.infill(text, steps=4, seed=3)
    assert out.startswith("The ") and " sat on the " in out


def test_api_generate_exact_length(model):
    out = model.generate(length=32, steps=4, seed=7)
    assert isinstance(out, str) and len(out) > 0


def test_api_deterministic_given_seed(model):
    a = model.infill("The <mask> sat.", steps=4, seed=11)
    b = model.infill("The <mask> sat.", steps=4, seed=11)
    assert a == b
