"""Tests for the XIIV knowledge layer (retrieval mocked; no network)."""
import os

import pytest

from xiiv import XIIV
from xiiv.api import IDENTITY_RESPONSE
from xiiv.knowledge import pipeline
from xiiv.knowledge.news import detect_dates
from xiiv.tokenizer import XIIVTokenizer

WEIGHTS_10M = os.path.join(os.path.dirname(__file__), "..", "weights",
                           "xiiv.safetensors")
TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "weights_52m",
                         "tokenizer.json")

FAKE_FACTS = [
    {"text": "Paris is the capital of France. The Eiffel Tower stands in Paris.",
     "source": "Wikipedia: Paris"},
]


@pytest.fixture(scope="module")
def tok():
    return XIIVTokenizer(TOKENIZER)


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(WEIGHTS_10M):
        pytest.skip("10M weights not present")
    return XIIV(WEIGHTS_10M)


# --- key term extraction: no model needed ----------------------------------

def test_key_terms_keep_entities_years_content_words(tok):
    terms = pipeline.key_terms(
        ["The Eiffel Tower was completed in 1889 in Paris."], tok)
    joined = " ".join(tok.decode(t) for t in terms)
    assert "Paris" in joined
    assert "1889" in joined
    assert "Eiffel" in joined or "Tower" in joined
    for t in terms:
        assert 1 <= len(t) <= 4


def test_key_terms_single_token_mode(tok):
    terms = pipeline.key_terms(
        ["The Eiffel Tower was completed in 1889 in Paris."], tok,
        single_token=True)
    for t in terms:
        assert len(t) == 1


def test_key_terms_drop_stopwords_everywhere(tok):
    terms = pipeline.key_terms(["The cat sat on the mat."], tok)
    for t in terms:
        assert tok.decode(t).lower() not in pipeline._STOPWORDS


def test_key_terms_deterministic(tok):
    f = ["Paris is the capital of France in 2024."]
    assert pipeline.key_terms(f, tok) == pipeline.key_terms(f, tok)


# --- date detection: pure function ------------------------------------------

def test_detect_dates_year_month():
    assert (2023, 3, 15) in detect_dates("What happened on March 15 2023?")


def test_detect_dates_no_date():
    assert detect_dates("What is the capital of France?") == []


# --- answer(): retrieval mocked ---------------------------------------------

def test_ask_identity_needs_no_facts():
    out = pipeline.answer("who are you", None)
    assert out["answer"] == IDENTITY_RESPONSE
    assert out["guaranteed"] is True


def test_ask_no_facts_says_so(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda q, **k: [])
    out = pipeline.answer("xqzvww unknowable thing", None)
    assert "couldn't find verified information" in out["answer"]
    assert out["facts"] == []


def test_ask_enforces_key_terms(model, monkeypatch, tok):
    monkeypatch.setattr(pipeline, "retrieve",
                        lambda q, **k: list(FAKE_FACTS))
    out = pipeline.answer("What is the capital of France?", model,
                          steps=4, seed=7)
    assert out["guaranteed"] is True
    assert out["sources"] == ["Wikipedia: Paris"]
    # the top fact is anchored verbatim at the answer start
    assert out["answer"].startswith("Paris is the capital of France.")
    gen = out["answer"].lower()
    # every required single-token term appears in the generated answer
    for term_ids in out["key_terms"]:
        assert len(term_ids) == 1
        assert tok.decode(term_ids).lower() in gen, \
            f"required term {tok.decode(term_ids)!r} missing from answer"


def test_ask_baseline_is_prompt_only(model, monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve",
                        lambda q, **k: list(FAKE_FACTS))
    out = pipeline.answer("What is the capital of France?", model,
                          steps=4, seed=7, enforce=False)
    assert out["guaranteed"] is False
    assert isinstance(out["answer"], str) and len(out["answer"]) > 0


def test_ask_reports_sources(model, monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve",
                        lambda q, **k: list(FAKE_FACTS))
    out = pipeline.answer("Capital of France?", model, steps=4, seed=7)
    assert len(out["facts"]) == 1
    assert "Wikipedia" in out["sources"][0]
