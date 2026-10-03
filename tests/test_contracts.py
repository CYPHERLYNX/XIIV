"""Tests for xiiv.contracts.Contract — the unified guarantee path."""
import os

import pytest

from xiiv.contracts import Contract, ContractError
from xiiv.tokenizer import XIIVTokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "weights_52m",
                         "tokenizer.json")


@pytest.fixture(scope="module")
def tok():
    return XIIVTokenizer(TOKENIZER)


def test_fluent_builder(tok):
    c = (Contract()
         .lock("Ask first.")
         .require("ask", "step")
         .forbid("cut")
         .exact_length(64))
    constraints, total = c.compile(tok, prompt_len=10)
    assert total == 64
    assert constraints["required_ids"]
    assert constraints["forbidden_ids"]
    # policy anchored at the first generated position
    assert constraints["anchors"][10] == tok.encode("Ask")[0]


def test_lock_is_byte_exact(tok):
    c = Contract().lock("Do not change this.")
    constraints, _ = c.compile(tok, prompt_len=5)
    text = "Do not change this."
    ids = [i for i in tok.encode(text)
           if i not in (tok.mask_id, tok.eos_id)]
    for p, t in enumerate(ids):
        assert constraints["anchors"][5 + p] == t


def test_multi_token_term_raises(tok):
    with pytest.raises(ContractError):
        Contract().require("verify").compile(tok, prompt_len=0)


def test_contradiction_raises(tok):
    with pytest.raises(ContractError):
        Contract().require("ask").forbid("ask").compile(tok, prompt_len=0)


def test_policy_too_long_raises(tok):
    with pytest.raises(ContractError):
        Contract().lock("word " * 100).compile(tok, prompt_len=120)


def test_default_length(tok):
    _, total = Contract().require("ask").compile(tok, prompt_len=0)
    assert total == 128


def test_verify_and_holds(tok):
    import torch
    c = Contract().lock("Hi.").require("ask").forbid("cut")
    constraints, total = c.compile(tok, prompt_len=4)
    # build a canvas satisfying everything: prompt + anchored policy +
    # required token present, forbidden absent
    ids = torch.full((total,), tok.encode("the")[0])
    for p, t in constraints["anchors"].items():
        ids[p] = t
    ids[50] = constraints["required_ids"][0]
    check = Contract.verify(ids, constraints, tok.mask_id)
    assert Contract.holds(check)
    # break it: remove the required token
    ids[50] = tok.encode("the")[0]
    if ids[50] == constraints["required_ids"][0]:
        ids[50] = tok.encode("and")[0]
    check = Contract.verify(ids, constraints, tok.mask_id)
    assert not Contract.holds(check)
