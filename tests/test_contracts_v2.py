"""Tests for Contract v2: phrases, ordering, cardinality, conflict detection.

The guarantee tests use ADVERSARIAL dummy models (not the trained weights)
on purpose: if the structural guarantees hold when the model is actively
trying to break them, they hold by construction, not by luck.
"""
import os

import pytest
import torch

from xiiv.contracts import Contract, ContractError
from xiiv.diffusion import DenoisingState, check_constraints
from xiiv.tokenizer import XIIVTokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "weights_52m",
                         "tokenizer.json")
VOCAB = 8192
MASK_ID = 2
EOS_ID = 3


@pytest.fixture(scope="module")
def tok():
    return XIIVTokenizer(TOKENIZER)


class RandomModel(torch.nn.Module):
    """Fresh random logits every call — no learned bias to lean on."""

    def __init__(self, seed=0):
        super().__init__()
        self.g = torch.Generator().manual_seed(seed)

    def forward(self, x):
        L = x.shape[1]
        return torch.randn(1, L, VOCAB, generator=self.g) * 3.0


class AdversarialModel(torch.nn.Module):
    """Puts almost all mass on `push_ids` at every position: actively
    tries to force the forbidden phrase / blow past at_most caps."""

    def __init__(self, push_ids):
        super().__init__()
        self.push_ids = list(push_ids)

    def forward(self, x):
        L = x.shape[1]
        logits = torch.full((1, L, VOCAB), -5.0)
        for t in self.push_ids:
            logits[:, :, t] = 10.0
        return logits


def run(constraints, model, length=64, steps=8, seed=0, prompt_len=8):
    prompt = torch.randint(100, 1000, (prompt_len,))
    st = DenoisingState(length=length, mask_id=MASK_ID, eos_id=EOS_ID,
                        steps=steps, seed=seed, prompt_ids=prompt,
                        constraints=constraints, vocab_size=VOCAB,
                        device="cpu")
    while st.step(model):
        pass
    return st.x


def clean(tok, text):
    return [i for i in tok.encode(text) if i not in (MASK_ID, EOS_ID)]


# -- compiler ----------------------------------------------------------

def test_phrase_compiles_to_anchors(tok):
    c = Contract().require_phrase("hello world").exact_length(64)
    cons, total = c.compile(tok, prompt_len=8)
    ids = clean(tok, "hello world")
    assert len(ids) > 1
    # the phrase block is pinned contiguously somewhere after the prompt
    starts = [p for p in cons["anchors"]
              if all(cons["anchors"].get(p + j) == t
                     for j, t in enumerate(ids))]
    assert starts, "phrase block not found in anchors"
    assert all(p >= 8 for p in starts)


def test_before_orders_blocks(tok):
    c = (Contract()
         .require_phrase("hello world")
         .require_phrase("goodbye world")
         .before("hello world", "goodbye world")
         .exact_length(64))
    cons, _ = c.compile(tok, prompt_len=8)
    assert len(cons["order_pairs"]) == 1
    a_ids = clean(tok, "hello world")
    # a's block starts before b's block
    a_pos = min(p for p, t in cons["anchors"].items() if t == a_ids[0]
                and all(cons["anchors"].get(p + j) == x
                        for j, x in enumerate(a_ids)))
    b_ids = clean(tok, "goodbye world")
    b_pos = min(p for p, t in cons["anchors"].items() if t == b_ids[0]
                and all(cons["anchors"].get(p + j) == x
                        for j, x in enumerate(b_ids)))
    assert a_pos < b_pos


def test_before_promotes_to_required(tok):
    c = Contract().before("alpha beta", "gamma delta").exact_length(64)
    cons, _ = c.compile(tok, prompt_len=8)
    # both phrases became required (anchored) even though require_phrase
    # was never called
    assert len(cons["anchors"]) >= 4


def test_cycle_raises(tok):
    c = (Contract()
         .require_phrase("alpha beta")
         .require_phrase("gamma delta")
         .before("alpha beta", "gamma delta")
         .before("gamma delta", "alpha beta"))
    with pytest.raises(ContractError):
        c.compile(tok, prompt_len=8)


def test_count_conflict_raises(tok):
    c = Contract().at_least("the", 3).at_most("the", 1)
    with pytest.raises(ContractError):
        c.compile(tok, prompt_len=8)


def test_forbidden_subsequence_of_required_raises(tok):
    # "world" is inside required phrase "hello world"
    c = (Contract()
         .require_phrase("hello world")
         .forbid_phrase("hello"))
    with pytest.raises(ContractError):
        c.compile(tok, prompt_len=8)


def test_phrase_pinned_position_overlap_raises(tok):
    c = (Contract()
         .lock("hello world")
         .require_phrase("hello world", at=8))
    with pytest.raises(ContractError):
        c.compile(tok, prompt_len=8)


# -- sampler guarantees vs adversarial models ---------------------------

def test_forbid_phrase_holds_vs_adversary(tok):
    phrase = "hello world"
    pids = clean(tok, phrase)
    assert len(pids) > 1
    c = Contract().forbid_phrase(phrase).exact_length(64)
    cons, _ = c.compile(tok, prompt_len=8)
    model = AdversarialModel(pids)  # tries to emit the phrase everywhere
    for seed in range(3):
        x = run(cons, model, seed=seed)
        check = check_constraints(x, cons, MASK_ID)
        assert check["phrases_ok"], f"forbidden phrase formed (seed {seed})"
        assert check["complete"]
        assert Contract.holds(check)


def test_at_most_holds_vs_adversary(tok):
    tid = clean(tok, "the")[0]
    c = Contract().at_most("the", 2).exact_length(64)
    cons, _ = c.compile(tok, prompt_len=8)
    model = AdversarialModel([tid])  # tries to emit "the" everywhere
    for seed in range(3):
        x = run(cons, model, seed=seed)
        check = check_constraints(x, cons, MASK_ID)
        assert check["caps_met"], f"cap breached (seed {seed})"
        assert x.tolist().count(tid) <= 2


def test_at_least_counts_met(tok):
    tid = clean(tok, "the")[0]
    c = Contract().at_least("the", 3).exact_length(64)
    cons, _ = c.compile(tok, prompt_len=8)
    assert cons["required_counts"][tid] == 3
    model = RandomModel(seed=1)
    for seed in range(3):
        x = run(cons, model, seed=seed + 10)
        check = check_constraints(x, cons, MASK_ID)
        assert check["counts_met"], f"count not met (seed {seed})"
        assert x.tolist().count(tid) >= 3
        assert Contract.holds(check)


def test_phrase_require_and_order_end_to_end(tok):
    c = (Contract()
         .require_phrase("hello world")
         .require_phrase("goodbye world")
         .before("hello world", "goodbye world")
         .at_least("the", 2)
         .forbid("cut")  # single-token forbid alongside v2 features
         .exact_length(64))
    cons, _ = c.compile(tok, prompt_len=8)
    model = RandomModel(seed=7)
    x = run(cons, model, seed=42)
    check = check_constraints(x, cons, MASK_ID)
    assert check["anchors_intact"]
    assert check["counts_met"]
    assert check["order_ok"]
    assert check["complete"]
    assert Contract.holds(check)
    xl = x.tolist()
    # phrases present contiguously, in order
    a = clean(tok, "hello world")
    b = clean(tok, "goodbye world")
    ia = next(i for i in range(len(xl)) if xl[i:i + len(a)] == a)
    ib = next(i for i in range(len(xl)) if xl[i:i + len(b)] == b)
    assert ia + len(a) <= ib


def test_phrase_at_most_allows_limited_occurrences(tok):
    phrase = "hello world"
    pids = clean(tok, phrase)
    c = Contract().at_most(phrase, 1).exact_length(64)
    cons, _ = c.compile(tok, prompt_len=8)
    model = AdversarialModel(pids)
    for seed in range(3):
        x = run(cons, model, seed=seed)
        check = check_constraints(x, cons, MASK_ID)
        assert check["phrases_ok"], f"phrase exceeded cap (seed {seed})"


def test_v1_api_unchanged(tok):
    # old-style contract still compiles and runs identically
    c = (Contract()
         .lock("Ask first.")
         .require("ask", "step")
         .forbid("cut")
         .exact_length(64))
    cons, total = c.compile(tok, prompt_len=10)
    assert total == 64
    assert "required_counts" in cons  # new key present, old keys intact
    assert cons["required_ids"]
    model = RandomModel(seed=3)
    x = run(cons, model, seed=5, prompt_len=10)
    check = check_constraints(x, cons, MASK_ID)
    assert Contract.holds(check)
