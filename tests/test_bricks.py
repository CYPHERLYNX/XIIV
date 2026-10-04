"""Tests for Guarantee Bricks: must-include brick + proof certificates.

The enforcement tests use ADVERSARIAL dummy models on purpose: if the
guarantee holds when the model actively avoids the phrase, it holds by
construction, not by luck. The brick is tested WITHOUT the monolithic
constraints dict -- the DenoisingState gets no constraints, so the brick
alone is the enforcer.

The verifier tests attack the certificate: tampered positions, removed
phrases, replayed contracts, broken chains. Verification is fail-closed.
"""
import copy
import os

import pytest
import torch

from xiiv.bricks import MustIncludeBrick, BRICKS
from xiiv.bricks import certificate as certmod
from xiiv.bricks.must_include import UnsatisfiableBrick
from xiiv.bricks.verify import verify_certificate
from xiiv.diffusion import DenoisingState
from xiiv.tokenizer import XIIVTokenizer

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "weights_52m",
                         "tokenizer.json")
VOCAB = 8192
MASK_ID = 2
EOS_ID = 3


@pytest.fixture(scope="module")
def tok():
    return XIIVTokenizer(TOKENIZER)


def clean_ids(tok, text):
    return [i for i in tok.encode(text) if i not in (MASK_ID, EOS_ID)]


class PhraseSuppressor(torch.nn.Module):
    """Puts ~zero mass on suppress_ids: the model never emits the phrase
    on its own. If the phrase appears in the output, only the brick put
    it there."""

    def __init__(self, suppress_ids, seed=0):
        super().__init__()
        self.suppress = list(suppress_ids)
        self.g = torch.Generator().manual_seed(seed)

    def forward(self, x):
        L = x.shape[1]
        logits = torch.randn(1, L, VOCAB, generator=self.g)
        for t in self.suppress:
            logits[:, :, t] = -1e9
        return logits


def run_with_brick(brick, model, length=64, steps=8, seed=0, prompt_len=8):
    """Denoise with NO monolithic constraints; the brick is the enforcer."""
    prompt = torch.randint(100, 1000, (prompt_len,))
    st = DenoisingState(length=length, mask_id=MASK_ID, eos_id=EOS_ID,
                        steps=steps, seed=seed, prompt_ids=prompt,
                        constraints=None, vocab_size=VOCAB, device="cpu")
    brick.allocate(length, occupied=set(), prompt_len=prompt_len)
    brick.enforce(st)
    while st.step(model):
        pass
    return st.x


def has_phrase(ids_list, phrase_ids):
    n = len(phrase_ids)
    return any(ids_list[i:i + n] == phrase_ids
               for i in range(len(ids_list) - n + 1))


# -- brick enforcement ---------------------------------------------------

def test_brick_registered(tok):
    assert BRICKS["must-include"] is MustIncludeBrick


def test_brick_enforces_phrase_vs_adversary(tok):
    phrase = "quantum leap"
    pids = clean_ids(tok, phrase)
    assert len(pids) > 1
    brick = MustIncludeBrick(phrase, pids)
    model = PhraseSuppressor(pids, seed=1)
    for seed in range(3):
        x = run_with_brick(brick, model, seed=seed)
        ids = x.tolist()
        assert has_phrase(ids, pids), \
            f"phrase missing despite brick (seed {seed})"
        # the model never predicts the phrase: pinned positions are the
        # only place its tokens can appear
        for pos, tid in zip(brick.positions, pids):
            assert ids[pos] == tid


def test_brick_allocates_contiguous_block(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    positions = brick.allocate(64, occupied=set(), prompt_len=8)
    assert positions == list(range(positions[0],
                                   positions[0] + len(pids)))
    assert all(p >= 8 for p in positions)


def test_brick_avoids_occupied(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    occupied = set(range(8, 8 + len(pids)))  # block the first-fit spot
    positions = brick.allocate(64, occupied=occupied, prompt_len=8)
    assert not any(p in occupied for p in positions)


def test_brick_fail_closed_when_no_fit(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    with pytest.raises(UnsatisfiableBrick):
        # canvas far too small for the phrase: refuse, don't weaken
        brick.allocate(len(pids) - 1, occupied=set(), prompt_len=0)


def test_enforce_before_allocate_raises(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    prompt = torch.randint(100, 1000, (8,))
    st = DenoisingState(length=64, mask_id=MASK_ID, eos_id=EOS_ID,
                        steps=4, seed=0, prompt_ids=prompt,
                        constraints=None, vocab_size=VOCAB, device="cpu")
    with pytest.raises(RuntimeError):
        brick.enforce(st)


# -- certificates ----------------------------------------------------------

CONTRACT = {"require_phrase": "quantum leap"}


def test_certificate_verifies_honest_output(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    model = PhraseSuppressor(pids, seed=1)
    x = run_with_brick(brick, model, seed=0)
    ids = x.tolist()
    cert = brick.certify(ids, CONTRACT)
    result = verify_certificate(cert, ids, CONTRACT)
    assert result.ok, f"honest certificate rejected: {result.reason}"


def test_verifier_rejects_tampered_positions(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    model = PhraseSuppressor(pids, seed=1)
    x = run_with_brick(brick, model, seed=0)
    ids = x.tolist()
    cert = brick.certify(ids, CONTRACT)
    tampered = copy.deepcopy(cert)
    tampered["positions"] = [p + 5 for p in tampered["positions"]]
    result = verify_certificate(tampered, ids, CONTRACT)
    assert not result.ok, "tampered positions accepted"


def test_verifier_rejects_removed_phrase(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    model = PhraseSuppressor(pids, seed=1)
    x = run_with_brick(brick, model, seed=0)
    ids = x.tolist()
    cert = brick.certify(ids, CONTRACT)
    # surgically remove the phrase after certification
    attacked = list(ids)
    for pos in cert["positions"]:
        attacked[pos] = 100  # some innocent token
    result = verify_certificate(cert, attacked, CONTRACT)
    assert not result.ok, "output with phrase removed accepted"


def test_verifier_rejects_replayed_contract(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    model = PhraseSuppressor(pids, seed=1)
    x = run_with_brick(brick, model, seed=0)
    ids = x.tolist()
    cert = brick.certify(ids, CONTRACT)
    other_contract = {"require_phrase": "classical mechanics"}
    result = verify_certificate(cert, ids, other_contract)
    assert not result.ok, "certificate replayed against another contract"


def test_verifier_rejects_unknown_brick(tok):
    pids = clean_ids(tok, "quantum leap")
    brick = MustIncludeBrick("quantum leap", pids)
    model = PhraseSuppressor(pids, seed=1)
    x = run_with_brick(brick, model, seed=0)
    ids = x.tolist()
    cert = brick.certify(ids, CONTRACT)
    cert["brick"] = "must-include-evil-twin"
    result = verify_certificate(cert, ids, CONTRACT)
    assert not result.ok, "unknown brick type accepted"


def test_chain_two_bricks(tok):
    pids_a = clean_ids(tok, "quantum leap")
    pids_b = clean_ids(tok, "neural nets")
    brick_a = MustIncludeBrick("quantum leap", pids_a)
    brick_b = MustIncludeBrick("neural nets", pids_b)
    model = PhraseSuppressor(pids_a + pids_b, seed=1)

    prompt = torch.randint(100, 1000, (8,))
    st = DenoisingState(length=96, mask_id=MASK_ID, eos_id=EOS_ID,
                        steps=8, seed=0, prompt_ids=prompt,
                        constraints=None, vocab_size=VOCAB, device="cpu")
    occupied = set()
    for brick in (brick_a, brick_b):
        positions = brick.allocate(96, occupied=occupied, prompt_len=8)
        occupied.update(positions)
        brick.enforce(st)
    while st.step(model):
        pass
    ids = st.x.tolist()
    assert has_phrase(ids, pids_a) and has_phrase(ids, pids_b)

    contract = {"require_phrase": ["quantum leap", "neural nets"]}
    cert_a = brick_a.certify(ids, contract)
    cert_b = brick_b.certify(ids, contract,
                             prev_hash=cert_a["cert_hash"])
    assert verify_certificate(cert_a, ids, contract).ok
    assert verify_certificate(cert_b, ids, contract,
                              prev_hash=cert_a["cert_hash"]).ok
    # cut the chain: brick B's certificate without brick A's hash
    assert not verify_certificate(cert_b, ids, contract).ok
    # reorder attack: swap the chain order
    assert not verify_certificate(
        cert_a, ids, contract, prev_hash=cert_b["cert_hash"]).ok
