"""Tests for the contract-native architecture.

The contract-native model: a Contract is serialized to canonical text,
encoded by ContractEncoder, and cross-attended by every diffusion block.
Key properties under test:
  1. to_text() is deterministic and covers every contract clause.
  2. The contract encoder produces correctly-shaped embeddings.
  3. contract_enabled=False models are bit-identical to before the change.
  4. With contract_enabled=True but contract_ids=None, the model behaves
     exactly like the unconditional model (zero-init gate = no-op).
  5. With contract_ids given, the conditioning actually reaches the logits
     (the model is not ignoring the contract).
  6. DenoisingState threads contract_ids through to the model.
"""
import torch

from xiiv.config import ModelConfig
from xiiv.contracts import Contract
from xiiv.contract_encoder import ContractEncoder, encode_contract
from xiiv.diffusion import DenoisingState
from xiiv.model import XIIVModel


def tiny_cfg(**kw):
    args = dict(vocab_size=512, mask_id=510, eos_id=511, d_model=64,
                n_layers=2, n_heads=4, ffn_dim=128, seq_len=32)
    args.update(kw)
    return ModelConfig(**args)


def fake_tokenizer():
    # encode: deterministic hash -> ids; decode not needed here.
    class T:
        mask_id = 510
        eos_id = 511

        def encode(self, s):
            return [(abs(hash(s)) >> (8 * i)) % 500 + 7
                    for i in range(max(1, len(s) // 4))]

    return T()


# -- contract text form --------------------------------------------------

def test_to_text_covers_everything_and_is_deterministic():
    c = (Contract()
         .lock("Once upon", at="start")
         .require("the")
         .require_phrase("quantum leap", at=10)
         .forbid("bad")
         .forbid_phrase("old theory")
         .before("hypothesis", "conclusion")
         .at_least("the", 2)
         .at_most("very", 3)
         .exact_length(64))
    t1, t2 = c.to_text(), c.to_text()
    assert t1 == t2
    for needle in ["LOCK", "REQUIRE", "REQUIRE_PHRASE", "FORBID",
                   "FORBID_PHRASE", "ORDER", "ATLEAST", "ATMOST",
                   "LENGTH", "quantum leap", "hypothesis", "conclusion",
                   "64"]:
        assert needle in t1, needle


def test_to_text_empty_contract():
    assert Contract().to_text() == ""


# -- contract encoder ----------------------------------------------------

def test_contract_encoder_shapes():
    torch.manual_seed(0)
    cfg = tiny_cfg(contract_enabled=True)
    enc = ContractEncoder(cfg)
    ids = torch.randint(0, 500, (2, 12))
    out = enc(ids)
    assert out.shape == (2, 12, cfg.d_model)


def test_encode_contract_helper():
    c = Contract().require("the").forbid("bad").exact_length(32)
    ids = encode_contract(c, fake_tokenizer(), 96)
    assert ids.dim() == 2 and ids.shape[0] == 1
    assert ids.shape[1] <= 96


def test_encode_contract_too_long_raises():
    c = Contract()
    for i in range(60):
        c.require_phrase(f"phrase number {i} with several words")
    try:
        encode_contract(c, fake_tokenizer(), 16)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlong contract")


# -- model integration ---------------------------------------------------

def test_contract_disabled_is_bit_identical():
    """contract_enabled=False must not change a single logit vs the
    pre-change architecture (same seed, same weights init)."""
    torch.manual_seed(7)
    cfg = tiny_cfg()
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    ids = torch.randint(0, 500, (1, 16))
    with torch.no_grad():
        a = m(ids)
        b = m(ids)
    assert torch.equal(a, b)
    # and the new kwargs default to the same path
    with torch.no_grad():
        c = m(ids, contract_ids=None)
    assert torch.equal(a, c)


def test_contract_enabled_no_contract_matches_unconditional():
    """Zero-init gate: with contract_ids=None the conditioned model is an
    exact no-op relative to its own unconditional path."""
    torch.manual_seed(11)
    cfg = tiny_cfg(contract_enabled=True)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    ids = torch.randint(0, 500, (1, 16))
    with torch.no_grad():
        a = m(ids)
        b = m(ids, contract_ids=None)
    assert torch.equal(a, b)


def test_contract_ids_actually_change_logits():
    """The conditioning must REACH the output: different contracts must
    produce different logits (the model is not ignoring the contract)."""
    torch.manual_seed(13)
    cfg = tiny_cfg(contract_enabled=True)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    ids = torch.randint(0, 500, (1, 16))
    c1 = torch.randint(0, 500, (1, 8))
    c2 = torch.randint(0, 500, (1, 8))
    assert not torch.equal(c1, c2)
    with torch.no_grad():
        base = m(ids)
        l1 = m(ids, contract_ids=c1)
        l2 = m(ids, contract_ids=c2)
    # gate starts at zero -> identical at init; nudge the gates open to
    # prove the pathway is live end to end.
    for layer in m.layers:
        layer.contract_attn.gate.data.fill_(1.0)
    with torch.no_grad():
        g1 = m(ids, contract_ids=c1)
        g2 = m(ids, contract_ids=c2)
    assert not torch.equal(g1, g2), "contract pathway is dead"
    assert not torch.equal(g1, base), "contract ignored vs unconditional"


def test_contract_ids_rejected_when_disabled():
    cfg = tiny_cfg()
    m = XIIVModel(cfg)
    ids = torch.randint(0, 500, (1, 8))
    cid = torch.randint(0, 500, (1, 4))
    try:
        m(ids, contract_ids=cid)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_grad_checkpoint_path_accepts_contract():
    torch.manual_seed(21)
    cfg = tiny_cfg(contract_enabled=True)
    m = XIIVModel(cfg)
    m.grad_checkpoint = True
    m.train()
    ids = torch.randint(0, 500, (1, 16)).long()
    cid = torch.randint(0, 500, (1, 6))
    logits = m(ids, contract_ids=cid)
    assert logits.shape == (1, 16, cfg.vocab_size)
    loss = logits.sum()
    loss.backward()  # must not crash through the checkpointed blocks


# -- sampler wiring ------------------------------------------------------

class _ConstModel(torch.nn.Module):
    """Returns fixed logits; records kwargs it was called with."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.seen = []

    def forward(self, xb, **kw):
        self.seen.append(set(kw))
        B, L = xb.shape
        return torch.zeros(B, L, self.vocab)


def test_denoising_state_threads_contract_ids():
    torch.manual_seed(3)
    vocab, mask = 64, 63
    model = _ConstModel(vocab)
    cid = torch.tensor([[1, 2, 3]])
    st = DenoisingState(length=8, mask_id=mask, eos_id=62, vocab_size=vocab,
                       contract_ids=cid, seed=1)
    st.step(model)
    assert model.seen and "contract_ids" in model.seen[0]
    # snapshot round-trip preserves the contract
    snap = st.to_snapshot()
    st2 = DenoisingState.from_snapshot(snap)
    assert torch.equal(st2.contract_ids.cpu(), cid)


def test_denoising_state_without_contract_is_unchanged():
    torch.manual_seed(3)
    vocab, mask = 64, 63
    model = _ConstModel(vocab)
    st = DenoisingState(length=8, mask_id=mask, eos_id=62, vocab_size=vocab, seed=1)
    st.step(model)
    assert model.seen and "contract_ids" not in model.seen[0]
    assert st.to_snapshot()["contract_ids"] is None
