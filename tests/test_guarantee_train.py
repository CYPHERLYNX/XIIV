"""Tests for guarantee-native pre-training.

The sampler must NEVER emit an unsatisfiable contract: every clause is
derived from the target sequence itself. These tests verify that property
at the token level using a stable word-level fake tokenizer.
"""
import torch

from xiiv.config import ModelConfig
from xiiv.contracts import Contract
from xiiv.diffusion import diffusion_loss
from xiiv.guarantee_train import sample_contracts, encode_contract_batch
from xiiv.model import XIIVModel


class WordTokenizer:
    """Stable word-level tokenizer: encode(decode(ids)) == ids."""

    def __init__(self, n_words=50):
        self.vocab = {}
        self.words = []
        self.mask_id = 10000
        self.eos_id = 10001
        # pre-populate so raw id i <-> word f"w{i}" is stable from the start
        for i in range(n_words):
            self._id(f"w{i}")

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.words)
            self.words.append(w)
        return self.vocab[w]

    def encode(self, text):
        return [self._id(w) for w in text.split() if w]

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids
                        if i < len(self.words))


def make_batch(g, rows=4, length=32, vocab_words=40):
    # random word-id sequences in [0, vocab_words)
    return torch.randint(0, vocab_words, (rows, length), generator=g)


def contract_satisfiable(c: Contract, seq_ids, tok: WordTokenizer):
    """Check every clause of c against the token sequence (word-level)."""
    words = [tok.words[i] for i in seq_ids if i < len(tok.words)]
    text = c.to_text()
    assert text, "empty contract"
    # required phrases: contiguous word runs present in the sequence
    for phrase, _at in c._phrases:
        pw = phrase.split()
        assert any(words[i:i + len(pw)] == pw
                   for i in range(len(words) - len(pw) + 1)), \
            f"required phrase {phrase!r} not in sequence"
    # forbidden terms: absent
    for t in c._forbidden:
        assert t not in words, f"forbidden term {t!r} present"
    # at_least: count holds
    for term, n in c._at_least:
        assert words.count(term) >= n, f"at_least {term!r} x{n} violated"
    # before: every a occurrence precedes every b occurrence
    for a, b in c._order:
        aw = a.split()
        bw = b.split()
        apos = [i for i in range(len(words) - len(aw) + 1)
                if words[i:i + len(aw)] == aw]
        bpos = [i for i in range(len(words) - len(bw) + 1)
                if words[i:i + len(bw)] == bw]
        assert apos and bpos, f"order pair {a!r} < {b!r} missing"
        assert max(apos) < min(bpos), f"order {a!r} < {b!r} violated"


def test_standard_contracts_are_satisfiable():
    g = torch.Generator().manual_seed(0)
    tok = WordTokenizer()
    batch = make_batch(g)
    contracts = sample_contracts(batch, tok, 50000, tok.mask_id, tok.eos_id,
                                 g, adversarial_frac=0.0)
    assert len(contracts) == batch.shape[0]
    for c, row in zip(contracts, batch.tolist()):
        assert isinstance(c, Contract)
        contract_satisfiable(c, row, tok)


def test_adversarial_contracts_are_satisfiable():
    g = torch.Generator().manual_seed(1)
    tok = WordTokenizer()
    batch = make_batch(g)
    contracts = sample_contracts(batch, tok, 50000, tok.mask_id, tok.eos_id,
                                 g, adversarial_frac=1.0)
    assert len(contracts) == batch.shape[0]
    for c, row in zip(contracts, batch.tolist()):
        contract_satisfiable(c, row, tok)


def test_adversarial_prefers_rare_spans():
    # adversarial require_phrase should pick a span occurring exactly once
    # when such spans exist (check the property holds structurally)
    g = torch.Generator().manual_seed(2)
    tok = WordTokenizer()
    batch = make_batch(g, rows=8, length=48)
    contracts = sample_contracts(batch, tok, 50000, tok.mask_id, tok.eos_id,
                                 g, adversarial_frac=1.0)
    rare = 0
    for c, row in zip(contracts, batch.tolist()):
        words = [tok.words[i] for i in row if i < len(tok.words)]
        for phrase, _ in c._phrases:
            pw = phrase.split()
            occ = sum(1 for i in range(len(words) - len(pw) + 1)
                      if words[i:i + len(pw)] == pw)
            if occ == 1:
                rare += 1
    assert rare >= 1, "adversarial sampler never picked a rare span"


def test_encode_contract_batch_shapes():
    g = torch.Generator().manual_seed(3)
    tok = WordTokenizer()
    batch = make_batch(g)
    contracts = sample_contracts(batch, tok, 50000, tok.mask_id, tok.eos_id,
                                 g, adversarial_frac=0.5)
    ids = encode_contract_batch(contracts, tok, 96, "cpu")
    assert ids.dim() == 2 and ids.shape[0] == batch.shape[0]
    assert ids.shape[1] <= 96


def test_diffusion_loss_with_contract_ids():
    torch.manual_seed(5)
    cfg = ModelConfig(vocab_size=512, mask_id=510, eos_id=511, d_model=64,
                      n_layers=2, n_heads=4, ffn_dim=128, seq_len=32,
                      contract_enabled=True)
    model = XIIVModel(cfg)
    model.init_weights()
    model.train()
    batch = torch.randint(0, 500, (2, 32))
    cid = torch.randint(0, 500, (2, 10))
    loss, info = diffusion_loss(model, batch, cfg.mask_id,
                                contract_ids=cid)
    assert torch.isfinite(loss).item()
    loss.backward()  # gradients flow through the contract path


def test_diffusion_loss_without_contract_unchanged():
    torch.manual_seed(5)
    cfg = ModelConfig(vocab_size=512, mask_id=510, eos_id=511, d_model=64,
                      n_layers=2, n_heads=4, ffn_dim=128, seq_len=32)
    model = XIIVModel(cfg)
    model.init_weights()
    batch = torch.randint(0, 500, (2, 32))
    loss, _ = diffusion_loss(model, batch, cfg.mask_id)
    assert torch.isfinite(loss).item()
