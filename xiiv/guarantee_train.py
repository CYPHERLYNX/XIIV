"""Guarantee-native pre-training: synthetic contract sampler.

The core idea: instead of training the model on raw text alone, a fraction
of batches is *contract-conditioned*. For each sequence we synthesize a
Contract FROM the sequence's own tokens, so the contract is satisfiable by
construction, then train the standard masked-diffusion loss conditioned on
the contract text: the model learns P(target | noisy input, contract).

Two modes:
  standard    -- ordinary satisfiable clauses (a required phrase, a
                 forbidden absent token, a loose at-least count).
  adversarial -- hard-but-satisfiable clauses designed to provoke
                 violations: a phrase occurring exactly once (the model
                 must pin a rare span), a tight at-least with zero slack
                 (exact count), an ordering over distant spans.

Nothing here can produce an unsatisfiable contract: every clause is
derived from the target sequence itself. Training on contradictions
would teach the model to ignore contracts; this sampler never does.
"""
import torch

from .contracts import Contract


def _clean(ids, mask_id, eos_id):
    return [i for i in ids if i not in (mask_id, eos_id)]


def _randint(lo, hi, g):
    # inclusive lo, exclusive hi
    return int(torch.randint(lo, hi, (), generator=g).item())


def _pick_span(clean, g, min_len=2, max_len=4):
    n = len(clean)
    if n < min_len + 1:
        return None
    span_len = _randint(min_len, min(max_len, n - 1) + 1, g)
    start = _randint(0, n - span_len + 1, g)
    return clean[start:start + span_len]


def _decode_span(tokenizer, span):
    try:
        text = tokenizer.decode(span).strip()
    except Exception:
        return ""
    return text


def _standard_clauses(c: Contract, clean, tokenizer, vocab_size,
                      mask_id, eos_id, g):
    # require a phrase sampled from the sequence itself
    span = _pick_span(clean, g)
    if span:
        text = _decode_span(tokenizer, span)
        if text:
            c.require_phrase(text)
    # forbid a token absent from the sequence (satisfiable by construction)
    present = set(clean)
    for _ in range(8):
        cand = _randint(0, vocab_size, g)
        if cand not in present and cand not in (mask_id, eos_id):
            term = _decode_span(tokenizer, [cand])
            if term and len(tokenizer.encode(term)) == 1:
                c.forbid(term)
            break
    # loose at-least on the most frequent token
    if torch.rand((), generator=g).item() < 0.5 and clean:
        freq = {}
        for t in clean:
            freq[t] = freq.get(t, 0) + 1
        top, cnt = max(freq.items(), key=lambda kv: kv[1])
        term = _decode_span(tokenizer, [top])
        if term and len(tokenizer.encode(term)) == 1:
            c.at_least(term, min(cnt, 2))


def _adversarial_clauses(c: Contract, clean, tokenizer, vocab_size,
                         mask_id, eos_id, g):
    n = len(clean)
    # 1) require a phrase occurring EXACTLY once: the model must pin a
    # rare span instead of leaning on repetition.
    placed = False
    for _ in range(12):
        span = _pick_span(clean, g, 2, 5)
        if not span:
            break
        occ = sum(1 for i in range(n - len(span) + 1)
                  if clean[i:i + len(span)] == span)
        if occ == 1:
            text = _decode_span(tokenizer, span)
            if text:
                c.require_phrase(text)
                placed = True
                break
    if not placed:
        _standard_clauses(c, clean, tokenizer, vocab_size,
                          mask_id, eos_id, g)
        return
    # 2) tight at-least: exact count, zero slack.
    freq = {}
    for t in clean:
        freq[t] = freq.get(t, 0) + 1
    if freq:
        top, cnt = max(freq.items(), key=lambda kv: kv[1])
        term = _decode_span(tokenizer, [top])
        if term and len(tokenizer.encode(term)) == 1 and cnt >= 2:
            c.at_least(term, cnt)  # exact: no slack allowed
    # 3) ordering over two distant spans (a must precede b).
    if n >= 16 and torch.rand((), generator=g).item() < 0.7:
        s1 = _pick_span(clean[:n // 2], g, 2, 3)
        s2 = _pick_span(clean[n // 2:], g, 2, 3)
        t1 = _decode_span(tokenizer, s1) if s1 else ""
        t2 = _decode_span(tokenizer, s2) if s2 else ""
        if t1 and t2 and t1 != t2:
            c.before(t1, t2)


def sample_contracts(batch: torch.Tensor, tokenizer, vocab_size: int,
                     mask_id: int, eos_id: int, g,
                     adversarial_frac: float = 0.3):
    """One satisfiable Contract per sequence in ``batch`` (B, T).

    ``batch`` holds CLEAN token ids (pre-masking). Returns a list of
    Contract, one per row -- never None (falls back to a trivial
    require_phrase when the sequence is too short for anything else).
    """
    out = []
    for b in range(batch.shape[0]):
        clean = _clean(batch[b].tolist(), mask_id, eos_id)
        c = Contract()
        if len(clean) >= 8:
            if torch.rand((), generator=g).item() < adversarial_frac:
                _adversarial_clauses(c, clean, tokenizer, vocab_size,
                                     mask_id, eos_id, g)
            else:
                _standard_clauses(c, clean, tokenizer, vocab_size,
                                  mask_id, eos_id, g)
        if not c.to_text():
            # degenerate fallback: still a real, satisfiable contract
            span = clean[:2] if len(clean) >= 2 else clean
            text = _decode_span(tokenizer, span) or "the"
            c.require_phrase(text)
        out.append(c)
    return out


def encode_contract_batch(contracts, tokenizer, max_tokens: int, device):
    """[Contract] -> (B, C) padded contract-id tensor for the model."""
    from .contract_encoder import encode_contract
    rows = []
    for c in contracts:
        try:
            ids = encode_contract(c, tokenizer, max_tokens).squeeze(0)
        except ValueError:
            # overlong: fall back to a trivial contract rather than
            # dropping the sequence from the conditioned batch
            tiny = Contract().require_phrase("the")
            ids = encode_contract(tiny, tokenizer, max_tokens).squeeze(0)
        rows.append(ids)
    width = max(int(r.shape[0]) for r in rows)
    pad_id = tokenizer.mask_id
    padded = [torch.cat([r, torch.full((width - r.shape[0],), pad_id,
                                       dtype=torch.long)])
              for r in rows]
    return torch.stack(padded).to(device)
