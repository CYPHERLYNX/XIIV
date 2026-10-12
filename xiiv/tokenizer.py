"""Byte-level BPE tokenizer for XIIV, trained on the training corpus.

Vocab layout: [0, vocab_size) = BPE pieces, then [MASK], then [EOS].
The trained tokenizer.json is committed to the repo so inference needs
zero network access and zero external downloads.
"""
import json
import os
import re

MASK_TOKEN = "[MASK]"
EOS_TOKEN = "[EOS]"

# <mask>            -> one editable token
# <mask:N>          -> N consecutive editable tokens (N >= 1)
# anything else (e.g. <mask:abc>) is literal text, not a marker
# NOTE: the mask-marker span regex is built per-call inside
# encode_with_masks from its mask_marker argument.


class XIIVTokenizer:
    def __init__(self, tokenizer_json_path: str):
        from tokenizers import Tokenizer as HFTokenizer
        self._tok = HFTokenizer.from_file(tokenizer_json_path)
        self.mask_id = self._tok.token_to_id(MASK_TOKEN)
        self.eos_id = self._tok.token_to_id(EOS_TOKEN)
        assert self.mask_id is not None and self.eos_id is not None, \
            "tokenizer.json missing special tokens"
        # special ids come from the tokenizer file itself (mask_id=2, eos_id=3
        # for this build) — never from ModelConfig defaults.
        self.vocab_size = self._tok.get_vocab_size()

    @classmethod
    def train(cls, text_files, out_path: str, vocab_size: int = 8192):
        """Train a byte-level BPE tokenizer on raw text files."""
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        tok = HFTokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        # Without this, decode() concatenates raw byte-level pieces
        # (l e.g. "Ġ" instead of " ") — the 32k tokenizer shipped broken
        # until this was added (2026-10-09).
        tok.decoder = ByteLevelDecoder(add_prefix_space=False,
                                       trim_offsets=True, use_regex=True)
        trainer = BpeTrainer(vocab_size=vocab_size,
                             special_tokens=["[UNK]", "[PAD]", MASK_TOKEN, EOS_TOKEN],
                             show_progress=True)
        tok.train(text_files, trainer)
        # sanity: special ids must be the last two before [UNK]/[PAD]... just record
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tok.save(out_path)
        return cls(out_path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids, skip_special: bool = True) -> str:
        ids = [int(i) for i in ids]
        if skip_special:
            # NOTE: only [MASK]/[EOS] are stripped; [UNK]/[PAD] pass through.
            ids = [i for i in ids if i not in (self.mask_id, self.eos_id)]
            text = self._tok.decode(ids)
        else:
            text = self._tok.decode(ids)
        return text

    def encode_with_masks(self, text: str, mask_marker: str = "<mask>"):
        """Encode text containing mask markers into ids + frozen mask.

        ``<mask>`` becomes one [MASK] token; ``<mask:N>`` becomes N
        consecutive [MASK] tokens (N >= 1) — variable-length editable spans.
        All other positions are frozen (infill mode).
        Returns (ids_tensor, frozen_bool_tensor).
        """
        import re as _re
        import torch
        # wire mask_marker into the span regex: "<mask>" -> "<mask(?::(\d+))?>"
        # (re.escape leaves ">" unescaped, so split the trailing bracket off)
        if mask_marker.endswith(">"):
            span_re = _re.compile(
                _re.escape(mask_marker[:-1]) + r"(?::(\d+))?>")
        else:
            span_re = _re.compile(_re.escape(mask_marker))
        ids, frozen = [], []
        last = 0
        for mch in span_re.finditer(text):
            for tok_id in self.encode(text[last:mch.start()]):
                ids.append(tok_id)
                frozen.append(True)
            n = int(mch.group(1)) if mch.group(1) is not None else 1
            if n < 1:
                raise ValueError(
                    f"invalid mask span <mask:{mch.group(1)}>: N must be >= 1")
            ids.extend([self.mask_id] * n)
            frozen.extend([False] * n)
            last = mch.end()
        for tok_id in self.encode(text[last:]):
            ids.append(tok_id)
            frozen.append(True)
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(frozen, dtype=torch.bool))
