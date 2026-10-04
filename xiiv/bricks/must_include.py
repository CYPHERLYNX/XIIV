"""Must-include brick: the phrase MUST appear in the output.

The brick compiles a required phrase into a contiguous block of canvas
positions, pins the phrase's tokens there through the sampler's own
pinning primitive, and emits a proof certificate binding the phrase,
its positions, the contract, and the exact output.

The brick is fail-closed: if no contiguous block fits the canvas, it
raises instead of silently weakening the guarantee.
"""
from . import certificate as certmod


class UnsatisfiableBrick(ValueError):
    """Raised when the guarantee cannot be met on this canvas."""


class MustIncludeBrick:
    brick_type = "must-include"

    def __init__(self, phrase: str, token_ids):
        self.phrase = phrase
        self.token_ids = [int(t) for t in token_ids]
        if not self.token_ids:
            raise ValueError("must-include needs a non-empty phrase")
        self.positions = None  # allocated by allocate()

    @classmethod
    def from_text(cls, phrase: str, tokenizer):
        """Build from raw text using the project's tokenizer."""
        ids = tokenizer.encode(phrase)
        mask_id = getattr(tokenizer, "mask_id", None)
        eos_id = getattr(tokenizer, "eos_id", None)
        ids = [t for t in ids if t not in (mask_id, eos_id)]
        return cls(phrase, ids)

    def allocate(self, canvas_length: int, occupied, prompt_len: int):
        """First-fit a contiguous block for the phrase.

        Returns the allocated positions. Raises UnsatisfiableBrick when
        the phrase does not fit the free canvas (fail-closed).
        """
        n = len(self.token_ids)
        occupied = set(int(p) for p in occupied)
        p = int(prompt_len)
        while True:
            if p + n > int(canvas_length):
                raise UnsatisfiableBrick(
                    f"phrase {self.phrase!r} ({n} tokens) does not fit "
                    f"the canvas of length {canvas_length}")
            block = list(range(p, p + n))
            if not any(q in occupied for q in block):
                self.positions = block
                return block
            p += 1

    def enforce(self, state) -> None:
        """Pin the phrase into the sampler state.

        ``state`` is a xiiv.diffusion.DenoisingState (or any object with
        a compatible pin(pos, token_id) method). The brick drives the
        sampler; it does not fork or patch it.
        """
        if self.positions is None:
            raise RuntimeError("enforce() called before allocate()")
        for pos, tid in zip(self.positions, self.token_ids):
            state.pin(pos, tid)

    def certify(self, output_ids, contract: dict, prev_hash=None) -> dict:
        """Build the proof certificate for a finished generation."""
        if self.positions is None:
            raise RuntimeError("certify() called before allocate()")
        cert = {
            "brick": self.brick_type,
            "version": certmod.CERT_VERSION,
            "contract_hash": certmod.contract_hash(contract),
            "phrase": self.phrase,
            "phrase_token_ids": list(self.token_ids),
            "positions": list(self.positions),
            "output_length": len(output_ids),
            "output_hash": certmod.output_hash(output_ids),
            "prev_hash": prev_hash or certmod.GENESIS_HASH,
        }
        return certmod.seal(cert)
