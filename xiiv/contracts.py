"""The Contract system: the unified path for guarantee-native generation.

A Contract declares what an output MUST satisfy, in one place::

    from xiiv.contracts import Contract

    c = Contract()
    c.lock("Policy: ask first. Record each step.")  # anchored byte-exact
    c.require("ask", "step")     # must appear — enforced by the sampler
    c.forbid("cut")              # must not appear — logits at -inf
    c.exact_length(128)          # total canvas length, declarative

    constraints, total = c.compile(tokenizer, prompt_len=len(prompt_ids))

The compiler lowers the declaration into sampler constraints
(anchors / required_ids / forbidden_ids); the DenoisingState's
slot-reservation invariant enforces required terms, and
check_constraints() verifies the finished canvas. This module is the
single source of truth — the knowledge layer and agent contracts both
build on it.

Single-token rule: require/forbid terms must encode to exactly one
token. The sampler enforces token ids; a multi-token word's tokens
would be placed independently and scatter instead of forming the word.
compile() raises ContractError otherwise — loudly, not silently.
"""
import torch

from .diffusion import check_constraints

DEFAULT_LENGTH = 128


class ContractError(ValueError):
    """Unsatisfiable or malformed contract."""


class Contract:
    """A declared output contract, compiled to sampler constraints."""

    def __init__(self):
        self._locks = []       # [(text, position)] position: "start"|int
        self._required = []    # [term]
        self._forbidden = []   # [term]
        self._length = None

    def lock(self, text, at="start"):
        """Pin ``text`` byte-exact at ``at`` ("start" of generated span
        or an absolute canvas position)."""
        if at != "start" and not isinstance(at, int):
            raise ContractError("lock position must be 'start' or int")
        self._locks.append((text, at))
        return self

    def require(self, *terms):
        """Each term MUST appear at least once in the final canvas."""
        self._required.extend(terms)
        return self

    def forbid(self, *terms):
        """Each term MUST NOT appear in the final canvas."""
        self._forbidden.extend(terms)
        return self

    def exact_length(self, n):
        """Total canvas length will be exactly ``n`` tokens."""
        if n <= 0:
            raise ContractError("length must be positive")
        self._length = int(n)
        return self

    # -- compilation ----------------------------------------------------

    @staticmethod
    def _term_ids(tokenizer, term):
        ids = [i for i in tokenizer.encode(term)
               if i not in (tokenizer.mask_id, tokenizer.eos_id)]
        if len(ids) != 1:
            raise ContractError(
                f"contract term {term!r} is {len(ids)} tokens; "
                "require/forbid need exactly 1 (the sampler enforces "
                "token ids — multi-token words would scatter)")
        return ids[0]

    def compile(self, tokenizer, prompt_len):
        """Lower the contract to (constraints, total_length).

        ``prompt_len`` is the frozen prompt's token count; "start"
        locks anchor at the first generated position after it.
        """
        total = self._length or DEFAULT_LENGTH
        anchors = {}
        cursor = prompt_len
        for text, at in self._locks:
            ids = [i for i in tokenizer.encode(text)
                   if i not in (tokenizer.mask_id, tokenizer.eos_id)]
            pos = cursor if at == "start" else int(at)
            if pos < prompt_len or pos + len(ids) > total:
                raise ContractError("locked text does not fit the canvas")
            for p, t in enumerate(ids):
                anchors[pos + p] = t
            if at == "start":
                cursor = pos + len(ids)

        required = sorted({self._term_ids(tokenizer, t)
                           for t in self._required})
        forbidden = sorted({self._term_ids(tokenizer, t)
                           for t in self._forbidden})
        if set(required) & set(forbidden):
            raise ContractError("term both required and forbidden")

        n_masks = total - prompt_len - len(anchors)
        if n_masks < len(required):
            raise ContractError(
                f"not enough free positions for {len(required)} "
                "required terms")

        constraints = {}
        if anchors:
            constraints["anchors"] = anchors
        if required:
            constraints["required_ids"] = required
        if forbidden:
            constraints["forbidden_ids"] = forbidden
        return constraints, total

    # -- verification ----------------------------------------------------

    @staticmethod
    def verify(ids, constraints, mask_id):
        """Machine-check the finished canvas. Returns the check dict."""
        return check_constraints(ids, constraints, mask_id)

    @staticmethod
    def holds(check):
        """True when every applicable guarantee verified."""
        keys = ("required_present", "forbidden_absent", "anchors_intact")
        return all(check.get(k, True) for k in keys)
