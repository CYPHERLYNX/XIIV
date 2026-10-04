"""The Contract system: the unified path for guarantee-native generation.

A Contract declares what an output MUST satisfy, in one place::

    from xiiv.contracts import Contract

    c = Contract()
    c.lock("Policy: ask first. Record each step.")   # anchored byte-exact
    c.require("ask", "step")                         # single tokens, flexible
    c.require_phrase("myocardial infarction")        # multi-token, contiguous
    c.require_phrase("call cardiology", at=96)       # pinned position
    c.before("myocardial infarction", "call cardiology")  # ordering
    c.at_least("aspirin", 2)                         # cardinality
    c.at_most("ibuprofen", 1)                        # cardinality cap
    c.forbid("cut")                                  # single token, never
    c.forbid_phrase("do not resuscitate")            # phrase, never
    c.exact_length(128)

    constraints, total = c.compile(tokenizer, prompt_len=len(prompt_ids))

The compiler lowers the declaration into sampler constraints; the
DenoisingState enforces them during unmasking (slot reservation for
required counts, preventive suppression for forbidden phrases and
at-most caps), and check_constraints() verifies the finished canvas.

Design notes (v2):
- Single-token require/forbid keep the flexible slot-reservation
  mechanism (positions chosen by the sampler).
- Phrases compile to compiler-allocated contiguous blocks (deterministic
  layout). "Must appear somewhere" is realized as "the compiler finds
  room and pins it" — the content guarantee is identical, and the
  position is knowable, which contracts usually want.
- before(a, b) promotes its arguments to required phrases when they
  are not already (ordering an absent term is vacuous).
- compile() proves satisfiability up front: cycles, contradictions,
  subsequence conflicts, and space exhaustion all raise ContractError
  loudly instead of failing silently at sample time.
"""
import torch

from .diffusion import check_constraints

DEFAULT_LENGTH = 128


class ContractError(ValueError):
    """Unsatisfiable or malformed contract."""


class Contract:
    """A declared output contract, compiled to sampler constraints."""

    def __init__(self):
        self._locks = []          # [(text, position)] "start"|int
        self._required = []       # [term] single-token, flexible
        self._forbidden = []      # [term] single-token
        self._phrases = []        # [(text, at|None)] multi-token require
        self._forbid_phrases = []  # [text]
        self._order = []          # [(a_text, b_text)]
        self._at_least = []       # [(term, n)]
        self._at_most = []        # [(term, n)]
        self._length = None

    # -- declaration ---------------------------------------------------

    def lock(self, text, at="start"):
        """Pin ``text`` byte-exact at ``at`` ("start" of generated span
        or an absolute canvas position)."""
        if at != "start" and not isinstance(at, int):
            raise ContractError("lock position must be 'start' or int")
        self._locks.append((text, at))
        return self

    def require(self, *terms):
        """Each term MUST appear at least once (single-token terms;
        positions chosen by the sampler)."""
        self._required.extend(terms)
        return self

    def forbid(self, *terms):
        """Each term MUST NOT appear (single-token terms)."""
        self._forbidden.extend(terms)
        return self

    def require_phrase(self, text, at=None):
        """A multi-token phrase MUST appear contiguously. Position is
        compiler-assigned (deterministic) unless ``at`` pins it."""
        if at is not None and not isinstance(at, int):
            raise ContractError("phrase position must be int or None")
        self._phrases.append((text, at))
        return self

    def forbid_phrase(self, text):
        """A multi-token phrase MUST NOT appear (preventive suppression
        in the sampler; verified on the finished canvas)."""
        self._forbid_phrases.append(text)
        return self

    def before(self, a, b):
        """Phrase ``a`` must precede phrase ``b`` (every occurrence of a
        before every occurrence of b). Arguments are promoted to required
        phrases when not already declared."""
        if a == b:
            raise ContractError("before() needs two distinct phrases")
        self._order.append((a, b))
        return self

    def at_least(self, term, n):
        """``term`` must occur at least ``n`` times (single token: count;
        multi-token: n phrase blocks)."""
        if n < 1:
            raise ContractError("at_least needs n >= 1")
        self._at_least.append((term, int(n)))
        return self

    def at_most(self, term, n):
        """``term`` must occur at most ``n`` times (single token: logit
        suppression at the cap; multi-token: phrase completion
        suppression)."""
        if n < 0:
            raise ContractError("at_most needs n >= 0")
        self._at_most.append((term, int(n)))
        return self

    def exact_length(self, n):
        """Total canvas length will be exactly ``n`` tokens."""
        if n <= 0:
            raise ContractError("length must be positive")
        self._length = int(n)
        return self

    # -- text form (for the contract-native model) ---------------------

    @staticmethod
    def _q(text: str) -> str:
        return '"' + text.replace('"', '\\"') + '"'

    def to_text(self) -> str:
        """Canonical text serialization of the contract.

        Deterministic: the same contract always produces the same string.
        This is the input to the contract-native model's contract encoder
        (tokenized with the main tokenizer) -- the contract as a first-class
        architectural input, not a prompt hack.
        """
        parts = []
        for text, at in self._locks:
            parts.append(f"LOCK {self._q(text)} @{at}")
        for t in self._required:
            parts.append(f"REQUIRE {self._q(t)}")
        for text, at in self._phrases:
            tag = f" @{at}" if at is not None else ""
            parts.append(f"REQUIRE_PHRASE {self._q(text)}{tag}")
        for t in self._forbidden:
            parts.append(f"FORBID {self._q(t)}")
        for text in self._forbid_phrases:
            parts.append(f"FORBID_PHRASE {self._q(text)}")
        for a, b in self._order:
            parts.append(f"ORDER {self._q(a)} < {self._q(b)}")
        for term, n in self._at_least:
            parts.append(f"ATLEAST {self._q(term)} x{n}")
        for term, n in self._at_most:
            parts.append(f"ATMOST {self._q(term)} x{n}")
        if self._length is not None:
            parts.append(f"LENGTH {self._length}")
        return " | ".join(parts)

    # -- compilation ----------------------------------------------------

    def _clean_ids(self, tokenizer, text):
        ids = [i for i in tokenizer.encode(text)
               if i not in (tokenizer.mask_id, tokenizer.eos_id)]
        if not ids:
            raise ContractError(f"contract text {text!r} encodes to nothing")
        return ids

    def _single_id(self, tokenizer, term):
        ids = self._clean_ids(tokenizer, term)
        if len(ids) != 1:
            raise ContractError(
                f"contract term {term!r} is {len(ids)} tokens; "
                "require/forbid/at_least/at_most on bare terms need "
                "exactly 1 token — use require_phrase/forbid_phrase for "
                "multi-token text")
        return ids[0]

    @staticmethod
    def _is_subseq(hay, needle):
        n = len(needle)
        return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))

    def compile(self, tokenizer, prompt_len):
        """Lower the contract to (constraints, total_length).

        ``prompt_len`` is the frozen prompt's token count; "start"
        locks anchor at the first generated position after it.
        Raises ContractError when the contract is unsatisfiable.
        """
        total = self._length or DEFAULT_LENGTH

        # --- 1. locks -> anchors (existing semantics) ---
        anchors = {}
        cursor = prompt_len
        for text, at in self._locks:
            ids = self._clean_ids(tokenizer, text)
            pos = cursor if at == "start" else int(at)
            if pos < prompt_len or pos + len(ids) > total:
                raise ContractError("locked text does not fit the canvas")
            for p, t in enumerate(ids):
                if pos + p in anchors:
                    raise ContractError("overlapping locked spans")
                anchors[pos + p] = t
            if at == "start":
                cursor = pos + len(ids)

        # --- 2. phrase units (explicit + promoted) ---
        # text -> {"ids": [...], "at": int|None}
        units = {}
        def get_unit(text):
            if text not in units:
                units[text] = {"ids": self._clean_ids(tokenizer, text),
                               "at": None}
            return text
        at_of = {}
        for text, at in self._phrases:
            get_unit(text)
            if at is not None:
                if text in at_of and at_of[text] != at:
                    raise ContractError(
                        f"phrase {text!r} pinned at two positions")
                at_of[text] = at
                units[text]["at"] = at
        for a, b in self._order:
            get_unit(a)
            get_unit(b)
        # at_least multi-token terms: each required occurrence becomes its
        # own phrase unit (unique key, shared display text).
        n_extra = 0
        for term, n in self._at_least:
            ids = self._clean_ids(tokenizer, term)
            if len(ids) > 1:
                for _ in range(n):
                    key = f"{term}\x00{n_extra}"
                    n_extra += 1
                    units[key] = {"ids": ids, "at": None}

        # --- 3. topological order from before() ---
        nodes = list(units.keys())
        edges = [(a, b) for a, b in self._order]
        for a, b in edges:
            if a not in units or b not in units:
                raise ContractError("before() on unknown phrase")
        # Kahn's algorithm; leftovers => cycle
        indeg = {t: 0 for t in nodes}
        succ = {t: [] for t in nodes}
        for a, b in edges:
            succ[a].append(b)
            indeg[b] += 1
        queue = sorted([t for t in nodes if indeg[t] == 0])
        topo = []
        while queue:
            t = queue.pop(0)
            topo.append(t)
            for s in sorted(succ[t]):
                indeg[s] -= 1
                if indeg[s] == 0:
                    queue.append(s)
            queue.sort()
        if len(topo) != len(nodes):
            cyclic = sorted(set(nodes) - set(topo))
            raise ContractError(
                f"before() cycle involving: {cyclic}")

        # --- 4. layout: explicit positions, then first-fit in topo order ---
        occupied = set(anchors.keys())
        blocks = {}  # text -> start position
        for text in nodes:
            at = units[text]["at"]
            if at is None:
                continue
            ids = units[text]["ids"]
            if at < prompt_len or at + len(ids) > total:
                raise ContractError(
                    f"phrase {text!r} does not fit the canvas")
            rng = set(range(at, at + len(ids)))
            if rng & occupied:
                raise ContractError(
                    f"phrase {text!r} overlaps a locked span or phrase")
            blocks[text] = at
            occupied |= rng
        fit_cursor = prompt_len
        def first_fit(length):
            nonlocal fit_cursor
            p = fit_cursor
            while True:
                if p + length > total:
                    raise ContractError(
                        "contract does not fit the canvas: "
                        "not enough free positions for phrases")
                if not any(q in occupied for q in range(p, p + length)):
                    occupied.update(range(p, p + length))
                    fit_cursor = p + length
                    return p
                p += 1
        for text in topo:
            if text in blocks:
                continue
            ids = units[text]["ids"]
            blocks[text] = first_fit(len(ids))

        # --- 5. required counts (single-token) ---
        required_counts = {}
        for t in self._required:
            tid = self._single_id(tokenizer, t)
            required_counts[tid] = max(required_counts.get(tid, 0), 1)
        for term, n in self._at_least:
            ids = self._clean_ids(tokenizer, term)
            if len(ids) == 1:
                tid = ids[0]
                required_counts[tid] = max(required_counts.get(tid, 0), n)

        # --- 6. at_most caps ---
        at_most = {}
        forbidden_phrases = []  # [(ids, max_allowed)]
        for term, n in self._at_most:
            ids = self._clean_ids(tokenizer, term)
            if len(ids) == 1:
                tid = ids[0]
                at_most[tid] = min(at_most.get(tid, n), n)
            else:
                forbidden_phrases.append((ids, n))
        for text in self._forbid_phrases:
            forbidden_phrases.append((self._clean_ids(tokenizer, text), 0))

        # --- 7. conflict detection ---
        forbidden_ids = sorted({self._single_id(tokenizer, t)
                                for t in self._forbidden})
        if set(required_counts) & set(forbidden_ids):
            raise ContractError("term both required and forbidden")
        # required/forbidden phrase contradiction (subsequence)
        req_phrase_idss = [u["ids"] for u in units.values()]
        lock_idss = []
        for text, at in self._locks:
            lock_idss.append(self._clean_ids(tokenizer, text))
        for fids, m in forbidden_phrases:
            if m != 0:
                continue
            for rids in req_phrase_idss + lock_idss:
                if self._is_subseq(rids, fids):
                    raise ContractError(
                        "forbidden phrase is contained in a required "
                        "phrase/lock: unsatisfiable")
            for rids in req_phrase_idss:
                if rids == fids:
                    raise ContractError(
                        "phrase both required and forbidden")
        # required single token inside a required phrase is fine (counts
        # independently); required single inside FORBIDDEN phrase: the
        # preventive suppression only blocks completions, single tokens
        # can still appear elsewhere — no contradiction.
        for tid, need in required_counts.items():
            if tid in at_most and need > at_most[tid]:
                raise ContractError(
                    f"at_least({need}) exceeds at_most({at_most[tid]})")
        # phrase at_least vs at_most on the same phrase text
        phrase_at_most = {}
        for term, n in self._at_most:
            ids = self._clean_ids(tokenizer, term)
            if len(ids) > 1:
                phrase_at_most[term] = min(phrase_at_most.get(term, n), n)
        for term, n in self._at_least:
            ids = self._clean_ids(tokenizer, term)
            if len(ids) > 1 and term in phrase_at_most \
                    and n > phrase_at_most[term]:
                raise ContractError(
                    f"at_least({n}) exceeds at_most({phrase_at_most[term]}) "
                    f"for phrase {term!r}")
        # order holds after layout (explicit-at phrases participate)
        for a, b in self._order:
            pa, pb = blocks.get(a), blocks.get(b)
            if pa is not None and pb is not None and pa >= pb:
                raise ContractError(
                    f"before({a!r}, {b!r}) violated by pinned positions")

        # --- 8. space check ---
        # anchors now include phrase blocks; required singles need slots.
        free = total - prompt_len - len(occupied)
        need = sum(required_counts.values())
        if free < need:
            raise ContractError(
                f"not enough free positions: {need} required token "
                f"occurrences but only {free} free slots")

        # --- 9. emit ---
        constraints = {}
        # anchors from locks already in `anchors`; add phrase blocks
        for text, pos in blocks.items():
            ids = units[text]["ids"]
            for p, t in enumerate(ids):
                anchors[pos + p] = t
        if anchors:
            constraints["anchors"] = anchors
        if required_counts:
            constraints["required_counts"] = required_counts
            # backward-compat set form
            constraints["required_ids"] = sorted(required_counts)
        if forbidden_ids:
            constraints["forbidden_ids"] = forbidden_ids
        if at_most:
            constraints["at_most"] = at_most
        if forbidden_phrases:
            constraints["forbidden_phrases"] = [
                (list(ids), m) for ids, m in forbidden_phrases]
        order_pairs = []
        for a, b in self._order:
            order_pairs.append((units[a]["ids"], units[b]["ids"]))
        if order_pairs:
            constraints["order_pairs"] = order_pairs
        return constraints, total

    # -- verification ----------------------------------------------------

    @staticmethod
    def verify(ids, constraints, mask_id):
        """Machine-check the finished canvas. Returns the check dict."""
        return check_constraints(ids, constraints, mask_id)

    @staticmethod
    def holds(check):
        """True when every applicable guarantee verified."""
        keys = ("required_present", "forbidden_absent", "anchors_intact",
                "counts_met", "caps_met", "phrases_ok", "order_ok")
        return all(check.get(k, True) for k in keys)
