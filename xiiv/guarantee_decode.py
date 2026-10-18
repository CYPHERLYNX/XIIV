"""Guarantee-Constrained Diffusion Decoding (GCDD) for XIIV.

LAVE-style lookahead-then-verify constrained decoding
(Zhang et al., "Lookahead-then-Verify: Reliable Constrained Decoding for
Diffusion LLMs under Context-Free Grammars", arXiv 2602.00612, 2026),
adapted for XIIV's masked-diffusion sampler and fused with XIIV's semantic
contract system.

What this module does
---------------------
During masked-diffusion denoising, every token the sampler wants to commit
is checked *before* it becomes permanent:

1. SYNTACTIC lookahead (this module's port of LAVE): the hypothetical canvas
   — current tokens plus the candidate, remaining [MASK]s as wildcards — is
   tested for JSON viability. A commit that provably destroys the
   possibility of a valid JSON output is vetoed.
2. SEMANTIC lookahead (XIIV's existing ``DenoisingState.contract_satisfiable``,
   which already cites LAVE/DINGO as the syntactic prior art it complements):
   required/forbidden/at-most contract predicates must stay satisfiable.

The two checks form ONE per-commit veto. Vetoed commits are resampled
(bounded attempts) and then deferred — never forced. Every veto, deferral,
and endgame conflict is appended to ``DenoisingState.interventions``, so the
guarantee-native metric (intervention rate) keeps measuring exactly this.

Prior-art honesty
-----------------
The lookahead-then-verify *mechanism* is published prior art (LAVE, 2026;
DINGO, 2025, for the constrained-decoding half). XIIV's original step is
the fusion: syntactic viability and semantic contract predicates enforced
as a single veto inside a masked-diffusion sampler, with the veto trail
feeding the intervention-rate metric. Documented as a principled port with
an original composition — not a from-scratch invention.

Semantics of the viability check (read before relying on it)
-------------------------------------------------------------
``JsonViabilityChecker.viable(text)`` answers: "could the masked slots of
this canvas still be completed into a valid JSON document?" Masked slots are
rendered as a wildcard character and modelled as matching **any string**
(Kleene star, possibly empty). Consequences:

* SOUND REJECTION (the guarantee direction): if ``viable()`` returns False,
  then NO assignment of strings to the masked slots can make the output
  valid JSON. The decoder therefore never commits a token that provably
  kills JSON validity. Vetoes are never wrong.
* CONSERVATIVE ACCEPTANCE: ``viable()`` may return True for canvases whose
  completion would need multi-character expansions of a single slot (a real
  token decodes to 1+ characters; the star model over-approximates). That
  only means a missed veto opportunity — never an unsound one.
* ABSTAIN ON TRUNCATION: parser stacks deeper than ``max_depth`` (or a
  runaway reprocess chain) make the check return True instead of deciding.
  Unknown is never reported as impossible.

Two further approximations, both documented rather than hidden:

* The check runs on *decoded text*. Tokenizers are not perfectly
  compositional (``decode([a, b])`` may differ from
  ``decode([a]) + decode([b])`` at BPE boundaries), so the examined text is
  an approximation of the final string. In practice the divergence is at
  whitespace/punctuation seams and cannot fabricate the *structural*
  contradictions this checker vetoes on.
* The representative alphabet used for wildcard reachability (``_REP``) is
  complete in the following sense: every Unicode character behaves, in every
  parser configuration, identically to some member of ``_REP`` — or reaches
  only configurations also reachable via ``_REP`` strings. Digits are covered
  by ``0``/``1`` (leading-zero vs nonzero), case variants by ``e``/``E``,
  whitespace by space, and every other character by its structural twin or
  by the generic string-content representative ``a``. Dropping a character
  class would only shrink the explored reachability set, which is why the
  set above is fixed and asserted non-empty in tests.

CPU-only, no training, no extra RAM: the checker is a small pushdown
automaton; per-commit cost is O(canvas length x live configurations), and
wildcard reachability is memoized per checker instance (one warm-up cost,
then fast).
"""

import torch

WILDCARD = "\x00"  # renders a masked slot; no real tokenizer emits NUL

_WS = (" ", "\t", "\n", "\r")
_HEX = frozenset("0123456789abcdefABCDEF")

# Representative alphabet for wildcard (Kleene-star) reachability.
# See the module docstring for the completeness argument.
_REP = tuple('{}[],:"\\-01.eE+ trufalsn')


class JsonViabilityChecker:
    """Incremental JSON-grammar viability with wildcard (masked-slot) support.

    Deterministic pushdown automaton for strict JSON (objects, arrays,
    strings with escapes incl. \\uXXXX, numbers, true/false/null,
    insignificant whitespace). A wildcard character stands for "any string";
    reachability under wildcards is computed by BFS over the automaton's
    transition graph, memoized per configuration.
    """

    def __init__(self, max_depth: int = 24, wildcard: str = WILDCARD):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if len(wildcard) != 1:
            raise ValueError("wildcard must be a single character")
        self.max_depth = int(max_depth)
        self.wildcard = wildcard
        self._reach_cache: dict = {}
        self._truncated = False

    # -- pushdown automaton ------------------------------------------------
    # Frames: "OBJ" (inside {...}), "ARR" (inside [...]), "VAL" (a scalar
    # value being built), "KEY" (an object key string being built),
    # "OBJV" (object value slot, pushed by ':').
    # Modes: "VAL", "OBJ_OPEN", "OBJ_COLON", "OBJ_AFTER", "ARR_OPEN",
    # "ARR_AFTER", "STR", "ESC", "DONE", ("NUM", kind), ("LIT", word, idx),
    # ("ESCU", k).

    def _finish(self, stack):
        """A value completed; stack[-1] is its frame. Returns (stack, mode)."""
        if not stack:
            return None
        frame, rest = stack[-1], stack[:-1]
        if frame not in ("VAL", "OBJ", "ARR"):
            return None
        if not rest:
            return ((), "DONE")
        top = rest[-1]
        if top == "ARR":
            return (rest, "ARR_AFTER")
        if top == "OBJV":
            if len(rest) < 2 or rest[-2] != "OBJ":
                return None
            return (rest[:-1], "OBJ_AFTER")
        return None

    def _end_number(self, stack):
        r = self._finish(stack)
        return None if r is None else (r[0], r[1], True)  # reprocess char

    def _trans_val(self, stack, ch):
        if ch in _WS:
            return (stack, "VAL", False)
        if ch == "{":
            return (stack + ("OBJ",), "OBJ_OPEN", False)
        if ch == "[":
            return (stack + ("ARR",), "ARR_OPEN", False)
        if ch == '"':
            return (stack + ("VAL",), "STR", False)
        if ch == "-":
            return (stack + ("VAL",), ("NUM", "neg"), False)
        if ch == "0":
            return (stack + ("VAL",), ("NUM", "zero"), False)
        if ch in "123456789":
            return (stack + ("VAL",), ("NUM", "int"), False)
        if ch == "t":
            return (stack + ("VAL",), ("LIT", "true", 1), False)
        if ch == "f":
            return (stack + ("VAL",), ("LIT", "false", 1), False)
        if ch == "n":
            return (stack + ("VAL",), ("LIT", "null", 1), False)
        return None

    def _trans(self, stack, mode, ch):
        """One PDA step. Returns (stack, mode, reprocess) or None if dead."""
        kind = mode[0] if isinstance(mode, tuple) else mode
        if kind == "VAL":
            return self._trans_val(stack, ch)
        if kind == "ARR_OPEN":
            if ch == "]":
                r = self._finish(stack)
                return None if r is None else (r[0], r[1], False)
            if ch in _WS:
                return (stack, mode, False)
            return self._trans_val(stack, ch)
        if kind == "OBJ_OPEN":
            if ch in _WS:
                return (stack, mode, False)
            if ch == "}":
                r = self._finish(stack)
                return None if r is None else (r[0], r[1], False)
            if ch == '"':
                return (stack + ("KEY",), "STR", False)
            return None
        if kind == "OBJ_COLON":
            if ch in _WS:
                return (stack, mode, False)
            if ch == ":":
                return (stack + ("OBJV",), "VAL", False)
            return None
        if kind == "OBJ_AFTER":
            if ch in _WS:
                return (stack, mode, False)
            if ch == ",":
                return (stack, "OBJ_COMMA", False)
            if ch == "}":
                r = self._finish(stack)
                return None if r is None else (r[0], r[1], False)
            return None
        if kind == "OBJ_COMMA":
            # Strict JSON: after ',' a key is REQUIRED ('}' is dead).
            if ch in _WS:
                return (stack, mode, False)
            if ch == '"':
                return (stack + ("KEY",), "STR", False)
            return None
        if kind == "ARR_AFTER":
            if ch in _WS:
                return (stack, mode, False)
            if ch == ",":
                return (stack, "ARR_COMMA", False)
            if ch == "]":
                r = self._finish(stack)
                return None if r is None else (r[0], r[1], False)
            return None
        if kind == "ARR_COMMA":
            # Strict JSON: after ',' a value is REQUIRED (']' is dead).
            if ch in _WS:
                return (stack, mode, False)
            return self._trans_val(stack, ch)
        if kind == "STR":
            if ch == '"':
                if not stack:
                    return None
                top, rest = stack[-1], stack[:-1]
                if top == "KEY":
                    if not rest or rest[-1] != "OBJ":
                        return None
                    return (rest, "OBJ_COLON", False)
                if top == "VAL":
                    r = self._finish(stack)
                    return None if r is None else (r[0], r[1], False)
                return None
            if ch == "\\":
                return (stack, "ESC", False)
            if ord(ch) < 0x20:
                return None  # strict JSON: no unescaped control chars
            return (stack, mode, False)
        if kind == "ESC":
            if ch == "u":
                return (stack, ("ESCU", 0), False)
            if ch in '"\\/bfnrt':
                return (stack, "STR", False)
            return None
        if kind == "ESCU":
            _, k = mode
            if ch in _HEX:
                nmode = ("ESCU", k + 1) if k + 1 < 4 else "STR"
                return (stack, nmode, False)
            return None
        if kind == "NUM":
            _, nkind = mode
            if nkind == "neg":
                if ch in "0123456789":
                    return (stack, ("NUM", "zero" if ch == "0"
                                    else "int"), False)
                return None
            if nkind == "zero":
                if ch == ".":
                    return (stack, ("NUM", "dot"), False)
                if ch in "eE":
                    return (stack, ("NUM", "exps"), False)
                return self._end_number(stack)
            if nkind == "int":
                if ch in "0123456789":
                    return (stack, mode, False)
                if ch == ".":
                    return (stack, ("NUM", "dot"), False)
                if ch in "eE":
                    return (stack, ("NUM", "exps"), False)
                return self._end_number(stack)
            if nkind == "dot":
                if ch in "0123456789":
                    return (stack, ("NUM", "frac"), False)
                return None
            if nkind == "frac":
                if ch in "0123456789":
                    return (stack, mode, False)
                if ch in "eE":
                    return (stack, ("NUM", "exps"), False)
                return self._end_number(stack)
            if nkind == "exps":
                if ch in "+-":
                    return (stack, ("NUM", "expsg"), False)
                if ch in "0123456789":
                    return (stack, ("NUM", "exp"), False)
                return None
            if nkind == "expsg":
                if ch in "0123456789":
                    return (stack, ("NUM", "exp"), False)
                return None
            if nkind == "exp":
                if ch in "0123456789":
                    return (stack, mode, False)
                return self._end_number(stack)
            return None
        if kind == "LIT":
            _, word, i = mode
            if ch == word[i]:
                if i + 1 == len(word):
                    r = self._finish(stack)
                    return None if r is None else (r[0], r[1], False)
                return (stack, ("LIT", word, i + 1), False)
            return None
        if kind == "DONE":
            return (stack, mode, False) if ch in _WS else None
        return None  # unknown mode: defensive dead

    def _step_raw(self, stack, mode, ch):
        """Apply one char incl. number-end reprocessing. None if dead.

        Sets self._truncated (abstain -> viable) if the depth bound is hit
        or a reprocess chain runs away; truncation never reports False.
        """
        for _ in range(8):
            r = self._trans(stack, mode, ch)
            if r is None:
                return None
            stack, mode, reprocess = r
            if len(stack) > self.max_depth:
                self._truncated = True
                return None
            if not reprocess:
                return (stack, mode)
        self._truncated = True
        return None

    # -- wildcard (Kleene-star) reachability --------------------------------

    def _reach(self, cfg):
        """All configs reachable from cfg by any string (depth-bounded).

        Memoized; on truncation returns None and sets self._truncated.
        Cached entries are complete (never partial), so splicing them is
        exact and rejection stays sound.
        """
        hit = self._reach_cache.get(cfg)
        if hit is not None:
            return hit
        if self._truncated:
            return None
        seen = {cfg}
        frontier = [cfg]
        while frontier:
            s, m = frontier.pop()
            for ch in _REP:
                r = self._step_raw(s, m, ch)
                if self._truncated:
                    return None
                if r is None or r in seen:
                    continue
                sub = self._reach_cache.get(r)
                if sub is not None:
                    for c2 in sub:
                        if c2 not in seen:
                            seen.add(c2)
                            frontier.append(c2)
                    continue
                seen.add(r)
                frontier.append(r)
        fz = frozenset(seen)
        self._reach_cache[cfg] = fz
        return fz

    def _close_set(self, configs):
        """Union of reachability sets. None on truncation."""
        out = set()
        for c in configs:
            r = self._reach(c)
            if r is None:
                return None
            out |= r
        return out

    # -- public API ----------------------------------------------------------

    def viable(self, text: str) -> bool:
        """True unless the canvas provably cannot become valid JSON.

        ``text`` is the decoded canvas with masked slots rendered as
        ``self.wildcard``. False is a sound rejection (no completion
        exists); True is conservative (a completion may exist, or the
        check abstained on a depth bound).
        """
        self._truncated = False
        configs = {((), "VAL")}
        for ch in text:
            if ch == self.wildcard:
                configs = self._close_set(configs)
            else:
                nxt = set()
                for (s, m) in configs:
                    r = self._step_raw(s, m, ch)
                    if r is not None:
                        nxt.add(r)
                configs = nxt
            if self._truncated:
                return True
            if configs is None or not configs:
                return False
        configs = self._close_set(configs)
        if self._truncated or configs is None:
            return True
        # A completion exists iff a full document is reachable: the final
        # wildcard already explored arbitrary insertions, so DONE must be
        # among the reachable configurations.
        return any(m == "DONE" for (_, m) in configs)


class GuaranteeLookahead:
    """One per-commit veto fusing syntactic + semantic lookahead.

    ``veto_reason(state, pos, token_id)`` returns None when the commit is
    allowed, else a short reason string ("contract_unsatisfiable" /
    "json_unviable"). ``try_commit`` wraps veto-aware resampling with a
    bounded budget; it returns None when every resample was vetoed, in
    which case the caller defers the position (leaves it masked).
    """

    def __init__(self, *, json_checker=None, token_text=None,
                 mask_id: int, check_contracts: bool = True,
                 max_veto_attempts: int = 8):
        """
        Args:
            json_checker: JsonViabilityChecker or None (syntax check off).
            token_text: callable ``int -> str`` decoding ONE token id.
                Required when json_checker is set.
            mask_id: the mask token id (rendered as wildcard).
            check_contracts: also run state.contract_satisfiable on the
                hypothetical canvas (semantic side of the veto).
            max_veto_attempts: resamples per position before deferral.
        """
        if json_checker is not None and token_text is None:
            raise ValueError("token_text is required with json_checker")
        self.json_checker = json_checker
        self.token_text = token_text
        self.mask_id = int(mask_id)
        self.check_contracts = bool(check_contracts)
        self.max_veto_attempts = int(max_veto_attempts)
        if self.max_veto_attempts < 0:
            raise ValueError("max_veto_attempts must be >= 0")

    def _canvas_text(self, state, pos: int, token_id: int) -> str:
        """Decoded canvas with the candidate placed; masks as wildcards."""
        wc = self.json_checker.wildcard
        parts = []
        for i, t in enumerate(state.x.tolist()):
            t = int(t)
            if i == pos:
                parts.append(self.token_text(int(token_id)))
            elif t == self.mask_id:
                parts.append(wc)
            else:
                parts.append(self.token_text(t))
        return "".join(parts)

    def veto_reason(self, state, pos: int, token_id: int):
        """None if the commit is allowed, else a reason string."""
        token_id = int(token_id)
        if self.check_contracts:
            x_hyp = state.x.clone()
            x_hyp[int(pos)] = token_id
            if not state.contract_satisfiable(x_hyp):
                return "contract_unsatisfiable"
        if self.json_checker is not None:
            if not self.json_checker.viable(
                    self._canvas_text(state, pos, token_id)):
                return "json_unviable"
        return None

    def json_viable_placement(self, state, pos: int, token_id: int) -> bool:
        """Syntactic half of the veto, for endgame slot preference."""
        if self.json_checker is None:
            return True
        return self.json_checker.viable(
            self._canvas_text(state, pos, token_id))

    @torch.no_grad()
    def try_commit(self, state, pos: int, p: torch.Tensor):
        """Veto-aware commit. Returns token id, or None to defer.

        Draws from ``p`` (already tempered/normalized, mask excluded) with
        the state's seeded generator — deterministic given the seed. Each
        vetoed draw is zeroed and logged to ``state.interventions``; after
        ``max_veto_attempts`` vetoes the position is deferred (caller
        leaves it masked for a later step).
        """
        p = p.clone()
        for _ in range(self.max_veto_attempts + 1):
            s = float(p.sum())
            if s <= 0:
                return None
            t = int(torch.multinomial(p / s, 1, generator=state.g).item())
            reason = self.veto_reason(state, pos, t)
            if reason is None:
                return t
            state.interventions.append({
                "step": state.steps_remaining,
                "kind": "guarantee_veto",
                "detail": {"position": int(pos), "token": t,
                           "reason": reason},
            })
            p[t] = 0.0
        return None
