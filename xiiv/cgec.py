"""Certificate-Guided Early Commit (CGEC) for masked-diffusion decoding.

A sampler that fuses three 2025-2026 results into something none of them
does, and emits a machine-checkable *certificate* with every generation:

- Prophet (ICLR 2026): confidence-gap early commit -- when every
  remaining position in the block is high-gap, go "all-in" and fill them
  all in one step instead of burning more forward passes.
- Mündler / Dekoninck / Vechev (2025): grammatical constraints applied
  directly to *diffusion* decoding (not ported from AR).
- XGrammar (MLSys 2025): incremental grammar checking at microseconds
  per token.

The novelty: no prior work emits per-generation (tokens, grammar-state
trace, min confidence gap) certificates for diffusion LMs. Prophet
commits without a parser, Mündler constrains without certifying
confidence, XGrammar is AR-only. Here the certificate is replayable by
an *independent* verifier (xiiv/cert_verify.py) WITHOUT the model
weights -- "every answer carries its proof" as a literal artifact.

Domain 1: arithmetic expressions. The grammar gate certifies, at every
commit, that the position-ordered committed subsequence can still be
completed to a full valid expression given the remaining masked
positions -- checked by exact forward/backward PDA reachability
(Dang & Ermon 2026: constraint satisfaction by construction for dLLMs;
here via reachability tables rather than graphical-model inference).

Grammar (symbol classes; one class per committed token):
    E := T (('+'|'-'|'*'|'/') T)*
    T := F (('*'|'/') F)*          [same transitions as E at PDA level]
    F := digit | '(' E ')'
i.e. standard precedence, single digits, fully explicit parens.
PDA state = (expect_operand: bool, paren_depth: int).
A complete expression always has ODD length (1 token, +2 per binary op
or paren pair), so the arithmetic domain requires odd canvas length --
enforced fail-closed at sampler start.

The sampler is generic over the *domain*: the caller supplies
``alphabet`` ({token_id: symbol_class}) and the transition/accept
functions live in the domain spec below. Only token ids present in the
alphabet can ever be committed in cgec mode -- the model proposing
non-domain tokens gets them deferred, and a canvas the model cannot
complete inside the domain ends ``completed=False`` (fail-closed, never
a half-certified emit).

Pre-registered falsifiers (see tests/test_cgec.py):
- F1 soundness: scripted valid-expression model -> completed, final
  canvas parses complete, cert_verify says valid.
- F2 gate: scripted saboteur (top-1 always grammar-invalid) ->
  completed anyway via lower-ranked candidates, gate_rejects > 0,
  final canvas still parses complete.
- F3 early commit: confident scripted model -> early_commits > 0 and
  forward_passes strictly below the vanilla policy on the same plan.
- F4 verifier: tampered certificate -> verify returns valid=False.
- F5 grammar unit: PDA accepts/rejects a fixed battery.
- KILL: F1 or F4 fails -> the module does not ship.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.
"""

import json
import os

import torch
import torch.nn.functional as F

__all__ = [
    "CGECConfig",
    "CGECTrace",
    "POLICIES",
    "ARITH_CLASSES",
    "ARITH_OPS",
    "arith_transition",
    "arith_accept",
    "arith_valid_length",
    "arithmetic_alphabet_32k",
    "is_feasible",
    "cgec_sample",
    "cgec_certificate",
]

POLICIES = ("vanilla", "cgec")

# ---------------------------------------------------------------------------
# Domain 1: arithmetic expressions ("arith-v1")
# ---------------------------------------------------------------------------

ARITH_CLASSES = ("digit", "plus", "minus", "mul", "div", "lparen", "rparen")
ARITH_OPS = frozenset({"plus", "minus", "mul", "div"})
ARITH_DOMAIN = "arith-v1"


def arith_transition(state, symbol):
    """One PDA step. state = (expect_operand: bool, depth: int).

    Returns the next state, or None if the symbol is rejected here.
    Unknown symbol classes are rejected (closed world).
    """
    expect_operand, depth = state
    if symbol == "digit":
        return (False, depth) if expect_operand else None
    if symbol == "lparen":
        return (True, depth + 1) if expect_operand else None
    if symbol in ARITH_OPS:
        return (True, depth) if not expect_operand else None
    if symbol == "rparen":
        if not expect_operand and depth > 0:
            return (False, depth - 1)
        return None
    return None


def arith_accept(state):
    """A complete expression: operand just placed, no open parens."""
    expect_operand, depth = state
    return (not expect_operand) and depth == 0


def arith_valid_length(length):
    """Every complete arith-v1 expression has odd token length."""
    return length % 2 == 1


def _classes_at(x_list, pos, alphabet, mask_id):
    tok = x_list[pos]
    if tok == mask_id:
        return ARITH_CLASSES  # wildcard: any class is committable
    return (alphabet[tok],)


def _forward_tables(x_list, alphabet, mask_id):
    """F[i] = PDA states reachable after some assignment to prefix [0, i)."""
    L = len(x_list)
    F = [set() for _ in range(L + 1)]
    F[0].add((True, 0))
    for i in range(L):
        for st in F[i]:
            for cls in _classes_at(x_list, i, alphabet, mask_id):
                ns = arith_transition(st, cls)
                if ns is not None:
                    F[i + 1].add(ns)
    return F


def _predecessors(state, symbol):
    """Invert arith_transition analytically: all (expect_operand, depth)
    states that step to ``state`` on ``symbol``. O(1)."""
    eo2, d2 = state
    if symbol == "digit":
        # needs expect_operand=True -> (False, d)
        return {(True, d2)} if not eo2 else set()
    if symbol == "lparen":
        # needs expect_operand=True -> (True, d+1)
        return {(True, d2 - 1)} if (eo2 and d2 >= 1) else set()
    if symbol in ARITH_OPS:
        # needs expect_operand=False -> (True, d)
        return {(False, d2)} if eo2 else set()
    if symbol == "rparen":
        # needs expect_operand=False, d>0 -> (False, d-1)
        return {(False, d2 + 1)} if not eo2 else set()
    return set()


def _backward_tables(x_list, alphabet, mask_id):
    """B[i] = PDA states from which some assignment to suffix [i, L)
    reaches accept."""
    L = len(x_list)
    B = [set() for _ in range(L + 1)]
    B[L].add((False, 0))
    for i in range(L - 1, -1, -1):
        for cls in _classes_at(x_list, i, alphabet, mask_id):
            for st2 in B[i + 1]:
                B[i] |= _predecessors(st2, cls)
    return B


def _candidate_ok(pos, cls, F, B):
    """Would committing class ``cls`` at ``pos`` keep a completion alive?"""
    for st in F[pos]:
        ns = arith_transition(st, cls)
        if ns is not None and ns in B[pos + 1]:
            return True
    return False


def is_feasible(x_list, alphabet, mask_id):
    """Exact check: can the masked positions be filled so the whole
    canvas parses as a complete arith-v1 expression?"""
    F = _forward_tables(x_list, alphabet, mask_id)
    return (False, 0) in F[len(x_list)]


def arithmetic_alphabet_32k(tokenizer_path=None):
    """Build the arith-v1 alphabet from the real XIIV 32k tokenizer.

    Maps single-char token ids -> symbol classes. MEASURED 2026-11-01:
    digits 0-9, +, -, /, (, ) exist as single tokens; ``*`` is ABSENT
    from the vocabulary entirely, so ``x`` (id 79) serves as the
    multiply symbol -- documented here, not silently assumed.
    Every mapping is round-trip verified (id decodes back to the char).
    """
    if tokenizer_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(here, "..", "weights",
                                      "tokenizer_32k.json")
    with open(tokenizer_path, encoding="utf-8") as fh:
        tok = json.load(fh)
    vocab = tok["model"]["vocab"]
    inv = {i: t for t, i in vocab.items()}

    def find(char, symbol):
        tid = vocab.get(char)
        assert tid is not None, f"token {char!r} missing from 32k vocab"
        assert inv[tid] == char, \
            f"token id {tid} round-trips to {inv[tid]!r}, not {char!r}"
        return tid, symbol

    pairs = []
    for d in "0123456789":
        pairs.append(find(d, "digit"))
    pairs.append(find("+", "plus"))
    pairs.append(find("-", "minus"))
    # '*' has no token in this vocab (measured); 'x' is the multiply key.
    assert "*" not in vocab, "vocab changed: '*' now exists, remap multiply"
    pairs.append(find("x", "mul"))
    pairs.append(find("/", "div"))
    pairs.append(find("(", "lparen"))
    pairs.append(find(")", "rparen"))
    return {tid: sym for tid, sym in pairs}


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class CGECConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        block_size: int = 32,
        tau: float = 0.95,     # Clover-style: parallel-fill threshold
        gamma: float = 0.10,   # Prophet-style: all-in confidence gap
        topk: int = 4,         # candidates scanned per position
        max_steps: int = 128,  # hard termination cap
        stall_budget: int = 3, # zero-progress steps before giving up
    ):
        self.block_size = block_size
        self.tau = tau
        self.gamma = gamma
        self.topk = topk
        self.max_steps = max_steps
        self.stall_budget = stall_budget


class CGECTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.steps = 0
        self.forward_passes = 0
        self.commits = 0
        self.early_commits = 0   # tokens committed via Prophet all-in
        self.gate_rejects = 0    # candidates refused by the grammar gate
        self.defers = 0          # positions with no feasible candidate
        self.min_gap = None      # min top1-top2 gap over committed tokens
        self.grammar_valid = False
        self.completed = False
        self.events = []

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "events"}
        d["n_events"] = len(self.events)
        return d


def _topk_for_pos(probs_row: torch.Tensor, k: int, alphabet: dict,
                  mask_id: int):
    """Top-k (token, prob) pairs restricted to the domain alphabet.

    Tokens outside the alphabet can never be committed in cgec mode.
    Deterministic.
    """
    row = probs_row.clone()
    allowed = torch.tensor(sorted(alphabet.keys()), dtype=torch.long)
    mask = torch.zeros_like(row, dtype=torch.bool)
    mask[allowed] = True
    mask[mask_id] = False
    row[~mask] = -1.0
    k = max(1, min(k, int(mask.sum().item())))
    if k == 0:
        return []
    vals, idx = torch.topk(row, k)
    return [(int(t), float(v)) for t, v in zip(idx.tolist(), vals.tolist())
            if float(v) > 0.0]


def cgec_sample(model, *, length: int, mask_id: int, alphabet: dict,
                config: "CGECConfig | None" = None, seed: int = 0,
                policy: str = "cgec", frozen: "dict | None" = None,
                domain: str = ARITH_DOMAIN):
    """Run one certificate-guided unmasking trajectory.

    Args:
        model: callable x(1, L) -> logits(1, L, V).
        length: canvas length (must be odd for arith-v1).
        mask_id: the mask token id.
        alphabet: {token_id: symbol_class}; the domain's fill alphabet.
        policy: "vanilla" (one highest-confidence commit per step, no
            gate -- the F3 baseline) or "cgec".
        frozen: optional {pos: token} pinned positions.
        domain: grammar domain name (only "arith-v1" today).

    Returns:
        (ids LongTensor(L), CGECTrace, certificate dict).
        Fail-closed: completed=False (never a half-certified emit) when
        the canvas cannot be completed inside the domain.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {POLICIES}")
    if domain != ARITH_DOMAIN:
        raise ValueError(f"unknown domain {domain!r}")
    cfg = config or CGECConfig()
    frozen = {int(p): int(t) for p, t in (frozen or {}).items()}
    for p, t in frozen.items():
        if t not in alphabet:
            raise ValueError(
                f"frozen token id {t} at pos {p} is outside the "
                f"{domain} alphabet -- the domain cannot certify it")

    torch.manual_seed(seed)
    trace = CGECTrace(policy)
    x = torch.full((length,), mask_id, dtype=torch.long)
    gap = torch.zeros(length, dtype=torch.float)
    for p, t in frozen.items():
        x[int(p)] = int(t)
        gap[int(p)] = 1.0  # given, not sampled: maximal gap

    commit_log = []  # (pos, token, symbol, gap, step, early)

    def note_commit(pos, tok, cls, g, step, early):
        x[pos] = tok
        gap[pos] = g
        trace.commits += 1
        if early:
            trace.early_commits += 1
        if trace.min_gap is None or g < trace.min_gap:
            trace.min_gap = g
        commit_log.append({"pos": int(pos), "token": int(tok),
                           "symbol": cls, "gap": float(g),
                           "step": int(step), "early": bool(early)})

    # ---- fail-closed preconditions -------------------------------------
    if not arith_valid_length(length):
        trace.events.append(("bad_length",
                             {"length": length, "domain": domain}))
        cert = cgec_certificate(trace, x.tolist(), alphabet, mask_id,
                                domain, commit_log)
        return x, trace, cert
    if not is_feasible(x.tolist(), alphabet, mask_id):
        trace.events.append(("infeasible_start", {}))
        cert = cgec_certificate(trace, x.tolist(), alphabet, mask_id,
                                domain, commit_log)
        return x, trace, cert

    masked = {p for p in range(length) if int(x[p]) == mask_id}
    stall = 0
    step = 0

    def block_range(b):
        lo = b * cfg.block_size
        return lo, min(lo + cfg.block_size, length)

    n_blocks = (length + cfg.block_size - 1) // cfg.block_size
    for b in range(n_blocks):
        blo, bhi = block_range(b)
        while any(p in masked for p in range(blo, bhi)):
            if step >= cfg.max_steps:
                trace.events.append(("max_steps", {"step": step}))
                break
            step += 1
            trace.steps = step
            with torch.no_grad():
                logits = model(x.unsqueeze(0))[0]
            trace.forward_passes += 1
            probs = torch.softmax(logits.float(), dim=-1)
            top1 = probs.max(dim=-1).values
            top2 = torch.topk(probs, 2, dim=-1).values[:, 1]
            gaps = (top1 - top2)

            made = 0
            if policy == "vanilla":
                # Baseline: one commit per step, highest confidence,
                # no grammar gate.
                order = sorted((float(top1[p]), p) for p in masked
                               if blo <= p < bhi)
                if order:
                    _, pos = order[-1]
                    t1 = int(torch.argmax(probs[pos]).item())
                    note_commit(pos, t1, alphabet.get(t1, "?"),
                                float(gaps[pos]), step, False)
                    masked.discard(pos)
                    made = 1
            else:
                x_list = x.tolist()
                F, B = _forward_tables(x_list, alphabet,
                                       mask_id), _backward_tables(
                                           x_list, alphabet, mask_id)

                def try_commit(pos):
                    """Top-k scan for the first grammar-feasible token."""
                    nonlocal F, B, x_list
                    cands = _topk_for_pos(probs[pos], cfg.topk,
                                          alphabet, mask_id)
                    for tok, pr in cands:
                        cls = alphabet[tok]
                        if _candidate_ok(pos, cls, F, B):
                            note_commit(pos, tok, cls,
                                        float(gaps[pos]), step, False)
                            masked.discard(pos)
                            x_list[pos] = tok
                            F = _forward_tables(x_list, alphabet, mask_id)
                            B = _backward_tables(x_list, alphabet, mask_id)
                            return True
                        trace.gate_rejects += 1
                    trace.defers += 1
                    return False

                # Phase 1 (Clover): parallel-fill confident positions.
                block_masked = sorted(p for p in masked
                                      if blo <= p < bhi)
                confident = [p for p in block_masked
                             if float(top1[p]) >= cfg.tau]
                # highest confidence first: early commits constrain most
                confident.sort(key=lambda p: -float(top1[p]))
                for pos in confident:
                    if pos in masked and try_commit(pos):
                        made += 1

                # Phase 2 (Prophet): all-in one-step fill when every
                # remaining masked position in the block is high-gap.
                rest = [p for p in masked if blo <= p < bhi]
                if rest and all(float(gaps[p]) >= cfg.gamma
                                for p in rest):
                    trace.events.append(("all_in",
                                         {"block": b, "n": len(rest)}))
                    for pos in sorted(rest):
                        if pos not in masked:
                            continue
                        cands = _topk_for_pos(probs[pos], cfg.topk,
                                              alphabet, mask_id)
                        placed = False
                        for tok, _pr in cands:
                            cls = alphabet[tok]
                            if _candidate_ok(pos, cls, F, B):
                                note_commit(pos, tok, cls,
                                            float(gaps[pos]), step, True)
                                masked.discard(pos)
                                x_list[pos] = tok
                                F = _forward_tables(x_list, alphabet,
                                                    mask_id)
                                B = _backward_tables(x_list, alphabet,
                                                     mask_id)
                                made += 1
                                placed = True
                                break
                            trace.gate_rejects += 1
                        if not placed:
                            # Confident but grammar-infeasible: stop the
                            # all-in; the model's confidence was misplaced.
                            trace.events.append(("all_in_abort",
                                                 {"pos": pos}))
                            trace.defers += 1
                            break

            if made > 0:
                stall = 0
            else:
                stall += 1
                if stall >= cfg.stall_budget:
                    trace.events.append(("stall_exit",
                                         {"step": step, "block": b}))
                    break
        if stall >= cfg.stall_budget or step >= cfg.max_steps:
            break

    trace.completed = not masked
    x_list = x.tolist()
    if trace.completed and policy == "cgec":
        # Final exact check: the full canvas parses complete.
        F = _forward_tables(x_list, alphabet, mask_id)
        trace.grammar_valid = (False, 0) in F[length]
        trace.events.append(("final_check",
                             {"grammar_valid": trace.grammar_valid}))
        # Sound fail direction: a completed cgec canvas that does not
        # parse must never be presented as certified.
        assert trace.grammar_valid, \
            "completed cgec canvas does not parse -- gate unsound"
    elif trace.completed:
        trace.grammar_valid = is_feasible(x_list, alphabet, mask_id)

    cert = cgec_certificate(trace, x_list, alphabet, mask_id, domain,
                            commit_log)
    return x, trace, cert


def cgec_certificate(trace: CGECTrace, x_list: list, alphabet: dict,
                     mask_id: int, domain: str, commit_log: list) -> dict:
    """Emit the machine-checkable certificate for a run.

    Contents: final tokens, the per-commit trace (position, token,
    symbol, confidence gap, step, early-commit flag), the minimum gap,
    and the run verdict. The independent verifier (xiiv/cert_verify.py)
    replays this without model weights.
    """
    symbols = [alphabet.get(t, "?") if t != mask_id else "<mask>"
               for t in x_list]
    return {
        "certificate": "cgec-v1",
        "domain": domain,
        "policy": trace.policy,
        "length": len(x_list),
        "mask_id": mask_id,
        "alphabet": {str(k): v for k, v in alphabet.items()},
        "tokens": list(x_list),
        "symbols": symbols,
        "commits": commit_log,
        "min_gap": trace.min_gap,
        "steps": trace.steps,
        "forward_passes": trace.forward_passes,
        "n_commits": trace.commits,
        "early_commits": trace.early_commits,
        "gate_rejects": trace.gate_rejects,
        "defers": trace.defers,
        "grammar_valid": trace.grammar_valid,
        "completed": trace.completed,
    }
