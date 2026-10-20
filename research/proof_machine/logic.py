"""
Boolean-expression certificates for the Compositional Proof Machine
(spec section 5.6, task 1: logic chains).

Domain: 8-bit vectors (values 0..255), LSB-first bit 0.
Each primitive is a unary bit-vector op (possibly with a fixed constant
operand, exactly as ADD_k carries a constant k in Phase 1).

A certificate is an abstract-interpretation trace over a symbolic boolean
domain: the abstraction of the 8-bit register is an 8-tuple of boolean
expressions over the input bits b0..b7. For this op set the symbolic
transformers are EXACT (no precision loss), so the trace is a chain of
canonical boolean expressions -- the boolean analogue of certificates.py.

Expression representation: nested tuples (JSON-serializable as lists).
    ("bconst", b)          b in (0, 1)
    ("bvar", i)            input bit i (LSB-first)
    ("bnot", e)
    ("band", (e1..en))     n >= 2, canonical: flattened, const-folded,
    ("bor",  (e1..en))     deduped, complement-annihilated, sorted
    ("bxor", (e1..en))

The VERIFIER below is structurally independent from the prover: it
re-derives every step from the recorded input vector using its own rule
table and never trusts the prover's recorded outputs. Verification is
linear in |trace| (number of steps); each step re-derives 8 bit exprs.
"""

W = 8
MASK = (1 << W) - 1


# ---------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------

def int_to_bits(x):
    """int in [0,255] -> tuple of W bits, LSB-first."""
    x = int(x) & MASK
    return tuple((x >> i) & 1 for i in range(W))


def bits_to_int(bits):
    v = 0
    for i, b in enumerate(bits):
        v |= (int(b) & 1) << i
    return v


# ---------------------------------------------------------------------------
# Canonical constructors (deterministic: prover and verifier must agree)
# ---------------------------------------------------------------------------

def bconst(b):
    return ("bconst", 1 if b else 0)


def bvar(i):
    assert 0 <= i < W, f"bit index {i} out of range"
    return ("bvar", i)


def _complement(e):
    # canonical forms never contain bnot(bnot(_)) (folded at construction)
    return e[1] if e[0] == "bnot" else ("bnot", e)


def _flatten(tag, e):
    if e[0] == tag:
        out = []
        for o in e[1]:
            out.extend(_flatten(tag, o))
        return out
    return [e]


def bnot(e):
    tag = e[0]
    if tag == "bconst":
        return ("bconst", 1 - e[1])
    if tag == "bnot":
        return e[1]  # double negation
    return ("bnot", e)


def band(e1, e2):
    ops = _flatten("band", e1) + _flatten("band", e2)
    uniq, seen = [], set()
    for o in ops:
        if o == ("bconst", 0):
            return ("bconst", 0)          # annihilator
        if o == ("bconst", 1):
            continue                       # identity
        if _complement(o) in seen:
            return ("bconst", 0)          # e & ~e = 0
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    if not uniq:
        return ("bconst", 1)
    if len(uniq) == 1:
        return uniq[0]
    uniq.sort(key=canon)
    return ("band", tuple(uniq))


def bor(e1, e2):
    ops = _flatten("bor", e1) + _flatten("bor", e2)
    uniq, seen = [], set()
    for o in ops:
        if o == ("bconst", 1):
            return ("bconst", 1)          # annihilator
        if o == ("bconst", 0):
            continue                       # identity
        if _complement(o) in seen:
            return ("bconst", 1)          # e | ~e = 1
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    if not uniq:
        return ("bconst", 0)
    if len(uniq) == 1:
        return uniq[0]
    uniq.sort(key=canon)
    return ("bor", tuple(uniq))


def bxor(e1, e2):
    ops = _flatten("bxor", e1) + _flatten("bxor", e2)
    parity = 0
    counts = {}
    for o in ops:
        if o[0] == "bconst":
            parity ^= o[1]
        else:
            counts[o] = counts.get(o, 0) + 1
    kept = [o for o, c in counts.items() if c % 2 == 1]  # e ^ e = 0
    if parity:
        kept.append(("bconst", 1))
    if parity and len(kept) == 2 and kept[-1] == ("bconst", 1):
        return bnot(kept[0])  # e ^ 1 = ~e
    if not kept:
        return ("bconst", 0)
    if len(kept) == 1:
        return kept[0]
    kept.sort(key=canon)
    return ("bxor", tuple(kept))


# ---------------------------------------------------------------------------
# Canonical rendering / evaluation
# ---------------------------------------------------------------------------

def _par(e):
    return canon(e) if e[0] in ("bconst", "bvar") else "(" + canon(e) + ")"


def canon(e):
    """Render a bit-expression to its canonical string form."""
    tag = e[0]
    if tag == "bconst":
        return str(e[1])
    if tag == "bvar":
        return f"b{e[1]}"
    if tag == "bnot":
        return "~" + _par(e[1])
    if tag == "band":
        return "(" + "&".join(canon(o) for o in e[1]) + ")"
    if tag == "bor":
        return "(" + "|".join(canon(o) for o in e[1]) + ")"
    if tag == "bxor":
        return "(" + "^".join(canon(o) for o in e[1]) + ")"
    raise ValueError(f"bad bexpr {e}")


def canon_bvec(v):
    """Human-readable vector: 0xNN when fully concrete, else bit exprs."""
    if all(b[0] == "bconst" for b in v):
        return f"0x{bits_to_int(b[1] for b in v):02X}"
    return "[" + ",".join(canon(b) for b in v) + "]"


def _ev(e, bits):
    tag = e[0]
    if tag == "bconst":
        return e[1]
    if tag == "bvar":
        return bits[e[1]]
    if tag == "bnot":
        return 1 - _ev(e[1], bits)
    if tag == "band":
        r = 1
        for o in e[1]:
            r &= _ev(o, bits)
        return r
    if tag == "bor":
        r = 0
        for o in e[1]:
            r |= _ev(o, bits)
        return r
    if tag == "bxor":
        r = 0
        for o in e[1]:
            r ^= _ev(o, bits)
        return r
    raise ValueError(f"bad bexpr {e}")


def eval_bexpr(e, x0):
    """Exact evaluation of a bit-expression at input value x0."""
    return _ev(e, int_to_bits(int(x0)))


def eval_bvec(v, x0):
    """Evaluate a bit-vector abstraction at x0 -> int."""
    bits = int_to_bits(int(x0))
    return bits_to_int(_ev(b, bits) for b in v)


def bexpr_size(e):
    tag = e[0]
    if tag in ("bconst", "bvar"):
        return 1
    if tag == "bnot":
        return 1 + bexpr_size(e[1])
    return 1 + sum(bexpr_size(o) for o in e[1])


def bvec_size(v):
    return sum(bexpr_size(b) for b in v)


# ---------------------------------------------------------------------------
# Fixed symbolic abstract transformers (the rule table).
# Sound by construction for the task-1 op set; NOT learned.
# Each maps a W-tuple of bit-expressions to a W-tuple (LSB-first).
# ---------------------------------------------------------------------------

def _t_not(v):
    return tuple(bnot(b) for b in v)


def _t_xor_c(c):
    cb = int_to_bits(c)
    def t(v):
        return tuple(bxor(b, bconst(cb[i])) for i, b in enumerate(v))
    return t


def _t_and_c(c):
    cb = int_to_bits(c)
    def t(v):
        return tuple(band(b, bconst(cb[i])) for i, b in enumerate(v))
    return t


def _t_or_c(c):
    cb = int_to_bits(c)
    def t(v):
        return tuple(bor(b, bconst(cb[i])) for i, b in enumerate(v))
    return t


def _t_shl1(v):
    return (bconst(0),) + v[:W - 1]


def _t_shr1(v):
    return v[1:] + (bconst(0),)


def _t_rol1(v):
    return (v[W - 1],) + v[:W - 1]


RULES_BOOL = {
    "NOT": _t_not,
    "XOR_A5": _t_xor_c(0xA5),
    "XOR_3C": _t_xor_c(0x3C),
    "AND_F0": _t_and_c(0xF0),
    "OR_0F": _t_or_c(0x0F),
    "SHL_1": _t_shl1,
    "SHR_1": _t_shr1,
    "ROL_1": _t_rol1,
}

BOOL_OP_NAMES = list(RULES_BOOL.keys())

START_VEC = tuple(bvar(i) for i in range(W))


# ---------------------------------------------------------------------------
# JSON (de)serialization: tuples <-> lists
# ---------------------------------------------------------------------------

def bexpr_to_json(e):
    tag = e[0]
    if tag == "bconst":
        return ["bconst", e[1]]
    if tag == "bvar":
        return ["bvar", e[1]]
    if tag == "bnot":
        return ["bnot", bexpr_to_json(e[1])]
    if tag in ("band", "bor", "bxor"):
        return [tag, [bexpr_to_json(o) for o in e[1]]]
    raise ValueError(f"bad bexpr {e}")


def bexpr_from_json(j):
    tag = j[0]
    if tag == "bconst":
        return ("bconst", int(j[1]))
    if tag == "bvar":
        return ("bvar", int(j[1]))
    if tag == "bnot":
        return ("bnot", bexpr_from_json(j[1]))
    if tag in ("band", "bor", "bxor"):
        return (tag, tuple(bexpr_from_json(o) for o in j[1]))
    raise ValueError(f"bad bexpr json {j}")


def vec_to_json(v):
    return [bexpr_to_json(b) for b in v]


def vec_from_json(j):
    v = tuple(bexpr_from_json(b) for b in j)
    assert len(v) == W, f"vector width {len(v)} != {W}"
    return v


# ---------------------------------------------------------------------------
# Independent verifier (boolean analogue of verify_certificate)
# ---------------------------------------------------------------------------

def _verify_bstep(s, i, prev_out, x0):
    op = s["op"]
    if op not in RULES_BOOL:
        return False, {"error": f"unknown op {op} at step {i}"}, None
    in_v = vec_from_json(s["in"])
    out_v = vec_from_json(s["out"])
    derived = RULES_BOOL[op](in_v)   # verifier applies its OWN rule table
    if derived != out_v:
        return False, {"error": f"step {i} ({op}): re-derivation mismatch",
                       "derived": canon_bvec(derived),
                       "claimed": canon_bvec(out_v)}, None
    if prev_out is not None and in_v != prev_out:
        return False, {"error": f"step {i}: input != previous output"}, None
    return True, {}, out_v


def verify_btrace(steps, x0, start_vec, expected_result):
    """
    Mechanical check of a boolean trace. Checks (each linear in |trace|):
      1. steps[0].in == start_vec
      2. RULES_BOOL[op](in) == out   (re-derived, not trusted)
      3. out_k == in_{k+1}           (chaining)
      4. eval(out_last, x0) == expected_result   (skipped when None)
    """
    try:
        x0 = int(x0) & MASK
        if not steps:
            return False, {"error": "empty trace"}
        first_in = vec_from_json(steps[0]["in"])
        if first_in != start_vec:
            return False, {"error": "trace does not start at claimed input"}
        prev_out = None
        max_nodes = 0
        for i, s in enumerate(steps):
            ok, info, out_v = _verify_bstep(s, i, prev_out, x0)
            if not ok:
                info["step_index"] = i
                return False, info
            prev_out = out_v
            max_nodes = max(max_nodes, bvec_size(out_v))
        if expected_result is not None and \
                eval_bvec(prev_out, x0) != int(expected_result):
            return False, {
                "error": "final vector does not evaluate to claimed result",
                "eval": eval_bvec(prev_out, x0),
                "claimed": int(expected_result)}
        return True, {"steps": len(steps), "max_expr_nodes": max_nodes,
                      "final": canon_bvec(prev_out),
                      "final_tuple": prev_out}
    except Exception as ex:  # a malformed certificate fails closed
        return False, {"error": f"verifier exception: {type(ex).__name__}: {ex}"}


def verify_bcertificate(cert):
    """
    Mechanically check a boolean certificate. Returns (ok, info).
    Checks: trace starts at the input-bit vector; every transformer
    application re-derived; chaining; final vector evaluates to the
    claimed integer result.
    """
    try:
        x0 = int(cert["x0"])
        result = int(cert["result"])
        return verify_btrace(cert["steps"], x0, START_VEC, result)
    except Exception as ex:
        return False, {"error": f"verifier exception: {type(ex).__name__}: {ex}"}
