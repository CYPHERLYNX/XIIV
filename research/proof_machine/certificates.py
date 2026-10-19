"""
Certificates for the Compositional Proof Machine (spec: research/composition_operator_v1.md).

A certificate is an abstract-interpretation trace: the sequence of abstract states
produced by pushing the input abstraction through each primitive's FIXED symbolic
abstract transformer. For our Phase-1 op set (ADD_k, MUL_k, NEG, MOD_m) the symbolic
domain is EXACT -- no precision loss -- so the trace is a chain of canonical
symbolic expressions over the input variable x0.

Expression representation: nested tuples (JSON-serializable as lists).
    ("const", c)      integer constant
    ("var",)          input variable x0
    ("add", e, c)     e + c,  c != 0, e not a const   (canonical)
    ("mul", e, k)     k * e,  k not in (0, 1), e not a const (canonical)
    ("neg", e)        -e, e not const/neg            (canonical)
    ("mod", e, m)     e mod m (Python semantics: 0 <= result < m)

The VERIFIER below is structurally independent from the prover: it re-derives every
step from the recorded input expression using its own rule table and never trusts
the prover's recorded outputs. Verification is linear in |tau| (number of steps).
"""

# ---------------------------------------------------------------------------
# Canonical constructors
# ---------------------------------------------------------------------------

def const(c):
    return ("const", int(c))

def var():
    return ("var",)

def neg(e):
    if e[0] == "const":
        return ("const", -e[1])
    if e[0] == "neg":
        return e[1]
    return ("neg", e)

def add(e, c):
    c = int(c)
    if c == 0:
        return e
    if e[0] == "const":
        return ("const", e[1] + c)
    if e[0] == "add":
        return add(e[1], e[2] + c)
    if e[0] == "neg":
        # -(f) + c  stays as-is (canonical: keep neg inside add)
        return ("add", e, c)
    return ("add", e, c)

def mul(e, k):
    k = int(k)
    if k == 0:
        return ("const", 0)
    if k == 1:
        return e
    if k == -1:
        return neg(e)
    if e[0] == "const":
        return ("const", e[1] * k)
    if e[0] == "mul":
        return mul(e[1], e[2] * k)
    if e[0] == "neg":
        return neg(mul(e[1], k))
    if e[0] == "add":
        # distribute: k*(f + c) = k*f + k*c  (keeps canonical shape)
        return add(mul(e[1], k), e[2] * k)
    return ("mul", e, k)

def mod(e, m):
    m = int(m)
    assert m > 0
    if e[0] == "const":
        return ("const", e[1] % m)
    return ("mod", e, m)

# ---------------------------------------------------------------------------
# Canonical rendering / parsing (for human-readable certs and JSON round-trip)
# ---------------------------------------------------------------------------

def canon(e):
    """Render expression to its canonical string form."""
    tag = e[0]
    if tag == "const":
        return str(e[1])
    if tag == "var":
        return "x0"
    if tag == "add":
        inner = canon(e[1])
        c = e[2]
        return f"{inner}+{c}" if c > 0 else f"{inner}{c}"
    if tag == "mul":
        inner = canon(e[1])
        if e[1][0] in ("add",):
            inner = f"({inner})"
        return f"{e[2]}*{inner}"
    if tag == "neg":
        inner = canon(e[1])
        if e[1][0] in ("add",):
            inner = f"({inner})"
        return f"-{inner}"
    if tag == "mod":
        return f"({canon(e[1])} mod {e[2]})"
    raise ValueError(f"bad expr {e}")

def eval_expr(e, x0):
    """Exact integer evaluation of the symbolic expression at x0."""
    tag = e[0]
    if tag == "const":
        return e[1]
    if tag == "var":
        return int(x0)
    if tag == "add":
        return eval_expr(e[1], x0) + e[2]
    if tag == "mul":
        return e[2] * eval_expr(e[1], x0)
    if tag == "neg":
        return -eval_expr(e[1], x0)
    if tag == "mod":
        return eval_expr(e[1], x0) % e[2]
    raise ValueError(f"bad expr {e}")

def expr_size(e):
    """Number of AST nodes -- certificate size grows linearly with depth."""
    tag = e[0]
    if tag in ("const", "var"):
        return 1
    if tag in ("add", "mul"):
        return 1 + expr_size(e[1])
    if tag in ("neg", "mod"):
        return 1 + expr_size(e[1])
    raise ValueError(f"bad expr {e}")

# ---------------------------------------------------------------------------
# Fixed symbolic abstract transformers (the rule table).
# Sound by construction for the Phase-1 op set; NOT learned.
# ---------------------------------------------------------------------------

def _t_add1(e): return add(e, 1)
def _t_add3(e): return add(e, 3)
def _t_add7(e): return add(e, 7)
def _t_mul2(e): return mul(e, 2)
def _t_mul5(e): return mul(e, 5)
def _t_mod7(e): return mod(e, 7)
def _t_mod13(e): return mod(e, 13)
def _t_neg(e): return neg(e)

RULES = {
    "ADD_1": _t_add1,
    "ADD_3": _t_add3,
    "ADD_7": _t_add7,
    "MUL_2": _t_mul2,
    "MUL_5": _t_mul5,
    "MOD_7": _t_mod7,
    "MOD_13": _t_mod13,
    "NEG": _t_neg,
}

OP_NAMES = list(RULES.keys())

# ---------------------------------------------------------------------------
# JSON (de)serialization: tuples <-> lists
# ---------------------------------------------------------------------------

def expr_to_json(e):
    tag = e[0]
    if tag == "const":
        return ["const", e[1]]
    if tag == "var":
        return ["var"]
    if tag in ("add", "mul", "mod"):
        return [tag, expr_to_json(e[1]), e[2]]
    if tag == "neg":
        return ["neg", expr_to_json(e[1])]
    raise ValueError(f"bad expr {e}")

def expr_from_json(j):
    tag = j[0]
    if tag == "const":
        return ("const", j[1])
    if tag == "var":
        return ("var",)
    if tag in ("add", "mul", "mod"):
        return (tag, expr_from_json(j[1]), j[2])
    if tag == "neg":
        return ("neg", expr_from_json(j[1]))
    raise ValueError(f"bad expr json {j}")

# ---------------------------------------------------------------------------
# Independent verifier
# ---------------------------------------------------------------------------

def _verify_flat_step(s, i, prev_out, x0):
    """Verify one primitive step. Returns (ok, info, out_e) or (False, info, None)."""
    op = s["op"]
    if op not in RULES:
        return False, {"error": f"unknown op {op} at step {i}"}, None
    in_e = expr_from_json(s["in"])
    out_e = expr_from_json(s["out"])
    # re-derive: the verifier applies its OWN rule table
    derived = RULES[op](in_e)
    if derived != out_e:
        return False, {"error": f"step {i} ({op}): re-derivation mismatch",
                       "derived": canon(derived), "claimed": canon(out_e)}, None
    # chaining
    if prev_out is not None and in_e != prev_out:
        return False, {"error": f"step {i}: input != previous output"}, None
    return True, {}, out_e


def _verify_branch_node(s, i, prev_out, x0):
    """
    Verify one IFZ branch node. The node claims:
      in          abstraction at branch entry (== prev_out by chaining)
      cond_expr   guard expression over x0
      cond_value  the NEURAL guard value the prover decided on
      taken       "then" | "else"
      cond_steps  guard sub-trace (in -> cond output)
      steps       taken-branch sub-trace (in -> out), on the PRE-guard value
      out         claimed branch output abstraction
    Checks (all re-derived, nothing trusted):
      B1. chaining: in == prev_out
      B2. guard sub-trace valid AND evaluates to cond_value
          (neural guard value must equal the symbolic re-derivation --
           a divergence here fails LOUD instead of shipping)
      B3. taken consistent with cond_value (then iff 0) -- the P_then/P_else
          interface condition, checked by the INDEPENDENT verifier
      B4. branch sub-trace valid from the pre-guard input, ending at out
    Returns (ok, info, out_e).
    """
    try:
        in_e = expr_from_json(s["in"])
        out_e = expr_from_json(s["out"])
        cond_value = int(s["cond_value"])
        taken = s["taken"]
        cond_e = expr_from_json(s["cond_expr"])
    except Exception as ex:
        return False, {"error": f"branch {i}: malformed node: {ex}"}, None
    # B1. chaining
    if prev_out is not None and in_e != prev_out:
        return False, {"error": f"branch {i}: input != previous output"}, None
    # B2. guard sub-trace: must be valid and evaluate to the claimed cond_value
    ok, info = verify_trace(s["cond_steps"], x0, in_e, cond_value)
    if not ok:
        return False, {"error": f"branch {i}: guard sub-trace invalid",
                       "detail": info}, None
    guard_out = info["final_expr_tuple"]
    if guard_out != cond_e:
        return False, {"error": f"branch {i}: guard expr != re-derived guard output",
                       "claimed": canon(cond_e), "derived": canon(guard_out)}, None
    if eval_expr(cond_e, x0) != cond_value:
        # Independent re-derivation disagrees with the neural guard value:
        # neural divergence at the branch condition -- fail closed.
        return False, {"error": f"branch {i}: neural-divergence at guard: "
                                f"symbolic={eval_expr(cond_e, x0)} != "
                                f"neural={cond_value}"}, None
    # B3. the interface condition, checked independently: then iff guard == 0
    want = "then" if cond_value == 0 else "else"
    if taken != want:
        return False, {"error": f"branch {i}: taken={taken!r} inconsistent with "
                                f"guard value {cond_value} (want {want!r}); "
                                f"P_then/P_else interface violated"}, None
    # B4. taken-branch sub-trace from the PRE-guard input, ending at out
    ok, info = verify_trace(s["steps"], x0, in_e, None)
    if not ok:
        return False, {"error": f"branch {i}: taken-branch sub-trace invalid",
                       "detail": info}, None
    if info["final_expr_tuple"] != out_e:
        return False, {"error": f"branch {i}: branch output != claimed out",
                       "derived": canon(info["final_expr_tuple"]),
                       "claimed": canon(out_e)}, None
    return True, {"branch": i, "taken": taken, "cond_value": cond_value}, out_e


def verify_trace(steps, x0, start_expr, expected_result):
    """
    Generalized mechanical check of a trace (flat op steps and/or IFZ branch
    nodes). Returns (ok: bool, info: dict).

    Checks (each linear in |trace|):
      1. steps[0].in == start_expr   (initial abstraction as claimed)
      2. RULES[op](in) == out        (each abstract transformer application
                                      re-derived, not trusted)
      3. out_k == in_{k+1}           (trace chaining, incl. branch nodes)
      4. eval(out_last, x0) == expected_result   (skipped when None)
    info carries final_expr (canonical string), final_expr_tuple, step count
    and max expression size.
    """
    try:
        x0 = int(x0)
        if not steps:
            return False, {"error": "empty trace"}
        first = steps[0]
        first_in = expr_from_json(first["in"])
        if first_in != start_expr:
            return False, {"error": f"trace does not start at claimed input: "
                                    f"{canon(first_in)} != {canon(start_expr)}"}
        prev_out = None
        max_nodes = 0
        n_flat = 0
        n_branch = 0
        for i, s in enumerate(steps):
            if s.get("branch"):
                ok, info, out_e = _verify_branch_node(s, i, prev_out, x0)
                n_branch += 1
            else:
                ok, info, out_e = _verify_flat_step(s, i, prev_out, x0)
                n_flat += 1
            if not ok:
                info["step_index"] = i
                return False, info
            prev_out = out_e
            max_nodes = max(max_nodes, expr_size(out_e))
        if expected_result is not None and eval_expr(prev_out, x0) != int(expected_result):
            return False, {"error": "final expression does not evaluate to claimed result",
                           "eval": eval_expr(prev_out, x0),
                           "claimed": int(expected_result)}
        return True, {"steps": len(steps), "flat_steps": n_flat,
                      "branch_nodes": n_branch, "max_expr_nodes": max_nodes,
                      "final_expr": canon(prev_out), "final_expr_tuple": prev_out}
    except Exception as ex:  # a malformed certificate fails closed
        return False, {"error": f"verifier exception: {type(ex).__name__}: {ex}"}


def verify_certificate(cert):
    """
    Mechanically check a certificate. Returns (ok: bool, info: dict).

    Delegates to verify_trace with start_expr = x0. For certificates produced
    by ProofMachine.run / run_program, checks:
      1. steps[0].in == x0           (initial abstraction covers the input)
      2. RULES[op](in) == out        (each abstract transformer application
                                      re-derived, not trusted)
      3. out_k == in_{k+1}           (trace chaining)
      4. eval(out_last, x0) == result (final abstraction implies postcondition
                                      Q(x,y) := y = E(x0); for MOD steps the
                                      0 <= y < m bound is checked by eval
                                      under Python mod semantics)
    Branch nodes additionally undergo B1-B4 (guard re-derivation, neural
    guard agreement, taken-consistency, branch sub-trace validity).
    """
    try:
        x0 = int(cert["x0"])
        result = int(cert["result"])
        return verify_trace(cert["steps"], x0, ("var",), result)
    except Exception as ex:  # a malformed certificate fails closed
        return False, {"error": f"verifier exception: {type(ex).__name__}: {ex}"}
