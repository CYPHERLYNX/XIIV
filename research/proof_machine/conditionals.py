"""
Conditional composition for the Compositional Proof Machine
(spec section 5.6, task 2: IFZERO branching).

A PROGRAM is a list of steps; each step is either
  - an op name (str) from certificates.RULES, or
  - an IFZ node: ("IFZ", cond_prog, then_prog, else_prog), where each element
    is itself a program (nesting allowed).

Semantics of IFZ(cond, then, else) at value x (Hoare-logic IF-rule shape):
    c  = cond(x)                       # guard computation (evidence)
    out = then(x)  if c == 0 else else(x)   # data flows into the branch

The guard is a side computation: the branch receives the PRE-guard value x,
not the guard output. Rationale: real conditionals branch on a predicate of
the data while the data itself flows into the taken branch. The guard trace is
kept as certificate evidence for the branch decision.

Preconditions (NON-VACUOUS -- this is the point of 5.6 task 2):
    P_then(c) := (c == 0),      P_else(c) := (c != 0)
The interface check  gamma(last(tau_cond)) subset { y | P_taken(y) }  now does
real routing work: it is evaluated on the concrete guard value at every branch
and a mismatch halts LOUD, localized to the branch. Phase-1's vacuous
P=true interface check is kept for flat op steps; branches exercise the real
forall-y.(Q_f -> P_g) condition.

Certificate: a branch node records the guard expression, the NEURAL guard
value, the taken branch, the guard sub-trace and the taken-branch sub-trace.
The independent verifier re-derives the guard value symbolically and requires
it to equal the recorded neural value -- a neural divergence at a branch
condition can never ship a certificate (defense in depth alongside the
run()-side halt, mirroring the VRC doctrine that divergences halt loud).

Cost model: max-path primitive count (worst-case executed ops). The task
generator builds branches of equal taken-cost so taken-path == max-path and
depth stays comparable with the Phase-1 flat sweep.
"""

from certificates import RULES


def is_ifz(step):
    return isinstance(step, tuple) and len(step) == 4 and step[0] == "IFZ"


def ifz(cond_prog, then_prog, else_prog):
    """Constructor with light shape validation."""
    assert isinstance(cond_prog, list) and len(cond_prog) > 0, \
        "cond program must be a non-empty step list"
    assert isinstance(then_prog, list) and isinstance(else_prog, list), \
        "branches must be step lists"
    return ("IFZ", cond_prog, then_prog, else_prog)


def max_path_cost(prog):
    """Worst-case primitive count over all branch choices."""
    total = 0
    for step in prog:
        if is_ifz(step):
            _, cond, then_p, else_p = step
            total += (max_path_cost(cond)
                      + max(max_path_cost(then_p), max_path_cost(else_p)))
        else:
            if step not in RULES:
                raise ValueError(f"unknown op {step}")
            total += 1
    return total


def check_branch_interface(cond_value, taken):
    """
    The non-vacuous interface check for a branch decision.
    P_then(c) := (c == 0); P_else(c) := (c != 0).
    Returns (ok, reason). By construction the executor always passes this;
    it exists so the routing condition is REAL (unit-tested adversarially)
    and failures are loud and localized.
    """
    cond_value = int(cond_value)
    if taken == "then":
        ok = (cond_value == 0)
        return ok, (f"P_then holds (guard={cond_value}==0)"
                    if ok else
                    f"P_then VIOLATED: taken='then' but guard={cond_value}!=0")
    if taken == "else":
        ok = (cond_value != 0)
        return ok, (f"P_else holds (guard={cond_value}!=0)"
                    if ok else
                    f"P_else VIOLATED: taken='else' but guard={cond_value}==0")
    return False, f"unknown branch label {taken!r}"


def decide_branch(cond_value):
    """then iff the guard value is zero."""
    return "then" if int(cond_value) == 0 else "else"


# ---------------------------------------------------------------------------
# JSON serialization (programs appear in certificates as "prog", informational;
# the verifier trusts only the trace, never this field).
# ---------------------------------------------------------------------------

def prog_to_json(prog):
    out = []
    for step in prog:
        if is_ifz(step):
            _, cond, then_p, else_p = step
            out.append(["IFZ", prog_to_json(cond), prog_to_json(then_p),
                        prog_to_json(else_p)])
        else:
            out.append(step)
    return out


def prog_from_json(j):
    out = []
    for step in j:
        if isinstance(step, list) and step and step[0] == "IFZ":
            _, cond, then_p, else_p = step
            out.append(("IFZ", prog_from_json(cond), prog_from_json(then_p),
                        prog_from_json(else_p)))
        else:
            out.append(step)
    return out


def prog_to_str(prog):
    """Compact human-readable rendering."""
    parts = []
    for step in prog:
        if is_ifz(step):
            _, cond, then_p, else_p = step
            parts.append(f"IFZ[{prog_to_str(cond)}]"
                         f"?{prog_to_str(then_p)}:{prog_to_str(else_p)}")
        else:
            parts.append(step)
    return "[" + ",".join(parts) + "]"
