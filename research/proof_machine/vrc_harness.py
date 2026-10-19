"""
Verity-Routed Composition (VRC) — deterministic verifier-routed synthesis harness.

Phase 2 of the Compositional Proof Machine (spec: research/composition_operator_v1.md,
section 4.4 "the composition policy becomes the next problem", and 5.6). Phase 1 proved
certified composition doesn't degrade with depth (kill gate PASS, 2026-10-17); the open
problem is the *composition policy*: from a specification (target postcondition
``y = E(x0)``) to a primitive sequence whose certificate the independent verifier
accepts.

Design (after SKILLFORGE, arXiv:2608.29841 — an honest PORT, see
research/daily_2026-10-17.md for the novelty assessment). A deterministic harness, not
an agent, with a formal oracle in the loop. The oracle here is threefold:

  1. the composition interface check  (compose.interface_check — the COMP rule's
     ``gamma(last(tau_f)) subset {y | P_g(y)}`` side),
  2. the independent certificate verifier (certificates.verify_certificate),
  3. exact symbolic target equality (the composed abstract transformer must equal
     the specified postcondition — no search, no guessing).

Oracle failures are parsed into categories that deterministically select the next
repair skill. Termination is guaranteed: every repair iteration strictly grows a taboo
set (bounded by positions x ops), and a hard iteration cap (MAX_ITERS = 20) bounds the
loop regardless. Soundness is unconditional: a returned sequence verifies by
construction — the independent verifier accepts its certificate and concrete neural
execution equals symbolic execution on every probe input. A neural divergence never
ships a certificate; it halts LOUD with the offending primitive named.

Skill library (all deterministic):
  SPECSPLIT  parse the target spec; REJECT specs outside the primitive closure with
             the exact offending subexpression (spec-reject, never silent search).
  GENSEQ     constructive recursive planner: realized target expr -> op sequence.
  GENIFACE   run and record the interface check at every step (evidence trail for
             the intervention-rate metric).
  REPAIRSTEP on interface-fail@k: taboo (position k, op), keep the validated prefix,
             replan under the taboo. Rollback never mutates proven work.
  STRENGTHEN / WEAKEN: Phase-2 hooks for non-vacuous preconditions. No-ops in Phase 1
             (all P are `true`); present so the router's contract is complete.

Key lemma — realize(): the spec language (parser output) is strictly larger than the
trace language (what RULES can produce from x0). Three gaps, closed by realize():

  1. ("add", E, c < 0) never appears in a trace. Every ADD_k has k > 0, and by
     induction over RULES every ("add", _, c) in a trace has c >= 0 (MUL
     distributes non-negative constants; NEG/MOD create no adds). realize rewrites
     add(E, c<0) -> neg(add(neg(E), -c)) — the NEG sandwich, which IS
     trace-realizable. E.g. the spec "x0-3" plans to [NEG, ADD_3, NEG].
  2. ("mul", E, k < 0) never appears in a trace (MUL_k has k in {2,5}; the mul()
     constructor only ever sees k > 0 from RULES). realize rewrites
     mul(E, k<0) -> neg(mul(E, -k)).
  3. ("const", c) targets are unreachable: every trace from var() contains var()
     (induction over RULES — no op discards the input), so a constant postcondition
     is spec-rejected, not searched.

The planner therefore operates on realized targets, and symbolic equality is checked
against the realized form. The original spec string is kept for reporting.

Phase-1 honesty notes (see also proof_machine/README.md):
  - The interface-check leg is implemented and executed at every step, but vacuous
    (all P = true) until Phase-2 primitives introduce nontrivial preconditions. The
    REPAIRSTEP routing is therefore exercised in tests via a simulated Phase-2-style
    interface failure (explicitly labeled); the live Phase-1 routing decisions are
    spec-reject and neural-divergence.
  - STRENGTHEN/WEAKEN are documented no-ops in Phase 1.
  - No training, no GPU: pure synthesis over the frozen Phase-1 library. KBs of RAM.

Termination argument: each loop iteration either returns a verified composition,
halts loud (VRCHalted), or executes REPAIRSTEP, which inserts a (position, op) pair
into `taboo` that the deterministic planner is then forbidden from emitting again —
so `taboo` grows strictly monotonically (bounded by 64 positions x 8 ops), and
MAX_ITERS caps the loop independently. The planner itself is structural recursion
over the target AST: it terminates.
"""

import os
import re

from certificates import RULES, canon, var, const, add, mul, neg, mod, verify_certificate
from compose import ProofMachine, interface_check  # noqa: F401  (monkeypatch point for tests)
from conditionals import ifz, prog_to_str


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

class SpecRejected(Exception):
    """The target spec is outside the primitive closure. Loud, with the cause."""
    def __init__(self, reason, subexpr=""):
        super().__init__(reason)
        self.reason = reason
        self.subexpr = subexpr


class VRCHalted(Exception):
    """The harness stopped LOUD. Never a silent wrong answer."""
    def __init__(self, category, diagnosis, history=()):
        super().__init__(f"[{category}] {diagnosis}")
        self.category = category
        self.diagnosis = diagnosis
        self.history = list(history)


# ---------------------------------------------------------------------------
# SPECSPLIT, part 1: minimal spec-language parser -> canonical expr tuples
#
# Supported fragment (documented restriction): affine expressions over x0 with
# constant-only addition, constant-only multiplication, mod by a literal, unary
# minus, parentheses. Anything else -> SpecRejected with the offending piece.
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\d+|x0|mod|[+\-*/()]")


class _Parser:
    def __init__(self, s):
        self.toks = _TOKEN.findall(s.replace(" ", ""))
        if not self.toks:
            raise SpecRejected("empty spec", s)
        self.i = 0
        self.src = s

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        try:
            e = self.expr()
        except SpecRejected:
            raise
        except Exception as ex:  # noqa: BLE001 — any constructor misuse is a spec problem
            raise SpecRejected(
                f"spec uses constructs outside the affine fragment: "
                f"{type(ex).__name__}: {ex}", self.src) from ex
        if self.peek() is not None:
            raise SpecRejected(
                f"trailing tokens after spec: {self.toks[self.i:]}", self.src)
        return e

    def expr(self):
        e = self.term()
        while self.peek() in ("+", "-"):
            op = self.next()
            t = self.term()
            if op == "+":
                e = self._sadd(e, t)
            else:
                if t[0] != "const":
                    raise SpecRejected(
                        "spec subtraction of a non-constant "
                        f"{canon(t)} is outside the affine fragment", canon(t))
                e = add(e, -t[1])  # may carry a negative const; realize() normalizes
        return e

    @staticmethod
    def _sadd(a, b):
        if b[0] == "const":
            return add(a, b[1])
        if a[0] == "const":
            return add(b, a[1])
        raise SpecRejected(
            f"addition of two non-constant expressions {canon(a)} + {canon(b)} "
            "is outside the affine spec fragment", f"{canon(a)}+{canon(b)}")

    def term(self):
        e = self.factor()
        while self.peek() in ("*", "mod"):
            if self.next() == "*":
                e = self._smul(e, self.factor())
            else:
                m = self.next()
                if m is None or not m.isdigit() or int(m) <= 0:
                    raise SpecRejected(f"bad modulus {m!r} after 'mod'", self.src)
                e = mod(e, int(m))
        return e

    @staticmethod
    def _smul(a, b):
        if a[0] == "const" and b[0] == "const":
            return const(a[1] * b[1])
        if b[0] == "const":
            return mul(a, b[1])
        if a[0] == "const":
            return mul(b, a[1])
        raise SpecRejected(
            f"nonlinear product {canon(a)} * {canon(b)}: the primitive closure "
            "is affine — products of non-constants are unreachable",
            f"{canon(a)}*{canon(b)}")

    def factor(self):
        t = self.peek()
        if t == "(":
            self.next()
            e = self.expr()
            if self.next() != ")":
                raise SpecRejected("unbalanced parenthesis in spec", self.src)
            return e
        if t == "-":
            self.next()
            return neg(self.factor())
        if t == "x0":
            self.next()
            return var()
        if t is not None and t.isdigit():
            self.next()
            return const(int(t))
        raise SpecRejected(f"unexpected token {t!r} in spec", self.src)


def parse_target(s):
    """Parse a spec string into a canonical expression tuple (SPECSPLIT)."""
    return _Parser(s).parse()


# ---------------------------------------------------------------------------
# SPECSPLIT, part 2: realize() — spec language -> trace language (see lemma)
# ---------------------------------------------------------------------------

def realize(e):
    """Rewrite a parsed target into trace-realizable canonical form.

    The planner and the equality oracle both operate on realized targets.
    """
    tag = e[0]
    if tag in ("const", "var"):
        return e
    if tag == "neg":
        return neg(realize(e[1]))
    if tag == "add":
        E2, c = realize(e[1]), e[2]
        if E2[0] == "const":
            return const(E2[1] + c)
        if c == 0:
            return E2
        if E2[0] == "add" and c >= 0:
            return ("add", E2[1], E2[2] + c)  # merge, stays non-negative
        if c < 0:
            # Lemma gap 1: the NEG sandwich IS trace-realizable.
            return ("neg", ("add", ("neg", E2), -c))
        return ("add", E2, c)
    if tag == "mul":
        E2, k = realize(e[1]), e[2]
        if k == 0:
            return const(0)
        if k == 1:
            return E2
        if k == -1:
            return neg(E2)
        if E2[0] == "const":
            return const(E2[1] * k)
        if k < 0:
            # Lemma gap 2.
            return ("neg", ("mul", E2, -k))
        return ("mul", E2, k)
    if tag == "mod":
        return mod(realize(e[1]), e[2])
    raise SpecRejected(f"unknown expression tag {tag}", canon(e))


# ---------------------------------------------------------------------------
# GENSEQ: constructive recursive planner, target expr -> op sequence.
# Raises SpecRejected for anything outside the 8-op closure. Position-aware
# taboo lets REPAIRSTEP forbid (position, op) pairs.
# ---------------------------------------------------------------------------

def _check_taboo(pos, op, forbidden):
    if (pos, op) in forbidden:
        raise SpecRejected(
            f"taboo blocks ({pos}, {op}): no admissible plan under the "
            f"current repair taboo {sorted(forbidden)}", op)


def _factor_25(k):
    """Factor k into 2s and 5s; the only multipliers the library provides."""
    i = j = 0
    n = k
    while n % 2 == 0:
        i += 1
        n //= 2
    while n % 5 == 0:
        j += 1
        n //= 5
    if n != 1:
        raise SpecRejected(
            f"coefficient {k} has prime factors outside {{2, 5}} "
            f"(remaining {n}): outside the 8-op closure", f"{k}*x")
    return i, j


def _decompose_const(c, pos, forbidden):
    """Non-negative int -> ADD_7/ADD_3/ADD_1 sequence (greedy, taboo-aware)."""
    assert c >= 0, "internal: _decompose_const expects c >= 0 (realize first)"
    ops = []
    rem, p = c, pos
    while rem > 0:
        for d in (7, 3, 1):
            if d <= rem and (p, f"ADD_{d}") not in forbidden:
                ops.append(f"ADD_{d}")
                rem -= d
                p += 1
                break
        else:
            raise SpecRejected(
                f"taboo blocks the constant decomposition of +{c} "
                f"at position {p}", f"+{c}")
    return ops, p


def _plan(e, pos, forbidden):
    """Realized target -> (op list, end position). Raises SpecRejected."""
    tag = e[0]
    if tag == "var":
        return [], pos
    if tag == "const":
        # Lemma gap 3: no op discards x0.
        raise SpecRejected(
            f"constant target {e[1]}: every trace from x0 contains x0 "
            "(induction over RULES) — constants are unreachable", canon(e))
    if tag == "neg":
        ops, p = _plan(e[1], pos, forbidden)
        _check_taboo(p, "NEG", forbidden)
        return ops + ["NEG"], p + 1
    if tag == "add":
        E, c = e[1], e[2]
        assert isinstance(c, int) and c >= 0, \
            "internal: _plan expects realized targets (run realize() first)"
        # Peephole: k*E2 + c with k | c  ->  [plan(E2 + c/k), MUL_k].
        # Shorter certificates: the MUL distributes over the smaller addend.
        # Falls back to the plain structural plan if the taboo blocks it.
        if E[0] == "mul" and c > 0 and c % E[2] == 0:
            k = E[2]
            try:
                fops, fp = _plan(realize(("add", E[1], c // k)), pos, forbidden)
                i, j = _factor_25(k)
                for cand in (["MUL_2"] * i + ["MUL_5"] * j,
                             ["MUL_5"] * j + ["MUL_2"] * i):
                    if all((fp + q, op) not in forbidden
                           for q, op in enumerate(cand)):
                        return fops + cand, fp + len(cand)
            except SpecRejected:
                pass  # taboo-blocked: fall through to the plain plan
        ops, p = _plan(E, pos, forbidden)
        dops, p2 = _decompose_const(c, p, forbidden)
        return ops + dops, p2
    if tag == "mul":
        E, k = e[1], e[2]
        assert isinstance(k, int) and k > 0, \
            "internal: _plan expects realized targets (run realize() first)"
        i, j = _factor_25(k)
        ops, p = _plan(E, pos, forbidden)
        for cand in (["MUL_2"] * i + ["MUL_5"] * j,
                     ["MUL_5"] * j + ["MUL_2"] * i):
            if all((p + q, op) not in forbidden for q, op in enumerate(cand)):
                return ops + cand, p + len(cand)
        raise SpecRejected(
            f"taboo blocks every factorization order of x{k}", canon(e))
    if tag == "mod":
        E, m = e[1], e[2]
        if m not in (7, 13):
            raise SpecRejected(
                f"modulus {m}: the library provides only MOD_7 and MOD_13",
                canon(e))
        ops, p = _plan(E, pos, forbidden)
        _check_taboo(p, f"MOD_{m}", forbidden)
        return ops + [f"MOD_{m}"], p + 1
    raise SpecRejected(f"unknown expression tag {tag}", canon(e))


# ---------------------------------------------------------------------------
# VRC harness
# ---------------------------------------------------------------------------

class VRC:
    """Verity-Routed Composition: spec -> verified primitive sequence."""

    MAX_ITERS = 20       # hard termination cap (bounded iterations guarantee)
    PLAN_DEPTH_CAP = 64  # plans deeper than this are rejected, not searched
    PROBES = (5, -3, 42, 1000, -1000)

    # Deterministic oracle-category -> repair-skill routing table.
    # Only "interface-fail" continues the loop; everything else halts loud.
    ROUTES = {
        "spec-rejected": "ABORT",     # outside the closure: no repair exists
        "interface-fail": "REPAIRSTEP",
        "target-mismatch": "ABORT",    # planner/oracle inconsistency: loud bug
        "cert-invalid": "ABORT",       # the verifier is ground truth: loud
        "neural-divergence": "ABORT",  # primitive unreliable: retrain, never ship
        "repair-exhausted": "ABORT",
        "budget-exhausted": "ABORT",
    }

    def __init__(self, pm: ProofMachine):
        self.pm = pm

    # -- skills ---------------------------------------------------------

    def skill_specsplit(self, spec):
        """SPECSPLIT: parse + realize. Returns (raw, realized)."""
        raw = parse_target(spec) if isinstance(spec, str) else spec
        return raw, realize(raw)

    def skill_genseq(self, target_realized, forbidden=frozenset()):
        """GENSEQ: constructive plan for a realized target under a taboo."""
        ops, _ = _plan(target_realized, 0, forbidden)
        if len(ops) > self.PLAN_DEPTH_CAP:
            raise SpecRejected(
                f"plan depth {len(ops)} exceeds cap {self.PLAN_DEPTH_CAP}",
                canon(target_realized))
        return ops

    def skill_geniface(self, seq):
        """GENIFACE: run the interface check at every step; evidence trail."""
        evidence = []
        e = var()
        for k, op in enumerate(seq):
            ok, reason = interface_check(e, op)
            evidence.append({"step": k, "op": op, "ok": ok, "reason": reason})
            if not ok:
                break
            e = RULES[op](e)
        return evidence

    @staticmethod
    def skill_strengthen(diag):
        """STRENGTHEN (Phase-2 hook): tighten a precondition. No-op in Phase 1."""
        return {"skill": "STRENGTHEN", "status": "noop-phase1",
                "note": "all Phase-1 preconditions are `true`; nothing to tighten",
                "diag": diag}

    @staticmethod
    def skill_weaken(diag):
        """WEAKEN (Phase-2 hook): loosen a precondition. No-op in Phase 1."""
        return {"skill": "WEAKEN", "status": "noop-phase1",
                "note": "all Phase-1 preconditions are `true`; nothing to loosen",
                "diag": diag}

    # -- oracle helpers -------------------------------------------------

    def _localize_divergence(self, x0, ops):
        """First op where concrete execution leaves the symbolic trace."""
        for k in range(1, len(ops) + 1):
            r, _ = self.pm.run(x0, ops[:k])
            if r != self.pm.symbolic_result(x0, ops[:k]):
                return ops[k - 1], k
        return None, None

    # -- main loop ------------------------------------------------------

    def synthesize(self, spec, probes=None):
        """Synthesize a verified primitive sequence for `spec`.

        Returns a result dict on success; raises VRCHalted (loud) otherwise.
        """
        probes = list(probes) if probes else list(self.PROBES)
        try:
            raw, target = self.skill_specsplit(spec)
        except SpecRejected as r:
            raise VRCHalted("spec-rejected",
                            f"{r.reason} [subexpr: {r.subexpr}]") from r
        taboo = set()
        history = []
        iters = 0
        best_prefix_len = 0
        in_repair = False

        while True:
            iters += 1
            if iters > self.MAX_ITERS:
                raise VRCHalted("budget-exhausted",
                                f"{self.MAX_ITERS} iterations exceeded", history)
            try:
                ops = self.skill_genseq(target, frozenset(taboo))
            except SpecRejected as r:
                category = "repair-exhausted" if in_repair else "spec-rejected"
                raise VRCHalted(
                    category,
                    f"{r.reason} [subexpr: {r.subexpr}]"
                    + (f" [taboo: {sorted(taboo)}]" if in_repair else ""),
                    history) from r

            # Oracle leg 1+2: interface checks (symbolic side), then target equality.
            evidence = self.skill_geniface(ops)
            bad = next((ev for ev in evidence if not ev["ok"]), None)
            if bad is not None:
                # REPAIRSTEP: taboo (k, op); the validated prefix [0, k) survives —
                # repair only ever extends-or-replaces from a validated point.
                k, op = bad["step"], bad["op"]
                taboo.add((k, op))
                best_prefix_len = max(best_prefix_len, k)
                in_repair = True
                history.append({"iter": iters, "category": "interface-fail",
                                "routed_to": "REPAIRSTEP",
                                "step": k, "op": op, "reason": bad["reason"],
                                "taboo": sorted(taboo)})
                continue

            e = var()
            for op in ops:
                e = RULES[op](e)
            if e != target:
                raise VRCHalted(
                    "target-mismatch",
                    f"planner emitted {canon(e)} but the target is "
                    f"{canon(target)} — internal planner/oracle inconsistency",
                    history)

            # Oracle leg 3: concrete execution + independent verification, all probes.
            # The certificate is CONSTRUCTED alongside execution, never searched.
            # Layer attribution order is deliberate: the concrete-vs-symbolic
            # check runs BEFORE cert verification, so an execution-layer failure
            # is diagnosed as neural-divergence (retrain the primitive) rather
            # than misattributed to the proof layer. The verifier remains the
            # final gate either way (defense in depth).
            for x0 in probes:
                result, cert = self.pm.run(x0, ops)
                sym = self.pm.symbolic_result(x0, ops)
                if result != sym:
                    bad_op, bad_k = self._localize_divergence(x0, ops)
                    raise VRCHalted(
                        "neural-divergence",
                        f"concrete ({result}) != symbolic ({sym}) at x0={x0}; "
                        f"first divergence at step {bad_k} (op {bad_op}) — "
                        f"primitive unreliable, retraining required; "
                        f"no certificate shipped", history)
                vok, vinfo = verify_certificate(cert)
                if not vok:
                    raise VRCHalted("cert-invalid",
                                    f"independent verifier rejected the certificate "
                                    f"at x0={x0}: {vinfo}", history)

            return {"status": "ok",
                    "ops": ops,
                    "iters": iters,
                    "spec": spec if isinstance(spec, str) else canon(raw),
                    "realized": canon(target),
                    "probes": [{"x0": x0,
                                "result": self.pm.symbolic_result(x0, ops)}
                               for x0 in probes],
                    "history": history,
                    "best_prefix_len": best_prefix_len,
                    "repairs": sum(1 for h in history
                                   if h["category"] == "interface-fail")}

    # -- conditional synthesis (spec 5.6 task 2) ----------------------------

    def synthesize_conditional(self, cond_spec, then_spec, else_spec,
                               probes=None):
        """Synthesize a verified IFZ program from three arithmetic specs.

        Each spec is planned independently with the existing SPECSPLIT/GENSEQ
        skills (same closure rules, same loud spec-reject). The assembled
        program is [IFZ(cond_ops, then_ops, else_ops)]: at runtime the guard
        program's NEURAL value decides the branch (0 -> then), and the
        P_then/P_else interface check routes on it. Every probe is executed
        AND independently verified (concrete == symbolic, certificate
        accepted by the verifier incl. B1-B4); any failure halts loud with
        its category -- nothing unverified is returned.

        Scope note: no repair loop for the three sub-plans in this iteration
        (each sub-plan is constructive and deterministic); a SpecRejected
        sub-plan halts loud as spec-rejected, naming the offending spec.
        """
        probes = list(probes) if probes else list(self.PROBES)
        plans = {}
        for name, spec in (("cond", cond_spec), ("then", then_spec),
                           ("else", else_spec)):
            try:
                raw, target = self.skill_specsplit(spec)
                ops = self.skill_genseq(target)
            except SpecRejected as r:
                raise VRCHalted(
                    "spec-rejected",
                    f"{name} spec rejected: {r.reason} "
                    f"[subexpr: {r.subexpr}]") from r
            # target-equality oracle on the sub-plan (same as synthesize)
            e = var()
            for op in ops:
                e = RULES[op](e)
            if e != target:
                raise VRCHalted(
                    "target-mismatch",
                    f"{name} planner emitted {canon(e)} but target is "
                    f"{canon(target)} -- internal inconsistency")
            plans[name] = ops
        prog = [ifz(plans["cond"], plans["then"], plans["else"])]
        probe_results = []
        for x0 in probes:
            try:
                result, cert = self.pm.run_program(x0, prog)
            except RuntimeError as ex:
                raise VRCHalted("neural-divergence",
                                f"probe x0={x0}: {ex}") from ex
            sym = self.pm.symbolic_result_program(x0, prog)
            if result != sym:
                raise VRCHalted(
                    "neural-divergence",
                    f"concrete ({result}) != symbolic ({sym}) at x0={x0}; "
                    f"no certificate shipped")
            vok, vinfo = verify_certificate(cert)
            if not vok:
                raise VRCHalted(
                    "cert-invalid",
                    f"independent verifier rejected the branch certificate "
                    f"at x0={x0}: {vinfo}")
            node = cert["steps"][0]
            probe_results.append({"x0": x0, "result": result,
                                  "taken": node["taken"],
                                  "guard_value": node["cond_value"]})
        return {"status": "ok",
                "prog": prog_to_str(prog),
                "cond_ops": plans["cond"],
                "then_ops": plans["then"],
                "else_ops": plans["else"],
                "probes": probe_results,
                "verified": True}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def main():
    import json
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    spec = sys.argv[1] if len(sys.argv) > 1 else "((x0+3)*2 mod 7)+1"
    pm = ProofMachine(os.path.join(here, "checkpoints"))
    vrc = VRC(pm)
    try:
        out = vrc.synthesize(spec)
    except VRCHalted as h:
        print(json.dumps({"status": "halted", "category": h.category,
                          "diagnosis": h.diagnosis, "history": h.history},
                         indent=2, default=str))
        sys.exit(2)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
