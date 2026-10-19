"""
The composition operator (spec section 2).

Given certified primitives F = (f, P_f, Q_f, tau_f) and G = (g, P_g, Q_g, tau_g),
    G o F = (g∘f, P, Q, tau)
with:
  P(x) := P_f(x) and forall y. (Q_f(x,y) -> P_g(y))     [interface condition]
  Q(x,z) := exists y. (Q_f(x,y) and Q_g(y,z))
  tau    := tau_f ++ tau_g*   (G's trace re-rooted at F's output abstraction)

Phase-1 instantiation: every primitive's precondition P is `true` (all integer
inputs are legal), so the interface check gamma(last(tau_f)) ⊆ {y | P_g(y)} is
vacuously satisfied -- but it is still EXECUTED (as a hook) so the machinery is
real and Phase-2 primitives with nontrivial preconditions will exercise it.

ProofMachine.run(x0, ops):
  1. Concrete execution: x_{k+1} = M_{op_k}(x_k)   (neural forward passes)
  2. Symbolic execution (in parallel): e_{k+1} = M_{op_k}#(e_k)  (exact)
  3. Returns (x_L, certificate) where the certificate is the trace
     [(op_k, e_k, e_{k+1})] plus x0 and the claimed result.
"""

import torch

from certificates import RULES, var, eval_expr, canon, expr_to_json
from conditionals import (is_ifz, decide_branch, check_branch_interface,
                          prog_to_json)
from primitives import build_net, apply_concrete


def interface_check(prev_out_expr, next_op):
    """
    Check gamma(last(tau_f)) ⊆ { y | P_{next}(y) }.
    Phase 1: all preconditions are `true`, so this always passes.
    The hook exists so failures are loud and localizable when Phase-2
    primitives introduce nontrivial preconditions.
    Returns (ok: bool, reason: str).
    """
    # P_next(y) := true  for all Phase-1 primitives.
    return True, "P=true (Phase 1)"


class ProofMachine:
    def __init__(self, checkpoint_dir, device="cpu"):
        self.device = device
        self.nets = {}
        for op in RULES:
            net = build_net(op)
            sd = torch.load(f"{checkpoint_dir}/{op}.pt", map_location=device,
                            weights_only=True)
            net.load_state_dict(sd)
            net.eval().to(device)
            self.nets[op] = net

    @torch.no_grad()
    def run(self, x0, ops):
        """Execute a composition; return (result:int, certificate:dict)."""
        x = int(x0)
        e = var()
        steps = []
        for op in ops:
            if op not in RULES:
                raise ValueError(f"unknown op {op}")
            ok, reason = interface_check(e, op)
            if not ok:
                # LOUD failure, localized to the exact composition step --
                # this is the designed failure mode, not silent degradation.
                raise RuntimeError(f"interface check failed at op {op}: {reason}")
            e_next = RULES[op](e)
            steps.append({
                "op": op,
                "in": expr_to_json(e),
                "out": expr_to_json(e_next),
                "in_str": canon(e),
                "out_str": canon(e_next),
            })
            e = e_next
            x = apply_concrete(self.nets[op], op, x)
        cert = {"x0": int(x0), "ops": list(ops), "steps": steps, "result": x,
                "final_expr": canon(e)}
        return x, cert

    def symbolic_result(self, x0, ops):
        """Exact reference: pure symbolic execution (Python big ints)."""
        e = var()
        for op in ops:
            e = RULES[op](e)
        return eval_expr(e, int(x0))

    # -- conditional programs (spec 5.6 task 2) ---------------------------

    def _exec(self, x, e, prog, x0):
        """
        Recursive executor. x: current concrete int (neural), e: current
        abstract expression over x0, prog: step list, x0: original input.
        Returns (x_out, e_out, steps) with steps a flat list of step dicts /
        branch nodes for the certificate.
        """
        steps = []
        for step in prog:
            if is_ifz(step):
                x, e, node = self._exec_branch(x, e, step, x0)
                steps.append(node)
            else:
                if step not in RULES:
                    raise ValueError(f"unknown op {step}")
                ok, reason = interface_check(e, step)
                if not ok:
                    # LOUD failure, localized to the exact composition step --
                    # this is the designed failure mode, not silent degradation.
                    raise RuntimeError(
                        f"interface check failed at op {step}: {reason}")
                e_next = RULES[step](e)
                steps.append({
                    "op": step,
                    "in": expr_to_json(e),
                    "out": expr_to_json(e_next),
                    "in_str": canon(e),
                    "out_str": canon(e_next),
                })
                e = e_next
                x = apply_concrete(self.nets[step], step, x)
        return x, e, steps

    def _exec_branch(self, x, e, node, x0):
        """
        Execute one IFZ(cond, then, else) node at concrete x / abstract e.
        The guard decides on the NEURAL value; the certificate must then prove
        it equals the symbolic value (else loud neural-divergence halt -- the
        VRC doctrine: a divergence never ships a certificate). The taken
        branch runs on the PRE-guard value. The P_then/P_else interface check
        routes on the guard value and fails loud on any inconsistency.
        Returns (x_out, e_out, branch_node_dict).
        """
        _, cond_prog, then_prog, else_prog = node
        # Guard: concrete (neural) and symbolic in parallel.
        xc, ec, cond_steps = self._exec(x, e, cond_prog, x0)
        cv = int(xc)
        sv = eval_expr(ec, int(x0))
        if cv != sv:
            raise RuntimeError(
                f"neural-divergence at branch condition: neural guard={cv} != "
                f"symbolic guard={sv} (guard expr {canon(ec)}); halting loud, "
                f"no certificate shipped")
        taken = decide_branch(cv)
        ok, reason = check_branch_interface(cv, taken)
        if not ok:
            # The non-vacuous interface condition doing real work: P_taken
            # must hold of the guard value. Unreachable by construction;
            # present so the routing check is real, not decorative.
            raise RuntimeError(f"branch interface check failed: {reason}")
        branch_prog = then_prog if taken == "then" else else_prog
        # Branch runs on the PRE-guard (x, e): the guard is a side condition.
        xb, eb, branch_steps = self._exec(x, e, branch_prog, x0)
        branch_node = {
            "branch": True,
            "in": expr_to_json(e),
            "in_str": canon(e),
            "cond_expr": expr_to_json(ec),
            "cond_str": canon(ec),
            "cond_value": cv,
            "taken": taken,
            "interface_reason": reason,
            "cond_steps": cond_steps,
            "steps": branch_steps,
            "out": expr_to_json(eb),
            "out_str": canon(eb),
        }
        return xb, eb, branch_node

    @torch.no_grad()
    def run_program(self, x0, prog):
        """Execute a program (flat ops and/or IFZ nodes); return (result, cert)."""
        x0 = int(x0)
        x, e, steps = self._exec(x0, var(), prog, x0)
        cert = {"x0": x0, "prog": prog_to_json(prog), "steps": steps,
                "result": x, "final_expr": canon(e)}
        return x, cert

    def symbolic_result_program(self, x0, prog):
        """Exact reference for programs: pure symbolic execution."""
        x0 = int(x0)

        def sym(e, p):
            for step in p:
                if is_ifz(step):
                    _, cond, then_p, else_p = step
                    ec = sym(e, cond)
                    taken = then_p if eval_expr(ec, x0) == 0 else else_p
                    e = sym(e, taken)  # branch on the pre-guard value
                else:
                    e = RULES[step](e)
            return e

        return eval_expr(sym(var(), prog), x0)
