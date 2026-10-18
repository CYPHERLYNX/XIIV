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
