"""
The composition operator for logic chains (spec section 5.6, task 1).

Boolean analogue of compose.ProofMachine: sequential composition of
certified boolean primitives over the 8-bit symbolic domain.

LogicProofMachine.run(x0, ops):
  1. Concrete execution: x_{k+1} = M_{op_k}(x_k)   (neural forward passes)
  2. Symbolic execution (in parallel): v_{k+1} = M_{op_k}#(v_k)  (exact)
  3. Returns (x_L, certificate) where the certificate is the bit-vector
     trace [(op_k, v_k, v_{k+1})] plus x0 and the claimed result.

All Phase-1 preconditions are `true` here too (every byte input is legal),
so the interface check is the same vacuous-but-executed hook.
"""

import torch

from logic import (RULES_BOOL, START_VEC, canon_bvec, vec_to_json,
                   eval_bvec)
from bool_primitives import build_net, apply_concrete


def interface_check(prev_vec, next_op):
    """P_next(v) := true for all task-1 primitives. Hook stays real."""
    return True, "P=true (task 1)"


class LogicProofMachine:
    def __init__(self, checkpoint_dir, device="cpu"):
        self.device = device
        self.nets = {}
        for op in RULES_BOOL:
            net = build_net(op)
            sd = torch.load(f"{checkpoint_dir}/{op}.pt", map_location=device,
                            weights_only=True)
            net.load_state_dict(sd)
            net.eval().to(device)
            self.nets[op] = net

    @torch.no_grad()
    def run(self, x0, ops):
        """Execute a boolean composition; return (result:int, certificate)."""
        x = int(x0) & 0xFF
        v = START_VEC
        steps = []
        for op in ops:
            if op not in RULES_BOOL:
                raise ValueError(f"unknown op {op}")
            ok, reason = interface_check(v, op)
            if not ok:
                raise RuntimeError(
                    f"interface check failed at op {op}: {reason}")
            v_next = RULES_BOOL[op](v)
            steps.append({
                "op": op,
                "in": vec_to_json(v),
                "out": vec_to_json(v_next),
                "in_str": canon_bvec(v),
                "out_str": canon_bvec(v_next),
            })
            v = v_next
            x = apply_concrete(self.nets[op], op, x)
        cert = {"x0": int(x0), "ops": list(ops), "steps": steps,
                "result": x, "final": canon_bvec(v)}
        return x, cert

    def symbolic_result(self, x0, ops):
        """Exact reference: pure symbolic execution (Python ints)."""
        v = START_VEC
        for op in ops:
            v = RULES_BOOL[op](v)
        return eval_bvec(v, int(x0))
