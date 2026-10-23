"""
Noisy-input execution for the Compositional Proof Machine
(spec section 5.6, task 3).

The observed input is a NOMINAL point x0, but the true input is only known to
lie in I0 = [x0 - r, x0 + r] (noise radius r >= 0). Two trajectories run in
parallel:

  concrete (neural): the 8 trained primitive nets execute on the nominal x0.
  abstract (interval): I_{k+1} = INTERVAL_RULES[op_k](I_k) -- sound and, for
    flat chains, EXACT (see intervals.py).

Agreement (the neural bridge under noise): the nominal neural point must lie
inside the interval at EVERY step. A violation halts LOUD -- a divergence
never ships a certificate (VRC doctrine).

Conditionals under noise: the guard is now an INTERVAL G, not a point.
  G == {0}        -> then-branch (decisive)
  0 not in G      -> else-branch (decisive)
  else (straddle) -> BOTH branches are reachable by some true input: execute
                     both on the pre-guard interval and JOIN (hull). The join
                     is the single source of precision loss in the whole
                     noisy machine -- the sweep measures it.

Certificate: the interval trace plus, for straddles, join nodes recording the
guard interval, the straddle evidence, and both branch sub-traces.
"""

import torch

from certificates import RULES, OP_NAMES
from conditionals import is_ifz, prog_to_json
from intervals import (iv, width, contains, join, guard_decision,
                       INTERVAL_RULES, iv_to_json)


class NoiseDivergence(RuntimeError):
    """Neural nominal point escaped its interval -- fail loud, no certificate."""


class NoiseProofMachine:
    def __init__(self, proof_machine):
        # Reuse the trained primitive nets from a loaded ProofMachine.
        self.pm = proof_machine
        self.nets = proof_machine.nets

    # -- flat chains ----------------------------------------------------

    @torch.no_grad()
    def run_noisy(self, x0, r, ops):
        """
        Execute ops under input noise radius r.
        Returns (nominal_result:int, certificate:dict).
        Raises NoiseDivergence if the neural trajectory ever leaves the
        interval trace (loud, localized to the step).
        """
        from primitives import apply_concrete
        x0, r = int(x0), int(r)
        assert r >= 0
        x = x0
        cur = iv(x0 - r, x0 + r)
        steps = []
        for k, op in enumerate(ops):
            if op not in INTERVAL_RULES:
                raise ValueError(f"unknown op {op}")
            nxt = INTERVAL_RULES[op](cur)
            x = apply_concrete(self.nets[op], op, x)
            if not contains(nxt, x):
                raise NoiseDivergence(
                    f"step {k} ({op}): neural nominal {x} not in interval "
                    f"[{nxt[0]}, {nxt[1]}]; halting loud, no certificate")
            steps.append({"op": op, "in": iv_to_json(cur),
                          "out": iv_to_json(nxt)})
            cur = nxt
        cert = {"x0": x0, "r": r, "ops": list(ops), "steps": steps,
                "result": x, "final_interval": iv_to_json(cur),
                "final_width": width(cur)}
        return x, cert

    # -- conditional programs -------------------------------------------

    @torch.no_grad()
    def run_program_noisy(self, x0, r, prog):
        """Execute a program (flat ops and/or IFZ nodes) under noise."""
        from primitives import apply_concrete
        x0, r = int(x0), int(r)
        assert r >= 0
        x, cur, steps, stats = self._exec_noisy(x0, iv(x0 - r, x0 + r), prog)
        cert = {"x0": x0, "r": r, "prog": prog_to_json(prog), "steps": steps,
                "result": x, "final_interval": iv_to_json(cur),
                "final_width": width(cur), "straddles": stats["straddles"],
                "decisive": stats["decisive"]}
        return x, cert

    def _exec_noisy(self, x, cur, prog):
        """
        Recursive noisy executor.
        x: nominal concrete int (neural trajectory). cur: current interval.
        Returns (x_out, cur_out, steps, stats).
        """
        from primitives import apply_concrete
        steps = []
        stats = {"straddles": 0, "decisive": 0}
        for step in prog:
            if is_ifz(step):
                x, cur, node, st = self._exec_branch_noisy(x, cur, step)
                steps.append(node)
                stats["straddles"] += st["straddles"]
                stats["decisive"] += st["decisive"]
            else:
                if step not in INTERVAL_RULES:
                    raise ValueError(f"unknown op {step}")
                nxt = INTERVAL_RULES[step](cur)
                x = apply_concrete(self.nets[step], step, x)
                if not contains(nxt, x):
                    raise NoiseDivergence(
                        f"op {step}: neural nominal {x} escaped interval "
                        f"[{nxt[0]}, {nxt[1]}]; halting loud")
                steps.append({"op": step, "in": iv_to_json(cur),
                              "out": iv_to_json(nxt)})
                cur = nxt
        return x, cur, steps, stats

    def _exec_branch_noisy(self, x, cur, node):
        """
        One IFZ(cond, then, else) node under noise. The guard interval G is
        computed from the cond program on the current interval. On straddle,
        both branches run on the PRE-guard interval and join (hull).
        The nominal neural value follows the nominal guard decision only for
        the concrete trajectory; the certificate covers all possibilities.
        """
        from primitives import apply_concrete
        _, cond_prog, then_prog, else_prog = node
        # Guard: nominal concrete value (recursive neural execution, handles
        # nested IFZ) + guard interval, in parallel.
        xc, g_iv, cond_steps, _ = self._exec_noisy(x, cur, cond_prog)
        cv_nominal = xc  # nominal guard value; agreement with the interval
        # is enforced by the per-step contains-checks in _exec_noisy.
        decision = guard_decision(g_iv)
        stats = {"straddles": 0, "decisive": 0}
        if decision == "straddle":
            stats["straddles"] = 1
            # Both branches on the PRE-guard (x, cur); join the intervals.
            xb_t, out_t, steps_t, st_t = self._exec_noisy(x, cur, then_prog)
            xb_e, out_e, steps_e, st_e = self._exec_noisy(x, cur, else_prog)
            joined = join(out_t, out_e)
            # Nominal follows its own guard value.
            taken_nominal = "then" if cv_nominal == 0 else "else"
            x_out = xb_t if taken_nominal == "then" else xb_e
            if not contains(joined, x_out):
                raise NoiseDivergence(
                    f"join: nominal {x_out} not in joined interval; halting")
            node_dict = {
                "branch": True, "straddle": True,
                "in": iv_to_json(cur),
                "cond_interval": iv_to_json(g_iv),
                "cond_nominal": int(cv_nominal),
                "taken_nominal": taken_nominal,
                "then_out": iv_to_json(out_t), "else_out": iv_to_json(out_e),
                "out": iv_to_json(joined),
                "cond_steps": cond_steps,
                "then_steps": steps_t, "else_steps": steps_e,
            }
            stats["straddles"] += st_t["straddles"] + st_e["straddles"]
            stats["decisive"] += st_t["decisive"] + st_e["decisive"]
            return x_out, joined, node_dict, stats
        # Decisive: single branch, on the pre-guard (x, cur).
        stats["decisive"] = 1
        taken = "then" if decision == "then" else "else"
        branch_prog = then_prog if taken == "then" else else_prog
        # Sanity: nominal guard must agree with the interval decision.
        nominal_taken = "then" if cv_nominal == 0 else "else"
        assert nominal_taken == taken, \
            f"guard interval {g_iv} says {taken} but nominal guard " \
            f"{cv_nominal} says {nominal_taken} -- soundness bug"
        x_out, out_iv, branch_steps, st = self._exec_noisy(x, cur, branch_prog)
        node_dict = {
            "branch": True, "straddle": False,
            "in": iv_to_json(cur),
            "cond_interval": iv_to_json(g_iv),
            "cond_nominal": int(cv_nominal),
            "taken": taken,
            "out": iv_to_json(out_iv),
            "cond_steps": cond_steps,
            "steps": branch_steps,
        }
        stats["straddles"] += st["straddles"]
        stats["decisive"] += st["decisive"]
        return x_out, out_iv, node_dict, stats

    # -- reference semantics (brute force, for validity checks) --------

    @staticmethod
    def true_output_set_flat(x0, r, ops):
        """{ symbolic_result(x, ops) : x in [x0-r, x0+r] } (exact ints)."""
        from certificates import RULES as SRULES, var, eval_expr
        out = set()
        for x in range(x0 - r, x0 + r + 1):
            e = var()
            for op in ops:
                e = SRULES[op](e)
            out.add(eval_expr(e, x))
        return out

    @staticmethod
    def _prog_eval(x, prog):
        """Exact point semantics of a program (recursive, big ints)."""
        from certificates import RULES as SR, var, eval_expr
        from conditionals import is_ifz

        def walk(e, p):
            for step in p:
                if is_ifz(step):
                    _, cond, then_p, else_p = step
                    ec = walk(e, cond)
                    branch = then_p if eval_expr(ec, x) == 0 else else_p
                    e = walk(e, branch)  # branch on the pre-guard value
                else:
                    e = SR[step](e)
            return e

        return eval_expr(walk(var(), prog), x)

    @staticmethod
    def true_output_set_program(x0, r, prog):
        """{ program output at x : x in [x0-r, x0+r] } via exact semantics."""
        return {NoiseProofMachine._prog_eval(x, prog)
                for x in range(x0 - r, x0 + r + 1)}
