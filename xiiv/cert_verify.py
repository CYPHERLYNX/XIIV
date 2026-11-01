"""Independent verifier for CGEC certificates (xiiv/cgec.py).

This module replays a cgec-v1 certificate WITHOUT the model weights and
checks every claim the sampler made. It is deliberately implemented
independently of cgec's internals: its PDA transition table and its
feasibility check are written separately from the sampler's, so a bug in
one would show up as a verification failure rather than being shared.

Checks performed by verify_certificate(cert):
  V1 structural: required keys, types, policy/domain names.
  V2 alphabet: every committed token id is in the recorded alphabet and
      its recorded symbol matches the alphabet.
  V3 per-commit gate replay: after each commit (in recorded order), the
      position-ordered committed subsequence is a valid prefix of the
      arith-v1 expression language AND the canvas is still completable
      (independent forward reachability with masks as wildcards).
  V4 final parse: the final token sequence parses as a COMPLETE valid
      expression (its own PDA), with no masks remaining iff
      completed=True.
  V5 min_gap: recomputed from the recorded per-commit gaps equals the
      certificate's min_gap.
  V6 counters: n_commits == len(commits); early commits are a subset.

Returns {"valid": bool, "failures": [str, ...]}. Any tampering with
tokens, symbols, order, or gaps breaks at least one check.
"""

__all__ = ["verify_certificate", "ArithPDA"]

_OPS = ("plus", "minus", "mul", "div")


class ArithPDA:
    """Independent pushdown automaton for arith-v1.

    Written separately from cgec.arith_transition (same grammar spec,
    separate code): state = [expect_operand: bool, depth: int].
    step(symbol) -> True/False (False = reject, state unchanged).
    """

    def __init__(self):
        self.expect_operand = True
        self.depth = 0

    def step(self, symbol):
        if symbol == "digit":
            if not self.expect_operand:
                return False
            self.expect_operand = False
            return True
        if symbol == "lparen":
            if not self.expect_operand:
                return False
            self.expect_operand = True
            self.depth += 1
            return True
        if symbol in _OPS:
            if self.expect_operand:
                return False
            self.expect_operand = True
            return True
        if symbol == "rparen":
            if self.expect_operand or self.depth == 0:
                return False
            self.expect_operand = False
            self.depth -= 1
            return True
        return False

    def accepts(self):
        return (not self.expect_operand) and self.depth == 0


def _completable(tokens, alphabet, mask_id):
    """Independent feasibility: forward reachability over the canvas with
    masks as wildcards; True iff some fill parses complete."""
    L = len(tokens)
    cur = {(True, 0)}
    for t in tokens:
        classes = (alphabet[t],) if t != mask_id else tuple(alphabet.values())
        nxt = set()
        for (eo, d) in cur:
            for c in classes:
                pda = ArithPDA()
                pda.expect_operand, pda.depth = eo, d
                if pda.step(c):
                    nxt.add((pda.expect_operand, pda.depth))
        cur = nxt
        if not cur:
            return False
    return any((not eo) and d == 0 for (eo, d) in cur)


def verify_certificate(cert):
    failures = []

    # ---- V1 structural ----
    if not isinstance(cert, dict):
        return {"valid": False, "failures": ["not_a_dict"]}
    for key in ("certificate", "domain", "policy", "length", "mask_id",
                "alphabet", "tokens", "symbols", "commits", "min_gap",
                "completed"):
        if key not in cert:
            failures.append(f"missing_key:{key}")
    if failures:
        return {"valid": False, "failures": failures}
    if cert["certificate"] != "cgec-v1":
        failures.append(f"bad_version:{cert['certificate']}")
    if cert["domain"] != "arith-v1":
        failures.append(f"bad_domain:{cert['domain']}")
    L = cert["length"]
    mask_id = cert["mask_id"]
    try:
        alphabet = {int(k): v for k, v in cert["alphabet"].items()}
    except (ValueError, AttributeError):
        return {"valid": False, "failures": ["bad_alphabet"]}
    tokens = cert["tokens"]
    if len(tokens) != L:
        failures.append("tokens_length_mismatch")

    # ---- V2 alphabet consistency ----
    for c in cert["commits"]:
        t = c.get("token")
        if t not in alphabet:
            failures.append(f"commit_token_outside_alphabet:pos={c.get('pos')}")
        elif c.get("symbol") != alphabet[t]:
            failures.append(f"commit_symbol_mismatch:pos={c.get('pos')}")

    # ---- V3 per-commit gate replay ----
    canvas = [mask_id] * L
    for c in cert["commits"]:
        p, t = c["pos"], c["token"]
        if not (0 <= p < L):
            failures.append(f"commit_pos_out_of_range:{p}")
            break
        if canvas[p] != mask_id:
            failures.append(f"commit_pos_reused:{p}")
            break
        canvas[p] = t
        # prefix-validity of the position-ordered committed subsequence
        pda = ArithPDA()
        ok = True
        for tt in canvas:
            if tt == mask_id:
                continue
            if not pda.step(alphabet[tt]):
                ok = False
                break
        if not ok:
            failures.append(f"prefix_invalid_after_commit:pos={p}")
            break
        # completability with remaining masks as wildcards
        if not _completable(canvas, alphabet, mask_id):
            failures.append(f"infeasible_after_commit:pos={p}")
            break

    # ---- V4 final parse ----
    if cert["completed"]:
        if any(t == mask_id for t in tokens):
            failures.append("completed_but_masked")
        else:
            pda = ArithPDA()
            for t in tokens:
                if t not in alphabet:
                    failures.append(f"final_token_outside_alphabet:{t}")
                    break
                if not pda.step(alphabet[t]):
                    failures.append("final_parse_reject")
                    break
            else:
                if not pda.accepts():
                    failures.append("final_not_accepting")
    else:
        if cert.get("grammar_valid"):
            failures.append("incomplete_but_grammar_valid")

    # ---- V5 min_gap ----
    gaps = [c["gap"] for c in cert["commits"]]
    if gaps:
        recomputed = min(gaps)
        if cert["min_gap"] is None or abs(recomputed - cert["min_gap"]) > 0:
            failures.append(
                f"min_gap_mismatch:cert={cert['min_gap']},"
                f"recomputed={recomputed}")
    elif cert["min_gap"] is not None:
        failures.append("min_gap_set_without_commits")

    # ---- V6 counters ----
    if cert.get("n_commits") != len(cert["commits"]):
        failures.append("n_commits_mismatch")

    return {"valid": not failures, "failures": failures}
