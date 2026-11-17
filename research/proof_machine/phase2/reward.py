"""
Phase 2, step 2 — dense certificate-obligation reward for RL training of
the learned composition policy (spec research/composition_operator_v1.md §6,
option 2: "RL with certificate-validity as a dense reward").

Recipe grounding (Part A, 2026-11-17; all from search-result abstracts,
numbers are the papers' own claims, not XIIV measurements):

- Verifier-Grounded Agentic RL (arXiv:2610.19298): a verifier-as-reward must
  be DENSE — the weighted fraction of proof obligations discharged, not a
  binary accept/reject — and needs an INTEGRITY AUDIT that outranks the
  verifier (reject any candidate that alters the spec or assumes the
  conclusion).
- Verified but Vacuous (arXiv:2611.05590, IBM): 99.1% verifier acceptance,
  only 1.9% non-vacuous — RL against a binary verifier reward learns to
  eliminate proof obligations. Our evaluation's honest metric is therefore
  verified_success (certificate verifies AND concrete == symbolic AND final
  value equals the target), never certificate validity alone.
- Compositional Reasoning under RL Post-Training (arXiv:2609.19465,
  NeurIPS 2026): decomposed-to-composed asymmetry — RL on decomposed skills
  fails to transfer to compound tasks (Level I compound extrapolation 23.8%),
  while compound-task RL transfers back to atomic skills (61.4%). The RL
  stage therefore trains on COMPOUND (long-plan) specs, not short ones.
- From f(x) and g(x) to f(g(x)) (ICLR 2026): RFT (supervised imitation)
  fails to learn composition even with compositional data; RL is the
  critical factor for acquiring generalizable compositional skills.
- RLVR for small models (arXiv:2605.30478): combined/dense rewards beat
  single rewards on 0.6B/1B models; reward shaping can induce degeneration
  (e.g. toward shorter completions) — so this reward has NO length term.

Reward definition (this module):

  An "obligation" is one probe's full check: execute the sampled plan,
  the independent certificate verifies, AND the final symbolic value
  equals the realized target. The reward is the FRACTION of the 5 probes'
  obligations discharged, in [0, 1].

  The integrity audit is target equality itself: a plan whose certificate
  verifies but which computes a different function than the target is
  paid only for the probes where it happens to agree (usually 0) — never
  for certificate validity alone. A verified-but-wrong plan scores ~0.

  The training-time reward uses the PURE SYMBOLIC path (RULES + eval_expr,
  no neural execution): the policy chooses ops, and the symbolic function
  is what the ops determine. Neural execution faithfulness
  (concrete == symbolic) is a property of the FROZEN primitives, not of
  the policy, and Phase-1 measured it at 100% (V = 100% at every depth;
  step-1 eval: 0 cert-invalid in 7,500 probe-executions). The full
  neural + certificate path is audited separately (full_verify, and the
  pre-registered eval_policy.py evaluation, which runs pm.run +
  verify_certificate on every held-out spec).

No GPU. CPU-only, pure Python + torch-free symbolic evaluation.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PM)

from certificates import RULES, eval_expr, verify_certificate  # noqa: E402
from vrc_harness import parse_target, realize  # noqa: E402

PROBES = (5, -3, 42, 1000, -1000)


def symbolic_eval(ops, x0):
    """Exact reference: pure symbolic execution (Python big ints)."""
    e = None
    from certificates import var
    e = var()
    for op in ops:
        if op not in RULES:
            raise ValueError(f"unknown op {op}")
        e = RULES[op](e)
    return eval_expr(e, int(x0))


def probe_reward(spec, ops, probes=PROBES):
    """Dense reward for one sampled plan.

    Returns (reward, detail): reward = fraction of probes whose full
    obligation is discharged (symbolic result == realized target).
    Empty plans, unknown ops, and unparseable specs score 0.0 — loudly
    labelled in `detail`, never silently.
    """
    if not ops:
        return 0.0, {"fail": "empty-plan", "probes_hit": 0,
                     "probes_total": len(probes)}
    try:
        target = realize(parse_target(spec))
    except Exception as ex:  # noqa: BLE001 — an unparseable spec pays 0
        return 0.0, {"fail": f"spec-parse-error:{type(ex).__name__}",
                     "probes_hit": 0, "probes_total": len(probes)}
    hits = 0
    for x0 in probes:
        try:
            if symbolic_eval(ops, x0) == eval_expr(target, x0):
                hits += 1
        except Exception:  # noqa: BLE001 — a failing probe pays 0
            pass
    return hits / len(probes), {"fail": None, "probes_hit": hits,
                                "probes_total": len(probes)}


def full_verify(spec, ops, pm, probes=PROBES):
    """Full audit path: neural execution + independent certificate
    verification + target equality, per probe. Used to audit the
    symbolic-reward assumption (certificates are structural) on sampled
    RL rollouts. Returns a dict with per-probe fractions.

    pm: a ProofMachine instance (loads the frozen primitive nets).
    """
    if not ops:
        return {"verified_frac": 0.0, "cert_valid_frac": 0.0,
                "fail": "empty-plan"}
    try:
        target = realize(parse_target(spec))
    except Exception as ex:  # noqa: BLE001
        return {"verified_frac": 0.0, "cert_valid_frac": 0.0,
                "fail": f"spec-parse-error:{type(ex).__name__}"}
    verified = cert_ok = 0
    fail = None
    for x0 in probes:
        try:
            result, cert = pm.run(x0, ops)
            sym = pm.symbolic_result(x0, ops)
            vok, _ = verify_certificate(cert)
            if vok:
                cert_ok += 1
            if result == sym and vok and sym == eval_expr(target, x0):
                verified += 1
        except Exception as ex:  # noqa: BLE001 — any execution failure is a fail
            fail = f"execution-error:{type(ex).__name__}"
            break
    return {"verified_frac": verified / len(probes),
            "cert_valid_frac": cert_ok / len(probes), "fail": fail}
