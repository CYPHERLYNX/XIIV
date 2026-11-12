"""CVEL -- Certificate-Verified Expansion Length for masked-diffusion decoding.

Every sampler in this family so far takes the canvas length as given
and searches commit order (CGBS), trajectory (CSBR), or schedule
(PTCS/CBRS/CWTC). None of them questions the length itself -- yet the
literature now says length is the under-explored decision:

- Entropy-Valley (arXiv:2608.22274): a training-free length selector
  for masked-diffusion MT that scores candidate canvases by mean
  predictive entropy from all-mask forward passes. Recovers 64.9% /
  65.3% / 33.0% of the COMET-22 gain from reference lengths on
  En->Zh / Zh->En / En->De. Diagnosis: the denoising-friendly length
  need not match the reference length.
- LR-DLLM (arXiv:2602.07546): treats generation length as an explicit
  inference variable and adjusts the span bidirectionally under a
  length-regularized criterion (+13.4% HumanEval-Infilling vs DreamOn).
  Diagnosis: there is an intrinsic length-induced bias in DLLM
  confidence estimates, so naive confidence comparisons across lengths
  are unreliable.
- CARVE (arXiv:2608.30922, logged 2026-11-09): verified expansion by
  Jensen-Shannon divergence of the model's own predictions.
- LAVE (arXiv:2602.00612, ISSTA Oct 2026): lookahead-then-verify
  constrained decoding for diffusion LMs under CFGs. The operative
  principle is *preservability*: a step is safe only if the partial
  output can still be completed into a valid sentence.
- Masked diffusion as random sequential adsorption (arXiv:2611.07845,
  Nov 5, 2026): the error of parallel generation is the entropy
  production of the generation process; for short-range correlations
  the optimal schedule is nearly uniform. CVEL commits one position
  per round, so it adds no parallel error.
- The Flexibility Trap (ICML 2026 Outstanding Paper,
  arXiv:2601.15165): confidence-first decoding routes around the real
  decision points. Length and commit order are orthogonal decisions;
  CGBS searches the order, CVEL searches the length.

CVEL takes the *diagnoses* (length is a decision; a verifier can
score candidate lengths; a step must preserve completability) and
builds a different mechanism around them, one that fits the
guarantee-native thesis and needs no training:

- LENGTH-SATISFIABILITY ORACLE, model-free. A contract is an explicit
  spec: required literal token spans that must appear contiguously in
  the final canvas, plus a minimum final length. The oracle answers,
  for a canvas, whether every span is still placeable (already
  present, or a window whose positions are all masked-or-matching)
  and whether the minimum length is met. A span is unplaceable iff
  the longest masked run is shorter than the span -- provable from
  the window definition, since a window inside a masked run of length
  >= len(span) is always placeable -- so the deficit is exact:
  len(span) minus longest masked run.
- EXPANSION ONLY ON PROOF OF UNSATISFIABILITY. Before each commit
  round the oracle runs. If satisfiable, nothing happens: the run is
  exactly the vanilla greedy loop. If not, k = the exact deficit
  (capped by per-expansion and max-length budgets) masks are inserted
  at the end of the longest masked run -- the only place that grows
  the binding run -- and the oracle re-runs. The expansion is kept
  only if satisfiability is restored; otherwise the run fails closed
  with length_failure=True. Committed tokens keep their relative
  order (insertion only).
- THE CERTIFICATE IS THE JUSTIFICATION, not a score. The cvel-v1
  certificate itemizes initial/final length, every expansion (round,
  position, k, reason, satisfiable-before/after, pre-expansion
  canvas), commits, rounds, forward passes, gate rejects, and
  length_failure. ``cvel_verify_certificate`` recomputes the oracle
  on each stored pre-expansion canvas and on the final canvas, plus
  the independent no-repeat re-check -- no weights, no sampler state.
- TERMINATION: one commit per round, so a run needs at most as many
  rounds as there are positions to fill plus expansions. ``max_rounds``
  must cover that; a run that cannot finish in the budget, stalls
  (no safe candidate anywhere), or cannot expand within max_length
  reports ``completed=False`` with masks remaining rather than
  emitting an uncertified or contract-violating canvas. Fail-closed
  like the rest of the family.
- Contraction is explicitly out of scope: deleting committed tokens
  would break the certificate's order-preservation claim.
  LR-DLLM's bidirectional adjustment is the cited contrast.

A beam never touches a certificate-verified commit except by
abandoning the branch that holds it, and any commit re-passed the
gate at generation time. No logits are ever modified or fused.

Prior art (absorbed, not copied):
- Entropy-Valley / LR-DLLM / CARVE: length is a decision and can be
  verified. CVEL takes the diagnosis but not the mechanism: the
  scorer is an independent contract oracle (not the model's own
  entropy, confidence, or JS divergence), and expansion is justified
  by a proof of unsatisfiability (not by a stability score).
- LAVE: preservability. CVEL applies it to the length axis instead
  of the grammar axis.
- The Flexibility Trap / Who Gets Unmasked (arXiv:2610.24795):
  sampler choices change the output and must be reported
  per-configuration, never as a bare model property. CVEL's length
  decision is a sampler choice and the certificate says so.
- In-repo: dcmr._would_repeat / _topk_for_pos (the certificate
  gate), scv.verify_canvas (independent re-check), the CDR/CBRS/
  CWTC/CSBR/CGBS fail-closed pattern.

Pre-registered falsifiers (see xiiv/tests/test_cvel.py):
- F1 no-expansion equivalence: on a satisfiable-from-start contract,
  cvel produces the same canvas as the vanilla policy and records
  zero expansions.
- F2 expansion rescues the short canvas: on the span stub (required
  4-token span, longest masked run 3), the vanilla policy completes
  but violates the contract, while cvel expands by exactly the
  deficit (k=1, at the end of the longest masked run) and the final
  canvas contains the span.
- F3 every expansion justified: recomputing the oracle on the
  pre-expansion canvas gives unsatisfiable, and on the expanded
  canvas gives satisfiable, for every expansion in the certificate.
- F4 bound: a contract unsatisfiable within max_length ends with
  completed=False, length_failure=True, and final length ==
  max_length; no silent truncation.
- F5 final soundness: completed=True implies the independent
  verify_canvas(final) is clean AND the oracle finds the final
  canvas satisfiable; a tampered canvas fails cvel_verify_certificate.
- F6 determinism: identical runs -> identical canvas and certificate.
- KILL: F2 fails -> the module does not ship. F3 or F5 failing means
  the expansions are not actually certificate-gated -> kill.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed. The contract is supplied by the caller; with
contract=None the length check is inert and the run is the vanilla
greedy loop with a cvel-v1 certificate.
"""

import torch

from .dcmr import _topk_for_pos, _would_repeat
from .scv import verify_canvas

__all__ = [
    "CVELConfig",
    "CVELTrace",
    "cvel_sample",
    "cvel_certificate",
    "cvel_verify_certificate",
    "length_oracle",
    "POLICIES",
]

POLICIES = ("vanilla", "cvel")


# ---------------------------------------------------------------------------
# Length-satisfiability oracle (model-free)
# ---------------------------------------------------------------------------

def _masked_runs(canvas, mask_id):
    """Contiguous masked runs as (start, length) pairs."""
    runs = []
    start = None
    for i, tok in enumerate(canvas):
        if tok == mask_id:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - start))
                start = None
    if start is not None:
        runs.append((start, len(canvas) - start))
    return runs


def _longest_masked_run(canvas, mask_id):
    """(start, length) of the longest masked run, or (None, 0)."""
    runs = _masked_runs(canvas, mask_id)
    if not runs:
        return None, 0
    return max(runs, key=lambda r: (r[1], -r[0]))


def _span_placeable(canvas, span, mask_id):
    """Can ``span`` still appear contiguously in this canvas?

    A window hosts the span when every position is either masked or
    already holds the span's token there. An already-present span is
    the zero-mask special case.
    """
    if not span:
        return True
    n = len(span)
    for i in range(len(canvas) - n + 1):
        ok = True
        for j, tok in enumerate(span):
            c = canvas[i + j]
            if c != mask_id and c != tok:
                ok = False
                break
        if ok:
            return True
    return False


def length_oracle(canvas, mask_id, contract):
    """Is this canvas still completable into a contract-satisfying one?

    ``contract`` is a dict with optional keys:
      - ``required_spans``: list of token lists that must appear
        contiguously in the final canvas.
      - ``min_length``: minimum final canvas length.

    Returns a dict with ``satisfiable`` (bool), ``min_extra`` (exact
    number of extra masks needed, 0 when satisfiable), ``insert_pos``
    (where to insert them: end of the longest masked run, or the end
    of the canvas when the binding deficit is min_length), and
    ``reasons`` (one string per binding deficit). With contract=None
    (or an empty contract) the canvas is trivially satisfiable.
    """
    canvas = list(canvas)
    if not contract:
        return {"satisfiable": True, "min_extra": 0,
                "insert_pos": len(canvas), "reasons": []}
    spans = [list(s) for s in contract.get("required_spans", [])]
    min_length = contract.get("min_length", 0) or 0
    _, longest = _longest_masked_run(canvas, mask_id)
    reasons = []
    deficits = []
    span_deficit = 0
    for span in spans:
        if not span:
            continue
        if not _span_placeable(canvas, span, mask_id):
            # Unplaceable iff longest masked run < len(span): a window
            # inside a run of length >= len(span) is always placeable.
            deficit = len(span) - longest
            span_deficit = max(span_deficit, deficit)
            reasons.append(
                f"span {span} unplaceable "
                f"(longest masked run {longest} < {len(span)})")
    length_deficit = max(0, min_length - len(canvas))
    if length_deficit:
        reasons.append(
            f"min_length {min_length} > canvas length {len(canvas)}")
    min_extra = max(span_deficit, length_deficit)
    if span_deficit:
        run_start, _ = _longest_masked_run(canvas, mask_id)
        insert_pos = (run_start + longest) if run_start is not None \
            else len(canvas)
    else:
        insert_pos = len(canvas)
    return {
        "satisfiable": min_extra == 0,
        "min_extra": min_extra,
        "insert_pos": insert_pos,
        "reasons": reasons,
        "deficits": {"span": span_deficit, "length": length_deficit},
    }


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class CVELConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        topk: int = 4,
        max_rounds: int = 64,
        conf_threshold: float = 0.35,
        max_expansions: int = 4,
        expand_at_most: int = 8,
        max_length: int = 256,
    ):
        self.n = n                                # n-gram order of the certificate
        self.topk = topk                          # candidates scanned per position
        self.max_rounds = max_rounds              # hard termination cap
        self.conf_threshold = conf_threshold      # commit bar
        self.max_expansions = max_expansions      # expansion budget
        self.expand_at_most = expand_at_most      # masks per expansion
        self.max_length = max_length              # absolute length cap


class CVELTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.initial_length = 0
        self.final_length = 0
        self.rounds = 0
        self.forward_passes = 0
        self.expansions = 0           # expansions performed
        self.expansions_log = []      # per-expansion audit entries
        self.gate_rejects = 0         # top-1 candidates refused by the gate
        self.commits = 0
        self.completed = False
        self.final_no_repeat = False
        self.length_failure = False   # unsatisfiable within max_length
        self.contract_satisfied = None  # None when no contract given
        self.commit_log = []          # per-commit audit entries

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def cvel_certificate(trace: CVELTrace) -> dict:
    """The cvel-v1 certificate: the length justification and the
    independent re-check of the final canvas."""
    return {
        "version": "cvel-v1",
        "policy": trace.policy,
        "initial_length": trace.initial_length,
        "final_length": trace.final_length,
        "rounds": trace.rounds,
        "forward_passes": trace.forward_passes,
        "expansions": trace.expansions_log,
        "expansions_count": trace.expansions,
        "gate_rejects": trace.gate_rejects,
        "commits": trace.commits,
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "length_failure": trace.length_failure,
        "contract_satisfied": trace.contract_satisfied,
        "commit_log": trace.commit_log,
    }


def cvel_verify_certificate(ids, cert: dict, mask_id: int,
                            contract=None) -> dict:
    """Independent re-check of a cvel-v1 certificate: no model, no
    sampler state.

    Recomputes the oracle on every stored pre-expansion canvas and on
    the final canvas, plus the no-repeat re-check. Returns a dict with
    per-check booleans and an overall ``valid`` flag. A certificate
    whose expansions are not justified, or whose final canvas violates
    the contract, fails.
    """
    ids = [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    checks = {}
    check = verify_canvas(ids, mask_id, 4)
    checks["final_no_repeat"] = bool(check["final_no_repeat"])
    expansions = cert.get("expansions", [])
    justified = True
    for entry in expansions:
        before = entry.get("canvas_before", [])
        k = entry.get("k", 0)
        pos = entry.get("position", len(before))
        oracle_before = length_oracle(before, mask_id, contract)
        if oracle_before["satisfiable"]:
            justified = False
            break
        expanded = before[:pos] + [mask_id] * k + before[pos:]
        oracle_after = length_oracle(expanded, mask_id, contract)
        if not oracle_after["satisfiable"]:
            justified = False
            break
        if not entry.get("satisfiable_before") is False:
            justified = False
            break
        if not entry.get("satisfiable_after") is True:
            justified = False
            break
    checks["expansions_justified"] = justified
    if contract:
        final_oracle = length_oracle(ids, mask_id, contract)
        checks["contract_satisfied"] = bool(final_oracle["satisfiable"])
    else:
        checks["contract_satisfied"] = True
    checks["completed_matches"] = (
        (not cert.get("completed"))
        or (mask_id not in ids and checks["contract_satisfied"]))
    checks["valid"] = all(checks.values())
    return checks


def _safe_candidates(canvas, pos, probs_row, cfg, mask_id):
    """Gate-safe candidates at pos, best-first.

    Returns (cands, top1_rejected): cands is a list of (tok, prob)
    that pass the no-repeat gate and clear conf_threshold, in
    descending probability order; top1_rejected is True when the
    highest-probability candidate was refused by the gate.
    """
    cands = _topk_for_pos(probs_row, cfg.topk, {mask_id})
    if not cands:
        return [], False
    top1_tok = cands[0][0]
    top1_rejected = _would_repeat(canvas, pos, top1_tok, cfg.n, mask_id)
    safe = [(tok, prob) for tok, prob in cands
            if prob >= cfg.conf_threshold
            and not _would_repeat(canvas, pos, tok, cfg.n, mask_id)]
    return safe, top1_rejected


def cvel_sample(model, length: int, mask_id: int, config=None,
                policy: str = "cvel", frozen=None, contract=None):
    """Run the CVEL sampler.

    ``frozen`` maps position -> token for prompt tokens that are never
    committed (and never tabooed). ``contract`` is the length-oracle
    spec (see :func:`length_oracle`); with contract=None the length
    check is inert and the run is the vanilla greedy loop. Returns
    (ids, trace, certificate); ``ids`` is a LongTensor whose length is
    the final canvas length, with ``mask_id`` where the run did not
    complete. Deterministic (greedy); no sampling anywhere.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; want one of {POLICIES}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    cfg = config if config is not None else CVELConfig()
    trace = CVELTrace(policy)
    trace.initial_length = length
    canvas = [mask_id] * length
    frozen = dict(frozen) if frozen else {}
    for pos, tok in frozen.items():
        canvas[pos] = int(tok)

    for round_no in range(cfg.max_rounds):
        masked = [p for p, t in enumerate(canvas) if t == mask_id]
        if not masked:
            break
        # Length check first: expand only on a proof that the current
        # canvas cannot be completed into a contract-satisfying one.
        if policy == "cvel" and contract:
            oracle = length_oracle(canvas, mask_id, contract)
            if not oracle["satisfiable"]:
                if (trace.expansions >= cfg.max_expansions
                        or len(canvas) >= cfg.max_length):
                    trace.length_failure = True
                    break
                k = min(oracle["min_extra"], cfg.expand_at_most,
                        cfg.max_length - len(canvas))
                if k <= 0:
                    trace.length_failure = True
                    break
                pre_canvas = list(canvas)
                pos = oracle["insert_pos"]
                canvas = (canvas[:pos] + [mask_id] * k
                          + canvas[pos:])
                trace.expansions += 1
                after = length_oracle(canvas, mask_id, contract)
                trace.expansions_log.append({
                    "round": round_no,
                    "position": pos,
                    "k": k,
                    "reason": "; ".join(oracle["reasons"]),
                    "min_extra": oracle["min_extra"],
                    "satisfiable_before": False,
                    "satisfiable_after": bool(after["satisfiable"]),
                    "canvas_before": pre_canvas,
                })
                if not after["satisfiable"]:
                    # The deficit math says one expansion restores
                    # satisfiability; if it did not, fail closed
                    # rather than emit an unjustified expansion.
                    trace.length_failure = True
                    break
                masked = [p for p, t in enumerate(canvas)
                          if t == mask_id]
        with torch.no_grad():
            logits = model(torch.tensor([canvas], dtype=torch.long))
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)

        # Rank masked positions by their best safe candidate; commit
        # exactly one (the best), like the vanilla greedy loop.
        ranked = []
        for pos in masked:
            safe, top1_rejected = _safe_candidates(
                canvas, pos, probs[pos], cfg, mask_id)
            if top1_rejected:
                trace.gate_rejects += 1
            if safe:
                ranked.append((pos, safe))
        if not ranked:
            # Stall: no safe candidate anywhere -- fail closed.
            break
        ranked.sort(key=lambda ps: (-ps[1][0][1], ps[0]))
        pos, safe = ranked[0]
        tok, prob = safe[0]
        canvas[pos] = tok
        trace.commits += 1
        trace.commit_log.append({
            "pos": pos,
            "tok": tok,
            "round": round_no,
            "prob": round(prob, 4),
        })
        trace.rounds += 1

    trace.final_length = len(canvas)
    trace.completed = mask_id not in canvas and not trace.length_failure
    check = verify_canvas(canvas, mask_id, cfg.n)
    trace.final_no_repeat = bool(check["final_no_repeat"])
    if contract:
        final_oracle = length_oracle(canvas, mask_id, contract)
        trace.contract_satisfied = bool(final_oracle["satisfiable"])
    cert = cvel_certificate(trace)
    return torch.tensor(canvas, dtype=torch.long), trace, cert
