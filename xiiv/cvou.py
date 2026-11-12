"""CVOU -- Certificate-Verified Ordered Unmasking for masked-diffusion decoding.

Every sampler in this family so far lets the model choose WHERE to
commit next (confidence ranking) and, since CVAB, HOW MANY. None of
them constrains the ORDER -- and the order is exactly what the newest
result says matters for reasoning:

- Left-to-right training boosts diffusion-LM reasoning (Tsinghua
  LeapLab, ICML 2026 Oral, arXiv:2601.15165, Nov 8, 2026): training
  LLaDA-8B-Instruct with strict left-to-right unmasking + GRPO beats
  confidence-based unmasking on GSM8K (89.1% vs 86.9% SPG, matched
  conditions) and MATH-500. Diagnosis: unmasking order is not a
  cosmetic choice -- on reasoning it beats confidence order by
  2.2pp. The literature's order is either confidence (a heuristic)
  or strict L2R (a fixed rule). Neither is justified per-position by
  anything the sampler can check. CVOU's order is justified by the
  certificate itself.
- Masked diffusion as random sequential adsorption
  (arXiv:2611.07845, Nov 5, 2026): the error of parallel generation
  is the entropy production of the generation process relative to
  the reversed masking process; for Markov data the process is
  random sequential adsorption with labelled gaps, solved exactly;
  the longer the correlations, the more the error concentrates at
  the start of generation. Diagnosis absorbed: commit order
  interacts with correlation structure -- early commits fix the
  context later commits see. CVAB read the correlation length off
  the certificate (interaction radius n-1); CVOU reads the ORDER
  off the same certificate.
- FEROM (arXiv:2611.10369) / PURL (arXiv:2611.11551): reveal-order
  marginal RL and planned unmasking-and-revision joint learning.
  Both need training/retraining -- shelf, future phase only.
  Diagnosis absorbed: reveal order is an endogenous variable worth
  optimizing. CVOU optimizes it greedily under a certificate
  instead of by RL: the order is whatever the certificate's closure
  rule admits, and the model only ranks within the admissible set.
- PV-Decoding (arXiv:2610.24722) / SpecCTMC (arXiv:2611.08362):
  commit-then-verify with self-referential verifiers; exact
  distribution-preserving speculative sampling for masked diffusion
  (1.73-4.38x fewer sequential evals, their numbers). Both verify
  with the model's own distributions or preserve the model's
  distribution. CVOU verifies with the independent no-repeat
  certificate instead, and preserves the certificate rather than
  the distribution.

CVOU's move: the certificate defines a partial order over positions
-- the closure frontier. A masked position ``p`` is order-admissible
iff every position in its left ``(n-1)``-window ``[p-n+1, p-1]`` is
already committed (frozen prompt tokens count as committed). Only
order-admissible positions may be committed; among them the model
ranks by safe-candidate confidence and commits the best. One commit
per round.

The payoff is PREFIX FINALITY -- incremental certification. When the
rightmost position of any n-gram window commits, that window is
fully committed and its verdict is final: no later commit can land
inside it, because every other position in the window is already
committed. That much is true of ANY commit order. What the closure
rule adds is that windows close in left-to-right order, so after
each round the committed PREFIX ``[0..m]`` (``m`` = leftmost masked
position minus one) is final -- every n-gram fully inside it is
decided forever, and a verifier can check the output incrementally,
round by round, instead of waiting for the whole canvas. In vanilla
confidence order commits are scattered, so the certified prefix
stays 0 until the leftmost position happens to commit -- typically
the last rounds -- and nothing is certifiable before the run ends.
CVOU's certified prefix grows every round; vanilla's does not.

Two corollaries, both provable from the admissibility rule alone:
- On a fully-masked canvas the admissible set is a singleton -- the
  leftmost masked position -- so CVOU IS strict left-to-right there.
  The certificate derives L2R; the Tsinghua empirical result (L2R
  beats confidence order on reasoning) is the measured shadow of
  this proof. CVOU is not "L2R plus a certificate": L2R is what the
  certificate's closure rule reduces to on a blank canvas.
- On a prompted canvas the admissible set is the set of segment
  heads -- the leftmost masked position of each maximal masked run --
  so segments advance independently and the model ranks the heads.
  The order is per-segment L2R with model-ranked segment
  interleaving, still with prefix finality per segment.
- Progress is guaranteed: the leftmost masked position is always
  admissible (everything left of it is committed), so the order rule
  alone can never stall the run; a stall means the gate refused
  every admissible candidate, and the run fails closed as usual.

- ADMISSIBILITY, not confidence. Each round computes the segment
  heads on the current canvas and ranks only those by their best
  safe candidate (reusing ``cvel._safe_candidates``). The model
  chooses among heads; the certificate chooses which positions are
  heads at all.
- PREFIX ACCOUNTING. The trace records ``prefix_area`` -- the sum
  over rounds of the certified-prefix length after each round -- and
  ``windows_closed`` -- the windows each commit finalizes. On a
  fully-masked canvas CVOU's prefix grows by exactly one per round;
  vanilla's stays 0 until the leftmost position commits. That is the
  falsifier, not a claim.
- PREEXISTING REPEAT fails closed at round zero: if the frozen
  prompt already contains a repeated n-gram, no commit sequence can
  un-violate it, so the run reports ``completed=False`` with zero
  forward passes and ``preexisting_repeat=True`` instead of emitting
  an uncertified canvas.
- THE CERTIFICATE IS THE REPLAY, not a score. The cvou-v1
  certificate itemizes every round (pre-round canvas, admissible
  set, chosen position, windows closed this round, certified prefix
  after the round, verified flag). ``cvou_verify_certificate``
  replays the chain model-free: each commit must have been
  order-admissible on its pre-round canvas, must have been masked
  there, applying the commit must reproduce the next round's
  pre-round canvas (or the final canvas), every intermediate canvas
  must pass the independent check, and the recorded prefix must
  match recomputation. ``completed=True`` additionally requires a
  mask-free, clean final canvas.
- TERMINATION: at most one forward pass per round, at most
  ``max_rounds`` rounds; a round that admits nothing stalls the run
  fail-closed with masks remaining rather than emitting an
  uncertified canvas. No logits are ever modified or fused.

Prior art (absorbed, not copied):
- L2R training (arXiv:2601.15165): order matters for reasoning.
  CVOU takes the diagnosis but not the mechanism: the order is
  derived from the certificate at inference time, with no
  retraining, and generalizes L2R to multi-segment canvases.
- RSA (arXiv:2611.07845): parallel error is structured and governed
  by a correlation length. CVOU takes the diagnosis but not the
  mechanism: the correlation length is read off the certificate
  (interaction radius n-1), not estimated from data.
- FEROM/PURL: reveal order is endogenous. CVOU takes the diagnosis
  but not the mechanism: the order is optimized greedily under a
  certificate instead of by RL.
- The Flexibility Trap / Who Gets Unmasked (arXiv:2610.24795):
  sampler choices change the output and must be reported
  per-configuration, never as a bare model property. CVOU's order
  is a sampler choice and the certificate says so.
- In-repo: dcmr._would_repeat / _topk_for_pos (the certificate
  gate), scv.verify_canvas (independent re-check), the CDR/CBRS/
  CWTC/CSBR/CGBS/CVEL/CVAB fail-closed pattern; cvel._safe_candidates
  (per-position gate-safe ranking) reused directly.

Pre-registered falsifiers (see xiiv/tests/test_cvou.py):
- F1 L2R order: on a fully-masked canvas, cvou commit positions are
  strictly increasing (0, 1, 2, ...).
- F2 same canvas, same cost: vanilla and cvou produce the identical
  final canvas on the distinct-position-stable stub, in the same
  number of forward passes.
- F3 prefix finality: on the adversarial right-first stub,
  vanilla's certified prefix is 0 until the final round
  (prefix_area 8 on length 8) while cvou's grows every round
  (prefix_area 36); cvou closes each window exactly once, in order.
- F4 segment interleave: on the two-segment prompted stub, cvou
  commits are L2R within each segment and every commit is a segment
  head at its round.
- F5 certificate replay: a real run verifies; tampering with the
  final canvas or with any round's chosen position fails
  verification.
- F6 fail-closed: a model offering no candidate above the confidence
  bar stalls with completed=False and masks remaining; a frozen
  prompt that already repeats fails closed at round zero with zero
  forward passes and preexisting_repeat=True.
- F7 determinism: identical runs -> identical canvas and certificate.
- KILL: F1 fails (order not L2R on a blank canvas) or F3 fails (no
  prefix-finality advantage) -> the module does not ship. F5 failing
  means the certificate is not actually a replay -> kill.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.
"""

import torch

from .cvel import _safe_candidates
from .scv import verify_canvas

__all__ = [
    "CVOUConfig",
    "CVOUTrace",
    "cvou_sample",
    "cvou_certificate",
    "cvou_verify_certificate",
    "order_admissible",
    "segment_heads",
    "closed_windows",
    "certified_prefix",
    "POLICIES",
]

POLICIES = ("vanilla", "cvou")


# ---------------------------------------------------------------------------
# Certificate geometry: the closure frontier
# ---------------------------------------------------------------------------

def order_admissible(canvas, pos: int, n: int, mask_id: int) -> bool:
    """Is masked position ``pos`` order-admissible on this canvas?

    Admissible iff every position in its left ``(n-1)``-window
    ``[pos-n+1, pos-1]`` (clipped to the canvas) is already committed
    -- not the mask token. Frozen prompt tokens count as committed.
    The leftmost masked position is always admissible: everything
    left of it is committed by definition.
    """
    lo = max(0, pos - n + 1)
    return all(canvas[q] != mask_id for q in range(lo, pos))


def segment_heads(canvas, n: int, mask_id: int) -> list:
    """The admissible set: the leftmost masked position of each
    maximal masked run. On a fully-masked canvas this is a singleton
    -- the leftmost masked position -- so CVOU reduces to strict
    left-to-right there."""
    return [p for p, t in enumerate(canvas)
            if t == mask_id and order_admissible(canvas, p, n, mask_id)]


def closed_windows(canvas, pos: int, n: int, mask_id: int) -> list:
    """Window starts finalized by committing ``pos`` on this canvas
    (``canvas`` is the pre-commit canvas).

    A window ``[s, s+n-1]`` containing ``pos`` is finalized by this
    commit iff every other position in it is already committed. The
    window's certificate verdict is final from this round on: no
    later commit can land inside it.
    """
    length = len(canvas)
    starts = []
    for s in range(max(0, pos - n + 1), min(pos, length - n) + 1):
        if all(canvas[q] != mask_id or q == pos for q in range(s, s + n)):
            starts.append(s)
    return starts


def certified_prefix(canvas, mask_id: int) -> int:
    """Length of the committed prefix: the first masked index, or the
    full length if nothing is masked. This prefix is final -- no
    future commit touches it -- so every n-gram inside it is decided
    forever."""
    for p, t in enumerate(canvas):
        if t == mask_id:
            return p
    return len(canvas)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class CVOUConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        topk: int = 4,
        max_rounds: int = 64,
        conf_threshold: float = 0.35,
    ):
        self.n = n                                # n-gram order of the certificate
        self.topk = topk                          # candidates scanned per position
        self.max_rounds = max_rounds              # hard termination cap
        self.conf_threshold = conf_threshold      # commit bar


class CVOUTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.n = 4
        self.rounds = 0
        self.forward_passes = 0
        self.commits = 0
        self.gate_rejects = 0         # top-1 candidates refused by the gate
        self.windows_closed = 0       # windows finalized across rounds
        self.prefix_area = 0          # sum of certified-prefix lengths per round
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen/prompt text already repeats
        self.stalled = False          # no admissible commit anywhere
        self.commit_log = []          # per-round audit entries

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def cvou_certificate(trace: CVOUTrace) -> dict:
    """The cvou-v1 certificate: the per-round order replay log."""
    return {
        "version": "cvou-v1",
        "policy": trace.policy,
        "n": trace.n,
        "rounds": trace.rounds,
        "forward_passes": trace.forward_passes,
        "commits": trace.commits,
        "gate_rejects": trace.gate_rejects,
        "windows_closed": trace.windows_closed,
        "prefix_area": trace.prefix_area,
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "preexisting_repeat": trace.preexisting_repeat,
        "stalled": trace.stalled,
        "commit_log": trace.commit_log,
    }


def cvou_verify_certificate(ids, cert: dict, mask_id: int) -> dict:
    """Independent re-check of a cvou-v1 certificate: no model, no
    sampler state.

    Replays every round: the chosen position must have been
    order-admissible on its stored pre-round canvas (for policy
    "cvou"; vanilla commits are exempt from the order rule but must
    still have been masked), must have been masked there, applying
    the commit must reproduce the next round's pre-round canvas (or
    the final canvas for the last round), every intermediate canvas
    must pass the independent no-repeat check, and the recorded
    certified prefix must match recomputation. ``completed=True``
    additionally requires a mask-free, clean final canvas. Returns
    per-check booleans and an overall ``valid`` flag.
    """
    ids = [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    n = int(cert.get("n", 4))
    policy = cert.get("policy", "cvou")
    checks = {}
    check = verify_canvas(ids, mask_id, n)
    checks["final_no_repeat"] = bool(check["final_no_repeat"])
    log = cert.get("commit_log", [])
    admissible = True
    chain = True
    accounting = True
    prev_after = None
    for i, entry in enumerate(log):
        before = [int(t) for t in entry.get("canvas_before", [])]
        pos = int(entry.get("pos", -1))
        tok = int(entry.get("tok", -1))
        if prev_after is not None and before != prev_after:
            chain = False
        if not (0 <= pos < len(before)):
            admissible = False
            continue
        if before[pos] != mask_id:
            admissible = False
        if policy == "cvou":
            if pos not in segment_heads(before, n, mask_id):
                admissible = False
            if sorted(entry.get("admissible", [])) != sorted(
                    segment_heads(before, n, mask_id)):
                accounting = False
        if entry.get("windows_closed") != closed_windows(
                before, pos, n, mask_id):
            accounting = False
        after = list(before)
        after[pos] = tok
        if not verify_canvas(after, mask_id, n)["final_no_repeat"]:
            admissible = False
        if entry.get("prefix_after") != certified_prefix(after, mask_id):
            accounting = False
        if i == len(log) - 1:
            if after != ids:
                chain = False
        prev_after = after
    checks["commits_admissible"] = admissible
    checks["chain_exact"] = chain
    checks["accounting_consistent"] = accounting
    checks["completed_sound"] = (
        (not cert.get("completed"))
        or (mask_id not in ids and checks["final_no_repeat"]))
    checks["valid"] = (checks["commits_admissible"]
                       and checks["chain_exact"]
                       and checks["accounting_consistent"]
                       and checks["completed_sound"])
    return checks


def cvou_sample(model, length: int, mask_id: int, config=None,
                policy: str = "cvou", frozen=None):
    """Run the CVOU sampler.

    ``frozen`` maps position -> token for prompt tokens that are never
    committed (and never tabooed). Returns (ids, trace, certificate);
    ``ids`` is a LongTensor with ``mask_id`` where the run did not
    complete. Deterministic (greedy); no sampling anywhere.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; want one of {POLICIES}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    cfg = config if config is not None else CVOUConfig()
    trace = CVOUTrace(policy)
    trace.n = cfg.n
    canvas = [mask_id] * length
    frozen = dict(frozen) if frozen else {}
    for pos, tok in frozen.items():
        canvas[pos] = int(tok)

    # Preexisting repeat: no commit sequence can un-violate a canvas
    # whose frozen text already repeats -- fail closed at round zero.
    if not verify_canvas(canvas, mask_id, cfg.n)["final_no_repeat"]:
        trace.preexisting_repeat = True
        trace.completed = False
        trace.final_no_repeat = False
        trace.stalled = True
        cert = cvou_certificate(trace)
        return torch.tensor(canvas, dtype=torch.long), trace, cert

    for round_no in range(cfg.max_rounds):
        masked = [p for p, t in enumerate(canvas) if t == mask_id]
        if not masked:
            break
        # The order rule: only segment heads may commit. Vanilla
        # exempts itself from the rule (every masked position is a
        # candidate) so the prefix-finality comparison is apples to
        # apples on the same gate and the same ranking.
        candidates = (segment_heads(canvas, cfg.n, mask_id)
                      if policy == "cvou" else masked)
        with torch.no_grad():
            logits = model(torch.tensor([canvas], dtype=torch.long))
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)

        ranked = []
        for pos in candidates:
            safe, top1_rejected = _safe_candidates(
                canvas, pos, probs[pos], cfg, mask_id)
            if top1_rejected:
                trace.gate_rejects += 1
            if safe:
                ranked.append((pos, safe))
        if not ranked:
            # Stall: no safe candidate at any admissible head -- fail
            # closed. (The leftmost masked position is always
            # admissible, so the order rule alone can never stall.)
            trace.stalled = True
            break
        ranked.sort(key=lambda ps: (-ps[1][0][1], ps[0]))
        pos, safe = ranked[0]
        tok, prob = safe[0]

        canvas_before = list(canvas)
        closed = closed_windows(canvas, pos, cfg.n, mask_id)
        canvas[pos] = tok
        prefix_after = certified_prefix(canvas, mask_id)
        trace.commits += 1
        trace.windows_closed += len(closed)
        trace.prefix_area += prefix_after
        trace.commit_log.append({
            "round": round_no,
            "canvas_before": canvas_before,
            "admissible": list(candidates),
            "pos": pos,
            "tok": tok,
            "prob": round(prob, 4),
            "windows_closed": closed,
            "prefix_after": prefix_after,
            "verified": bool(
                verify_canvas(canvas, mask_id, cfg.n)["final_no_repeat"]),
        })
        trace.rounds += 1

    trace.completed = mask_id not in canvas
    check = verify_canvas(canvas, mask_id, cfg.n)
    trace.final_no_repeat = bool(check["final_no_repeat"])
    cert = cvou_certificate(trace)
    return torch.tensor(canvas, dtype=torch.long), trace, cert
