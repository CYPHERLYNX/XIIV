"""CVAB -- Certificate-Verified Adaptive Batching for masked-diffusion decoding.

Every sampler in this family so far commits exactly one position per
round. The width of a commit round -- how many positions to unmask in
one forward pass -- has been a fixed hyperparameter everywhere, and
the recent literature says it should not be:

- Masked diffusion as random sequential adsorption
  (arXiv:2611.07845, Nov 5, 2026): the error of parallel generation
  is the entropy production of the generation process relative to the
  reversed masking process; for Markov data the process is random
  sequential adsorption with labelled gaps, solved exactly; the
  longer the correlations, the more the error concentrates at the
  start of generation; for the short-range correlations of English
  text the optimal schedule is nearly uniform. Diagnosis: commit
  width is a real decision with a measurable, structured error, and
  the error is governed by a correlation length.
- Batch-Paced (arXiv:2610.17917): fixed-budget batch decoding for
  dLLMs -- jointly ranks masked-token confidence across requests and
  commits the same fixed number of tokens for every request at each
  denoising pass; training-free; gains grow with batch size.
  Diagnosis: width is a first-class decoding variable and fixing it
  per request wastes budget. Mechanism NOT ported: the width is a
  fixed hyperparameter, decided by cross-request confidence ranking
  -- never justified for the batch being committed.
- PV-Decoding (arXiv:2610.24722): confidence-driven speculative
  decoding with progressive verification for block diffusion -- masked
  positions are partitioned into directly decoded / speculative /
  unresolved by two confidence thresholds, and speculative tokens are
  verified with progressively stronger policies; 8%-37% fewer
  sequential forward passes at <=1.3% accuracy drop (their numbers).
  Diagnosis: commit-then-verify beats commit-only, and verification
  strength should scale with risk. Mechanism NOT ported: the verifier
  is the model's own distributions (progressively stronger policies)
  -- self-referential. CVAB verifies each batch with the independent
  no-repeat certificate instead.
- Frequency-aware remasking test-time scaling (arXiv:2611.10486,
  logged 2026-11-10): tracks how often each token's prediction
  changes across denoising steps and commits frequently-changing
  tokens to explore alternative reasoning paths (+3.73pp LLaDA-8B /
  +7.33pp Dream-7B on MATH-500, their numbers). Diagnosis: prediction
  change across steps marks the decision points. Not ported: it picks
  *which* positions to reconsider; CVAB picks *how many* to commit
  per round. Complementary; a future run can stack them.
- FEROM (arXiv:2611.10369) / PURL (arXiv:2611.11551): reveal-order RL
  and planned unmasking-and-revision joint learning. Both need
  training/retraining -- shelf, future phase only. Diagnosis
  absorbed: reveal order is an endogenous variable; CVAB's batch is
  an order over positions too (rank order within a round), optimized
  greedily under a certificate instead of by RL.
- DMax (arXiv:2604.08302) / DSL-LLaDA (arXiv:2606.01024): aggressive
  parallel decoding and continuous denoising, both training-required.
  Not drop-in; shelf.

CVAB's move: the certificate's interaction radius IS the
correlation length. The no-repeat-n-gram certificate of order ``n``
has interaction radius ``n - 1``: two commits can only interact --
jointly complete a repeated n-gram -- if they are less than ``n``
apart, because a repeated n-gram needs ``n`` consecutive positions
and a window of length ``n`` contains two positions iff their
distance is at most ``n - 1``. So a batch is *provably*
certificate-safe iff (a) every member passes the per-commit gate
against the pre-batch canvas, and (b) members are pairwise at least
``n`` apart. No window then contains two new tokens, so each window's
check is exactly the single-commit check the gate already ran. The
width decision therefore never consults the model distribution: the
model only ranks candidates; the certificate decides how many may go
together. The width adapts per round to canvas geometry -- the
greedy maximal separated set among the ranked candidates -- instead
of being a fixed hyperparameter (Batch-Paced) or a confidence
heuristic (PV-Decoding).

- ADMISSIBILITY, not confidence. Each round walks the ranked masked
  positions best-first and admits a candidate only when it is at
  least ``n`` away from every already-admitted member. The walk
  stops when the batch reaches ``max_width`` or the ranked list is
  exhausted. Dropped candidates are counted as shrinks and itemized
  in the certificate, so a narrow batch is an audited fact, not a
  silent fallback.
- BATCH VERIFICATION (defense in depth). The admitted batch is
  simulated on the pre-round canvas and re-checked with the
  independent ``verify_canvas``. Admissibility plus a clean
  pre-batch canvas provably implies a clean simulated canvas, so a
  failed check is an alarm, not an expected path: the run falls back
  to the single best member (re-verified) and, if even that fails,
  stalls fail-closed. The alarm is counted in the trace.
- PREEXISTING REPEAT fails closed at round zero: if the frozen
  prompt already contains a repeated n-gram, no commit sequence can
  un-violate it, so the run reports ``completed=False`` with zero
  forward passes and ``preexisting_repeat=True`` instead of emitting
  an uncertified canvas.
- THE CERTIFICATE IS THE REPLAY, not a score. The cvab-v1
  certificate itemizes every round (pre-round canvas, batch members,
  target/examined/admitted widths, shrinks, min separation,
  verified flag). ``cvab_verify_certificate`` replays the chain
  model-free: each batch must be admissible on its pre-round canvas,
  each member must have been masked there, applying the batch must
  reproduce the next round's pre-round canvas (or the final canvas),
  and every intermediate canvas must pass the independent check.
  ``completed=True`` additionally requires a mask-free, clean final
  canvas. Width accounting (examined/shrinks) is audit metadata
  checked for internal consistency; the ranking itself is a sampler
  choice the certificate does not claim to verify.
- TERMINATION: at most one forward pass per round, at most
  ``max_rounds`` rounds; a round that admits nothing stalls the run
  fail-closed with masks remaining rather than emitting an
  uncertified canvas. No logits are ever modified or fused.

A beam never touches a certificate-verified commit except by
abandoning the branch that holds it, and any commit re-passed the
gate at generation time.

Prior art (absorbed, not copied):
- RSA (arXiv:2611.07845): parallel error is structured and governed
  by a correlation length. CVAB takes the diagnosis but not the
  mechanism: the correlation length is read off the certificate
  (interaction radius n-1), not estimated from data.
- Batch-Paced: width is a first-class variable. CVAB takes the
  diagnosis but not the mechanism: width is adaptive per round and
  justified per batch by the certificate, not fixed per request.
- PV-Decoding: commit-then-verify. CVAB takes the diagnosis but not
  the mechanism: the verifier is the independent no-repeat
  certificate, not the model's own distributions.
- The Flexibility Trap / Who Gets Unmasked (arXiv:2610.24795):
  sampler choices change the output and must be reported
  per-configuration, never as a bare model property. CVAB's batch
  width is a sampler choice and the certificate says so.
- In-repo: dcmr._would_repeat / _topk_for_pos (the certificate
  gate), scv.verify_canvas (independent re-check), the CDR/CBRS/
  CWTC/CSBR/CGBS/CVEL fail-closed pattern; cvel._safe_candidates
  (per-position gate-safe ranking) reused directly.

Pre-registered falsifiers (see xiiv/tests/test_cvab.py):
- F1 width-1 equivalence: policy="vanilla" and policy="cvab" with
  max_width=1 produce the same canvas on the same model.
- F2 batch speedup, same canvas: on a 16-mask stub with distinct
  position-stable proposals, cvab completes in 4 forward passes
  where vanilla needs 16, with an identical final canvas.
- F3 cross-commit safety: on the adjacent-conflict stub, the four
  top-ranked positions would jointly complete a repeated 4-gram; a
  naive batch violates the certificate, while cvab shrinks to width
  1 per round, records the shrinks, and finishes clean.
- F4 certificate replay: a real run verifies; tampering with the
  final canvas or with any round's batch fails verification.
- F5 fail-closed: a model offering no candidate above the confidence
  bar stalls with completed=False and masks remaining; a frozen
  prompt that already repeats fails closed at round zero with zero
  forward passes and preexisting_repeat=True.
- F6 determinism: identical runs -> identical canvas and certificate.
- KILL: F2 fails (no speedup) or F3 fails (a batch violates the
  certificate) -> the module does not ship. F4 failing means the
  certificate is not actually a replay -> kill.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.
"""

import torch

from .cvel import _safe_candidates
from .dcmr import _would_repeat
from .scv import verify_canvas

__all__ = [
    "CVABConfig",
    "CVABTrace",
    "cvab_sample",
    "cvab_certificate",
    "cvab_verify_certificate",
    "interaction_radius",
    "min_separation",
    "POLICIES",
]

POLICIES = ("vanilla", "cvab")


# ---------------------------------------------------------------------------
# Certificate geometry: the interaction radius is the correlation length
# ---------------------------------------------------------------------------

def interaction_radius(n: int) -> int:
    """Distance over which two commits can interact through the
    certificate: a window of length ``n`` contains two positions iff
    their distance is at most ``n - 1``."""
    return max(0, n - 1)


def min_separation(n: int) -> int:
    """Minimum pairwise distance for a batch to be admissible: ``n``,
    one more than the interaction radius, so no n-gram window spans
    two batch members."""
    return max(1, n)


def _batch_admissible(canvas, batch, n: int, mask_id: int) -> bool:
    """Is this batch provably certificate-safe on this canvas?

    Every member must have been masked in the pre-batch canvas, pass
    the per-commit gate there, and sit at least ``n`` away from every
    other member. The gate calls are the single-commit checks; the
    separation makes them independent (no window spans two members).
    """
    positions = [pos for pos, _, _ in batch]
    for pos in positions:
        if canvas[pos] != mask_id:
            return False
    for i, pos in enumerate(positions):
        if any(abs(pos - other) < n for j, other in enumerate(positions)
               if j != i):
            return False
    for pos, tok, _ in batch:
        if _would_repeat(canvas, pos, tok, n, mask_id):
            return False
    return True


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class CVABConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        topk: int = 4,
        max_rounds: int = 64,
        conf_threshold: float = 0.35,
        max_width: int = 4,
    ):
        self.n = n                                # n-gram order of the certificate
        self.topk = topk                          # candidates scanned per position
        self.max_rounds = max_rounds              # hard termination cap
        self.conf_threshold = conf_threshold      # commit bar
        self.max_width = max_width                # batch width cap per round


class CVABTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.n = 4
        self.rounds = 0
        self.forward_passes = 0
        self.commits = 0
        self.batches = 0              # rounds that committed >= 1 token
        self.shrinks = 0              # ranked candidates dropped by separation
        self.verify_alarms = 0        # admissible batches that failed re-check
        self.gate_rejects = 0         # top-1 candidates refused by the gate
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen/prompt text already repeats
        self.stalled = False          # no admissible commit anywhere
        self.batch_log = []           # per-round audit entries

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def cvab_certificate(trace: CVABTrace) -> dict:
    """The cvab-v1 certificate: the per-round batch replay log."""
    return {
        "version": "cvab-v1",
        "policy": trace.policy,
        "n": trace.n,
        "rounds": trace.rounds,
        "forward_passes": trace.forward_passes,
        "commits": trace.commits,
        "batches": trace.batches,
        "shrinks": trace.shrinks,
        "verify_alarms": trace.verify_alarms,
        "gate_rejects": trace.gate_rejects,
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "preexisting_repeat": trace.preexisting_repeat,
        "stalled": trace.stalled,
        "batch_log": trace.batch_log,
    }


def cvab_verify_certificate(ids, cert: dict, mask_id: int) -> dict:
    """Independent re-check of a cvab-v1 certificate: no model, no
    sampler state.

    Replays every round: the batch must be admissible on its stored
    pre-round canvas, every member must have been masked there,
    applying the batch must reproduce the next round's pre-round
    canvas (or the final canvas for the last round), and every
    intermediate canvas must pass the independent no-repeat check.
    ``completed=True`` additionally requires a mask-free, clean final
    canvas. Returns per-check booleans and an overall ``valid`` flag.
    A certificate whose batches are not admissible, whose chain does
    not replay, or whose completed claim is unsound, fails.
    """
    ids = [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    n = int(cert.get("n", 4))
    checks = {}
    check = verify_canvas(ids, mask_id, n)
    checks["final_no_repeat"] = bool(check["final_no_repeat"])
    log = cert.get("batch_log", [])
    admissible = True
    chain = True
    accounting = True
    prev_after = None
    for i, entry in enumerate(log):
        before = [int(t) for t in entry.get("canvas_before", [])]
        batch = [(int(m["pos"]), int(m["tok"]), float(m["prob"]))
                 for m in entry.get("batch", [])]
        if prev_after is not None and before != prev_after:
            chain = False
        if not _batch_admissible(before, batch, n, mask_id):
            admissible = False
        after = list(before)
        for pos, tok, _ in batch:
            after[pos] = tok
        if not verify_canvas(after, mask_id, n)["final_no_repeat"]:
            admissible = False
        if entry.get("admitted_width") != len(batch):
            accounting = False
        if entry.get("shrinks") != entry.get("examined", 0) - len(batch):
            accounting = False
        seps = [abs(batch[a][0] - batch[b][0])
                for a in range(len(batch)) for b in range(a + 1, len(batch))]
        recorded = entry.get("min_separation")
        expected = min(seps) if seps else None
        if recorded != expected:
            accounting = False
        if i == len(log) - 1:
            if after != ids:
                chain = False
        prev_after = after
    checks["batches_admissible"] = admissible
    checks["chain_exact"] = chain
    checks["accounting_consistent"] = accounting
    checks["completed_sound"] = (
        (not cert.get("completed"))
        or (mask_id not in ids and checks["final_no_repeat"]))
    checks["valid"] = (checks["batches_admissible"]
                       and checks["chain_exact"]
                       and checks["accounting_consistent"]
                       and checks["completed_sound"])
    return checks


def cvab_sample(model, length: int, mask_id: int, config=None,
                policy: str = "cvab", frozen=None):
    """Run the CVAB sampler.

    ``frozen`` maps position -> token for prompt tokens that are never
    committed (and never tabooed). Returns (ids, trace, certificate);
    ``ids`` is a LongTensor with ``mask_id`` where the run did not
    complete. Deterministic (greedy); no sampling anywhere.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; want one of {POLICIES}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    cfg = config if config is not None else CVABConfig()
    trace = CVABTrace(policy)
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
        cert = cvab_certificate(trace)
        return torch.tensor(canvas, dtype=torch.long), trace, cert

    width_cap = 1 if policy == "vanilla" else cfg.max_width
    for round_no in range(cfg.max_rounds):
        masked = [p for p, t in enumerate(canvas) if t == mask_id]
        if not masked:
            break
        with torch.no_grad():
            logits = model(torch.tensor([canvas], dtype=torch.long))
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)

        # Rank masked positions by their best safe candidate; the
        # ranking is the model's job, the batch width is not.
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
            trace.stalled = True
            break
        ranked.sort(key=lambda ps: (-ps[1][0][1], ps[0]))

        # Greedy maximal separated set in rank order: admit a
        # candidate only when it is >= n away from every member
        # already admitted. The walk stops at max_width.
        batch = []
        examined = 0
        for pos, safe in ranked:
            if len(batch) >= width_cap:
                break
            examined += 1
            tok, prob = safe[0]
            if all(abs(pos - kept) >= cfg.n for kept, _, _ in batch):
                batch.append((pos, tok, prob))
        canvas_before = list(canvas)

        # Batch verification (defense in depth): admissibility plus a
        # clean pre-batch canvas provably implies a clean simulated
        # canvas, so a failed check is an alarm, not an expected path.
        sim = list(canvas)
        for pos, tok, _ in batch:
            sim[pos] = tok
        verified = bool(verify_canvas(sim, mask_id, cfg.n)["final_no_repeat"])
        committed = batch
        if not verified:
            trace.verify_alarms += 1
            pos, tok, prob = batch[0]
            single = list(canvas)
            single[pos] = tok
            if verify_canvas(single, mask_id, cfg.n)["final_no_repeat"]:
                committed = [(pos, tok, prob)]
                verified = True
            else:
                trace.stalled = True
                break

        for pos, tok, _ in committed:
            canvas[pos] = tok
        trace.commits += len(committed)
        trace.batches += 1
        shrinks = examined - len(committed)
        trace.shrinks += shrinks
        seps = [abs(committed[a][0] - committed[b][0])
                for a in range(len(committed))
                for b in range(a + 1, len(committed))]
        trace.batch_log.append({
            "round": round_no,
            "canvas_before": canvas_before,
            "batch": [{"pos": pos, "tok": tok, "prob": round(prob, 4)}
                      for pos, tok, prob in committed],
            "target_width": width_cap,
            "examined": examined,
            "admitted_width": len(committed),
            "shrinks": shrinks,
            "min_separation": min(seps) if seps else None,
            "verified": bool(verified),
        })
        trace.rounds += 1

    trace.completed = mask_id not in canvas
    check = verify_canvas(canvas, mask_id, cfg.n)
    trace.final_no_repeat = bool(check["final_no_repeat"])
    cert = cvab_certificate(trace)
    return torch.tensor(canvas, dtype=torch.long), trace, cert
