"""CMSS -- Certificate Mutual-Stability Sampling for masked-diffusion decoding.

Every sampler in this family so far commits on a SINGLE round's evidence:
the model's top-1 confidence right now, gated by the certificate. None of
them asks whether the model's mind is MADE UP -- and the newest result says
that is exactly the question that decides quality per forward pass:

- LESS Is More: Mutual-Stability Sampling for Diffusion Language Models
  (Mohamed, Shang & Vazirgiannis, arXiv:2606.16908, Jun 2026): a
  training-free, model-agnostic adaptive sampler that treats token
  commitment as an online stopping problem. A masked position becomes
  eligible for unmasking only when its top-1 prediction has high
  confidence, its top-1 token PERSISTS across recent reverse steps, and
  its predictive distribution is stable under top-K inter-step
  Jensen-Shannon divergence. On Dream-7B, LLaDA-8B and LLaDA-1.5-8B
  across seven math/code/knowledge benchmarks, LESS improves the
  accuracy-efficiency trade-off over strong training-free adaptive
  samplers while using 72.1% fewer reverse steps than fixed-budget
  decoding (their numbers). Diagnosis absorbed: a commit is a bet on a
  prediction, and a prediction that flips between rounds is not a bet
  worth taking yet -- wait one more reverse step and let it settle.
- CadLLM (ACL 2026 Findings, arXiv:2512.07173): training-free
  confidence-aware calibration that adapts block size, step count,
  vocabulary range and submission threshold; 1.1-2.28x throughput on
  LLaDA/DREAM, adaptive threshold the core driver. Diagnosis absorbed:
  confidence is a usable control signal at inference time with no
  retraining. CMSS uses confidence as one leg of a three-legged rule
  instead of as the whole rule.
- SPOT (arXiv:2610.22543): speculative decoding for diffusion LMs --
  predict where the model commits next, verify a tree of block states
  in one pass, up to 6.8x fewer model passes, distribution exact.
  Diagnosis absorbed: verification can be decoupled from commitment.
  CMSS verifies with the independent certificate instead of the
  model's own distribution, and the thing being verified is stability,
  not distribution preservation. Shelf: tree verification is a future
  phase; CMSS's batch commit is the cheap version of the same idea.
- CoDiLA (ICML 2026, arXiv:2603.20216): a lightweight AR model attached
  to the DLM, fed the DLM's marginals as soft embeddings, fixes local
  incoherence of parallel sampling (syntax errors 70% -> 16% at B=8 on
  Dream-Coder-7B). Diagnosis absorbed: parallel commits are locally
  incoherent unless something coordinates them. CMSS coordinates them
  with stability instead of a second model: positions that commit
  together must each have settled independently. Shelf: no AR head is
  trained here; CMSS is the no-training answer to the same diagnosis.
- SCALE (ICML 2026, arXiv:2506.16659): minimalist SGD variant matching
  or beating Adam at 60M-1B with 35-45% of Adam's memory. Training-side
  only, logged for a future leg; not adopted in this sampler.
- The 2026 constrained-decoding consensus (structured outputs):
  grammar-constrained decoding guarantees SHAPE, not truth -- a
  hallucinated ID passes every type check. Diagnosis absorbed, and it
  is the standing justification for this whole family: XIIV's
  certificates must keep constraining SEMANTIC content (here: the
  stability of the prediction itself), not just syntax.

CMSS's move: stability is a CERTIFICATE, not a heuristic. LESS's three
conditions become three checkable claims attached to every committed
token, recorded in the generation trace and replayable model-free:

1. CONFIDENCE: P(top-1) >= conf_threshold (the same bar the gate uses).
2. PERSISTENCE: the top-1 token is identical over the last ``persist``
   consecutive observations of that position (default 2: the token
   survived at least one full reverse step unchanged).
3. DIVERGENCE: the Jensen-Shannon divergence between the top-K
   distributions of consecutive observations in the persistence window
   is <= jsd_threshold (default 0.02 nats) -- the distribution did not
   just keep its argmax, it kept its shape.

A position commits only when all three hold AND its stable top-1 passes
the independent no-repeat gate (``cvel._safe_candidates``). The commit
token must BE the stable top-1: if the gate refuses the top-1, the
position does not fall through to a lower candidate, because the
stability certificate is about the top-1 prediction, and a different
token has no stability evidence behind it. The batch of stable
positions commits together; the post-batch canvas is re-checked with
the independent verifier, shrinking to the single best commit on
violation, and failing closed if even that violates.

PATIENCE ROUNDS. A round with no stable position consumes one forward
pass, records observations, and commits nothing. Patience is never
wasted on a deterministic model: with an unchanged canvas the next
round's observations are identical, so persistence reaches 2 and the
JSD is 0, and every confident position stabilizes. Patience only
becomes permanent when the model's predictions OSCILLATE -- top-1
flipping between rounds, or the distribution reshuffling under a
constant argmax -- and then the run exhausts ``max_rounds`` and fails
closed with masks remaining, instead of emitting an unsettled canvas.
That is the falsifier: an oscillating model must never be committed.

Two corollaries, both provable from the stability rule alone:
- On a stable model (identical predictions every round) CMSS commits
  the whole canvas in TWO forward passes: round 0 observes, round 1
  stabilizes and commits everything. Vanilla needs one pass per token.
  The step saving is the measured quantity, not a claim.
- Stability is monotone in evidence: a position that was stable at
  round r stays stable at round r+1 unless the canvas changed under
  it (another commit landed nearby) or the model itself moved. The
  certificate records the exact window that justified each commit, so
  a replay can check the claim without the model.

- STABILITY, not confidence. Each round computes per-position
  observations (top-1, confidence, top-K distribution) and the
  three-legged rule decides eligibility; the certificate gate then
  decides commit-worthiness. Confidence alone never commits.
- BATCH ACCOUNTING. The trace records ``patience_rounds`` -- rounds
  that observed but committed nothing -- and ``batch_shrinks`` -- batch
  commits that had to shrink to one position. A run with zero patience
  rounds and zero shrinks is the clean path; anything else is visible.
- PREEXISTING REPEAT fails closed at round zero: if the frozen prompt
  already contains a repeated n-gram, no commit sequence can
  un-violate it, so the run reports ``completed=False`` with zero
  forward passes and ``preexisting_repeat=True`` instead of emitting
  an uncertified canvas.
- THE CERTIFICATE IS THE REPLAY, not a score. The cmss-v1
  certificate itemizes every round (pre-round canvas, per-position
  observations, committed positions with their stability triples,
  patience flag). ``cmss_verify_certificate`` replays the chain
  model-free: each commit must have been masked on its pre-round
  canvas, its token must equal the recorded top-1, its persistence
  run and JSDs must recompute from the stored observations and meet
  the thresholds, the gate must have admitted it, applying the round's
  commits must reproduce the next round's pre-round canvas (or the
  final canvas), every intermediate canvas must pass the independent
  check. ``completed=True`` additionally requires a mask-free, clean
  final canvas.
- TERMINATION: at most one forward pass per round, at most
  ``max_rounds`` rounds; exhaustion fails closed with masks remaining
  rather than emitting an unsettled canvas. No logits are ever
  modified or fused.

Prior art (absorbed, not copied):
- LESS (arXiv:2606.16908): the joint stability rule. CMSS takes the
  rule but not the authority structure: LESS's rule is a sampler
  heuristic; CMSS's rule is a certificate -- every commit carries its
  stability evidence, replayable without the model, and the
  certificate gate still has the final word.
- CadLLM: confidence as a control signal. CMSS takes the diagnosis
  but not the mechanism: confidence is one of three legs, never the
  whole rule.
- SPOT/CoDiLA: verification decoupled from commitment; parallel
  commits need coordination. CMSS takes the diagnosis but not the
  mechanism: coordination is stability, verified by the certificate.
- The Flexibility Trap / Who Gets Unmasked (arXiv:2610.24795):
  sampler choices change the output and must be reported
  per-configuration, never as a bare model property. CMSS's stability
  thresholds are sampler choices and the certificate says so.
- In-repo: dcmr._would_repeat / _topk_for_pos (the certificate
  gate), scv.verify_canvas (independent re-check), the CDR/CBRS/
  CWTC/CSBR/CGBS/CVEL/CVAB/CVOU fail-closed pattern;
  cvel._safe_candidates (per-position gate-safe ranking) reused
  directly.

Pre-registered falsifiers (see xiiv/tests/test_cmss.py):
- F1 stable stub: on a fully-masked canvas with a stable model, cmss
  completes in 2 forward passes (round 0 observes, round 1 commits
  all 16), every commit with persistence >= 2 and jsd == 0.
- F2 cost vs vanilla: vanilla needs 16 forward passes on the same
  stub; cmss needs 2; final canvases identical.
- F3 flip-flop: a model whose top-1 alternates every round never
  stabilizes -- the run exhausts max_rounds with masks remaining,
  completed=False, patience_rounds == max_rounds.
- F4 reshuffle: a model whose top-1 is constant but whose runner-up
  mass swaps every round has inter-step JSD ~0.052 > 0.02 -- never
  stabilizes, exhausts, completed=False.
- F5 low confidence: a model whose top-1 prob is ~1/3 < 0.35 never
  stabilizes on the confidence leg -- exhausts, completed=False.
- F6 certificate replay: a real run verifies; tampering with a
  commit's jsd, or with any round's chosen token, fails verification.
- F7 fail-closed: a model whose stable top-1 would create a repeat
  stalls with completed=False and masks remaining; a frozen prompt
  that already repeats fails closed at round zero with zero forward
  passes and preexisting_repeat=True.
- F8 determinism: identical runs -> identical canvas and certificate.
- KILL: F3 fails (a flip-flopping model commits anyway) or F6 fails
  (the certificate is not actually a replay) -> the module does not
  ship. F1 failing (stable stub needs more than 2 passes) means the
  stability rule is mis-implemented, not that it is slow -> kill.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.
"""

import math

import torch

from .cvel import _safe_candidates
from .dcmr import _topk_for_pos, _would_repeat
from .scv import verify_canvas

__all__ = [
    "CMSSConfig",
    "CMSSTrace",
    "cmss_sample",
    "cmss_certificate",
    "cmss_verify_certificate",
    "jsd_topk",
    "POLICIES",
]

POLICIES = ("vanilla", "cmss")


# ---------------------------------------------------------------------------
# Stability geometry: Jensen-Shannon divergence over top-K distributions
# ---------------------------------------------------------------------------

def jsd_topk(prev, cur) -> float:
    """Jensen-Shannon divergence (nats) between two top-K distributions.

    ``prev``/``cur`` are lists of (token, prob). The union of the two
    token sets is the support; each distribution is renormalized over
    the union (tokens absent from one side get probability 0 there),
    then JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) with M = (P+Q)/2.
    Identical distributions give exactly 0.0; disjoint support gives
    ln 2 (~0.6931). Deterministic, model-free.
    """
    p = dict(prev)
    q = dict(cur)
    tokens = set(p) | set(q)
    sp = sum(p.values())
    sq = sum(q.values())
    if sp <= 0.0 or sq <= 0.0:
        return 0.0
    jsd = 0.0
    for t in tokens:
        pi = p.get(t, 0.0) / sp
        qi = q.get(t, 0.0) / sq
        mi = 0.5 * (pi + qi)
        if pi > 0.0:
            jsd += 0.5 * pi * math.log(pi / mi)
        if qi > 0.0:
            jsd += 0.5 * qi * math.log(qi / mi)
    return max(0.0, jsd)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class CMSSConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        topk: int = 4,
        topk_jsd: int = 8,
        persist: int = 2,
        jsd_threshold: float = 0.02,
        max_rounds: int = 64,
        conf_threshold: float = 0.35,
    ):
        self.n = n                                # n-gram order of the certificate
        self.topk = topk                          # gate candidates scanned per position
        self.topk_jsd = topk_jsd                  # K for the stability distribution
        self.persist = persist                    # consecutive identical top-1s required
        self.jsd_threshold = jsd_threshold       # max inter-step JSD (nats)
        self.max_rounds = max_rounds              # hard termination cap
        self.conf_threshold = conf_threshold      # commit bar (= stability conf leg)


class CMSSTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.n = 4
        self.persist = 2
        self.jsd_threshold = 0.02
        self.conf_threshold = 0.35
        self.rounds = 0               # rounds that committed >= 1 position
        self.forward_passes = 0
        self.commits = 0
        self.patience_rounds = 0      # rounds that observed, committed nothing
        self.gate_rejects = 0         # stable top-1s refused by the gate
        self.batch_shrinks = 0        # batch commits shrunk to one position
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen/prompt text already repeats
        self.stalled = False          # stable predictions exist, gate refuses all
        self.exhausted = False        # max_rounds hit with masks remaining
        self.commit_log = []          # per-round audit entries

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def cmss_certificate(trace: CMSSTrace) -> dict:
    """The cmss-v1 certificate: the per-round stability replay log."""
    return {
        "version": "cmss-v1",
        "policy": trace.policy,
        "n": trace.n,
        "persist": trace.persist,
        "jsd_threshold": trace.jsd_threshold,
        "conf_threshold": trace.conf_threshold,
        "rounds": trace.rounds,
        "forward_passes": trace.forward_passes,
        "commits": trace.commits,
        "patience_rounds": trace.patience_rounds,
        "gate_rejects": trace.gate_rejects,
        "batch_shrinks": trace.batch_shrinks,
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "preexisting_repeat": trace.preexisting_repeat,
        "stalled": trace.stalled,
        "exhausted": trace.exhausted,
        "commit_log": trace.commit_log,
    }


def _recompute_stability(log, round_idx, pos, cfg):
    """Recompute a commit's stability triple from the stored log.

    Walks backwards from ``round_idx`` through the round entries'
    observations for ``pos``; returns (persistence, max_jsd) where
    persistence is the run length of identical top-1s ending at this
    round and max_jsd is the largest inter-step JSD inside that run.
    Returns (0, inf) when the position has no observation this round.
    """
    pos_key = str(pos)
    entry = log[round_idx]
    obs = entry.get("observations", {}).get(pos_key)
    if obs is None:
        return 0, math.inf
    top1 = obs["top1"]
    persistence = 1
    max_jsd = 0.0
    prev_topk = obs["topk"]
    for j in range(round_idx - 1, -1, -1):
        prev_obs = log[j].get("observations", {}).get(pos_key)
        if prev_obs is None or prev_obs["top1"] != top1:
            break
        jsd = jsd_topk(prev_obs["topk"], prev_topk)
        max_jsd = max(max_jsd, jsd)
        persistence += 1
        prev_topk = prev_obs["topk"]
    return persistence, max_jsd


def cmss_verify_certificate(ids, cert: dict, mask_id: int) -> dict:
    """Independent re-check of a cmss-v1 certificate: no model, no
    sampler state.

    Replays every round: each commit must have been masked on its
    pre-round canvas, its token must equal the recorded top-1, its
    persistence run and JSDs must recompute from the stored
    observations and meet the thresholds (for policy "cmss"; vanilla
    commits are exempt from the stability rule), the gate must have
    admitted it, applying the round's commits must reproduce the next
    round's pre-round canvas (or the final canvas), every intermediate
    canvas must pass the independent no-repeat check. ``completed=True``
    additionally requires a mask-free, clean final canvas. Returns
    per-check booleans and an overall ``valid`` flag.
    """
    ids = [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    n = int(cert.get("n", 4))
    policy = cert.get("policy", "cmss")
    cfg = CMSSConfig(n=n,
                     persist=int(cert.get("persist", 2)),
                     jsd_threshold=float(cert.get("jsd_threshold", 0.02)),
                     conf_threshold=float(cert.get("conf_threshold", 0.35)))
    checks = {}
    check = verify_canvas(ids, mask_id, n)
    checks["final_no_repeat"] = bool(check["final_no_repeat"])
    log = cert.get("commit_log", [])
    stable = True
    admissible = True
    chain = True
    prev_after = None
    for i, entry in enumerate(log):
        before = [int(t) for t in entry.get("canvas_before", [])]
        if prev_after is not None and before != prev_after:
            chain = False
        for commit in entry.get("commits", []):
            pos = int(commit.get("pos", -1))
            tok = int(commit.get("tok", -1))
            if not (0 <= pos < len(before)):
                admissible = False
                continue
            if before[pos] != mask_id:
                admissible = False
            if _would_repeat(before, pos, tok, n, mask_id):
                admissible = False
            if policy == "cmss":
                obs = entry.get("observations", {}).get(str(pos))
                if obs is None or int(obs.get("top1", -1)) != tok:
                    stable = False
                elif float(obs.get("conf", 0.0)) < cfg.conf_threshold:
                    stable = False
                else:
                    persistence, max_jsd = _recompute_stability(
                        log, i, pos, cfg)
                    if commit.get("persistence") != persistence:
                        stable = False
                    if abs(float(commit.get("jsd", -1.0)) - max_jsd) > 1e-9:
                        stable = False
                    if persistence < cfg.persist:
                        stable = False
                    if max_jsd > cfg.jsd_threshold + 1e-9:
                        stable = False
        after = list(before)
        for commit in entry.get("commits", []):
            after[int(commit["pos"])] = int(commit["tok"])
        if not verify_canvas(after, mask_id, n)["final_no_repeat"]:
            admissible = False
        if i == len(log) - 1:
            if after != ids:
                chain = False
        prev_after = after
    checks["commits_stable"] = stable
    checks["commits_admissible"] = admissible
    checks["chain_exact"] = chain
    checks["completed_sound"] = (
        (not cert.get("completed"))
        or (mask_id not in ids and checks["final_no_repeat"]))
    checks["valid"] = (checks["commits_stable"]
                       and checks["commits_admissible"]
                       and checks["chain_exact"]
                       and checks["completed_sound"])
    return checks


def cmss_sample(model, length: int, mask_id: int, config=None,
                policy: str = "cmss", frozen=None):
    """Run the CMSS sampler.

    ``frozen`` maps position -> token for prompt tokens that are never
    committed (and never tabooed). Returns (ids, trace, certificate);
    ``ids`` is a LongTensor with ``mask_id`` where the run did not
    complete. Deterministic (greedy); no sampling anywhere.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; want one of {POLICIES}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    cfg = config if config is not None else CMSSConfig()
    trace = CMSSTrace(policy)
    trace.n = cfg.n
    trace.persist = cfg.persist
    trace.jsd_threshold = cfg.jsd_threshold
    trace.conf_threshold = cfg.conf_threshold
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
        cert = cmss_certificate(trace)
        return torch.tensor(canvas, dtype=torch.long), trace, cert

    # Per-position observation history: pos -> list of
    # {"top1", "conf", "topk"} in round order.
    history = {p: [] for p in range(length)}

    for round_no in range(cfg.max_rounds):
        masked = [p for p, t in enumerate(canvas) if t == mask_id]
        if not masked:
            break
        with torch.no_grad():
            logits = model(torch.tensor([canvas], dtype=torch.long))
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)

        # Observe every masked position this round.
        observations = {}
        for pos in masked:
            topk = _topk_for_pos(probs[pos], cfg.topk_jsd, {mask_id})
            if not topk:
                continue
            obs = {"top1": topk[0][0], "conf": topk[0][1],
                   "topk": [[t, v] for t, v in topk]}
            observations[str(pos)] = obs
            history[pos].append(obs)

        canvas_before = list(canvas)
        if policy == "vanilla":
            # Baseline: one commit per round, best safe candidate.
            ranked = []
            for pos in masked:
                safe, top1_rejected = _safe_candidates(
                    canvas, pos, probs[pos], cfg, mask_id)
                if top1_rejected:
                    trace.gate_rejects += 1
                if safe:
                    ranked.append((pos, safe))
            if not ranked:
                trace.stalled = True
                trace.commit_log.append({
                    "round": round_no,
                    "canvas_before": canvas_before,
                    "observations": observations,
                    "commits": [],
                    "patience": False,
                })
                break
            ranked.sort(key=lambda ps: (-ps[1][0][1], ps[0]))
            pos, safe = ranked[0]
            tok, prob = safe[0]
            canvas[pos] = tok
            trace.commits += 1
            trace.rounds += 1
            trace.commit_log.append({
                "round": round_no,
                "canvas_before": canvas_before,
                "observations": observations,
                "commits": [{"pos": pos, "tok": tok, "conf": prob,
                             "persistence": None, "jsd": None}],
                "patience": False,
            })
            continue

        # CMSS: the three-legged stability rule decides eligibility.
        eligible = []
        for pos in masked:
            obs = observations.get(str(pos))
            if obs is None:
                continue
            if obs["conf"] < cfg.conf_threshold:
                continue
            hist = history[pos]
            if len(hist) < cfg.persist:
                continue
            run = hist[-cfg.persist:]
            if any(h["top1"] != obs["top1"] for h in run):
                continue
            jsds = [jsd_topk(run[i]["topk"], run[i + 1]["topk"])
                    for i in range(len(run) - 1)]
            max_jsd = max(jsds) if jsds else 0.0
            if max_jsd > cfg.jsd_threshold:
                continue
            eligible.append((pos, obs, len(run), max_jsd))

        if not eligible:
            # Patience round: observe, commit nothing. On a
            # deterministic model the next round stabilizes; on an
            # oscillating model this repeats until max_rounds.
            trace.patience_rounds += 1
            trace.commit_log.append({
                "round": round_no,
                "canvas_before": canvas_before,
                "observations": observations,
                "commits": [],
                "patience": True,
            })
            continue

        # Gate: the commit token must BE the stable top-1. A
        # gate-refused top-1 never falls through to a lower candidate
        # -- that candidate has no stability evidence behind it.
        commits = []
        for pos, obs, persistence, max_jsd in sorted(
                eligible, key=lambda e: (-e[1]["conf"], e[0])):
            safe, top1_rejected = _safe_candidates(
                canvas, pos, probs[pos], cfg, mask_id)
            if top1_rejected:
                trace.gate_rejects += 1
            if safe and safe[0][0] == obs["top1"]:
                commits.append({"pos": pos, "tok": obs["top1"],
                                "conf": obs["conf"],
                                "persistence": persistence,
                                "jsd": max_jsd})
        if not commits:
            # Stable predictions exist but the gate refused them all --
            # further rounds cannot help (same canvas, same refusals).
            trace.stalled = True
            trace.commit_log.append({
                "round": round_no,
                "canvas_before": canvas_before,
                "observations": observations,
                "commits": [],
                "patience": False,
            })
            break

        # Batch commit, re-verified; shrink to the single best commit
        # on violation, fail closed if even that violates.
        trial = list(canvas)
        for c in commits:
            trial[c["pos"]] = c["tok"]
        if not verify_canvas(trial, mask_id, cfg.n)["final_no_repeat"]:
            trace.batch_shrinks += 1
            best = commits[0]
            trial = list(canvas)
            trial[best["pos"]] = best["tok"]
            if not verify_canvas(trial, mask_id, cfg.n)["final_no_repeat"]:
                trace.stalled = True
                trace.commit_log.append({
                    "round": round_no,
                    "canvas_before": canvas_before,
                    "observations": observations,
                    "commits": [],
                    "patience": False,
                })
                break
            commits = [best]
        for c in commits:
            canvas[c["pos"]] = c["tok"]
        trace.commits += len(commits)
        trace.rounds += 1
        trace.commit_log.append({
            "round": round_no,
            "canvas_before": canvas_before,
            "observations": observations,
            "commits": commits,
            "patience": False,
        })

    trace.exhausted = (mask_id in canvas
                       and trace.forward_passes >= cfg.max_rounds
                       and not trace.stalled)
    trace.completed = mask_id not in canvas
    check = verify_canvas(canvas, mask_id, cfg.n)
    trace.final_no_repeat = bool(check["final_no_repeat"])
    cert = cmss_certificate(trace)
    return torch.tensor(canvas, dtype=torch.long), trace, cert
