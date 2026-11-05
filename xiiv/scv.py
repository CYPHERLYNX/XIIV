"""SCV -- Speculative Certificate Verification for masked-diffusion decoding.

Every sampler in this family (DCMR, CGAU, CGEC, CDR, PTCS) commits one
planned batch per forward pass and decides the batch with confidence
heuristics. SCV inverts the authority: the model *proposes* (a ranked
speculation) and the INDEPENDENT certificate verifier *disposes* -- it
decides, candidate by candidate, how many positions commit in this round.

One round:
1. A single forward pass scores every masked position. The frontier is
   the top-``frontier`` masked positions by top-1 probability
   (deterministic order; ties break by position).
2. Each frontier position drafts its top-``draft_depth`` token candidates
   -- the speculation (no extra forward passes; drafting is free).
3. Position-major prefix walk, in confidence order. For each position,
   its candidates are tried in rank order against the independent
   verifier (``dcmr._would_repeat`` on the working canvas -- canvas-only,
   the model is never consulted). The first certificate-safe candidate
   is accepted and written into the working canvas, so later candidates
   in the same round are judged against the batch accepted so far.
4. A position whose whole candidate list is verifier-rejected is
   *speculation-exhausted*: the round stops there (prefix property --
   the lower-confidence tail waits for denser context next round). There
   is no fixed per-step commit budget: the batch size is whatever prefix
   the verifier endorses, up to the frontier -- a certificate-native
   dynamic batch size.

Fail-closed: a canvas that is not fully unmasked, or that violates the
no-repeat certificate, is never presented as certified
(``completed=False``), unless the breach pre-existed in frozen text (the
sampler cannot repair frozen prompt tokens; the pre-existing repeat is
reported, not hidden).

The verifier check is conservative under pre-existing repeats: a
candidate completing a window identical to a pre-existing repeated window
is rejected, because the verifier cannot distinguish "new" from "old"
without history.

Prior art (absorbed, not copied):
- Self Speculative Decoding for dLLMs (Gao et al., SJTU,
  arXiv:2510.04147): draft-then-verify works for masked diffusion
  (3.46x, lossless). SCV takes the *shape* but rejects the mechanism:
  SSD's verifier is the model grading its own drafts (circular under the
  Forge doctrine); SCV's verifier is the independent certificate
  checker, and "lossless" is replaced by certificate soundness.
- EB-Sampler (Ben-Hamu et al., Meta, Aug 2026): parallel unmasking needs
  a principled batch budget bounding the *joint* risk. SCV takes the
  *batch-size-as-a-bound* idea but replaces the entropy error tolerance
  with a hard certificate gate -- the batch that commits is exactly the
  longest verifier-endorsed prefix.
- In-repo: dcmr._would_repeat / _topk_for_pos / find_repeated_ngrams
  (the certificate), the CDR/PTCS fail-closed pattern. PTCS's pre-commit
  gate already falls back to next-best per position; SCV is distinct in
  that the verifier (not a planned budget) decides the batch size, with
  prefix commit instead of per-position fallback.

Pre-registered falsifiers (see xiiv/tests/test_scv.py):
- F1 speculative dividend: on the deceptive stub (model top-1 always
  certificate-violating, top-2 safe), SCV completes in strictly fewer
  forward passes than gated top-1 vanilla, which stalls.
- F2 acceptance soundness: every accepted candidate was
  certificate-clean at accept time (replayed); rejects > 0 (the
  arbitration is not vacuous).
- F3 verifier-over-model: on the fork stub, the accepted token is the
  model's rank-1 candidate, not its rank-0 -- the verifier overruled the
  ranking.
- F4 prefix property: a speculation-exhausted position stops the round;
  lower-confidence positions are not committed that round.
- F5 fail-closed: the repeater stub never completes; no violating token
  is ever committed.
- F6 batching: on the clean stub, forward_passes ==
  ceil(n_masked / frontier) -- the verifier endorses the full frontier.
- KILL: F2 or F5 fails -> the module does not ship.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed. Deterministic: no RNG is used (seed kept for CLI parity).

Measured on CPU scripted stubs only -- real-checkpoint behavior
untested.
"""

import torch

from .dcmr import _would_repeat, _topk_for_pos, find_repeated_ngrams

__all__ = [
    "SCVConfig",
    "SCVTrace",
    "POLICIES",
    "scv_sample",
    "scv_certificate",
    "verify_canvas",
]

POLICIES = ("vanilla", "scv")


class SCVConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        frontier: int = 8,
        draft_depth: int = 3,
        max_steps: int = 64,
        stall_budget: int = 3,
        seed: int = 7,
    ):
        self.n = n                        # n-gram order of the certificate
        self.frontier = frontier          # positions drafted per round
        self.draft_depth = draft_depth    # candidate tokens per position
        self.max_steps = max_steps        # hard round cap (1 round = 1 fwd pass)
        self.stall_budget = stall_budget  # zero-accept rounds before giving up
        self.seed = seed                  # API parity; sampler is deterministic


class SCVTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.steps = 0              # rounds executed
        self.forward_passes = 0     # model forward passes (1 per round)
        self.drafts = 0             # candidate (pos, tok) pairs drafted
        self.accepts = 0            # candidates accepted by the verifier
        self.rejects = 0            # candidates rejected by the verifier
        self.tail_stops = 0         # rounds stopped by a prefix break
        self.commits = 0            # positions committed (== accepts)
        self.stalls = 0             # zero-accept rounds
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen text already repeats
        self.accept_order = []      # (round, pos, tok, rank) in accept order
        self.round_accepts = []     # per-round [(pos, tok)] lists
        self.events = []

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("events", "accept_order", "round_accepts")}
        d["n_events"] = len(self.events)
        d["n_accepts_order"] = len(self.accept_order)
        d["n_rounds_recorded"] = len(self.round_accepts)
        return d


def verify_canvas(x_list: list, mask_id: int, n: int) -> dict:
    """Independent re-check of a canvas: no model, no sampler state.

    Recomputes the no-repeat certificate from the token list alone, so a
    third party can replay the certificate claim without weights.
    """
    repeated = find_repeated_ngrams(x_list, n, mask_id)
    n_windows = sum(len(v) for v in repeated.values())
    return {
        "final_no_repeat": not bool(repeated),
        "n_repeated_windows": n_windows,
        "n_repeated_keys": len(repeated),
    }


def scv_certificate(trace: SCVTrace) -> dict:
    """Build the scv-v1 certificate from a run trace.

    Fail-closed: ``completed`` is True only if the trace completed; a
    non-completed trace never yields a certification claim.
    """
    cert = {
        "cert": "scv-v1",
        "policy": trace.policy,
        "completed": bool(trace.completed),
        "final_no_repeat": bool(trace.final_no_repeat),
        "preexisting_repeat": bool(trace.preexisting_repeat),
        "n_accepts": trace.accepts,
        "n_rejects": trace.rejects,
        "n_tail_stops": trace.tail_stops,
        "forward_passes": trace.forward_passes,
        "accept_order": [list(e) for e in trace.accept_order],
    }
    if not trace.completed:
        cert["completed"] = False
    return cert


def _top1_prob(probs_row: torch.Tensor) -> float:
    return float(torch.max(probs_row).item())


def scv_sample(model, length: int, mask_id: int, config=None,
               seed: int = 7, frozen=None, policy: str = "scv"):
    """Run SCV (or the gated top-1 vanilla baseline) over a blank canvas.

    Returns (ids LongTensor (1, L), trace, cert).
    """
    assert policy in POLICIES, f"unknown policy {policy!r}"
    cfg = config or SCVConfig()
    x_list = [mask_id] * length
    frozen = frozen or {}
    for p, t in frozen.items():
        x_list[int(p)] = int(t)
    x = torch.tensor([x_list], dtype=torch.long)

    trace = SCVTrace(policy)
    trace.preexisting_repeat = bool(
        find_repeated_ngrams(x_list, cfg.n, mask_id))

    stall = 0
    banned = {mask_id}
    for rnd in range(cfg.max_steps):
        masked = [i for i in range(length) if x_list[i] == mask_id]
        if not masked:
            break
        trace.steps += 1
        with torch.no_grad():
            logits = model(x)
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)

        # Frontier: top-`frontier` masked positions by top-1 probability.
        # Deterministic order: probability desc, then position asc.
        scored = sorted(masked, key=lambda p: (-_top1_prob(probs[p]), p))
        frontier_pos = scored[:max(1, cfg.frontier)]

        working = list(x_list)
        accepted = []  # (pos, tok, rank)

        if policy == "scv":
            for p in frontier_pos:
                cands = _topk_for_pos(probs[p], cfg.draft_depth, banned)
                trace.drafts += len(cands)
                chosen = None
                for rank, (tok, _pr) in enumerate(cands):
                    if _would_repeat(working, p, tok, cfg.n, mask_id):
                        trace.rejects += 1
                        trace.events.append(
                            ("candidate_reject",
                             {"round": rnd, "pos": p, "tok": tok,
                              "rank": rank}))
                        continue
                    chosen = (tok, rank)
                    break
                if chosen is None:
                    # Speculation-exhausted: prefix stop. The tail waits
                    # for denser context next round.
                    trace.tail_stops += 1
                    trace.events.append(("tail_stop", {"round": rnd,
                                                       "pos": p}))
                    break
                tok, rank = chosen
                working[p] = tok
                trace.accepts += 1
                accepted.append((p, tok, rank))
                trace.accept_order.append((rnd, p, tok, rank))
        else:  # vanilla: gated top-1, same frontier, no depth, no prefix
            for p in frontier_pos:
                cands = _topk_for_pos(probs[p], 1, banned)
                if not cands:
                    continue
                tok, _pr = cands[0]
                trace.drafts += 1
                if _would_repeat(working, p, tok, cfg.n, mask_id):
                    trace.rejects += 1
                    trace.events.append(
                        ("candidate_reject",
                         {"round": rnd, "pos": p, "tok": tok, "rank": 0}))
                    continue
                working[p] = tok
                trace.accepts += 1
                accepted.append((p, tok, 0))
                trace.accept_order.append((rnd, p, tok, 0))

        x_list = working
        x = torch.tensor([x_list], dtype=torch.long)
        trace.commits += len(accepted)
        trace.round_accepts.append([(p, tok) for p, tok, _r in accepted])
        if accepted:
            stall = 0
        else:
            stall += 1
            trace.stalls += 1
            trace.events.append(("stall", {"round": rnd}))
            if stall >= cfg.stall_budget:
                break

    remaining = [i for i in range(length) if x_list[i] == mask_id]
    check = verify_canvas(x_list, mask_id, cfg.n)
    trace.final_no_repeat = check["final_no_repeat"]
    # Fail-closed: complete only when fully unmasked and certificate-clean,
    # unless the breach pre-existed in frozen text (not the sampler's to fix).
    trace.completed = (not remaining and
                       (trace.final_no_repeat or trace.preexisting_repeat))
    cert = scv_certificate(trace)
    return x, trace, cert
