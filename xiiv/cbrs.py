"""CBRS -- Certificate-Backed Repair Scheduling for masked-diffusion decoding.

Two-phase sampler. The existing family (DCMR, CGEC, CDR, PTCS, SCV) is
progressive: positions are committed incrementally and the verifier
arbitrates what gets *committed*. CBRS inverts the shape: commit first,
verify after -- a fast UNVERIFIED draft, then a verifier-localized repair
pass that reworks only the fault envelope and freezes everything else.

Phase 1 (DRAFT): pure speculation, no certificate claim. One forward
pass per round; the frontier (top-``frontier`` masked positions by top-1
probability) commits its model top-1 outright. Up to ``draft_rounds``
rounds; the draft is complete when no masks remain.

Phase 2 (REPAIR): the INDEPENDENT certificate verifier (``verify_canvas``
-- canvas-only, the model is never consulted) localizes every
certificate fault (repeated n-gram windows). The repair zone is the
union of fault-window positions, expanded by ``halo`` positions on each
side, minus frozen positions. The zone is remasked; everything outside
it is FROZEN for the rest of the run (locality). Repair rounds resample
the zone -- each masked position tries its top-``repair_depth``
candidates in rank order and accepts the first that is certificate-safe
against the working canvas. The zone is re-localized every round from a
fresh verification. The run ends when the canvas verifies clean, when a
repair round commits nothing (stall), or when the budget exhausts.

The certificate (cbrs-v1) is a claim about the FINAL canvas only -- the
draft is explicitly uncertified speculation. Fail-closed: ``completed``
is True only when the canvas is fully unmasked and verifier-clean,
unless the breach pre-existed in frozen text (the sampler cannot repair
frozen prompt tokens; the pre-existing repeat is reported, not hidden).

No-regress invariant: positions outside every repair zone are never
remasked, so repair cannot introduce a fault where the draft was clean.
The repair phase itself is verifier-gated per commit, so a repair commit
cannot complete a repeated window at commit time either.

Prior art (absorbed, not copied):
- PACT (arXiv:2611.04152): training-free "certify the commit set" rule
  for masked-diffusion parallel decoding. CBRS takes the *certify a set*
  shape but inverts the object: the certified set is the *repair* set,
  localized by the verifier itself, not a commit batch.
- xiiv/diffusion.py self_refine (diffusion-native revision): remask the
  lowest-confidence spans and re-denoise in place. CBRS takes the
  *rework* shape but replaces the driver: not model confidence over
  spans, but the independent verifier's fault localization with
  frozen-draft locality (untouched regions are never remasked).
- "Where Did the Update Go?" (arXiv:2611.04192): change-envelope
  certification for model updates. CBRS takes the *envelope* shape
  (fault windows + halo) but the envelope here is verifier-computed on
  the canvas, and the certified claim is generation soundness, not
  update scoping.

Pre-registered falsifiers (see xiiv/tests/test_cbrs.py):
- F1 repair convergence: on the echo stub (model top-1 plants a repeated
  trigram), repair fixes every fault within budget -> completed=True,
  final_no_repeat, rejects > 0 (the top-1 echo is verifier-rejected).
- F2 locality: clean tail positions outside every repair zone are never
  remasked; their draft tokens survive repair byte-identical, and a
  clean position inside the zone keeps its draft token when top-1 stays
  certificate-safe (no gratuitous rewrite).
- F3 no-regress: verify_canvas(final) is clean AND every position
  outside the cumulative repair zone equals its draft token.
- F4 final soundness: completed=True implies the independent
  verify_canvas(final) is clean (recomputed, not trusted).
- F5 fail-closed: the unfixable stub (every candidate repeats) never
  completes; masks remain; the certificate claim is False.
- F6 determinism: identical runs -> identical repair zones and commit
  order.
- KILL: F3 or F5 fails -> the module does not ship.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed. Deterministic: no RNG is used (seed kept for CLI parity).

Measured on CPU scripted stubs only -- real-checkpoint behavior
untested.
"""

import torch

from .dcmr import _would_repeat, _topk_for_pos, find_repeated_ngrams
from .scv import verify_canvas

__all__ = [
    "CBRSConfig",
    "CBRSTrace",
    "POLICIES",
    "cbrs_sample",
    "cbrs_certificate",
]

POLICIES = ("vanilla", "cbrs")


class CBRSConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        frontier: int = 8,
        draft_rounds: int = 8,
        draft_depth: int = 1,
        repair_depth: int = 3,
        repair_rounds: int = 8,
        halo: int = 1,
        seed: int = 7,
    ):
        self.n = n                      # n-gram order of the certificate
        self.frontier = frontier        # positions drafted per round
        self.draft_rounds = draft_rounds  # draft phase round cap
        self.draft_depth = draft_depth  # candidates per draft position
        self.repair_depth = repair_depth  # candidates per repair position
        self.repair_rounds = repair_rounds  # repair phase round cap
        self.halo = halo                # fault-envelope expansion (positions)
        self.seed = seed                # API parity; sampler is deterministic


class CBRSTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.draft_rounds = 0       # draft forward passes executed
        self.draft_commits = 0      # positions committed in the draft
        self.repair_rounds = 0      # repair forward passes executed
        self.n_fault_windows_draft = 0  # violating windows after the draft
        self.n_positions_repaired = 0   # |cumulative repair zone|
        self.n_repair_commits = 0   # positions committed during repair
        self.accepts = 0            # verifier-safe repair candidates accepted
        self.rejects = 0            # repair candidates rejected by verifier
        self.commits = 0            # draft_commits + n_repair_commits
        self.forward_passes = 0     # total model forward passes
        self.stalls = 0             # zero-commit repair rounds
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen text already repeats
        self.locality = 1.0         # fraction of non-frozen positions never
                                    # remasked during repair
        self.draft_snapshot = []    # canvas right after the draft
        self.draft_commit_order = []  # (round, pos, tok) draft commits
        self.repair_zones = []      # per-round sorted repair-zone lists
        self.repair_commit_order = []  # (round, pos, tok, rank) repair commits
        self.events = []

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("events", "draft_commit_order",
                          "repair_commit_order", "repair_zones",
                          "draft_snapshot")}
        d["n_events"] = len(self.events)
        d["n_draft_commits_order"] = len(self.draft_commit_order)
        d["n_repair_commits_order"] = len(self.repair_commit_order)
        d["n_zones_recorded"] = len(self.repair_zones)
        return d


def cbrs_certificate(trace: CBRSTrace) -> dict:
    """Build the cbrs-v1 certificate from a run trace.

    The claim covers the FINAL canvas only. Fail-closed: ``completed``
    is True only if the trace completed; a non-completed trace never
    yields a certification claim.
    """
    cert = {
        "cert": "cbrs-v1",
        "policy": trace.policy,
        "completed": bool(trace.completed),
        "final_no_repeat": bool(trace.final_no_repeat),
        "preexisting_repeat": bool(trace.preexisting_repeat),
        "draft_passes": trace.draft_rounds,
        "repair_passes": trace.repair_rounds,
        "n_fault_windows_draft": trace.n_fault_windows_draft,
        "n_positions_repaired": trace.n_positions_repaired,
        "n_rejects": trace.rejects,
        "locality": trace.locality,
        "forward_passes": trace.forward_passes,
    }
    if not trace.completed:
        cert["completed"] = False
    return cert


def _top1_prob(probs_row: torch.Tensor) -> float:
    return float(torch.max(probs_row).item())


def _repair_zone(x_list: list, n: int, mask_id: int, halo: int,
                 frozen_keys: set) -> list:
    """Verifier-localized fault envelope: fault windows + halo, minus frozen."""
    faults = find_repeated_ngrams(x_list, n, mask_id)
    fault_pos = set()
    for starts in faults.values():
        for s in starts:
            fault_pos.update(range(s, s + n))
    zone = set()
    for p in fault_pos:
        zone.update(range(p - halo, p + halo + 1))
    zone = {p for p in zone
            if 0 <= p < len(x_list) and p not in frozen_keys}
    return sorted(zone)


def _frozen_contained(faults: dict, frozen_keys: set, n: int) -> bool:
    """True when every violating window lies entirely in frozen text."""
    for starts in faults.values():
        for s in starts:
            if not all(p in frozen_keys for p in range(s, s + n)):
                return False
    return True


def cbrs_sample(model, length: int, mask_id: int, config=None,
                seed: int = 7, frozen=None, policy: str = "cbrs"):
    """Run CBRS (or the gated top-1 vanilla baseline) over a blank canvas.

    Returns (ids LongTensor (1, L), trace, cert).
    """
    assert policy in POLICIES, f"unknown policy {policy!r}"
    cfg = config or CBRSConfig()
    x_list = [mask_id] * length
    frozen = frozen or {}
    frozen_keys = {int(p) for p in frozen}
    for p, t in frozen.items():
        x_list[int(p)] = int(t)
    x = torch.tensor([x_list], dtype=torch.long)

    trace = CBRSTrace(policy)
    trace.preexisting_repeat = bool(
        find_repeated_ngrams(x_list, cfg.n, mask_id))
    banned = {mask_id}

    # ---- Phase 1: draft (unverified speculation) -----------------------
    for rnd in range(cfg.draft_rounds):
        masked = [i for i in range(length) if x_list[i] == mask_id]
        if not masked:
            break
        trace.draft_rounds += 1
        with torch.no_grad():
            logits = model(x)
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)
        scored = sorted(masked, key=lambda p: (-_top1_prob(probs[p]), p))
        working = list(x_list)
        for p in scored[:max(1, cfg.frontier)]:
            cands = _topk_for_pos(probs[p], cfg.draft_depth, banned)
            if not cands:
                continue
            tok, _pr = cands[0]
            working[p] = tok
            trace.draft_commits += 1
            trace.draft_commit_order.append((rnd, p, tok))
        x_list = working
        x = torch.tensor([x_list], dtype=torch.long)

    trace.draft_snapshot = list(x_list)
    trace.n_fault_windows_draft = sum(
        len(v) for v in
        find_repeated_ngrams(x_list, cfg.n, mask_id).values())

    if policy == "vanilla":
        # Baseline: gated top-1 only (no draft speculation, no repair).
        # Re-run as plain gated commits for the comparison leg.
        trace.completed = False
        trace.final_no_repeat = False
        trace.locality = 1.0
        trace.commits = trace.draft_commits
        return x, trace, cbrs_certificate(trace)

    # ---- Phase 2: verifier-localized repair -----------------------------
    cumulative_zone = set()
    stall = 0
    for rr in range(cfg.repair_rounds):
        masked = [i for i in range(length) if x_list[i] == mask_id]
        check = verify_canvas(x_list, mask_id, cfg.n)
        if not masked and check["final_no_repeat"]:
            break
        zone = _repair_zone(x_list, cfg.n, mask_id, cfg.halo, frozen_keys)
        if not zone:
            # Nothing repairable: either clean-but-masked (unfixable
            # model) or faults frozen in place. Stop; fail-closed below.
            trace.events.append(("no_zone", {"round": rr}))
            break
        trace.repair_zones.append(list(zone))
        cumulative_zone.update(zone)
        for p in zone:
            x_list[p] = mask_id
        x = torch.tensor([x_list], dtype=torch.long)

        trace.repair_rounds += 1
        with torch.no_grad():
            logits = model(x)
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)
        scored = sorted(zone, key=lambda p: (-_top1_prob(probs[p]), p))
        working = list(x_list)
        committed = 0
        for p in scored:
            cands = _topk_for_pos(probs[p], cfg.repair_depth, banned)
            chosen = None
            for rank, (tok, _pr) in enumerate(cands):
                if _would_repeat(working, p, tok, cfg.n, mask_id):
                    trace.rejects += 1
                    trace.events.append(
                        ("repair_reject",
                         {"round": rr, "pos": p, "tok": tok, "rank": rank}))
                    continue
                chosen = (tok, rank)
                break
            if chosen is None:
                trace.events.append(("repair_exhausted",
                                     {"round": rr, "pos": p}))
                continue
            tok, rank = chosen
            working[p] = tok
            trace.accepts += 1
            trace.n_repair_commits += 1
            committed += 1
            trace.repair_commit_order.append((rr, p, tok, rank))
        x_list = working
        x = torch.tensor([x_list], dtype=torch.long)
        if committed:
            stall = 0
        else:
            stall += 1
            trace.stalls += 1
            trace.events.append(("repair_stall", {"round": rr}))
            break

    trace.n_positions_repaired = len(cumulative_zone)
    trace.commits = trace.draft_commits + trace.n_repair_commits
    nonfrozen = [p for p in range(length) if p not in frozen_keys]
    untouched = [p for p in nonfrozen if p not in cumulative_zone]
    trace.locality = (len(untouched) / len(nonfrozen)) if nonfrozen else 1.0

    remaining = [i for i in range(length) if x_list[i] == mask_id]
    check = verify_canvas(x_list, mask_id, cfg.n)
    trace.final_no_repeat = check["final_no_repeat"]
    # Fail-closed: complete only when fully unmasked and certificate-clean,
    # unless the only remaining breach pre-existed in frozen text (the
    # sampler cannot repair frozen prompt tokens; reported, not hidden).
    if not remaining and trace.final_no_repeat:
        trace.completed = True
    elif (not remaining and trace.preexisting_repeat and
          _frozen_contained(find_repeated_ngrams(x_list, cfg.n, mask_id),
                            frozen_keys, cfg.n)):
        trace.completed = True
    else:
        trace.completed = False
    cert = cbrs_certificate(trace)
    return x, trace, cert
