"""PTCS -- Planned-Trajectory Certificate Scheduler for masked-diffusion decoding.

Every sampler in this family so far (DCMR, CGAU, CGEC, CDR) makes
per-position commit decisions inside a *fixed* outer schedule: a constant
commit budget per step and a fixed step cap, with no notion of a plan.
PTCS plans the trajectory itself:

1. Calibrated informativeness. Each masked position scores
       info(p) = top1_prob(p) - trivial_weight * decay(step) * is_trivial(top1(p))
   with decay(step) = max(0, 1 - step / max_steps). Early in the run,
   trivial tokens (punctuation, filler -- a caller-supplied ``trivial_ids``
   set) are strongly discounted, so high-signal tokens commit first and the
   verifier sees dense evidence early; the discount decays to zero so no
   token is ever stranded.
2. Global budget planner. Instead of a fixed k commits/step, the planner
   reads the *global* informativeness distribution each step:
   - expand: if a full base-budget's worth of positions clears commit_bar,
     harvest the whole confident cluster (up to budget_max);
   - contract: if even the best position is below hold_floor, commit just
     1 and save the remaining passes for denser context later;
   - otherwise: the base budget ceil(masked / remaining_steps).
3. Verifier-side settling (the SureLock idea, with a non-circular
   criterion). A committed position *settles* when the INDEPENDENT
   verifier's margin -- cert_margin on the no-repeat-n-gram certificate --
   stays >= settle_margin for settle_streak consecutive verification
   rounds. Settled positions need no further re-evaluation: a SureLock-style
   deployment skips their query projection + FFN on later steps, and the
   trace reports est_query_skips = settled x steps as the audited saving.
   The settle criterion deliberately uses the verifier, never the model's
   own posterior (SureLock monitors the model's local KL -- the model
   grading its own convergence). A settled position whose margin later
   collapses is un-settled, remasked, and its offender tabooed.
4. Pre-commit certificate gate: a candidate commit that would complete a
   repeated n-gram is rejected (dcmr._would_repeat) and the next-best
   candidate is tried instead, so the gate never strands a position; the
   final canvas is certified repeat-free by construction.

Fail-closed: a completed canvas that violates the certificate is never
presented as certified (completed=False), unless the breach pre-existed in
frozen prompt tokens.

Prior art (absorbed, not copied):
- PC-Sampler (Huang et al., arXiv:2508.13021, v4 Oct 2026): diagnosed the
  two weaknesses this module addresses -- (a) uncertainty-based samplers
  lack global trajectory control, (b) they are biased toward trivial tokens
  early in decoding -- and proposed position-aware weighting with calibrated
  confidence (+10% avg over existing MDM samplers). PTCS ports the
  *diagnosis*, not the mechanism: the planning objective here is
  certificate acceptance under an independent verifier, and the
  trivial-token calibration is a decaying discount on a caller-supplied
  trivial set rather than PC-Sampler's learned calibration.
- SureLock (Oba et al., ICLR 2026): lock converged token positions, skip
  their query projection + FFN while caching K/V -- 30-50% FLOP saving on
  LLaDA-8B. PTCS ports the *lock-and-skip* idea; the convergence criterion
  is verifier-side (certificate-margin streak), so a confidently wrong
  model cannot lock in its own error. est_query_skips is the
  stub-measurable counterpart of SureLock's FLOP claim.
- In-repo: dcmr._would_repeat (pre-commit gate), cdr.cert_margin
  (verifier endorsement), the CDR fail-closed pattern.

Pre-registered falsifiers (see tests/test_ptcs.py):
- F1 trivial-first bias: on the trivial-dominant stub, PTCS commits every
  informative position before any trivial one; vanilla TPP commits trivial
  first. KILL: PTCS's trivial-first rate >= vanilla's.
- F2 planner dividend: under a tight step budget, PTCS finishes in fewer
  forward passes than fixed-budget vanilla on the same stub.
- F3 settle soundness: on the stable stub, settles > 0, unsets == 0,
  est_query_skips > 0.
- F4 settle defense-in-depth: a repeat injected around a settled position
  (bypassing the gate, as a unit test of the un-settle path) MUST
  un-settle, remask, and taboo the offender; the canvas is clean after.
- F5 certificate soundness: repeater stub -> completed, no repeated n-gram.
- KILL: F1 or F5 fails -> the module does not ship.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.

Measured on CPU scripted stubs only -- real-checkpoint behavior
untested. The no-repeat certificate's margin is binary (0.0/1.0); graded
certificates would make the settle streak genuinely graded -- noted, not
claimed.
"""

import torch

from .dcmr import _would_repeat, find_repeated_ngrams
from .cdr import cert_margin

__all__ = [
    "PTCSConfig",
    "PTCSTrace",
    "POLICIES",
    "informativeness",
    "plan_budget",
    "ptcs_sample",
    "ptcs_certificate",
    "verify_positions",
]

POLICIES = ("vanilla", "ptcs")


class PTCSConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        budget_max: int = 8,
        trivial_weight: float = 1.0,
        commit_bar: float = 0.5,
        hold_floor: float = 0.05,
        commit_floor: float = 0.5,
        settle_margin: float = 1.0,
        settle_streak: int = 2,
        verify_every: int = 1,
        gate_candidates: int = 8,
        taboo_cooldown: int = 4,
        max_steps: int = 64,
        stall_budget: int = 3,
    ):
        self.n = n                              # n-gram order of the certificate
        self.budget_max = budget_max            # planner expansion cap
        self.trivial_weight = trivial_weight    # trivial-token penalty scale
        self.commit_bar = commit_bar            # info bar to expand the budget
        self.hold_floor = hold_floor            # below this top info -> commit 1
        self.commit_floor = commit_floor        # cert_margin gate on commit
        self.settle_margin = settle_margin      # verifier margin to settle
        self.settle_streak = settle_streak      # consecutive clean rounds
        self.verify_every = verify_every        # steps between verify rounds
        self.gate_candidates = gate_candidates  # top-k tried per position
        self.taboo_cooldown = taboo_cooldown    # steps a remasked token stays banned
        self.max_steps = max_steps              # hard termination cap
        self.stall_budget = stall_budget        # zero-progress steps before giving up


class PTCSTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.steps = 0
        self.commits = 0
        self.commit_passes = 0
        self.forward_passes = 0
        self.verify_rounds = 0
        self.regen_passes = 0
        self.settles = 0
        self.unsets = 0
        self.rot_remasks = 0
        self.gate_rejects = 0
        self.est_query_skips = 0
        self.stalls = 0
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen/prompt text already repeats
        self.commit_order = []           # positions in commit order
        self.settled_final = []          # positions settled at end of run
        self.plan = []                   # per-step planner record
        self.events = []

    @property
    def remasks(self) -> int:
        return self.rot_remasks

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("events", "commit_order", "settled_final", "plan")}
        d["remasks"] = self.remasks
        d["n_events"] = len(self.events)
        d["n_commits_order"] = len(self.commit_order)
        d["n_settled_final"] = len(self.settled_final)
        d["n_plan_steps"] = len(self.plan)
        return d


def informativeness(top_prob: float, tok: int, trivial_ids: set,
                    step: int, max_steps: int, trivial_weight: float) -> float:
    """Calibrated informativeness of a candidate commit.

    Trivial tokens are discounted by trivial_weight * decay(step), with
    decay(step) = max(0, 1 - step / max_steps): full penalty at step 0,
    none at the horizon, so trivial tokens commit late but never strand.
    """
    decay = max(0.0, 1.0 - step / max(1, max_steps))
    penalty = trivial_weight * decay if tok in trivial_ids else 0.0
    return top_prob - penalty


def plan_budget(n_masked: int, remaining_steps: int,
                infos_desc: list, cfg: PTCSConfig) -> int:
    """The global trajectory planner: how many positions to commit now.

    infos_desc: informativeness values of the masked positions, sorted
    descending. Reads the *global* distribution, not just the top pick:
    expand on a confident cluster, contract to 1 when nothing is worth
    committing yet, else the base budget that finishes exactly on time.
    """
    if n_masked <= 0 or not infos_desc:
        return 0
    base = max(1, -(-n_masked // max(1, remaining_steps)))  # ceil
    base = min(base, n_masked)
    if infos_desc[base - 1] >= cfg.commit_bar:
        harvest = sum(1 for v in infos_desc if v >= cfg.commit_bar)
        return max(base, min(cfg.budget_max, harvest, n_masked))
    if infos_desc[0] < cfg.hold_floor:
        return 1
    return base


def _top1_for_pos(probs_row: torch.Tensor, banned: set):
    """Top-1 (token, prob) excluding banned ids. Deterministic."""
    row = probs_row.clone()
    if banned:
        row[list(banned)] = float("-inf")
    top_prob, top_tok = torch.max(row, dim=0)
    if not torch.isfinite(top_prob):
        return None, None
    return int(top_tok.item()), float(top_prob.item())


def _candidates_for_pos(probs_row: torch.Tensor, banned: set, k: int):
    """Top-k (token, prob) candidates excluding banned ids, prob order."""
    row = probs_row.clone()
    if banned:
        row[list(banned)] = float("-inf")
    k = max(1, min(k, row.numel()))
    vals, idx = torch.topk(row, k)
    out = []
    for v, t in zip(vals.tolist(), idx.tolist()):
        if v == float("-inf"):
            continue
        out.append((int(t), float(v)))
    return out


def verify_positions(x_list: list, committed: list, frozen: set,
                     settled: set, streak: dict, taboo: dict,
                     mask_id: int, cfg: PTCSConfig,
                     trace: "PTCSTrace | None" = None):
    """One verifier round over the committed positions.

    Settles positions whose certificate margin holds for settle_streak
    consecutive rounds; un-settles (remask + taboo the offender) any
    settled position whose margin collapsed. The verifier reads the
    canvas directly -- zero model forward passes.

    Mutates x_list (remasked positions), settled, streak, taboo, masked
    membership is handled by the caller via the returned remasked list.
    Returns (n_newly_settled, n_unsettled, remasked_positions).
    """
    newly_settled = 0
    unsettled = 0
    remasked = []
    for p in committed:
        if p in frozen:
            continue
        m = cert_margin(p, x_list, mask_id, cfg.n)
        if p in settled:
            if m < cfg.settle_margin:
                # Margin rot: the proof no longer endorses this token.
                # Un-settle, remask, taboo the offender (defense in
                # depth -- unreachable through the gated commit flow,
                # tested by unit injection in F4).
                settled.discard(p)
                streak[p] = 0
                unsettled += 1
                old = x_list[p]
                taboo.setdefault(p, {})[old] = cfg.taboo_cooldown
                x_list[p] = mask_id
                remasked.append(p)
                if trace is not None:
                    trace.unsets += 1
                    trace.rot_remasks += 1
                    trace.events.append(
                        ("unsettle", {"pos": p, "tok": old,
                                      "margin": round(m, 4)}))
        else:
            if m >= cfg.settle_margin:
                streak[p] = streak.get(p, 0) + 1
                if streak[p] >= cfg.settle_streak:
                    settled.add(p)
                    newly_settled += 1
                    if trace is not None:
                        trace.settles += 1
                        trace.events.append(("settle", {"pos": p}))
            else:
                streak[p] = 0
    if trace is not None:
        trace.verify_rounds += 1
    return newly_settled, unsettled, remasked


def ptcs_certificate(trace: "PTCSTrace", ids: torch.Tensor, mask_id: int,
                     n: int) -> dict:
    """Build the proof-carrying record for a PTCS run.

    The no-repeat claim replays model-free from ``tokens``; the planner
    record and settle counts replay from the trace.
    """
    return {
        "certificate": "ptcs-v1",
        "policy": trace.policy,
        "n": n,
        "length": int(ids.numel()),
        "mask_id": int(mask_id),
        "tokens": [int(t) for t in ids.tolist()],
        "commit_order": list(trace.commit_order),
        "settled_final": list(trace.settled_final),
        "plan": [dict(s) for s in trace.plan],
        "n_settles": trace.settles,
        "n_unsets": trace.unsets,
        "n_rot_remasks": trace.rot_remasks,
        "n_gate_rejects": trace.gate_rejects,
        "est_query_skips": trace.est_query_skips,
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "preexisting_repeat": trace.preexisting_repeat,
    }


def ptcs_sample(model, *, length: int, mask_id: int,
                config: "PTCSConfig | None" = None, seed: int = 0,
                policy: str = "ptcs",
                trivial_ids: "set | None" = None,
                frozen: "dict | None" = None):
    """Run one planned unmasking trajectory with certificate scheduling.

    Args:
        model: callable x(1, L) -> logits(1, L, V).
        length: canvas length. mask_id: the mask token id.
        policy: "ptcs" (planned trajectory + settle + gate) or "vanilla"
            (fixed-budget TPP by raw confidence -- the baseline).
        trivial_ids: token ids treated as trivial for the calibrated
            informativeness discount. None/empty = no discount.
        frozen: optional {pos: token} pinned positions (treated as
            committed context; never settled, verified, or remasked).

    Returns:
        (ids LongTensor(L), PTCSTrace, certificate dict).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {POLICIES}")
    cfg = config or PTCSConfig()
    if cfg.n < 1:
        raise ValueError("n-gram order must be >= 1")
    if cfg.settle_streak < 1:
        raise ValueError("settle_streak must be >= 1")
    if cfg.verify_every < 1:
        raise ValueError("verify_every must be >= 1")
    trivial_ids = set(int(t) for t in (trivial_ids or ()))
    frozen = {int(p): int(t) for p, t in (frozen or {}).items()}
    frozen_pos = set(frozen)

    torch.manual_seed(seed)
    x = torch.full((length,), mask_id, dtype=torch.long)
    for p, t in frozen.items():
        x[int(p)] = int(t)
    x_list = x.tolist()
    preexisting = bool(find_repeated_ngrams(x_list, cfg.n, mask_id))
    masked = {p for p in range(length) if int(x[p]) == mask_id}
    taboo: dict[int, dict[int, int]] = {}   # pos -> {tok: cooldown}
    settled: set[int] = set()
    streak: dict[int, int] = {}
    trace = PTCSTrace(policy)
    trace.preexisting_repeat = preexisting
    stall = 0

    def decay_taboo():
        for pos in list(taboo):
            for t in list(taboo[pos]):
                taboo[pos][t] -= 1
                if taboo[pos][t] <= 0:
                    del taboo[pos][t]
            if not taboo[pos]:
                del taboo[pos]

    def banned_for(pos: int) -> set:
        tb = taboo.get(pos) or {}
        return {mask_id} | {t for t, cd in tb.items() if cd > 0}

    def commit_phase(probs, infos, k: int, tag: str) -> int:
        """Commit up to k positions, ordered by informativeness (ptcs)
        or raw top-1 probability (vanilla). Pre-commit certificate gate
        applies under ptcs only."""
        if policy == "ptcs":
            ordered = sorted(masked, key=lambda p: -infos[p])
        else:
            top = {}
            for p in masked:
                tok, pr = _top1_for_pos(probs[p], banned_for(p))
                top[p] = (tok, pr if pr is not None else float("-inf"))
            ordered = sorted(masked, key=lambda p: -top[p][1])
        made = 0
        for pos in ordered:
            if made >= k or pos not in masked:
                continue
            # Try candidates in probability order: a gate-rejected top-1
            # falls back to the next best token instead of stranding the
            # position (the stall the naive gate would cause).
            chosen = None
            for tok, pr in _candidates_for_pos(probs[pos], banned_for(pos),
                                               cfg.gate_candidates):
                if policy == "ptcs" and _would_repeat(x_list, pos, tok,
                                                      cfg.n, mask_id):
                    trace.gate_rejects += 1
                    trace.events.append(
                        ("gate_reject", {"pos": pos, "tok": tok}))
                    continue
                chosen = (tok, pr)
                break
            if chosen is None:
                continue
            tok, _pr = chosen
            x[pos] = tok
            x_list[pos] = tok
            masked.discard(pos)
            made += 1
            trace.commits += 1
            trace.commit_order.append(pos)
            streak[pos] = 0
            trace.events.append(
                (f"{tag}_commit",
                 {"pos": pos, "tok": tok,
                  "info": round(infos.get(pos, 0.0), 4)}))
        return made

    def regen_after_verify() -> int:
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
        trace.forward_passes += 1
        trace.regen_passes += 1
        probs = torch.softmax(logits.float(), dim=-1)
        infos = {p: informativeness(
            float(torch.max(probs[p]).item()),
            int(torch.argmax(probs[p]).item()),
            trivial_ids, trace.steps, cfg.max_steps,
            cfg.trivial_weight) for p in masked}
        k = (plan_budget(len(masked), cfg.max_steps - trace.steps,
                         sorted(infos.values(), reverse=True), cfg)
             if policy == "ptcs" else 1)
        return commit_phase(probs, infos, max(1, k), "regen")

    step = 0
    while masked and step < cfg.max_steps:
        step += 1
        trace.steps = step
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
        trace.forward_passes += 1
        trace.commit_passes += 1
        probs = torch.softmax(logits.float(), dim=-1)

        infos = {}
        for p in masked:
            tok, pr = _top1_for_pos(probs[p], banned_for(p))
            if tok is None:
                infos[p] = float("-inf")
            else:
                infos[p] = informativeness(
                    pr, tok, trivial_ids, step, cfg.max_steps,
                    cfg.trivial_weight)
        remaining = cfg.max_steps - step
        if policy == "ptcs":
            k = plan_budget(len(masked), remaining,
                            sorted(infos.values(), reverse=True), cfg)
        else:
            k = max(1, -(-len(masked) // max(1, remaining)))
            k = min(k, len(masked))
        trace.plan.append({"step": step, "k": k,
                           "masked": len(masked),
                           "top_info": round(max(infos.values()), 4)
                           if infos else 0.0})

        made = commit_phase(probs, infos, k, "main")

        remade = 0
        if policy == "ptcs" and step % cfg.verify_every == 0 and made >= 0:
            committed = [p for p in range(length)
                         if x_list[p] != mask_id and p not in frozen_pos]
            _, _, remasked = verify_positions(
                x_list, committed, frozen_pos, settled, streak, taboo,
                mask_id, cfg, trace)
            for p in remasked:
                x[p] = mask_id
                masked.add(p)
            if remasked:
                remade = regen_after_verify()

        # SureLock-style saving, audited: settled positions need no
        # further query projection + FFN on a deployment that wires the
        # settled set into the attention kernel. Here it is counted, not
        # executed -- the stubs re-query the full canvas every pass.
        trace.est_query_skips += len(settled)

        if made + remade > 0:
            stall = 0
        else:
            stall += 1
            trace.stalls += 1
            if stall >= cfg.stall_budget:
                trace.events.append(("stall_exit", {"step": step}))
                break
        decay_taboo()

    # Final verification: bounded re-verify rounds; a repair that cannot
    # re-complete fails closed below.
    if policy == "ptcs":
        for _ in range(3):
            committed = [p for p in range(length)
                         if x_list[p] != mask_id and p not in frozen_pos]
            _, _, remasked = verify_positions(
                x_list, committed, frozen_pos, settled, streak, taboo,
                mask_id, cfg, trace)
            for p in remasked:
                x[p] = mask_id
                masked.add(p)
            if not remasked:
                break
            regen_after_verify()

    trace.settled_final = sorted(settled)
    trace.completed = not masked
    remaining_repeats = find_repeated_ngrams(x_list, cfg.n, mask_id)
    trace.final_no_repeat = not remaining_repeats
    if trace.completed:
        trace.events.append(("final_check",
                             {"repeated": len(remaining_repeats)}))
    # Sound fail direction: a completed canvas that violates the
    # certificate is never presented as certified -- it fails closed
    # (completed=False). Vanilla is the deliberately uncertified
    # baseline, so it is exempt, as is a breach preexisting in frozen
    # prompt tokens.
    if (trace.completed and not trace.final_no_repeat
            and not trace.preexisting_repeat and policy != "vanilla"):
        trace.completed = False
        trace.events.append(("fail_closed",
                             {"repeated": len(remaining_repeats)}))
    cert = ptcs_certificate(trace, x, mask_id, cfg.n)
    return x, trace, cert
