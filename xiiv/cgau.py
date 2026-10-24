"""Certificate-Gated Adaptive Unmasking (CGAU).

A proof-certificate-gated sampler for masked diffusion LMs.

Prior art (absorbed, not copied):
- LAVE (arXiv:2602.00612): lookahead verification + verified-prefix cache +
  recovery on repeated failure -- but at the CFG *syntax* level.
- LookUM / Lee et al. (arXiv:2511.05563, ICML 2026): verifier-based *path
  selection* over candidate unmasking orders; the verifier is a
  sequence-certainty measure (unsupervised, coherence-based). No rollback to
  a cached state, no proof certificates, no re-masking.
- UnMaskFork: deterministic branching over unmask actions; assumes
  *monotonic* unmasking (tokens are never re-masked).
- RL unmasking policies (arXiv:2512.09106): learned policies with a
  correctness+compute reward; no certificates.
- CGMS (in-repo, xiiv/diffusion.py `_cgms_commit`): per-placement
  `partial_certificate_check` gating at the *semantic* certificate level --
  the gating itself is CGMS's. CGMS has no recovery, no re-masking, and no
  certificate-driven early exit.

CGAU's claimable novel surface is the triad, at the semantic
proof-certificate level:
  1. Certified-state recovery: a cache of the last certifiable canvas;
     on a gate-failure stall, roll back to it (LAVE's recovery pattern,
     lifted from syntax to proof state).
  2. Certificate-margin early exit: positions whose top-1 is proof-stable
     (certifiable AND (model-confident OR the only certifiable choice))
     commit immediately and skip further denoising passes -- adaptive
     compute driven by proof margin, not model confidence alone.
  3. Targeted re-mask backtracking (last resort): the budgeted pass scans
     top-k for the first certifiable token (the gated core -- an earlier
     commit-top-1-then-backtrack design churned against the early exit,
     diagnosed 2026-10-24). Only when NOTHING in top-k is certifiable
     (hard stall at the position) does tier 1 taboo the top-1; only on
     PERSISTENT stall with a canvas-guilty reason (forbidden_present /
     cap_exceeded naming committed tokens) does tier 2 re-mask the
     lowest-confidence holder and taboo that token there 3x longer (so the
     early exit cannot immediately re-commit it). This breaks the
     monotonic-unmasking assumption instead of passively deferring.

Falsifiers (pre-registered; see research/cgau_design_2026-10-24.md):
  P1 final_valid(cgau) >= final_valid(gated) > final_valid(vanilla)
  P2 the gate fires on adversarial instances (gated/cgau reject > 0)
  P3 forward_passes(cgau) <= forward_passes(gated)  (early-exit compute win)
  P4 rollbacks + remasks + early_exits > 0 across the suite (mechanisms fire)
  KILL: final_valid(cgau) <= final_valid(vanilla) -> the triad adds nothing.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V)
and verifies against the existing contract compiler output via
`partial_certificate_check` (sound fail direction) and `check_constraints`
(final verdict) from xiiv.diffusion. No GPU needed.
"""

import re

import torch

from .diffusion import check_constraints, partial_certificate_check

__all__ = [
    "CGAUConfig",
    "CGAUTrace",
    "cgau_sample",
    "POLICIES",
]

POLICIES = ("vanilla", "gated", "cgau")


class CGAUConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        topk: int = 4,
        budget: int = 2,
        max_steps: int = 64,
        tau_margin: float = 0.35,
        stall_budget: int = 3,
        taboo_cooldown: int = 4,
    ):
        self.topk = topk                # candidate tokens considered per position
        self.budget = budget            # gated commits per step (early exits are extra)
        self.max_steps = max_steps      # hard termination cap
        self.tau_margin = tau_margin    # model-confidence margin for early exit
        self.stall_budget = stall_budget  # zero-commit steps before rollback
        self.taboo_cooldown = taboo_cooldown  # steps a tabooed token stays banned


class CGAUTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.steps = 0
        self.forward_passes = 0
        self.proposals = 0          # gate-evaluated (pos, tok) proposals
        self.gate_rejects = 0       # proposals failing partial_certificate_check
        self.commits = 0
        self.early_exits = 0
        self.remasks = 0
        self.rollbacks = 0
        self.taboo_hits = 0         # proposals skipped due to taboo
        self.stalls = 0
        self.completed = False
        self.final_valid = False
        self.unrecoverable = False
        self.events = []            # ("remask"|"rollback"|"early_exit", detail)

    def cert_failure_rate(self) -> float:
        return self.gate_rejects / self.proposals if self.proposals else 0.0

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "events"}
        d["cert_failure_rate"] = self.cert_failure_rate()
        d["n_events"] = len(self.events)
        return d


def _culprit_positions(reason: str, x: torch.Tensor, committed: dict,
                       anchors: dict) -> list:
    """Committed positions implicated by a gate-failure reason.

    Backtracking fires ONLY when the reason proves the *canvas* (not the
    proposal) is guilty: a forbidden token already present, or an at-most
    cap already reached by committed tokens. For order / required-slot
    reasons the proposal is at fault and is handled by taboo alone --
    blaming the most recent commit there caused re-mask thrash (diagnosed
    2026-10-24: 46 remasks / 4 instances, completion collapsed).
    Anchor positions are never blamed (they cannot be re-masked).
    """
    x_list = x.tolist()
    m = re.search(r"(?:forbidden_present|cap_exceeded):(\d+)", reason or "")
    if m:
        tok = int(m.group(1))
        return [p for p in committed
                if x_list[p] == tok and p not in anchors]
    m = re.search(r"anchor_conflict@(\d+)", reason or "")
    if m:
        p = int(m.group(1))
        return [p] if p in committed and p not in anchors else []
    return []


def _tight_need(x_list: list, c: dict, masked: set) -> set | None:
    """Slot reservation (mirrors DenoisingState's shared sampler logic).

    When the remaining masked slots exactly equal the still-missing required
    tokens, only those tokens may be proposed -- otherwise the canvas could
    never complete. Returns the allowed token set, or None when not tight.
    This is proposal shaping shared by all policies, not a CGAU mechanism.
    """
    required_counts = dict(c.get("required_counts") or {})
    for t in (c.get("required_ids") or []):
        required_counts.setdefault(int(t), 1)
    missing = {}
    for t, n in required_counts.items():
        short = int(n) - x_list.count(int(t))
        if short > 0:
            missing[int(t)] = short
    if missing and sum(missing.values()) == len(masked):
        return set(missing)
    return None


def _topk_for_pos(probs_row: torch.Tensor, k: int, banned: set):
    """Top-k (token, prob) pairs excluding banned ids. Deterministic.

    Only strictly-positive probabilities are returned: softmax underflow
    turns -inf logits into exact 0.0, and the sampler must never commit a
    token the model assigns zero probability (diagnosed 2026-10-24: the
    top-k scan was committing 0.0-prob filler tokens from scripted stubs).
    """
    row = probs_row.clone()
    if banned:
        row[list(banned)] = -1.0
    k = max(1, min(k, int((row > 0).sum().item())))
    vals, idx = torch.topk(row, k)
    return [(int(t), float(v)) for t, v in zip(idx.tolist(), vals.tolist())
            if float(v) > 0.0]


def cgau_sample(model, *, length: int, mask_id: int, constraints: dict,
                config: CGAUConfig | None = None, seed: int = 0,
                policy: str = "cgau", frozen: dict | None = None):
    """Run one constrained unmasking trajectory.

    Args:
        model: callable x(1, L) -> logits(1, L, V).
        length: canvas length. mask_id: the mask token id.
        constraints: contract-compiler dict (anchors, forbidden_ids,
            required_ids, required_counts, at_most, forbidden_phrases,
            order_pairs) -- the same dict `partial_certificate_check`
            and `check_constraints` consume.
        policy: "vanilla" (confidence unmasking, no checks),
            "gated" (CGMS-4th-gate spirit: per-placement certificate gate
            with top-k fallback scan, defer on failure, no backtracking,
            no taboo, no recovery), or "cgau" (full triad).
        frozen: optional {pos: token} pinned positions (treated as anchors).

    Returns:
        (ids LongTensor(L), CGAUTrace).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {POLICIES}")
    cfg = config or CGAUConfig()
    c = constraints or {}
    # normalize anchor keys to int (the contract compiler may emit str keys);
    # anchors are never re-masked by backtracking or rollback
    anchors = {int(p): int(t) for p, t in (c.get("anchors") or {}).items()}
    if frozen:
        anchors.update({int(p): int(t) for p, t in frozen.items()})
    forbidden = {int(t) for t in (c.get("forbidden_ids") or [])}

    torch.manual_seed(seed)
    x = torch.full((length,), mask_id, dtype=torch.long)
    for p, t in anchors.items():
        x[int(p)] = int(t)
    masked = {p for p in range(length) if int(x[p]) == mask_id}
    committed: dict[int, tuple[int, int, float]] = {}  # pos -> (tok, step, conf)
    taboo: dict[int, dict[int, int]] = {}             # pos -> {tok: cooldown}
    consec_reject: dict[int, int] = {}  # pos -> consecutive gate rejections
    cache = None          # (x_clone, frozenset(committed))
    stall = 0
    trace = CGAUTrace(policy)
    
    def cert_ok(pos: int, tok: int) -> tuple[bool, str]:
        trial = x.clone()
        trial[pos] = tok
        return partial_certificate_check(trial, c, mask_id)

    step = 0
    while masked and step < cfg.max_steps:
        step += 1
        trace.steps = step
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
        trace.forward_passes += 1
        probs = torch.softmax(logits.float(), dim=-1)

        # ---- candidate proposals for every masked position ----
        cand: dict[int, list[tuple[int, float]]] = {}
        order = []
        vocab_size = probs.shape[-1]
        need_set = _tight_need(x.tolist(), c, masked)
        for pos in sorted(masked):
            tb = taboo.get(pos) or {}
            for t in list(tb):
                if tb[t] <= 0:
                    del tb[t]
            # honest taboo-hit count: tokens that would be top-k without taboo
            row_nb = probs[pos].clone()
            row_nb[mask_id] = -1.0
            for t in forbidden:
                row_nb[t] = -1.0
            topk_nb = _topk_for_pos(row_nb, cfg.topk, set())
            trace.taboo_hits += sum(1 for t, _ in topk_nb
                                    if tb.get(t, 0) > 0)
            banned = {mask_id} | forbidden | {t for t, cd in tb.items()
                                               if cd > 0}
            if need_set is not None:
                banned |= {t for t in range(vocab_size)
                           if t not in need_set}
            topk = _topk_for_pos(probs[pos], cfg.topk, banned)
            if not topk:
                continue
            cand[pos] = topk
            order.append((topk[0][1], pos))

        step_commits = 0

        # ---- CGAU mechanism 2: certificate-margin early exit ----
        if policy == "cgau":
            for _conf, pos in sorted(order, reverse=True):
                if pos not in masked:
                    continue
                topk = cand[pos]
                t1, p1 = topk[0]
                t2 = topk[1][0] if len(topk) > 1 else None
                ok1, _ = cert_ok(pos, t1)
                if not ok1:
                    continue
                margin = p1 - (topk[1][1] if len(topk) > 1 else 0.0)
                ok2 = cert_ok(pos, t2)[0] if t2 is not None else True
                proof_stable = (margin >= cfg.tau_margin) or (not ok2)
                if proof_stable:
                    x[pos] = t1
                    masked.discard(pos)
                    committed[pos] = (t1, step, p1)
                    consec_reject[pos] = 0
                    step_commits += 1
                    trace.commits += 1
                    trace.early_exits += 1
                    trace.events.append(("early_exit",
                                         {"pos": pos, "tok": t1,
                                          "step": step,
                                          "margin": round(margin, 4),
                                          "unique": not ok2}))

        # ---- budgeted commits, confidence order (all policies) ----
        budgeted = 0
        order = [(cf, p) for cf, p in order if p in masked]
        order.sort(reverse=True)
        for _cf, pos in order:
            if pos not in masked:
                continue
            if budgeted >= cfg.budget:
                break
            budgeted += 1
            t1, p1 = cand[pos][0]
            if policy == "vanilla":
                x[pos] = t1
                masked.discard(pos)
                committed[pos] = (t1, step, p1)
                step_commits += 1
                trace.commits += 1
                continue
            if policy == "gated":
                # CGMS-4th-gate spirit: scan top-k for the first
                # certifiable token; defer the position if none qualifies.
                # No backtracking, no taboo, no recovery.
                for tok, pr in cand[pos]:
                    trace.proposals += 1
                    ok, _ = cert_ok(pos, tok)
                    if ok:
                        x[pos] = tok
                        masked.discard(pos)
                        committed[pos] = (tok, step, pr)
                        step_commits += 1
                        trace.commits += 1
                        trace.events.append(("commit",
                                             {"pos": pos, "tok": tok,
                                              "step": step}))
                        break
                    trace.gate_rejects += 1
                continue
            # cgau: scan top-k for the first certifiable token (the gated
            # core -- committing-then-backtracking churned against the
            # early exit, diagnosed 2026-10-24). Backtracking is now a
            # last resort for hard stalls, not the first response.
            placed = False
            first_reason: str | None = None
            for tok, pr in cand[pos]:
                trace.proposals += 1
                ok, reason = cert_ok(pos, tok)
                if ok:
                    x[pos] = tok
                    masked.discard(pos)
                    committed[pos] = (tok, step, pr)
                    step_commits += 1
                    trace.commits += 1
                    consec_reject[pos] = 0
                    placed = True
                    trace.events.append(("commit",
                                         {"pos": pos, "tok": tok,
                                          "step": step}))
                    break
                trace.gate_rejects += 1
                if first_reason is None:
                    first_reason = reason
            if not placed:
                # Hard stall at this position: nothing in top-k is
                # certifiable. Tier 1: taboo the top-1 briefly.
                consec_reject[pos] = consec_reject.get(pos, 0) + 1
                taboo.setdefault(pos, {})[t1] = cfg.taboo_cooldown
                # ---- CGAU mechanism 3: targeted re-mask backtracking ----
                # Tier 2 (last resort): persistent stall with a
                # canvas-guilty reason -- the committed canvas itself
                # blocks every proposal, so un-commit one holder.
                if consec_reject[pos] >= 2:
                    culprits = _culprit_positions(first_reason or "", x,
                                                  committed, anchors)
                    if culprits:
                        victim = min(culprits,
                                     key=lambda p: committed[p][2])
                        vtok = int(x[victim])
                        x[victim] = mask_id
                        masked.add(victim)
                        del committed[victim]
                        consec_reject[victim] = 0
                        # victim taboo outlasts the proposer taboo: without
                        # this the early exit re-commits the victim token
                        # and the two mechanisms churn (diagnosed 2026-10-24).
                        taboo.setdefault(victim, {})[vtok] = \
                            3 * cfg.taboo_cooldown
                        trace.remasks += 1
                        trace.events.append(("remask",
                                             {"pos": victim, "tok": vtok,
                                              "reason": first_reason,
                                              "step": step}))

        # ---- CGAU mechanism 1: certified-state cache + rollback ----
        if step_commits > 0:
            stall = 0
            if use_gate := (policy in ("gated", "cgau")):
                ok_all, _ = partial_certificate_check(x, c, mask_id)
                assert ok_all, \
                    "gated commits must keep the canvas certifiable"
            cache = (x.clone(), frozenset(committed))
        else:
            stall += 1
            trace.stalls += 1
            if policy == "cgau" and stall >= cfg.stall_budget:
                if cache is None:
                    trace.unrecoverable = True
                    break
                cx, ckeys = cache
                # positions committed after the cache point get re-masked
                # by the restore; taboo their tokens so the retry differs
                for p, (tok, _s, _cf) in list(committed.items()):
                    if p not in ckeys:
                        taboo.setdefault(p, {})[tok] = cfg.taboo_cooldown
                x = cx.clone()
                committed = {p: v for p, v in committed.items()
                             if p in ckeys}
                masked = {p for p in range(length)
                          if int(x[p]) == mask_id}
                trace.rollbacks += 1
                trace.events.append(("rollback", {"step": step,
                                                  "restored_commits":
                                                  len(ckeys)}))
                stall = 0

        # decay taboos
        for pos in list(taboo):
            for t in list(taboo[pos]):
                taboo[pos][t] -= 1
                if taboo[pos][t] <= 0:
                    del taboo[pos][t]
            if not taboo[pos]:
                del taboo[pos]

    trace.completed = not masked
    if trace.completed:
        report = check_constraints(x, c, mask_id)
        trace.final_valid = bool(all(report.values()))
        trace.events.append(("final_check", report))
    return x, trace
