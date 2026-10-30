"""Decoupled Certificate-Margin Remasking (DCMR).

A sampler for masked diffusion LMs that certifies a *no-repeated-n-gram*
property of the generated canvas -- free generation, no contract needed.

The certificate: at every point, the position-ordered committed canvas
(masks act as separators) contains no repeated n-gram of order ``n``.
Commits are gated on it, a post-step scan repairs repeats that parallel
commits could jointly create, and the final check asserts it. Fail-closed:
if the canvas cannot complete under the certificate, the run reports
``completed=False`` rather than emitting a degenerate repeat.

Prior art (absorbed, not copied):
- NAVIRA (arXiv:2606.06031): decoupled stochastic remasking for masked
  diffusion LMs -- a first forward pass estimates token quality and
  selects unreliable positions; those are masked out and a second
  forward pass recomputes generation logits from the corrected context,
  so tokens marked for correction cannot contaminate their own
  regeneration. Remask targets are sampled from a temperature-controlled
  distribution over quality scores (diversity preservation, no entropy
  collapse). DCMR lifts both mechanisms from *model-estimated quality*
  to *certificate-violation signals*: remask candidates are positions
  participating in a repeated n-gram (a checkable certificate breach),
  and the regeneration pass runs on the cleaned canvas.
- Dang & Ermon (arXiv:2607.07026): exact constrained decoding for dLLMs
  under finite-automaton constraints. A no-repeat-n-gram constraint is a
  regular language (de Bruijn automaton), so "constraint satisfaction by
  construction" is the formal backbone here; DCMR implements it via a
  per-commit gate + repair scan rather than their graphical-model
  inference, trading exactness of the posterior for a sampler that runs
  on any ``model(x) -> logits`` without extra machinery.
- HF ``no_repeat_ngram_size``: logit masking for *autoregressive*
  left-to-right decoding -- blunt and inapplicable to parallel,
  non-monotonic diffusion unmasking, where repeats emerge from
  simultaneous commitments and must be repaired by re-masking, not by
  masking logits.
- In-repo CGAU (xiiv/cgau.py): certificate-gated unmasking with recovery,
  but only under an explicit contract-compiler constraint dict
  (anchors/forbidden/required). DCMR is the free-generation sibling:
  the certificate is intrinsic (no-repeat-n-gram), so it works with no
  contract at all.

Pre-registered falsifiers (see tests/test_dcmr.py):
- F1 certificate soundness: the final canvas contains no repeated
  n-gram of order n on every run (scripted repeater model).
- F2 diversity: distinct-n-gram ratio strictly above the vanilla
  confidence-unmasking baseline on the repeater model.
- F3 pass budget (structural): forward_passes <= 2 * steps -- at most
  one decoupled regeneration pass per step, by construction.
- KILL: F1 fails -> the module does not ship. F2 failing means the
  mechanism adds nothing over vanilla -> kill.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.
"""

import torch

__all__ = [
    "DCMRConfig",
    "DCMRTrace",
    "dcmr_sample",
    "POLICIES",
    "find_repeated_ngrams",
]

POLICIES = ("vanilla", "dcmr")


class DCMRConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        topk: int = 4,
        budget: int = 4,
        remask_budget: int = 4,
        max_steps: int = 64,
        temperature: float = 1.0,
        conf_floor: float = 0.02,
        taboo_cooldown: int = 4,
        stall_budget: int = 3,
    ):
        self.n = n                        # n-gram order of the certificate
        self.topk = topk                  # candidates scanned per position
        self.budget = budget              # commits per phase
        self.remask_budget = remask_budget  # max remasks per regen cycle
        self.max_steps = max_steps        # hard termination cap
        self.temperature = temperature    # stochastic remask selection
        self.conf_floor = conf_floor      # committed conf below this is suspect
        self.taboo_cooldown = taboo_cooldown  # steps a remasked token stays banned
        self.stall_budget = stall_budget  # zero-progress steps before giving up


class DCMRTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.steps = 0
        self.forward_passes = 0
        self.commits = 0
        self.gate_rejects = 0       # commits refused by the no-repeat gate
        self.remasks = 0
        self.regen_passes = 0       # decoupled regeneration forward passes
        self.taboo_hits = 0
        self.stalls = 0
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen/prompt text already repeats
        self.distinct_n_ratio = 0.0
        self.events = []

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "events"}
        d["n_events"] = len(self.events)
        return d


def find_repeated_ngrams(x_list: list, n: int, mask_id: int,
                         committed_only: bool = True):
    """Find repeated n-grams in a canvas.

    Windows are computed in position order; a window counts only when all
    ``n`` positions are committed (non-mask). Returns a dict
    {ngram_tuple: [start_positions]} restricted to n-grams occurring >= 2.
    """
    if n < 1 or n > len(x_list):
        return {}
    groups = {}
    for i in range(len(x_list) - n + 1):
        window = x_list[i:i + n]
        if committed_only and any(t == mask_id for t in window):
            continue
        key = tuple(window)
        groups.setdefault(key, []).append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _would_repeat(x_list: list, pos: int, tok: int, n: int, mask_id: int):
    """Would committing tok at pos complete a repeated n-gram?

    Only windows containing pos can become newly complete, so only those
    are checked (the pre-commit canvas is assumed repeat-free).
    """
    trial = list(x_list)
    trial[pos] = tok
    L = len(trial)
    lo = max(0, pos - n + 1)
    hi = min(pos, L - n)
    seen = {}
    for i in range(L - n + 1):
        w = trial[i:i + n]
        if any(t == mask_id for t in w):
            continue
        key = tuple(w)
        seen.setdefault(key, []).append(i)
    for i in range(lo, hi + 1):
        key = tuple(trial[i:i + n])
        if len(seen.get(key, ())) >= 2:
            return True
    return False


def _topk_for_pos(probs_row: torch.Tensor, k: int, banned: set):
    """Top-k (token, prob) pairs excluding banned ids. Deterministic."""
    row = probs_row.clone()
    if banned:
        row[list(banned)] = -1.0
    k = max(1, min(k, int((row > 0).sum().item())))
    vals, idx = torch.topk(row, k)
    return [(int(t), float(v)) for t, v in zip(idx.tolist(), vals.tolist())
            if float(v) > 0.0]


def _distinct_n_ratio(x_list: list, n: int, mask_id: int) -> float:
    grams = [tuple(x_list[i:i + n])
             for i in range(len(x_list) - n + 1)
             if all(t != mask_id for t in x_list[i:i + n])]
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def dcmr_sample(model, *, length: int, mask_id: int,
                config: "DCMRConfig | None" = None, seed: int = 0,
                policy: str = "dcmr", frozen: "dict | None" = None):
    """Run one unmasking trajectory with decoupled certificate-margin remasking.

    Args:
        model: callable x(1, L) -> logits(1, L, V).
        length: canvas length. mask_id: the mask token id.
        policy: "vanilla" (confidence unmasking, no gate, no remask --
            the F2/F3 baseline) or "dcmr" (no-repeat-n-gram certificate
            gate + decoupled stochastic remasking).
        frozen: optional {pos: token} pinned positions (treated as
            committed context; never remasked, but participate in repeat
            detection).

    Returns:
        (ids LongTensor(L), DCMRTrace).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {POLICIES}")
    cfg = config or DCMRConfig()
    if cfg.n < 1:
        raise ValueError("n-gram order must be >= 1")
    frozen = {int(p): int(t) for p, t in (frozen or {}).items()}

    torch.manual_seed(seed)
    x = torch.full((length,), mask_id, dtype=torch.long)
    conf = torch.zeros(length, dtype=torch.float)
    for p, t in frozen.items():
        x[int(p)] = int(t)
        conf[int(p)] = 1.0
    if find_repeated_ngrams(x.tolist(), cfg.n, mask_id):
        # The prompt itself repeats: the certificate cannot be claimed
        # for the whole canvas. Record it; the gate still protects
        # sampler-committed positions.
        preexisting = True
    else:
        preexisting = False
    masked = {p for p in range(length) if int(x[p]) == mask_id}
    taboo: dict[int, dict[int, int]] = {}   # pos -> {tok: cooldown}
    trace = DCMRTrace(policy)
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

    def commit_phase(probs, tag: str) -> int:
        """Commit up to cfg.budget gated tokens. Returns commit count."""
        made = 0
        order = []
        for pos in sorted(masked):
            tb = taboo.get(pos) or {}
            banned = {mask_id} | {t for t, cd in tb.items() if cd > 0}
            topk = _topk_for_pos(probs[pos], cfg.topk, banned)
            if not topk:
                continue
            order.append((topk[0][1], pos, topk))
        order.sort(reverse=True)
        x_list = x.tolist()
        for _cf, pos, topk in order:
            if pos not in masked or made >= cfg.budget:
                continue
            if policy == "vanilla":
                t1, p1 = topk[0]
                x[pos] = t1
                conf[pos] = p1
                masked.discard(pos)
                made += 1
                trace.commits += 1
                continue
            # dcmr: scan top-k for the first token that does not
            # complete a repeated n-gram (the certificate gate).
            placed = False
            for tok, pr in topk:
                if _would_repeat(x_list, pos, tok, cfg.n, mask_id):
                    trace.gate_rejects += 1
                    continue
                x[pos] = tok
                conf[pos] = pr
                x_list[pos] = tok
                masked.discard(pos)
                made += 1
                trace.commits += 1
                placed = True
                trace.events.append((f"{tag}_commit",
                                     {"pos": pos, "tok": tok}))
                break
            if not placed:
                trace.events.append((f"{tag}_defer", {"pos": pos}))
        return made

    def suspect_scan():
        """Positions whose committed state breaches (or risks) the certificate.

        Returns {pos: suspicion}: repeat participants (certificate breach)
        plus committed tokens under conf_floor (NAVIRA-style unreliability
        signal). Frozen positions are never suspects.
        """
        x_list = x.tolist()
        repeated = find_repeated_ngrams(x_list, cfg.n, mask_id)
        susp: dict[int, float] = {}
        for starts in repeated.values():
            for s in starts:
                for p in range(s, s + cfg.n):
                    if p in frozen:
                        continue
                    susp[p] = susp.get(p, 0.0) + 1.0
        for p in range(length):
            if p in masked or p in frozen or p in susp:
                continue
            if float(conf[p]) < cfg.conf_floor:
                susp[p] = susp.get(p, 0.0) + 0.5
        return susp

    step = 0
    while masked and step < cfg.max_steps:
        step += 1
        trace.steps = step
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
        trace.forward_passes += 1
        probs = torch.softmax(logits.float(), dim=-1)

        made = commit_phase(probs, "main")

        remade = 0
        if policy == "dcmr" and masked:
            # ---- decoupled regeneration (NAVIRA, certificate-lifted) ----
            susp = suspect_scan()
            if susp:
                # Stochastic selection: sample up to remask_budget suspects
                # from a temperature-controlled distribution over suspicion
                # (diversity preservation, no repeated hammering).
                order = sorted(susp)
                scores = torch.tensor([susp[p] for p in order],
                                      dtype=torch.float)
                temp = max(cfg.temperature, 1e-6)
                weights = torch.softmax(scores / temp, dim=0)
                k = min(cfg.remask_budget, len(order))
                # deterministic under the run seed
                g = torch.Generator().manual_seed(seed + step)
                chosen_idx = torch.multinomial(weights, k,
                                               replacement=False,
                                               generator=g)
                chosen = [order[int(i)] for i in chosen_idx.tolist()]
                for p in chosen:
                    old = int(x[p])
                    # taboo the offending token so the regen pass cannot
                    # re-commit it and churn (loop-breaker).
                    taboo.setdefault(p, {})[old] = cfg.taboo_cooldown
                    x[p] = mask_id
                    conf[p] = 0.0
                    masked.add(p)
                    trace.remasks += 1
                    trace.events.append(("remask",
                                         {"pos": p, "tok": old,
                                          "suspicion":
                                          round(float(susp[p]), 3)}))
                # Second forward pass on the CLEANED canvas: remasked
                # tokens cannot contaminate their own regeneration.
                with torch.no_grad():
                    logits2 = model(x.unsqueeze(0))[0]
                trace.forward_passes += 1
                trace.regen_passes += 1
                probs2 = torch.softmax(logits2.float(), dim=-1)
                remade = commit_phase(probs2, "regen")

        if made + remade > 0:
            stall = 0
        else:
            stall += 1
            trace.stalls += 1
            if stall >= cfg.stall_budget:
                trace.events.append(("stall_exit", {"step": step}))
                break
        decay_taboo()

    trace.completed = not masked
    x_list = x.tolist()
    trace.distinct_n_ratio = _distinct_n_ratio(x_list, cfg.n, mask_id)
    if trace.completed:
        remaining = find_repeated_ngrams(x_list, cfg.n, mask_id)
        trace.final_no_repeat = not remaining
        trace.events.append(("final_check",
                             {"repeated": len(remaining)}))
        # Sound fail direction: a completed canvas that violates the
        # certificate must never be presented as certified. (Vanilla is
        # the deliberately uncertified baseline, so it is exempt.)
        assert policy == "vanilla" or trace.final_no_repeat \
            or trace.preexisting_repeat, \
            "completed canvas violates the no-repeat certificate"
    return x, trace
