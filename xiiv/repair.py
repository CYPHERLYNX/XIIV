"""Verifier-Guided Repair (VGR): a repair operator with teeth.

Built 2026-10-22 after three independent toy-scale failures shared one
diagnosis (Invention #1 gate 2, CSM Phase C, Guarantee-MMD pilot):
allocation/routing signals cannot pay when the repair operator is too
weak — free re-denoising of remasked positions simply regenerates the
same violations, because nothing structural prevents them.

VGR inverts the old refinement in two ways:

1. TARGETING BY DIAGNOSIS (default): the independent verifier
   (check_constraints + position-level analysis in `diagnose`) names
   the exact violating positions — forbidden tokens, cap excesses,
   forbidden-phrase spans, and slots for missing required tokens.
   No signal has to *find* the violations; they are computed.
2. ENFORCED REFILL: the refill runs through DenoisingState WITH the
   full constraints attached, so the sampler's structural machinery
   (forbidden suppression, at_most caps, phrase-completion prevention,
   slot reservation, endgame force-placement) makes the repaired
   positions incapable of recreating the diagnosed violation class.

Signal-targeted variants ('confidence', 'load') share the identical
enforced-refill mechanics and budget; only the choice of WHICH
positions to repair differs. That is what makes the fair retest in
research/repair_retest_probe.py a clean comparison: one capable
operator, three targeting policies.
"""
from dataclasses import dataclass, field

import torch

from .diffusion import (DenoisingState, _count_occurrences, _phrase_spans,
                        check_constraints)


def diagnose(ids, constraints, mask_id, frozen=None):
    """Position-level violation diagnosis on a finished (or partial)
    canvas. Returns a dict:
      forbidden_positions: filled, non-frozen positions holding a
        forbidden token;
      cap_excess_positions: {token: [positions]} occurrences beyond
        the at_most cap (all occurrences listed; caller picks which
        to sacrifice, e.g. lowest-confidence);
      phrase_excess_spans: [(start, phrase)] fully-filled forbidden
        phrase occurrences beyond their allowance;
      missing_required: {token: still-needed count};
      check: the raw check_constraints dict.
    """
    c = constraints or {}
    ids_list = ids.cpu().tolist() if torch.is_tensor(ids) else list(ids)
    if frozen is None:
        frozen_list = [False] * len(ids_list)
    elif torch.is_tensor(frozen):
        frozen_list = frozen.cpu().tolist()
    else:
        frozen_list = list(frozen)

    forbidden_positions = [
        p for p, t in enumerate(ids_list)
        if t in set(c.get("forbidden_ids") or []) and not frozen_list[p]
        and t != mask_id]

    cap_excess = {}
    for t, cap in (c.get("at_most") or {}).items():
        positions = [p for p, v in enumerate(ids_list) if v == int(t)]
        if len(positions) > int(cap):
            cap_excess[int(t)] = positions

    phrase_excess = []
    for phrase, max_allowed in (c.get("forbidden_phrases") or []):
        phrase = list(phrase)
        spans = _phrase_spans(ids_list, phrase)
        for s in spans[int(max_allowed):]:
            phrase_excess.append((s, phrase))

    required_counts = dict(c.get("required_counts") or {})
    for t in (c.get("required_ids") or []):
        required_counts.setdefault(int(t), 1)
    missing = {}
    for t, n in required_counts.items():
        have = ids_list.count(int(t))
        if have < int(n):
            missing[int(t)] = int(n) - have

    return {
        "forbidden_positions": forbidden_positions,
        "cap_excess_positions": cap_excess,
        "phrase_excess_spans": phrase_excess,
        "missing_required": missing,
        "check": check_constraints(ids, constraints, mask_id),
    }


@dataclass
class RepairReport:
    success: bool = False
    rounds_used: int = 0
    positions_touched: set = field(default_factory=set)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)


@torch.no_grad()
def _enforced_refill(model, ids, repair_positions, constraints, mask_id,
                     eos_id, steps, seed):
    """Re-denoise ONLY `repair_positions` with full structural
    enforcement; every other position is frozen at its draft value."""
    length = ids.shape[0]
    frozen = torch.ones(length, dtype=torch.bool)
    frozen[list(repair_positions)] = False
    noisy = ids.clone()
    noisy[list(repair_positions)] = mask_id
    st = DenoisingState(
        length=length, mask_id=mask_id, eos_id=eos_id, steps=steps,
        temperature=1.0, top_p=0.95, seed=seed,
        prompt_ids=noisy, frozen=frozen, constraints=constraints,
        device="cpu")
    while st.step(model):
        pass
    return st.x, st.conf


@torch.no_grad()
def verifier_guided_repair(model, ids, constraints, mask_id, eos_id,
                           frozen=None, *, rounds=2, budget_frac=0.15,
                           steps=8, seed=0, targeting="verifier",
                           conf=None, load=None):
    """Repair a draft so it satisfies `constraints`, or report failure.

    targeting:
      "verifier"   — repair exactly the diagnosed violating positions
                     (+ lowest-confidence open slots for missing
                     required tokens). The VGR default.
      "confidence" — repair the `budget` lowest-confidence open
                     positions per round (verifier not consulted for
                     targeting; enforcement still applies).
      "load"       — repair the `budget` highest guarantee-load open
                     positions per round (same conditions).

    Budget: per round at most k = ceil(budget_frac * #open) positions
    for signal targeting; verifier targeting is capped at k as well
    (worst violations first: forbidden, phrase spans, cap excess,
    required slots). Rounds repeat diagnosis after each refill.
    Frozen positions are never touched; a violation that lives only on
    frozen positions is reported as unrepairable (success=False).
    Deterministic given `seed`.
    """
    length = ids.shape[0]
    if frozen is None:
        frozen = torch.zeros(length, dtype=torch.bool)
    open_pos = [p for p in range(length) if not bool(frozen[p])]
    k = max(1, int(budget_frac * len(open_pos) + 0.999))
    cur = ids.clone()
    report = RepairReport(before=diagnose(cur, constraints, mask_id,
                                          frozen)["check"])
    cur_conf = conf.clone() if conf is not None else None

    for r in range(rounds):
        diag = diagnose(cur, constraints, mask_id, frozen)
        if all(diag["check"].values()):
            report.success = True
            break

        if targeting == "verifier":
            positions = list(diag["forbidden_positions"])
            for s, phrase in diag["phrase_excess_spans"]:
                positions += [p for p in range(s, s + len(phrase))
                              if not bool(frozen[p])]
            for t, positions_t in diag["cap_excess_positions"].items():
                excess = len(positions_t) - int(
                    (constraints.get("at_most") or {})[t])
                movable = [p for p in positions_t if not bool(frozen[p])]
                if cur_conf is not None:
                    movable.sort(key=lambda p: float(cur_conf[p]))
                positions += movable[:excess]
            need_slots = sum(diag["missing_required"].values())
            if need_slots > 0:
                have = {p for p in positions}
                rest = [p for p in open_pos if p not in have]
                if cur_conf is not None:
                    rest.sort(key=lambda p: float(cur_conf[p]))
                positions += rest[:need_slots]
            positions = sorted(set(positions))[:k + need_slots]
        elif targeting in ("confidence", "load"):
            if targeting == "confidence":
                if cur_conf is None:
                    raise ValueError("confidence targeting needs conf")
                ranked = sorted(open_pos, key=lambda p: float(cur_conf[p]))
            else:
                if load is None:
                    raise ValueError("load targeting needs load")
                ranked = sorted(open_pos, key=lambda p: -float(load[p]))
            positions = sorted(ranked[:k])
        else:
            raise ValueError(f"unknown targeting {targeting!r}")

        positions = [p for p in positions if not bool(frozen[p])]
        if not positions:
            break
        cur, new_conf = _enforced_refill(
            model, cur, positions, constraints, mask_id, eos_id, steps,
            seed + 7919 * (r + 1))
        if cur_conf is not None:
            cur_conf = new_conf  # signals re-rank on refreshed conf,
            # exactly as gate 2 / CSM Phase C refreshed theirs per round
        report.positions_touched.update(positions)
        report.rounds_used = r + 1

    final = diagnose(cur, constraints, mask_id, frozen)
    report.after = final["check"]
    report.success = all(final["check"].values())
    return cur, report
