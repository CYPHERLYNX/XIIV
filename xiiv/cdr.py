"""Certificate-Drift Remasking (CDR) for masked-diffusion decoding.

A sampler that asks, at every check cycle, not "is this token good now?"
but "does the *proof* still endorse the token I committed earlier?" --
and remasks the positions where the endorsement decayed between the
sparse commit-time context and the denser current context.

Two bundled signals over the same check-cycle machinery (k-partition
model probing for the confidence signal, full-canvas verifier
re-reads for the certificate signal):

- ``signal="confidence"``: the faithful CoDR port. Endorsement is the
  model's own probability of the committed token; drift is measured by
  re-masking each partition in turn and re-running the model (k forward
  passes per check cycle).
- ``signal="certificate"`` (default): the synthesis. Endorsement is the
  *independent verifier's* margin on the committed token -- for the
  no-repeat-n-gram certificate, 1.0 when no completed repeated window
  contains the position, 0.0 on breach. Each check cycle re-verifies the
  margin on the full current canvas (then-vs-now); the verifier reads
  the canvas directly, so certificate drift needs ZERO probe forward
  passes and no partition. (Partitioned probing is only needed for the
  model-based confidence signal, where the model must re-endorse each
  position without seeing its partition peers -- the CoDR design.)

The synthesis claim: confidence drift fails exactly when the model is
confidently wrong (the known failure of all confidence remasking). The
certificate margin decays anyway -- the proof rots even when the model
doesn't flinch -- so certificate-signal CDR repairs what confidence
remasking cannot see. Pre-registered falsifier F2 below can honestly
kill this claim: if certificate mode adds nothing over confidence mode
on the deceptive stub, only the CoDR port survives.

Prior art (absorbed, not copied):
- CoDR -- Confidence-Drift Remasking (Wu et al., arXiv:2610.08833, Sep
  2026): training-free, sampler-agnostic refinement for masked diffusion
  LMs. Errors come from *confidence drift*: a token committed under
  sparse context loses model endorsement once context densifies. Drift
  is estimated for all committed positions in k forward passes via
  k-partition probing; drifted tokens are remasked and regenerated.
  CDR ports this *mechanism* (partition -> probe under denser context ->
  remask -> regenerate on the cleaned canvas); signal="confidence" is
  the faithful port.
- DCMR (in-repo, xiiv/dcmr.py): the no-repeat-n-gram certificate used
  here, plus the taboo-the-offender + decoupled-regeneration machinery
  and the conf_floor "stillborn" idea, lifted from model confidence to
  certificate margin. DCMR checks the *current* canvas; CDR checks
  commit-time vs current (drift) -- the mechanisms compose, they don't
  overlap.
- "Remask, Don't Replace" / Token-to-Mask (arXiv:2604.18738): the
  detection/correction decoupling (suspect tokens are reset to [M],
  semantically inert, and re-predicted jointly under a cleaned context),
  the LOGITDIFF detector (exactly what confidence drift measures), and
  the stuck-set theory motivating verifier-side rather than
  model-side detection.
- TPP vs LCR (Jeon et al., arXiv:2609.34509, Oct 2026): commit order is
  position-first (pick position by max top-token probability, commit
  that position's own top token) -- never the LCR pattern of sampling a
  proposal at every position and keeping only the global winner. Locked
  in by falsifier F4.

Pre-registered falsifiers (see tests/test_cdr.py):
- F1 certificate soundness: final canvas has no repeated n-gram of
  order n (scripted repeater model), signal="certificate".
- F2 synthesis gate: on the deceptive stub (model stays confident on a
  repeat-token forever), signal="certificate" must repair via drift
  (drift_remasks > 0, completed, no repeats) while signal="confidence"
  must see no drift (drift_remasks == 0) and fail closed (not completed,
  repeats remain). KILL: if certificate mode does not beat confidence
  mode here, the synthesis claim dies.
- F3 pass budget (structural): signal="certificate" -> probe_passes == 0
  (verifier reads need no forward passes); signal="confidence" ->
  probe_passes <= (steps + 3) * k; always
  forward_passes == commit_passes + probe_passes + regen_passes.
- F4 TPP order: commit order is position-by-own-max-prob; every
  position keeps its own top token (no global-proposal collapse).
- KILL: F1 or F2 fails -> the module does not ship.

Fail-closed: a completed canvas that violates the no-repeat certificate
is never presented as certified (completed=False), unless the breach
was preexisting in frozen prompt tokens.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed.

Measured on CPU scripted stubs only -- real-checkpoint behavior
untested.
"""

import torch

from .dcmr import find_repeated_ngrams

__all__ = [
    "CDRConfig",
    "CDRTrace",
    "POLICIES",
    "SIGNALS",
    "cert_margin",
    "cdr_sample",
    "cdr_certificate",
]

POLICIES = ("vanilla", "cdr")
SIGNALS = ("confidence", "certificate")


class CDRConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        budget: int = 4,
        check_every: int = 1,
        probe_partitions: int = 2,
        tau: float = 0.3,
        conf_floor: float = 0.02,
        cert_floor: float = 0.5,
        taboo_cooldown: int = 4,
        max_steps: int = 64,
        stall_budget: int = 3,
    ):
        self.n = n                              # n-gram order of the certificate
        self.budget = budget                    # commits per phase
        self.check_every = check_every          # steps between drift checks
        self.probe_partitions = probe_partitions  # k in k-partition probing
        self.tau = tau                          # drift threshold for remask
        self.conf_floor = conf_floor            # stillborn floor (confidence)
        self.cert_floor = cert_floor            # stillborn floor (certificate)
        self.taboo_cooldown = taboo_cooldown    # steps a remasked token stays banned
        self.max_steps = max_steps              # hard termination cap
        self.stall_budget = stall_budget        # zero-progress steps before giving up


class CDRTrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str, signal: str):
        self.policy = policy
        self.signal = signal
        self.steps = 0
        self.commit_passes = 0
        self.forward_passes = 0
        self.probe_passes = 0
        self.commits = 0
        self.drift_remasks = 0
        self.stillborn_remasks = 0
        self.regen_passes = 0
        self.check_cycles = 0
        self.stalls = 0
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen/prompt text already repeats
        self.commit_order = []           # positions in commit order
        self.events = []

    @property
    def remasks(self) -> int:
        return self.drift_remasks + self.stillborn_remasks

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("events", "commit_order")}
        d["remasks"] = self.remasks
        d["n_events"] = len(self.events)
        d["n_commits_order"] = len(self.commit_order)
        return d


def cert_margin(pos: int, x_list: list, mask_id: int, n: int) -> float:
    """The verifier's endorsement of the committed token at ``pos``.

    1.0 = clean (no completed repeated n-gram window contains pos),
    0.0 = breach (some completed repeated window contains pos).
    Computed from the canvas alone -- no model involved.
    """
    repeated = find_repeated_ngrams(x_list, n, mask_id)
    for starts in repeated.values():
        for s in starts:
            if s <= pos < s + n:
                return 0.0
    return 1.0


def _banned_for(pos: int, taboo: dict, mask_id: int) -> set:
    tb = taboo.get(pos) or {}
    return {mask_id} | {t for t, cd in tb.items() if cd > 0}


def _top1_for_pos(probs_row: torch.Tensor, banned: set):
    """Top-1 (token, prob) excluding banned ids. Deterministic."""
    row = probs_row.clone()
    if banned:
        row[list(banned)] = float("-inf")
    top_prob, top_tok = torch.max(row, dim=0)
    if not torch.isfinite(top_prob):
        return None, None
    return int(top_tok.item()), float(top_prob.item())


def cdr_certificate(trace: "CDRTrace", ids: torch.Tensor, mask_id: int,
                    n: int, *, signal_needs_model: bool = False) -> dict:
    """Build the proof-carrying record for a CDR run.

    The no-repeat claim replays model-free from ``tokens``. Drift values
    under signal="confidence" replay only with the model (flagged
    honestly); under signal="certificate" they replay from the canvas
    states implied by commit_order + remask events.
    """
    return {
        "certificate": "cdr-v1",
        "policy": trace.policy,
        "signal": trace.signal,
        "n": n,
        "length": int(ids.numel()),
        "mask_id": int(mask_id),
        "tokens": [int(t) for t in ids.tolist()],
        "commit_order": list(trace.commit_order),
        "remasks": [e[1] for e in trace.events if e[0] == "remask"],
        "n_drift_remasks": trace.drift_remasks,
        "n_stillborn_remasks": trace.stillborn_remasks,
        "check_cycles": trace.check_cycles,
        "completed": trace.completed,
        "final_no_repeat": trace.final_no_repeat,
        "replay_needs_model": bool(signal_needs_model),
    }


def cdr_sample(model, *, length: int, mask_id: int,
               config: "CDRConfig | None" = None, seed: int = 0,
               policy: str = "cdr", signal: str = "certificate",
               frozen: "dict | None" = None):
    """Run one unmasking trajectory with certificate-drift remasking.

    Args:
        model: callable x(1, L) -> logits(1, L, V).
        length: canvas length. mask_id: the mask token id.
        policy: "vanilla" (confidence unmasking, no drift checks -- the
            baseline) or "cdr".
        signal: "confidence" (CoDR-faithful: model probability drift) or
            "certificate" (verifier-margin drift; the synthesis).
        frozen: optional {pos: token} pinned positions (treated as
            committed context; never probed or remasked).

    Returns:
        (ids LongTensor(L), CDRTrace, certificate dict).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {POLICIES}")
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; expected {SIGNALS}")
    cfg = config or CDRConfig()
    if cfg.n < 1:
        raise ValueError("n-gram order must be >= 1")
    if cfg.probe_partitions < 1:
        raise ValueError("probe_partitions must be >= 1")
    if cfg.check_every < 1:
        raise ValueError("check_every must be >= 1")
    frozen = {int(p): int(t) for p, t in (frozen or {}).items()}

    torch.manual_seed(seed)
    x = torch.full((length,), mask_id, dtype=torch.long)
    endorse: dict[int, float] = {}
    for p, t in frozen.items():
        x[int(p)] = int(t)
        endorse[int(p)] = 1.0
    preexisting = bool(find_repeated_ngrams(x.tolist(), cfg.n, mask_id))
    masked = {p for p in range(length) if int(x[p]) == mask_id}
    taboo: dict[int, dict[int, int]] = {}   # pos -> {tok: cooldown}
    trace = CDRTrace(policy, signal)
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

    def record_endorse(pos: int, probs_row: torch.Tensor, x_list: list):
        if signal == "confidence":
            endorse[pos] = float(probs_row[int(x[pos])].item())
        else:
            endorse[pos] = cert_margin(pos, x_list, mask_id, cfg.n)

    def commit_phase(probs, tag: str) -> int:
        """TPP commit: positions ordered by their OWN top-token
        probability (desc); each committed position keeps its own top
        token -- never a global proposal pick (anti-LCR)."""
        order = []
        for pos in sorted(masked):
            tok, pr = _top1_for_pos(probs[pos], _banned_for(pos, taboo, mask_id))
            if tok is None:
                continue
            order.append((pr, pos, tok))
        order.sort(reverse=True)
        made = 0
        x_list = x.tolist()
        for _pr, pos, tok in order:
            if pos not in masked or made >= cfg.budget:
                continue
            x[pos] = tok
            x_list[pos] = tok
            masked.discard(pos)
            made += 1
            trace.commits += 1
            trace.commit_order.append(pos)
            record_endorse(pos, probs[pos], x_list)
            trace.events.append(
                (f"{tag}_commit",
                 {"pos": pos, "tok": tok,
                  "endorse": round(endorse[pos], 4)}))
        return made

    def probe_drift():
        """k-partition probing. Pure: measures, never mutates.

        Returns [(pos, drift, rule)] with rule in {"drift", "stillborn"}:
        - drift: endorsement decayed commit-time -> now beyond tau
          (the CoDR lift).
        - stillborn: the proof/model never endorsed this token even at
          commit time (endorse_commit below floor).
        """
        committed = [p for p in range(length)
                     if int(x[p]) != mask_id and p not in frozen]
        if not committed:
            return []
        trace.check_cycles += 1
        suspects = []
        floor = cfg.conf_floor if signal == "confidence" else cfg.cert_floor
        if signal == "confidence":
            k = min(cfg.probe_partitions, len(committed))
            groups = [committed[i::k] for i in range(k)]  # round-robin
            for g in groups:
                xp = x.clone()
                xp[g] = mask_id
                with torch.no_grad():
                    logits = model(xp.unsqueeze(0))[0]
                trace.forward_passes += 1
                trace.probe_passes += 1
                pr = torch.softmax(logits.float(), dim=-1)
                for p in g:
                    now = float(pr[p, int(x[p])].item())
                    drift = endorse[p] - now
                    if drift > cfg.tau:
                        suspects.append((p, drift, "drift"))
                    elif endorse[p] < floor:
                        suspects.append((p, drift, "stillborn"))
        else:
            # Certificate signal: re-verify the proof margin on the full
            # current canvas (then-vs-now). A repeat that formed after
            # p's commit -- invisible under p's sparse commit-time
            # context -- is exactly the rot this signal detects. The
            # verifier reads the canvas directly: no partition, no model
            # forward passes (the synthesis dividend).
            x_list = x.tolist()
            for p in committed:
                now = cert_margin(p, x_list, mask_id, cfg.n)
                drift = endorse[p] - now
                if drift > cfg.tau:
                    suspects.append((p, drift, "drift"))
                elif endorse[p] < floor:
                    suspects.append((p, drift, "stillborn"))
        return suspects

    def repair(suspects) -> int:
        """Remask suspects (taboo the offender) and regenerate on the
        cleaned canvas -- the decoupled correction half."""
        if not suspects:
            return 0
        for p, drift, rule in suspects:
            old = int(x[p])
            # Taboo the offending token so the regen pass cannot
            # re-commit it and churn (loop-breaker).
            taboo.setdefault(p, {})[old] = cfg.taboo_cooldown
            x[p] = mask_id
            masked.add(p)
            endorse.pop(p, None)
            if rule == "drift":
                trace.drift_remasks += 1
            else:
                trace.stillborn_remasks += 1
            trace.events.append(
                ("remask", {"pos": p, "tok": old, "rule": rule,
                            "drift": round(drift, 4)}))
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
        trace.forward_passes += 1
        trace.regen_passes += 1
        commit_phase(torch.softmax(logits.float(), dim=-1), "regen")
        return len(suspects)

    step = 0
    while masked and step < cfg.max_steps:
        step += 1
        trace.steps = step
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
        trace.forward_passes += 1
        trace.commit_passes += 1
        probs = torch.softmax(logits.float(), dim=-1)

        made = commit_phase(probs, "main")

        remade = 0
        if policy == "cdr" and step % cfg.check_every == 0:
            remade = repair(probe_drift())

        if made + remade > 0:
            stall = 0
        else:
            stall += 1
            trace.stalls += 1
            if stall >= cfg.stall_budget:
                trace.events.append(("stall_exit", {"step": step}))
                break
        decay_taboo()

    # Final verification: the proof is re-verified on the completed
    # canvas (bounded rounds; a repair that cannot re-complete fails
    # closed below).
    if policy == "cdr":
        for _ in range(3):
            if not repair(probe_drift()):
                break

    trace.completed = not masked
    x_list = x.tolist()
    remaining = find_repeated_ngrams(x_list, cfg.n, mask_id)
    trace.final_no_repeat = not remaining
    if trace.completed:
        trace.events.append(("final_check", {"repeated": len(remaining)}))
    # Sound fail direction: a completed canvas that violates the
    # certificate is never presented as certified -- it fails closed
    # (completed=False) instead of crashing or emitting. (Vanilla is
    # the deliberately uncertified baseline, so it is exempt, as is a
    # breach preexisting in frozen prompt tokens.)
    if (trace.completed and not trace.final_no_repeat
            and not trace.preexisting_repeat and policy != "vanilla"):
        trace.completed = False
        trace.events.append(("fail_closed", {"repeated": len(remaining)}))
    cert = cdr_certificate(trace, x, mask_id, cfg.n,
                           signal_needs_model=(signal == "confidence"))
    return x, trace, cert
