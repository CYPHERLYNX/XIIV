"""CCCA -- Cross-Chain Consensus Arbitration for masked-diffusion decoding.

Every sampler in this family (DCMR, CGEC, CDR, PTCS, SCV, CBRS) decides
commits from a SINGLE forward pass: top-1 confidence, a margin, or a
planned budget. The verifier arbitrates *whether* a commit is
certificate-safe, but nothing arbitrates whether the commit is
*reliable* -- and the literature now says confidence alone does not
determine that:

- RPD (He et al., arXiv:2609.36452): parallel commitments chosen by
  confidence are unreliable; confidence alone does not determine
  reliability. RPD defers predictions using a layerwise signal.
- OSCAR (arXiv:2604.01624): N parallel denoising chains with randomized
  reveal orders; cross-chain Shannon entropy localizes unreliable
  commitments; targeted remasking follows.

CCCA takes the *diagnosis* (commit reliability needs a second signal
beyond single-pass confidence; cross-chain disagreement localizes
unreliable commits) and builds a different mechanism around it:

K parallel chains denoise the same canvas with DIFFERENT deterministic
reveal orders (confidence / left-to-right / right-to-left / seeded
shuffle). Each chain commits into its own working canvas, verifier-gated
against that canvas. An arbitrator then decides what reaches the
SHARED canvas:

- CONSENSUS: a position committed by >= ``quorum`` chains with the SAME
  token, and verifier-safe against the shared canvas, commits to the
  shared canvas. The commit carries two independent keys: cross-chain
  agreement AND the certificate gate.
- DISAGREEMENT: chains proposing DIFFERENT tokens for one position are
  a detected unreliability -- the position is NOT committed while
  disputed. Every chain rolls it back to masked (disagreement rollback)
  and resyncs; the position is re-proposed in a later round, when the
  context is denser. A disputed token never reaches the shared canvas.
- STANDING PROPOSAL: a position proposed by only one chain waits. If a
  second chain confirms it (same token), it commits by consensus. If it
  stands unconfirmed for ``confirm_rounds`` rounds, it is PROMOTED to
  the shared canvas as an explicitly NON-consensus commit -- verifier-
  gated, counted, and itemized in the certificate (the maximal
  certifiable residue, in Koomullil's arXiv:2605.16407 sense: what could
  not be consensus-certified is named, not hidden). This is the liveness
  fallback: the sampler can finish even where chains never agree.

The certificate (ccca-v1) claims: the final canvas is verifier-clean,
every consensus commit was confirmed by >= ``quorum`` independent reveal
orders, and the non-consensus residue is listed position by position.

Prior art (absorbed, not copied):
- RPD (arXiv:2609.36452): confidence alone is unreliable for parallel
  commits. CCCA takes the diagnosis but not the mechanism: RPD's second
  signal is layerwise agreement, unavailable through the
  ``model(x) -> logits`` interface; CCCA's is cross-chain agreement,
  which needs only that interface. RPD *defers* unreliable predictions;
  CCCA defers AND rolls back AND certifies the consensus that remains.
- OSCAR (arXiv:2604.01624): cross-chain entropy localizes uncertainty
  for remasking with retrieved evidence. CCCA takes the *cross-chain
  disagreement* shape but inverts the response: disagreement is a commit
  gate with rollback, not a remask trigger; there is no retrieved
  evidence and no external classifier -- the second key is the
  independent certificate verifier, not a hallucination score.
- LAVE (arXiv:2602.00612): a commit is safe only if the remaining canvas
  stays completable (lookahead-then-verify). CCCA takes the
  completability diagnosis: when a run ends incomplete, one diagnostic
  forward pass on the shared canvas counts *dead* positions (no
  certificate-safe candidate in the top-``draft_depth``), so an
  incomplete run reports WHY it could not complete instead of just
  stalling silently. LAVE checks CFG completability; CCCA checks
  certificate completability.
- Koomullil (arXiv:2605.16407): Maximal Certifiable Residue -- when
  everything cannot be certified, certify the maximal residue and name
  the rest. CCCA's non-consensus promotions are exactly that residue.
- In-repo: dcmr._would_repeat / _topk_for_pos / find_repeated_ngrams
  (the certificate), scv.verify_canvas (independent re-check), the
  CBRS/SCV fail-closed pattern.

Pre-registered falsifiers (see xiiv/tests/test_ccca.py):
- F1 consensus convergence: on the clean stub (order-independent
  model), completes with consensus_rate == 1.0, n_disagreements == 0,
  final_no_repeat.
- F2 disagreement deferral + rollback: on the fork stub (reveal order
  changes the proposed token), the disputed position is never committed
  while disputed; the trace shows n_disagreements >= 1 and
  n_rollbacks >= 1, and the shared-canvas commit of that position lands
  strictly after the disagreement round, with the post-rollback token.
- F3 convergence after rollback: the same fork stub completes with
  consensus_rate == 1.0 and n_nonconsensus_commits == 0 -- denser
  context resolves the disagreement; rollback helped, it did not stall.
- F4 final soundness: completed=True implies the independent
  verify_canvas(final) is clean (recomputed, not trusted).
- F5 fail-closed: the repeater stub never completes; masks remain; the
  certificate claim is False; the diagnostic pass reports
  dead_positions_final >= 1.
- F6 determinism: identical runs -> identical shared canvas, commit
  order, and disagreement events.
- F7 promotion honesty: with K=1 (no second chain can ever confirm),
  every commit goes through promotion after the confirm window;
  completed=True with consensus_rate == 0.0, n_nonconsensus_commits ==
  n_committed, and the certificate itemizes every promoted position.
- KILL: F2, F4, or F5 fails -> the module does not ship.

Cost: K forward passes per round (one per chain), counted honestly in
the trace. The consensus claim costs Kx passes; the certificate reports
the count so the trade is auditable.

The sampler drives any model with the interface
    model(x: LongTensor (1, L)) -> logits FloatTensor (1, L, V).
No GPU needed. Deterministic: the only randomness is the seeded shuffle
reveal order (seed kept for CLI parity otherwise).

Measured on CPU scripted stubs only -- real-checkpoint behavior
untested.
"""

import random

import torch

from .dcmr import _would_repeat, _topk_for_pos, find_repeated_ngrams
from .scv import verify_canvas

__all__ = [
    "CCCAConfig",
    "CCCATrace",
    "POLICIES",
    "ORDER_POLICIES",
    "ccca_sample",
    "ccca_certificate",
]

POLICIES = ("vanilla", "ccca")
ORDER_POLICIES = ("confidence", "ltr", "rtl", "shuffle")


class CCCAConfig:
    """Sampler hyperparameters."""

    def __init__(
        self,
        n: int = 4,
        frontier: int = 8,
        draft_depth: int = 3,
        chains: int = 3,
        quorum: int = 2,
        confirm_rounds: int = 3,
        max_rounds: int = 64,
        stall_budget: int = 3,
        seed: int = 7,
    ):
        self.n = n                      # n-gram order of the certificate
        self.frontier = frontier        # positions drafted per chain/round
        self.draft_depth = draft_depth  # candidates per position
        self.chains = chains            # parallel chains (reveal orders)
        self.quorum = quorum            # chains that must agree to commit
        self.confirm_rounds = confirm_rounds  # rounds a lone proposal waits
        self.max_rounds = max_rounds    # hard round cap
        self.stall_budget = stall_budget  # zero-commit rounds before give-up
        self.seed = seed                # shuffle reveal order seed


class CCCATrace:
    """Per-run measurement record (the falsifier inputs)."""

    def __init__(self, policy: str):
        self.policy = policy
        self.rounds = 0             # arbitration rounds executed
        self.forward_passes = 0     # model forward passes (K per round)
        self.commits = 0            # positions committed to shared canvas
        self.consensus_commits = 0  # ... by >= quorum agreement
        self.nonconsensus_commits = 0  # ... by promotion after the window
        self.disagreements = 0      # disputed positions detected
        self.rollbacks = 0          # positions rolled back to masked
        self.promotions = 0         # == nonconsensus_commits
        self.gate_rejects = 0       # candidates refused by the cert gate
        self.stalls = 0             # zero-commit rounds
        self.completed = False
        self.final_no_repeat = False
        self.preexisting_repeat = False  # frozen text already repeats
        self.consensus_rate = 0.0   # consensus / total commits (0 if none)
        self.dead_positions_final = 0  # masked positions with no safe
                                        # candidate (diagnostic pass, only
                                        # when the run ends incomplete)
        self.nonconsensus_positions = []  # promoted positions, sorted
        self.commit_order = []      # (round, pos, tok, consensus?) in order
        self.disagreement_order = []  # (round, pos, sorted tokens) disputed
        self.events = []

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("events", "commit_order", "disagreement_order",
                          "nonconsensus_positions")}
        d["n_events"] = len(self.events)
        d["n_commit_order"] = len(self.commit_order)
        d["n_disagreements_recorded"] = len(self.disagreement_order)
        d["n_nonconsensus_positions"] = len(self.nonconsensus_positions)
        return d


def ccca_certificate(trace: CCCATrace) -> dict:
    """Build the ccca-v1 certificate from a run trace.

    Fail-closed: ``completed`` is True only if the trace completed; a
    non-completed trace never yields a certification claim. The
    non-consensus residue is itemized, not hidden.
    """
    cert = {
        "cert": "ccca-v1",
        "policy": trace.policy,
        "completed": bool(trace.completed),
        "final_no_repeat": bool(trace.final_no_repeat),
        "preexisting_repeat": bool(trace.preexisting_repeat),
        "consensus_rate": trace.consensus_rate,
        "n_consensus_commits": trace.consensus_commits,
        "n_nonconsensus_commits": trace.nonconsensus_commits,
        "nonconsensus_positions": sorted(trace.nonconsensus_positions),
        "n_disagreements": trace.disagreements,
        "n_rollbacks": trace.rollbacks,
        "forward_passes": trace.forward_passes,
        "rounds": trace.rounds,
        "dead_positions_final": trace.dead_positions_final,
    }
    if not trace.completed:
        cert["completed"] = False
    return cert


def _top1_prob(probs_row: torch.Tensor) -> float:
    return float(torch.max(probs_row).item())


def _order_positions(masked: list, probs, order: str, seed: int,
                     chain: int) -> list:
    """Deterministic reveal order for one chain's frontier selection."""
    if order == "ltr":
        return sorted(masked)
    if order == "rtl":
        return sorted(masked, reverse=True)
    if order == "shuffle":
        rng = random.Random(seed * 1000 + chain)
        out = list(masked)
        rng.shuffle(out)
        return out
    # confidence: top-1 probability desc, position asc on ties.
    return sorted(masked, key=lambda p: (-_top1_prob(probs[p]), p))


def _chain_draft(model, w_list: list, length: int, mask_id: int, cfg,
                 order: str, chain: int, banned: set, trace) -> dict:
    """One chain's round: forward pass + verifier-gated frontier commits.

    Returns {pos: tok} committed into this chain's working canvas.
    """
    masked = [i for i in range(length) if w_list[i] == mask_id]
    if not masked:
        return {}
    x = torch.tensor([w_list], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    trace.forward_passes += 1
    probs = torch.softmax(logits[0].float(), dim=-1)
    ordered = _order_positions(masked, probs, order, cfg.seed, chain)
    committed = {}
    working = list(w_list)
    for p in ordered[:max(1, cfg.frontier)]:
        cands = _topk_for_pos(probs[p], cfg.draft_depth, banned)
        chosen = None
        for rank, (tok, _pr) in enumerate(cands):
            if _would_repeat(working, p, tok, cfg.n, mask_id):
                trace.gate_rejects += 1
                trace.events.append(
                    ("chain_gate_reject",
                     {"chain": chain, "pos": p, "tok": tok, "rank": rank}))
                continue
            chosen = tok
            break
        if chosen is None:
            continue
        working[p] = chosen
        committed[p] = chosen
    for p, tok in committed.items():
        w_list[p] = tok
    return committed


def ccca_sample(model, length: int, mask_id: int, config=None,
                seed: int = 7, frozen=None, policy: str = "ccca"):
    """Run CCCA (or the gated top-1 vanilla baseline) over a blank canvas.

    Returns (ids LongTensor (1, L), trace, cert).
    """
    assert policy in POLICIES, f"unknown policy {policy!r}"
    cfg = config or CCCAConfig()
    if policy == "vanilla":
        # Baseline: one chain, no consensus machinery -- the first
        # proposal commits immediately (unreachable quorum, zero confirm
        # window, so every commit is a promotion). No consensus claim
        # is made (consensus_rate stays 0.0).
        n_chains, quorum, confirm_rounds = 1, 10 ** 9, 0
    else:
        n_chains = max(1, cfg.chains)
        # NOTE: quorum is NOT clamped to n_chains. quorum > n_chains
        # means consensus is unreachable and every commit goes through
        # promotion -- the configuration says so, and the certificate
        # reports it (consensus_rate 0.0). See F7.
        quorum = max(1, cfg.quorum)
        confirm_rounds = cfg.confirm_rounds
    orders = [ORDER_POLICIES[k % len(ORDER_POLICIES)]
              for k in range(n_chains)]

    frozen = frozen or {}
    frozen_keys = {int(p) for p in frozen}
    shared = [mask_id] * length
    for p, t in frozen.items():
        shared[int(p)] = int(t)
    # Each chain's working canvas starts at the shared canvas and
    # diverges through its own standing proposals.
    chains_w = [list(shared) for _ in range(n_chains)]

    trace = CCCATrace(policy)
    trace.preexisting_repeat = bool(
        find_repeated_ngrams(shared, cfg.n, mask_id))
    banned = {mask_id}
    proposal_since = {}  # pos -> round index the lone proposal started

    stall = 0
    for rnd in range(cfg.max_rounds):
        masked_shared = [i for i in range(length) if shared[i] == mask_id]
        if not masked_shared:
            break
        trace.rounds += 1
        # ---- Phase 1: every chain drafts from its own canvas ---------
        for k in range(n_chains):
            _chain_draft(model, chains_w[k], length, mask_id, cfg,
                         orders[k], k, banned, trace)

        # ---- Phase 2: arbitration over standing proposals -----------
        # A proposal is a position committed in some chain's canvas but
        # still masked in the shared canvas.
        proposals = {}  # pos -> {chain: tok}
        for k in range(n_chains):
            wk = chains_w[k]
            for p in masked_shared:
                if wk[p] != mask_id:
                    proposals.setdefault(p, {})[k] = wk[p]
        committed_this_round = 0
        for p in sorted(proposals):
            toks = proposals[p]
            distinct = sorted(set(toks.values()))
            if len(distinct) > 1:
                # DISAGREEMENT: disputed token never reaches the shared
                # canvas. Roll every chain back to masked and resync.
                trace.disagreements += 1
                trace.rollbacks += 1
                trace.disagreement_order.append((rnd, p, distinct))
                trace.events.append(
                    ("disagreement",
                     {"round": rnd, "pos": p, "tokens": distinct}))
                for k in range(n_chains):
                    chains_w[k][p] = mask_id
                proposal_since.pop(p, None)
                continue
            tok = distinct[0]
            if len(toks) >= quorum:
                # CONSENSUS: >= quorum chains agree. Re-check against
                # the shared canvas (defensive: safe-against-chain
                # implies safe-against-shared, but never trust it).
                if _would_repeat(shared, p, tok, cfg.n, mask_id):
                    trace.events.append(
                        ("consensus_gate_reject",
                         {"round": rnd, "pos": p, "tok": tok}))
                    for k in range(n_chains):
                        chains_w[k][p] = mask_id
                    proposal_since.pop(p, None)
                    continue
                shared[p] = tok
                trace.consensus_commits += 1
                trace.commits += 1
                committed_this_round += 1
                trace.commit_order.append((rnd, p, tok, True))
                proposal_since.pop(p, None)
            else:
                # STANDING PROPOSAL: one chain only. Wait for a second
                # chain to confirm, or promote after the confirm window.
                since = proposal_since.setdefault(p, rnd)
                if rnd - since >= confirm_rounds:
                    if _would_repeat(shared, p, tok, cfg.n, mask_id):
                        trace.events.append(
                            ("promotion_gate_reject",
                             {"round": rnd, "pos": p, "tok": tok}))
                        for k in range(n_chains):
                            chains_w[k][p] = mask_id
                        proposal_since.pop(p, None)
                        continue
                    shared[p] = tok
                    trace.nonconsensus_commits += 1
                    trace.promotions += 1
                    trace.commits += 1
                    committed_this_round += 1
                    trace.commit_order.append((rnd, p, tok, False))
                    trace.nonconsensus_positions.append(p)
                    trace.events.append(
                        ("promotion",
                         {"round": rnd, "pos": p, "tok": tok,
                          "waited_rounds": rnd - since}))
                    proposal_since.pop(p, None)
        # Sync every shared commit into every chain's canvas.
        for k in range(n_chains):
            wk = chains_w[k]
            for p in masked_shared:
                if shared[p] != mask_id:
                    wk[p] = shared[p]
        if committed_this_round:
            stall = 0
        else:
            stall += 1
            trace.stalls += 1
            trace.events.append(("stall", {"round": rnd}))
            if stall >= cfg.stall_budget:
                break

    remaining = [i for i in range(length) if shared[i] == mask_id]
    check = verify_canvas(shared, mask_id, cfg.n)
    trace.final_no_repeat = check["final_no_repeat"]
    # Fail-closed: complete only when fully unmasked and certificate-
    # clean, unless the breach pre-existed in frozen text (the sampler
    # cannot repair frozen prompt tokens; reported, not hidden).
    trace.completed = (not remaining and
                       (trace.final_no_repeat or trace.preexisting_repeat))
    if trace.commits:
        trace.consensus_rate = (trace.consensus_commits / trace.commits)
    trace.nonconsensus_positions = sorted(trace.nonconsensus_positions)

    if not trace.completed and remaining:
        # LAVE diagnosis: one diagnostic pass on the shared canvas --
        # count DEAD positions (no certificate-safe candidate in the
        # top-``draft_depth``), so an incomplete run reports WHY it
        # could not complete. Counted honestly as a forward pass.
        x = torch.tensor([shared], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        trace.forward_passes += 1
        probs = torch.softmax(logits[0].float(), dim=-1)
        dead = 0
        for p in remaining:
            cands = _topk_for_pos(probs[p], cfg.draft_depth, banned)
            if not any(not _would_repeat(shared, p, tok, cfg.n, mask_id)
                       for tok, _pr in cands):
                dead += 1
        trace.dead_positions_final = dead

    cert = ccca_certificate(trace)
    x = torch.tensor([shared], dtype=torch.long)
    return x, trace, cert
