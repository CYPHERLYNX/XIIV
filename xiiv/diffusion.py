"""Masked-diffusion objective and sampler for XIIV.

Forward (training) process: sample a mask ratio t ~ Uniform(t_min, t_max),
replace each token with [MASK] independently w.p. t, and train the model to
reconstruct the original tokens at masked positions only. This is the
LLaDA-style absorbing-state discrete diffusion objective.

Reverse (inference) process: start from all-[MASK] and iteratively unmask the
most confident positions in parallel — no left-to-right order, no KV cache.

Constraint model (all enforced DURING denoising, verified after):
  * exact output length is a TOKEN count (``length`` tokens, always);
    decoded character/word length is NOT controlled.
  * EOS is an ordinary vocabulary token: the sampler never truncates at EOS
    and EOS carries no structural meaning unless the caller assigns it.
  * ``forbidden_ids`` are excluded from every model-sampled position by
    forcing their logits to -inf each step. Prompt/frozen context is the
    caller's responsibility — forbidden filtering does not rewrite it.
  * ``required_ids`` are a SET: each id must occur >= 1 in the final canvas.
    ``required_counts`` generalizes this to {id: min_occurrences}.
    Occurrences already present in prompt/frozen context count. Missing
    occurrences are force-placed into reserved masked slots (see the
    slot-reservation invariant in DenoisingState.step).
  * ``at_most`` caps occurrences: a token at its cap is suppressed at every
    remaining masked position.
  * ``forbidden_phrases`` are never allowed to complete: any fill that
    would create a new occurrence beyond its allowance is suppressed at
    that position (preventive, so the guarantee holds by construction).
  * ``order_pairs`` are verified post-hoc (every a before every b); the
    Contract compiler lays phrase blocks out in an order satisfying them.
  * ``anchors`` are pre-placed and locked before denoising starts.
  * [MASK] is not in the fill alphabet: a position the sampler fills never
    becomes [MASK] again, so the step budget always suffices and a finished
    run is always "complete" (no masks left). Only explicit remasking
    (remask_ratio / remask_positions / unmask_positions) reintroduces masks.
"""
import math

import torch
import torch.nn.functional as F


def mask_batch(ids: torch.Tensor, mask_id: int, t_min: float = 0.15,
               t_max: float = 1.0, generator=None, lock_prob: float = 0.0,
               lock_spans: tuple = (1, 2), lock_len: tuple = (4, 12)):
    """Apply the diffusion forward process to a batch of token ids.

    Returns (noisy_ids, targets, mask) where mask marks corrupted positions.

    Constraint-aware training: with probability ``lock_prob`` per
    sequence, 1-2 random contiguous spans are marked *locked* — never
    masked, never in the loss. The model must denoise everything around
    them, exactly like inference with anchored facts / locked policy
    text. This teaches fluency-around-fixed-text as a training
    objective, not just a sampling trick.
    """
    B = ids.shape[0]
    t = torch.empty(B, device=ids.device).uniform_(t_min, t_max, generator=generator)
    rand = torch.rand(ids.shape, device=ids.device, generator=generator)
    mask = rand < t.view(B, *([1] * (ids.dim() - 1)))
    if lock_prob > 0:
        L = ids.shape[1]
        locked = torch.zeros_like(mask)
        for b in range(B):
            if torch.rand((), generator=generator).item() > lock_prob:
                continue
            n_spans = torch.randint(lock_spans[0], lock_spans[1] + 1, (),
                                    generator=generator).item()
            for _ in range(n_spans):
                ln = torch.randint(lock_len[0], lock_len[1] + 1, (),
                                   generator=generator).item()
                ln = min(ln, L)
                st = torch.randint(0, L - ln + 1, (), generator=generator).item()
                locked[b, st:st + ln] = True
        mask = mask & ~locked
    noisy = ids.clone()
    noisy[mask] = mask_id
    return noisy, ids, mask


def diffusion_loss(model, ids: torch.Tensor, mask_id: int,
                   t_min: float = 0.15, t_max: float = 1.0, generator=None,
                   image_embeds: torch.Tensor | None = None,
                   lock_prob: float = 0.0,
                   contract_ids: torch.Tensor | None = None):
    """Mean cross-entropy over masked positions only.

    image_embeds: optional (B, N, d) visual prefix for multimodal training
    (substituted for the first N embeddings by the model; those positions
    should be excluded from masking by the caller).

    lock_prob: fraction of sequences trained with locked spans
    (constraint-aware training; see mask_batch).

    contract_ids: optional (B, C) contract text token ids for
    guarantee-native pre-training (the contract describes the CLEAN target;
    the model learns P(target | noisy input, contract)).
    """
    noisy, targets, mask = mask_batch(ids, mask_id, t_min, t_max, generator,
                                      lock_prob=lock_prob)
    fwd_kwargs = {}
    if image_embeds is not None:
        fwd_kwargs["image_embeds"] = image_embeds
    if contract_ids is not None:
        fwd_kwargs["contract_ids"] = contract_ids
    logits = model(noisy, **fwd_kwargs)
    n_masked = mask.sum()
    if n_masked == 0:
        # degenerate guard; keeps graph alive — always return (loss, info) tuple
        return logits.sum() * 0.0, {"mask_ratio": 0.0, "n_masked": 0}
    loss = F.cross_entropy(logits[mask], targets[mask], reduction="mean")
    return loss, {"mask_ratio": mask.float().mean().item(),
                  "n_masked": int(n_masked)}


@torch.no_grad()
def build_priority(length: int, order: str, device="cpu",
                   generator=None) -> torch.Tensor | None:
    """Per-position unmask priority (higher = unmasked earlier).

    Orders:
      confidence     — default; purely confidence-driven (order=None legacy)
      left-to-right  — fully positional: every position is its own tier
      right-to-left  — fully positional, reversed
      ends-first     — tiered: ending, then beginning, middle third last
      middle-last    — tiered: everything except the middle third first
      random         — random permutation, drawn from ``generator``
                       (pass the sampler's seeded generator for determinism)

    Implemented as priority-weighted top-k selection:
    score = priority * 1e6 + confidence, so priority tiers dominate and
    confidence breaks ties WITHIN a tier. l2r/r2l are therefore strictly
    positional; the tiered presets only order groups, not positions.
    """
    if order in ("confidence", None):
        return None
    pos = torch.arange(length, dtype=torch.float32)
    if order == "left-to-right":
        pr = (length - 1) - pos
    elif order == "right-to-left":
        pr = pos.clone()
    elif order == "ends-first":
        third = length // 3
        pr = torch.zeros(length)
        pr[2 * third:] = 2.0   # ending first
        pr[:third] = 1.0        # then the beginning
        # middle third (0.0) last
    elif order == "middle-last":
        third = length // 3
        pr = torch.ones(length)
        pr[third:2 * third] = 0.0
    elif order == "random":
        pr = torch.rand(length, device=device, generator=generator)
        return pr  # already on device
    else:
        raise ValueError(f"unknown order: {order!r} "
                         f"(confidence|left-to-right|right-to-left|"
                         f"ends-first|middle-last|random)")
    return pr.to(device)


def _count_occurrences(ids_list, phrase):
    """Non-overlapping contiguous occurrences of phrase in ids_list."""
    L = len(phrase)
    n, i, N = 0, 0, len(ids_list)
    while i <= N - L:
        if ids_list[i:i + L] == phrase:
            n += 1
            i += L
        else:
            i += 1
    return n


def _phrase_spans(ids_list, phrase):
    """Start positions of non-overlapping contiguous occurrences."""
    L = len(phrase)
    out, i, N = [], 0, len(ids_list)
    while i <= N - L:
        if ids_list[i:i + L] == phrase:
            out.append(i)
            i += L
        else:
            i += 1
    return out


def check_constraints(ids: torch.Tensor, constraints: dict | None,
                      mask_id: int) -> dict:
    """Verify hard constraints on a finished sequence. All structural
    guarantees below are enforced DURING unmasking (see DenoisingState),
    so this is a verification, not a filter."""
    c = constraints or {}
    required = c.get("required_ids", [])
    forbidden = c.get("forbidden_ids", [])
    anchors = c.get("anchors", {})
    required_counts = c.get("required_counts") or {}
    at_most = c.get("at_most") or {}
    forbidden_phrases = c.get("forbidden_phrases") or []
    order_pairs = c.get("order_pairs") or []
    ids = ids.cpu()
    ids_list = ids.tolist()
    counts_met = all(ids_list.count(t) >= n
                     for t, n in required_counts.items())
    if not required_counts:
        counts_met = True
    caps_met = all(ids_list.count(t) <= n for t, n in at_most.items())
    phrases_ok = all(
        _count_occurrences(ids_list, list(p)) <= m
        for p, m in forbidden_phrases)
    order_ok = True
    for a, b in order_pairs:
        a, b = list(a), list(b)
        sa, sb = _phrase_spans(ids_list, a), _phrase_spans(ids_list, b)
        if not sa or not sb:
            order_ok = False
            break
        # every occurrence of a ends before every occurrence of b starts
        if max(s + len(a) for s in sa) > min(sb):
            order_ok = False
            break
    return {
        "required_present": all(bool((ids == t).any()) for t in required),
        "forbidden_absent": not any(bool((ids == t).any()) for t in forbidden),
        "anchors_intact": all(ids[p] == t for p, t in anchors.items()
                              if 0 <= p < len(ids)),
        "complete": not bool((ids == mask_id).any()),
        "counts_met": counts_met,
        "caps_met": caps_met,
        "phrases_ok": phrases_ok,
        "order_ok": order_ok,
    }


class DenoisingState:
    """Full resumable state of one constrained denoising run.

    Captures everything needed to pause, inspect, intervene, and resume:
    the token canvas ``x``, the ``frozen`` mask, the ``masked`` set, the RNG
    ``generator`` state, ``steps_remaining``, the priority/order policy, the
    constraint sets, and per-position ``confidences``.

    Required-token guarantee — slot-reservation invariant:
      Let m_s = #masked positions and q_s = #missing required occurrences
      (sum over required_counts of still-needed counts) at the start of
      iteration s. INVARIANT: q_s <= m_s.
      * Base: __init__ raises unless m_0 >= sum(required_counts) >= q_0.
      * Step (not final): normal top-k fill is capped at (m_s - q_s)
        positions, so at least q_s masked slots survive; force-placement
        adds none. missing only shrinks as tokens appear, so
        m_{s+1} = m_s - filled >= q_s >= q_{s+1}.
      * Final step (steps_remaining == 1): ALL missing occurrences are
        force-placed into the most-confident remaining masked positions
        that keep v2 constraints intact (possible by the invariant;
        raises loudly if no safe position exists), so q_end = 0.
      Manual ops (pin/unmask/remask) re-validate the invariant and raise
      instead of breaking it. Hence: every required count is met in the
      final canvas. QED.
    """

    def __init__(self, *, length: int, mask_id: int, eos_id: int,
                 steps: int = 16, temperature: float = 1.0,
                 top_p: float = 0.95, seed: int | None = None,
                 prompt_ids=None, frozen: torch.Tensor | None = None,
                 remask_ratio: float = 0.0, order: str = "confidence",
                 softmask_k: int = 0,
                 constraints: dict | None = None, device="cpu",
                 vocab_size: int | None = None,
                 return_trajectory: bool = False,
                 image_embeds: torch.Tensor | None = None,
                 contract_ids: torch.Tensor | None = None,
                 allow_frozen_masks: bool = False):
        self.length = int(length)
        self.mask_id = int(mask_id)
        self.eos_id = int(eos_id)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.order = order
        self.remask_ratio = float(remask_ratio)
        self.softmask_k = int(softmask_k)
        self._soft_embeds = None  # (S, d) blended embeddings for soft positions
        self._soft_positions = None  # (S,) position indices
        self.allow_frozen_masks = bool(allow_frozen_masks)
        self.device = device
        self.vocab_size = vocab_size
        # Optional visual prefix (multimodal scaffold): (N, d) or (1, N, d),
        # substituted for the first N token embeddings by the model. The
        # caller freezes those N canvas positions (see xiiv.vision).
        if image_embeds is not None:
            ie = torch.as_tensor(image_embeds)
            if ie.dim() == 2:
                ie = ie.unsqueeze(0)
            if ie.dim() != 3:
                raise ValueError(
                    f"image_embeds must be (N, d) or (1, N, d), got {tuple(ie.shape)}")
            self.image_embeds = ie.to(device)
        else:
            self.image_embeds = None
        # Contract-native path: (1, C) contract token ids, passed to the
        # model's contract encoder on every step. None = unconditional.
        if contract_ids is not None:
            cid = torch.as_tensor(contract_ids, dtype=torch.long)
            if cid.dim() == 1:
                cid = cid.unsqueeze(0)
            if cid.dim() != 2 or cid.shape[0] != 1:
                raise ValueError(
                    f"contract_ids must be (C,) or (1, C), got {tuple(cid.shape)}")
            self.contract_ids = cid.to(device)
        else:
            self.contract_ids = None

        c = constraints or {}
        # SET semantics: each required id must occur >= 1 in the final
        # canvas; duplicates are meaningless and dropped.
        self.forbidden_ids = sorted({int(t) for t in c.get("forbidden_ids", [])})
        self.required_ids = sorted({int(t) for t in c.get("required_ids", [])})
        self.anchors = {int(p): int(t) for p, t in c.get("anchors", {}).items()}

        if mask_id in self.forbidden_ids or mask_id in self.required_ids:
            raise ValueError("mask_id cannot be forbidden or required")
        overlap = set(self.forbidden_ids) & set(self.required_ids)
        if overlap:
            raise ValueError(
                "impossible constraint set: token ids "
                f"{sorted(overlap)} are both required and forbidden")

        # --- v2 constraints ---
        # required COUNTS {id: min_occurrences}; required_ids merge as 1.
        self.required_counts = {t: 1 for t in self.required_ids}
        for t, n in (c.get("required_counts") or {}).items():
            t, n = int(t), int(n)
            if n < 1:
                raise ValueError(
                    f"required count for token {t} must be >= 1")
            if t in self.forbidden_ids:
                raise ValueError(
                    f"impossible constraint set: token {t} both required "
                    "and forbidden")
            if t == mask_id:
                raise ValueError("mask_id cannot be required")
            self.required_counts[t] = max(self.required_counts.get(t, 0), n)
        self.required_ids = sorted(self.required_counts)
        # at-most caps {id: max_occurrences}
        self.at_most = {}
        for t, n in (c.get("at_most") or {}).items():
            t, n = int(t), int(n)
            if n < 0:
                raise ValueError(f"at_most cap for token {t} must be >= 0")
            if t == mask_id:
                raise ValueError("mask_id cannot be capped")
            if t in self.required_counts and self.required_counts[t] > n:
                raise ValueError(
                    f"impossible constraint set: token {t} required "
                    f"{self.required_counts[t]}x but capped at {n}x")
            self.at_most[t] = n
        # forbidden phrases [(ids, max_allowed)]; 0 = never
        self.forbidden_phrases = []
        for ids, m in (c.get("forbidden_phrases") or []):
            ids = [int(i) for i in ids]
            m = int(m)
            if not ids:
                raise ValueError("empty forbidden phrase")
            if m < 0:
                raise ValueError("forbidden-phrase allowance must be >= 0")
            if mask_id in ids:
                raise ValueError("mask_id cannot appear in a phrase")
            if any(t in self.forbidden_ids for t in ids):
                # a phrase containing a forbidden single token can never
                # form (that token never appears): vacuously satisfiable.
                continue
            self.forbidden_phrases.append((ids, m))
        # ordering [(a_ids, b_ids)]: every a before every b
        self.order_pairs = []
        for a, b in (c.get("order_pairs") or []):
            a = [int(i) for i in a]
            b = [int(i) for i in b]
            if not a or not b:
                raise ValueError("empty phrase in order pair")
            self.order_pairs.append((a, b))

        if vocab_size is not None:
            for t in self.forbidden_ids + self.required_ids:
                if not (0 <= t < vocab_size):
                    raise ValueError(
                        f"constraint token id {t} out of vocabulary "
                        f"range [0, {vocab_size})")
            for t in self.at_most:
                if not (0 <= t < vocab_size):
                    raise ValueError(
                        f"at_most token id {t} out of vocabulary range")
            for ids, _ in self.forbidden_phrases:
                for t in ids:
                    if not (0 <= t < vocab_size):
                        raise ValueError(
                            "forbidden-phrase token id out of vocabulary")
            for p in self.anchors:
                if not (0 <= p < self.length):
                    raise ValueError(f"anchor position {p} out of range")
            if len(self.forbidden_ids) >= vocab_size - 1:
                # mask_id itself can never be forbidden (checked above), so
                # forbidding every other id leaves no fillable token: filled
                # positions structurally exclude [MASK] (see step()).
                raise ValueError(
                    "impossible constraint set: forbidden_ids leave no "
                    "fillable token (every non-[MASK] id is forbidden)")
        else:
            for p in self.anchors:
                if not (0 <= p < self.length):
                    raise ValueError(f"anchor position {p} out of range")
        if self.required_ids and self.remask_ratio > 0:
            # re-masking could erase a force-placed required token; the
            # guarantee requires refinement to be off.
            self.remask_ratio = 0.0

        self.g = torch.Generator(device=device)
        if seed is not None:
            self.g.manual_seed(seed)
        self.seed = seed

        x = torch.full((self.length,), mask_id, dtype=torch.long,
                       device=device)
        if prompt_ids is not None:
            prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long,
                                         device=device)
            L = min(len(prompt_ids), self.length)
            x[:L] = prompt_ids[:L]
        if frozen is None:
            # default: everything that isn't [MASK] is context and stays fixed.
            frozen = (x != mask_id)
        else:
            frozen = torch.as_tensor(frozen, dtype=torch.bool,
                                     device=device).clone()
            if frozen.shape != (self.length,):
                raise ValueError(
                    f"frozen mask shape {tuple(frozen.shape)} != "
                    f"({self.length},)")
        # anchors: pre-place + lock (structure priors)
        for p, t in self.anchors.items():
            x[p] = t
            frozen[p] = True
        # the model must SEE the context to condition on it
        # (image-prefix placeholders are exempt: their "context" is the
        # image embedding substituted at the embedding level, not a token)
        img_n = (self.image_embeds.shape[1]
                 if self.image_embeds is not None else 0)
        frozen_masks = frozen & (x == mask_id)
        if img_n:
            frozen_masks[:img_n] = False
        if bool(frozen_masks.any()) and not allow_frozen_masks:
            raise ValueError("frozen positions must hold context ids, "
                             "not [MASK] (pass allow_frozen_masks=True to "
                             "deliberately leave positions masked)")
        self.x = x
        self.frozen = frozen

        masked0 = (x == mask_id) & ~frozen
        need0 = sum(self.required_counts.values())
        if need0 and int(masked0.sum()) < need0:
            raise ValueError(
                f"unsatisfiable: {need0} required token occurrences "
                f"but only {int(masked0.sum())} maskable positions")
        self.masked = masked0.clone()

        # seeded generator threaded through: seed -> identical output for
        # every order preset, including "random".
        self.priority = build_priority(self.length, order, device,
                                       generator=self.g)
        self.conf = torch.zeros(self.length, dtype=torch.float32,
                                device=device)
        self.steps_remaining = int(steps)
        if self.steps_remaining < 0:
            raise ValueError("steps must be >= 0")
        self.trajectory = [x.clone()] if return_trajectory else None
        # Intervention log: every time the sampler actively prevents a
        # constraint violation (not just passive masking). Used by
        # benchmark 2 (intervention rate). Each entry:
        # {"step": int, "kind": str, "detail": dict}
        self.interventions = []

    # -- core iteration -------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.steps_remaining <= 0 or not bool(self.masked.any())

    # -- v2 constraint helpers -------------------------------------------
    def _need_counts(self, x):
        """{token_id: still-needed occurrences} for the required counts."""
        need = {}
        for t, n in self.required_counts.items():
            have = int((x == t).sum())
            if have < n:
                need[t] = n - have
        return need

    @staticmethod
    def _count_phrase(x_list, phrase):
        """Non-overlapping contiguous occurrences of phrase in x_list."""
        L = len(phrase)
        n, i, N = 0, 0, len(x_list)
        while i <= N - L:
            if x_list[i:i + L] == phrase:
                n += 1
                i += L
            else:
                i += 1
        return n

    def _phrase_occurrences(self, x):
        """Current occurrence count for each forbidden phrase."""
        xl = x.tolist()
        return [self._count_phrase(xl, p) for p, _ in self.forbidden_phrases]

    def _completing_tokens(self, pos):
        """Token ids that must NOT be placed at masked position ``pos``:
        any token whose placement would create a NEW forbidden-phrase
        occurrence beyond its allowance. Preventive: the phrase can never
        complete, so the guarantee holds by construction."""
        out = []
        if not self.forbidden_phrases:
            return out
        x = self.x
        occ = self._phrase_occurrences(x)
        for (phrase, max_allowed), cur in zip(self.forbidden_phrases, occ):
            if cur < max_allowed:
                continue
            L = len(phrase)
            for off in range(L):
                start = pos - off
                if start < 0 or start + L > self.length:
                    continue
                ok = True
                for j, tid in enumerate(phrase):
                    p = start + j
                    if p == pos:
                        continue
                    pv = int(x[p])
                    if pv == self.mask_id or pv != tid:
                        ok = False
                        break
                if ok:
                    out.append(phrase[off])
                    break  # one suppression per phrase per position
        return out

    def _capped_tokens(self):
        """Token ids currently at their at_most cap."""
        return [t for t, n in self.at_most.items()
                if int((self.x == t).sum()) >= n]

    @torch.no_grad()
    def contract_satisfiable(self, x=None) -> bool:
        """LAVE-style lookahead: can the contract still be satisfied?

        Checks whether the current (or hypothetical) canvas state can still
        be completed into a valid output. This prevents dead ends where the
        sampler fills all masks without placing required tokens, only to
        discover the violation at the end.

        Unlike LAVE/DINGO (which handle syntactic constraints), this covers
        SEMANTIC contracts: required/forbidden token counts with proof
        certificates. To our knowledge, no diffusion LM does this.

        Args:
            x: canvas to check (defaults to self.x)
        Returns:
            True if some completion of the masked positions can satisfy
            all contract constraints.
        """
        x = self.x if x is None else x
        n_masks = int((x == self.mask_id).sum())

        # Required counts: enough masks remain to place all missing?
        need = self._need_counts(x)
        total_missing = sum(need.values())
        if total_missing > n_masks:
            return False

        # Forbidden: none already placed? (masks can't introduce violations
        # since forbidden logits are -inf, but prompt/frozen might)
        if self.forbidden_ids:
            for tok in self.forbidden_ids:
                if int((x == tok).sum()) > 0:
                    # Check if it's in a frozen/prompt position (caller's
                    # responsibility) vs a filled position (our fault)
                    filled_forbidden = False
                    for pos in range(self.length):
                        if int(x[pos]) == tok and not bool(self.frozen[pos]):
                            # Was this filled by us? Check if it was masked
                            filled_forbidden = True
                            break
                    if filled_forbidden:
                        return False

        # at_most caps: not already exceeded?
        for tok, cap in self.at_most.items():
            if int((x == tok).sum()) > cap:
                return False

        return True

    def step(self, model) -> bool:
        """Run one denoise iteration. Returns True if more work remains."""
        if self.finished:
            return False
        model.eval()
        xb = self.x.unsqueeze(0)                              # (1, L)
        fwd_kwargs = {}
        if self.image_embeds is not None:
            fwd_kwargs["image_embeds"] = self.image_embeds
        if self.contract_ids is not None:
            fwd_kwargs["contract_ids"] = self.contract_ids
        logits = model(xb, **fwd_kwargs).squeeze(0)           # (L, V)
        # Intervention: check if the raw top-1 would have been forbidden
        # (i.e., the mask actively prevented a violation, not just passive)
        if self.forbidden_ids:
            raw_top1 = logits.argmax(dim=-1)
            for pos in range(self.length):
                if not bool(self.masked[pos]):
                    continue
                if int(raw_top1[pos]) in self.forbidden_ids:
                    self.interventions.append({
                        "step": self.steps_remaining,
                        "kind": "forbidden_prevented",
                        "detail": {"position": pos,
                                   "blocked_token": int(raw_top1[pos])},
                    })
            logits[:, self.forbidden_ids] = float("-inf")
            if not bool(torch.isfinite(logits).any(dim=-1).all()):
                n = int((~torch.isfinite(logits).any(dim=-1)).sum())
                raise ValueError(
                    "impossible constraint set: forbidden_ids exclude "
                    f"every vocabulary token at {n} position(s)")
        logits = logits / max(self.temperature, 1e-6)
        if self.top_p < 1.0:
            logits = _top_p_filter(logits, self.top_p)
        probs = torch.softmax(logits, dim=-1)
        conf, _ = probs.max(dim=-1)                             # (L,)
        self.conf = conf

        # required counts absent from the whole canvas (prompt/frozen count).
        # missing is a multiset: one entry per still-needed occurrence.
        need = self._need_counts(self.x)
        missing = [t for t, k in need.items() for _ in range(k)]

        # ENDGAME (invariant: sum(missing) <= #masked, proved in class
        # docstring): on the final step, force-place ALL missing occurrences
        # into the most-confident remaining masked positions that keep every
        # v2 constraint intact (no forbidden-phrase completion, no at_most
        # breach). A placement failure here is a loud error, not a silent
        # guarantee break.
        if missing and self.steps_remaining == 1:
            mconf = torch.where(self.masked, conf,
                                torch.full_like(conf, -1.0))
            cand = [p for p in torch.argsort(mconf, descending=True).tolist()
                    if bool(self.masked[p])]
            used = set()
            for tok_i in missing:
                placed_tok = False
                for pos_i in cand:
                    if pos_i in used:
                        continue
                    if tok_i in self._completing_tokens(pos_i):
                        continue
                    cap = self.at_most.get(tok_i)
                    if cap is not None and \
                            int((self.x == tok_i).sum()) >= cap:
                        raise ValueError(
                            f"unsatisfiable at endgame: required token "
                            f"{tok_i} would exceed its at_most cap")
                    self.x[pos_i] = tok_i
                    self.conf[pos_i] = 1.0  # placed deterministically
                    self.interventions.append({
                        "step": self.steps_remaining,
                        "kind": "endgame_forced_placement",
                        "detail": {"position": pos_i, "token": tok_i},
                    })
                    used.add(pos_i)
                    placed_tok = True
                    break
                if not placed_tok:
                    raise ValueError(
                        "unsatisfiable at endgame: no safe position for "
                        f"required token {tok_i}")
            self.masked = (self.x == self.mask_id) & ~self.frozen
            missing = []

        # normal parallel fill, reserving masked slots for missing counts.
        m = int(self.masked.sum())
        k = 0
        if m > 0:
            k = max(1, math.ceil(m / self.steps_remaining))
            if missing:
                # SLOT RESERVATION: never consume slots that missing required
                # occurrences still need. This is what keeps q_s <= m_s.
                k = min(k, max(0, m - len(missing)))
        if k > 0:
            # LOOKAHEAD SATISFIABILITY (world-first for semantic contracts):
            # Before accepting this batch, verify the contract remains
            # satisfiable. If filling k positions would create a dead end,
            # shrink k until satisfiable. This prevents wasted steps that
            # only discover violations at endgame.
            #
            # The check is O(L) — negligible vs a forward pass.
            # Slot reservation already handles required counts; this catches
            # the subtle cases (at_most interactions, forbidden placements).
            while k > 0:
                # Hypothetical: if we fill k positions, masks remaining = m - k
                masks_after = m - k
                need_after = self._need_counts(self.x)
                total_missing = sum(need_after.values())
                if total_missing <= masks_after:
                    break
                k -= 1
            if k <= 0:
                # Cannot fill any position without breaking satisfiability;
                # skip this step (endgame will force-place)
                self.steps_remaining -= 1
                return not self.finished
            if self.priority is not None:
                score = torch.where(self.masked, self.priority * 1e6 + conf,
                                    torch.full_like(conf, -1e18))
                _, idx = torch.topk(score, k)
            else:
                masked_conf = torch.where(self.masked, conf,
                                          torch.full_like(conf, -1.0))
                _, idx = torch.topk(masked_conf, k)
            # sample the chosen positions from their tempered distributions.
            # Structural rule: a filled position never becomes [MASK] again
            # (the model is free to predict EOS; [MASK] is simply not in the
            # fill alphabet). This is what makes "complete" a guarantee
            # rather than luck: the step budget always suffices.
            for i in idx.tolist():
                p = probs[i].clone()
                p[self.mask_id] = 0.0
                # v2: preventive suppression specific to this position —
                # at_most caps and forbidden-phrase completions. Applied
                # per fill (the loop is sequential, so each fill sees the
                # previous ones: no parallel-completion race).
                v2_blocked = []
                if self.forbidden_phrases or self.at_most:
                    v2_blocked = self._completing_tokens(i) + \
                        self._capped_tokens()
                    if v2_blocked:
                        p[v2_blocked] = 0.0
                s = p.sum()
                if s <= 0:
                    # degenerate: model put all mass on [MASK]; fall back to
                    # uniform over the allowed alphabet (still via g, so
                    # deterministic given seed).
                    allowed = torch.ones(p.shape[0], dtype=torch.bool,
                                         device=p.device)
                    allowed[self.mask_id] = False
                    if self.forbidden_ids:
                        allowed[self.forbidden_ids] = False
                    if v2_blocked:
                        allowed[v2_blocked] = False
                    if not bool(allowed.any()):
                        raise ValueError(
                            "impossible constraint set: no fillable token "
                            f"at position {i}")
                    p = allowed.float() / allowed.sum()
                else:
                    p = p / s
                self.x[i] = torch.multinomial(p, 1,
                                              generator=self.g).item()

        # optional refinement: re-mask the least confident non-frozen tokens
        # (disabled whenever required_ids are present — see __init__).
        if self.remask_ratio > 0:
            unmasked = (self.x != self.mask_id) & ~self.frozen
            if unmasked.any():
                uc = torch.where(unmasked, self.conf,
                                 torch.full_like(self.conf, float("inf")))
                r = max(1, int(unmasked.sum().item() * self.remask_ratio))
                _, ridx = torch.topk(uc, r, largest=False)
                self.x[ridx] = self.mask_id

        self.masked = (self.x == self.mask_id) & ~self.frozen
        self.steps_remaining -= 1
        if self.trajectory is not None:
            self.trajectory.append(self.x.clone())
        return not self.finished

    def soft_mask_blend(self, model, positions: torch.Tensor,
                        probs: torch.Tensor, k: int = 5) -> tuple:
        """Compute confidence-weighted blended embeddings for soft-masking.

        Instead of discrete mask->token transitions, each position gets:
            x̃ = (1−λ)·m + λ·Σ πᵢ·vᵢ
        where m is the mask embedding, vᵢ are top-k token embeddings,
        πᵢ their renormalized probabilities, λ from predictive entropy.

        From Hersche et al., ICLR 2026. Composes with contract constraints:
        forbidden tokens are excluded from the top-k before blending,
        so guarantees hold.

        Returns (blended_embeds, position_indices).
        """
        device = self.x.device
        d = model.cfg.d_model
        mask_emb = model.embed.weight[self.mask_id]  # (d,)

        blended = []
        pos_list = []
        for pos in positions.tolist():
            p = probs[pos].clone()
            p[self.mask_id] = 0.0
            if self.forbidden_ids:
                p[self.forbidden_ids] = float("-inf")
            # top-k
            topk_p, topk_idx = torch.topk(p, min(k, p.shape[0]))
            topk_p = torch.softmax(topk_p, dim=-1)  # renormalize
            # entropy-derived lambda: high entropy -> more mask, low -> more token
            entropy = -(topk_p * torch.log(topk_p + 1e-10)).sum()
            max_ent = torch.log(torch.tensor(float(len(topk_p))))
            lam = 1.0 - (entropy / (max_ent + 1e-10)).item()  # 0..1
            lam = max(0.0, min(0.95, lam))  # never fully commit
            # blended embedding
            tok_embs = model.embed.weight[topk_idx]  # (k, d)
            weighted = (topk_p.unsqueeze(-1) * tok_embs).sum(dim=0)  # (d,)
            blend = (1 - lam) * mask_emb + lam * weighted
            blended.append(blend)
            pos_list.append(pos)

        if not blended:
            return None, None
        return torch.stack(blended), torch.tensor(pos_list, device=device)
        """Run one denoise iteration. Returns True if more work remains."""
        if self.finished:
            return False
        model.eval()
        xb = self.x.unsqueeze(0)                              # (1, L)
        fwd_kwargs = {}
        if self.image_embeds is not None:
            fwd_kwargs["image_embeds"] = self.image_embeds
        if self.contract_ids is not None:
            fwd_kwargs["contract_ids"] = self.contract_ids
        if self._soft_embeds is not None and self._soft_positions is not None:
            fwd_kwargs["soft_embeds"] = self._soft_embeds.unsqueeze(0)
            fwd_kwargs["soft_positions"] = self._soft_positions.unsqueeze(0)
        logits = model(xb, **fwd_kwargs).squeeze(0)           # (L, V)
        # Intervention: check if the raw top-1 would have been forbidden
        # (i.e., the mask actively prevented a violation, not just passive)
        if self.forbidden_ids:
            raw_top1 = logits.argmax(dim=-1)
            for pos in range(self.length):
                if not bool(self.masked[pos]):
                    continue
                if int(raw_top1[pos]) in self.forbidden_ids:
                    self.interventions.append({
                        "step": self.steps_remaining,
                        "kind": "forbidden_prevented",
                        "detail": {"position": pos,
                                   "blocked_token": int(raw_top1[pos])},
                    })
            logits[:, self.forbidden_ids] = float("-inf")
            if not bool(torch.isfinite(logits).any(dim=-1).all()):
                n = int((~torch.isfinite(logits).any(dim=-1)).sum())
                raise ValueError(
                    "impossible constraint set: forbidden_ids exclude "
                    f"every vocabulary token at {n} position(s)")
        logits = logits / max(self.temperature, 1e-6)
        if self.top_p < 1.0:
            logits = _top_p_filter(logits, self.top_p)
        probs = torch.softmax(logits, dim=-1)
        conf, _ = probs.max(dim=-1)                             # (L,)
        self.conf = conf

        # required counts absent from the whole canvas (prompt/frozen count).
        # missing is a multiset: one entry per still-needed occurrence.
        need = self._need_counts(self.x)
        missing = [t for t, k in need.items() for _ in range(k)]

        # ENDGAME (invariant: sum(missing) <= #masked, proved in class
        # docstring): on the final step, force-place ALL missing occurrences
        # into the most-confident remaining masked positions that keep every
        # v2 constraint intact (no forbidden-phrase completion, no at_most
        # breach). A placement failure here is a loud error, not a silent
        # guarantee break.
        if missing and self.steps_remaining == 1:
            mconf = torch.where(self.masked, conf,
                                torch.full_like(conf, -1.0))
            cand = [p for p in torch.argsort(mconf, descending=True).tolist()
                    if bool(self.masked[p])]
            used = set()
            for tok_i in missing:
                placed_tok = False
                for pos_i in cand:
                    if pos_i in used:
                        continue
                    if tok_i in self._completing_tokens(pos_i):
                        continue
                    cap = self.at_most.get(tok_i)
                    if cap is not None and \
                            int((self.x == tok_i).sum()) >= cap:
                        raise ValueError(
                            f"unsatisfiable at endgame: required token "
                            f"{tok_i} would exceed its at_most cap")
                    self.x[pos_i] = tok_i
                    self.conf[pos_i] = 1.0  # placed deterministically
                    self.interventions.append({
                        "step": self.steps_remaining,
                        "kind": "endgame_forced_placement",
                        "detail": {"position": pos_i, "token": tok_i},
                    })
                    used.add(pos_i)
                    placed_tok = True
                    break
                if not placed_tok:
                    raise ValueError(
                        "unsatisfiable at endgame: no safe position for "
                        f"required token {tok_i}")
            self.masked = (self.x == self.mask_id) & ~self.frozen
            missing = []

        # normal parallel fill, reserving masked slots for missing counts.
        m = int(self.masked.sum())
        k = 0
        if m > 0:
            k = max(1, math.ceil(m / self.steps_remaining))
            if missing:
                # SLOT RESERVATION: never consume slots that missing required
                # occurrences still need. This is what keeps q_s <= m_s.
                k = min(k, max(0, m - len(missing)))
        if k > 0:
            # LOOKAHEAD SATISFIABILITY (world-first for semantic contracts):
            # Before accepting this batch, verify the contract remains
            # satisfiable. If filling k positions would create a dead end,
            # shrink k until satisfiable. This prevents wasted steps that
            # only discover violations at endgame.
            #
            # The check is O(L) — negligible vs a forward pass.
            # Slot reservation already handles required counts; this catches
            # the subtle cases (at_most interactions, forbidden placements).
            while k > 0:
                # Hypothetical: if we fill k positions, masks remaining = m - k
                masks_after = m - k
                need_after = self._need_counts(self.x)
                total_missing = sum(need_after.values())
                if total_missing <= masks_after:
                    break
                k -= 1
            if k <= 0:
                # Cannot fill any position without breaking satisfiability;
                # skip this step (endgame will force-place)
                self.steps_remaining -= 1
                return not self.finished
            if self.priority is not None:
                score = torch.where(self.masked, self.priority * 1e6 + conf,
                                    torch.full_like(conf, -1e18))
                _, idx = torch.topk(score, k)
            else:
                masked_conf = torch.where(self.masked, conf,
                                          torch.full_like(conf, -1.0))
                _, idx = torch.topk(masked_conf, k)
            # sample the chosen positions from their tempered distributions.
            # Structural rule: a filled position never becomes [MASK] again
            # (the model is free to predict EOS; [MASK] is simply not in the
            # fill alphabet). This is what makes "complete" a guarantee
            # rather than luck: the step budget always suffices.
            for i in idx.tolist():
                p = probs[i].clone()
                p[self.mask_id] = 0.0
                # v2: preventive suppression specific to this position —
                # at_most caps and forbidden-phrase completions. Applied
                # per fill (the loop is sequential, so each fill sees the
                # previous ones: no parallel-completion race).
                v2_blocked = []
                if self.forbidden_phrases or self.at_most:
                    v2_blocked = self._completing_tokens(i) + \
                        self._capped_tokens()
                    if v2_blocked:
                        p[v2_blocked] = 0.0
                s = p.sum()
                if s <= 0:
                    # degenerate: model put all mass on [MASK]; fall back to
                    # uniform over the allowed alphabet (still via g, so
                    # deterministic given seed).
                    allowed = torch.ones(p.shape[0], dtype=torch.bool,
                                         device=p.device)
                    allowed[self.mask_id] = False
                    if self.forbidden_ids:
                        allowed[self.forbidden_ids] = False
                    if v2_blocked:
                        allowed[v2_blocked] = False
                    if not bool(allowed.any()):
                        raise ValueError(
                            "impossible constraint set: no fillable token "
                            f"at position {i}")
                    p = allowed.float() / allowed.sum()
                else:
                    p = p / s
                self.x[i] = torch.multinomial(p, 1,
                                              generator=self.g).item()

        # optional refinement: re-mask the least confident non-frozen tokens
        # (disabled whenever required_ids are present — see __init__).
        if self.remask_ratio > 0:
            unmasked = (self.x != self.mask_id) & ~self.frozen
            if unmasked.any():
                uc = torch.where(unmasked, self.conf,
                                 torch.full_like(self.conf, float("inf")))
                r = max(1, int(unmasked.sum().item() * self.remask_ratio))
                _, ridx = torch.topk(uc, r, largest=False)
                self.x[ridx] = self.mask_id

        self.masked = (self.x == self.mask_id) & ~self.frozen
        self.steps_remaining -= 1

        # Soft-masking: compute blended embeddings for remaining masked
        # positions. Next forward pass will see uncertainty-weighted blends
        # instead of pure [MASK] embeddings. This gives richer signal.
        if self.softmask_k > 0 and bool(self.masked.any()):
            masked_pos = torch.where(self.masked)[0]
            # Recompute probs for blending (we have them from this step)
            blend_embeds, blend_pos = self.soft_mask_blend(
                model, masked_pos, probs, k=self.softmask_k)
            self._soft_embeds = blend_embeds
            self._soft_positions = blend_pos
        else:
            self._soft_embeds = None
            self._soft_positions = None

        if self.trajectory is not None:
            self.trajectory.append(self.x.clone())
        return not self.finished

    # -- mid-process intervention ----------------------------------------
    def _check_slot_invariant(self, x, masked, context: str):
        missing = sum(max(0, n - int((x == t).sum()))
                      for t, n in self.required_counts.items())
        if missing > int(masked.sum()):
            raise ValueError(
                f"{context} would break the required-token slot invariant "
                f"({missing} missing required occurrences, "
                f"{int(masked.sum())} masked slots left)")

    def pin(self, pos: int, token_id: int):
        """Pin a token mid-process: set it and freeze it immediately.

        The pinned token survives all remaining steps. Cannot pin [MASK]
        or a forbidden id. Raises if the pin would leave a required id
        with no masked slot to occupy.
        """
        pos, token_id = int(pos), int(token_id)
        if not (0 <= pos < self.length):
            raise ValueError(f"pin position {pos} out of range")
        if token_id == self.mask_id:
            raise ValueError("cannot pin [MASK]")
        if token_id in self.forbidden_ids:
            raise ValueError(f"cannot pin forbidden token id {token_id}")
        x = self.x.clone()
        x[pos] = token_id
        masked = (x == self.mask_id) & ~(self.frozen |
                                         _one_hot_mask(pos, self.length,
                                                       self.device))
        self._check_slot_invariant(x, masked, "pin()")
        self.x[pos] = token_id
        self.frozen[pos] = True
        self.conf[pos] = 1.0
        self.masked = (self.x == self.mask_id) & ~self.frozen

    def unmask_positions(self, indices):
        """Manual intervention: turn positions into editable [MASK] slots.

        Grows the masked set, so the slot invariant is preserved.
        """
        idx = torch.as_tensor(list(indices), dtype=torch.long,
                              device=self.device)
        if idx.numel() == 0:
            return
        if bool((idx < 0).any()) or bool((idx >= self.length).any()):
            raise ValueError("unmask_positions: index out of range")
        self.x[idx] = self.mask_id
        self.frozen[idx] = False
        self.conf[idx] = 0.0
        self.masked = (self.x == self.mask_id) & ~self.frozen

    def remask_positions(self, indices):
        """Manual intervention: re-mask non-frozen positions for re-fill.

        A remasked required token simply becomes missing again (both the
        missing count and the masked count grow by one), so the slot
        invariant is preserved and the endgame will replace it.
        """
        idx = torch.as_tensor(list(indices), dtype=torch.long,
                              device=self.device)
        if idx.numel() == 0:
            return
        if bool((idx < 0).any()) or bool((idx >= self.length).any()):
            raise ValueError("remask_positions: index out of range")
        if bool(self.frozen[idx].any()):
            raise ValueError("remask_positions: cannot re-mask a frozen "
                             "position (use unmask_positions to unfreeze "
                             "it first)")
        self.x[idx] = self.mask_id
        self.conf[idx] = 0.0
        self.masked = (self.x == self.mask_id) & ~self.frozen

    # -- pause / resume ----------------------------------------------------
    def to_snapshot(self) -> dict:
        """Serialize the full resumable state (tensors + RNG state).

        The dict is torch.save-able, so a run can pause in one process and
        resume in another with bit-identical continuation.
        """
        return {
            "version": 1,
            "length": self.length,
            "mask_id": self.mask_id,
            "eos_id": self.eos_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "order": self.order,
            "remask_ratio": self.remask_ratio,
            "device": self.device,
            "seed": self.seed,
            "vocab_size": self.vocab_size,
            "x": self.x.cpu().clone(),
            "frozen": self.frozen.cpu().clone(),
            "masked": self.masked.cpu().clone(),
            "conf": self.conf.cpu().clone(),
            "priority": self.priority.cpu().clone()
                      if self.priority is not None else None,
            "generator_state": self.g.get_state().cpu().clone(),
            "steps_remaining": self.steps_remaining,
            "forbidden_ids": list(self.forbidden_ids),
            "required_ids": list(self.required_ids),
            "required_counts": dict(self.required_counts),
            "at_most": dict(self.at_most),
            "forbidden_phrases": [(list(p), m)
                                  for p, m in self.forbidden_phrases],
            "order_pairs": [(list(a), list(b))
                            for a, b in self.order_pairs],
            "anchors": dict(self.anchors),
            "trajectory": ([t.cpu().clone() for t in self.trajectory]
                           if self.trajectory is not None else None),
            "image_embeds": (self.image_embeds.cpu().clone()
                             if self.image_embeds is not None else None),
            "contract_ids": (self.contract_ids.cpu().clone()
                             if self.contract_ids is not None else None),
            "allow_frozen_masks": self.allow_frozen_masks,
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "DenoisingState":
        """Rebuild a state from to_snapshot(). Continues bit-identically."""
        if snap.get("version") != 1:
            raise ValueError(f"unsupported snapshot version: "
                             f"{snap.get('version')!r}")
        st = cls.__new__(cls)
        st.length = snap["length"]
        st.mask_id = snap["mask_id"]
        st.eos_id = snap["eos_id"]
        st.temperature = snap["temperature"]
        st.top_p = snap["top_p"]
        st.order = snap["order"]
        st.remask_ratio = snap["remask_ratio"]
        st.device = snap["device"]
        st.seed = snap["seed"]
        st.vocab_size = snap["vocab_size"]
        st.forbidden_ids = list(snap["forbidden_ids"])
        st.required_ids = list(snap["required_ids"])
        st.required_counts = {int(t): int(n) for t, n in
                              snap.get("required_counts",
                                       {t: 1 for t in snap["required_ids"]}).items()}
        st.at_most = {int(t): int(n) for t, n in
                      snap.get("at_most", {}).items()}
        st.forbidden_phrases = [([int(i) for i in p], int(m)) for p, m in
                                snap.get("forbidden_phrases", [])]
        st.order_pairs = [([int(i) for i in a], [int(i) for i in b])
                          for a, b in snap.get("order_pairs", [])]
        st.anchors = dict(snap["anchors"])
        st.x = snap["x"].clone()
        st.frozen = snap["frozen"].clone()
        st.masked = snap["masked"].clone()
        st.conf = snap["conf"].clone()
        st.priority = (snap["priority"].clone()
                       if snap["priority"] is not None else None)
        st.steps_remaining = snap["steps_remaining"]
        st.trajectory = snap["trajectory"]
        st.image_embeds = (snap["image_embeds"].clone()
                           if snap.get("image_embeds") is not None else None)
        st.contract_ids = (snap["contract_ids"].clone()
                           if snap.get("contract_ids") is not None else None)
        st.allow_frozen_masks = bool(snap.get("allow_frozen_masks", False))
        g = torch.Generator(device=snap["device"])
        g.set_state(snap["generator_state"].clone())
        st.g = g
        return st


def _one_hot_mask(pos: int, length: int, device) -> torch.Tensor:
    m = torch.zeros(length, dtype=torch.bool, device=device)
    m[pos] = True
    return m


@torch.no_grad()
def sample_trajectory(model, length: int, mask_id: int, eos_id: int,
                      steps: int = 16, temperature: float = 1.0, top_p: float = 0.95,
                      seed: int | None = None, prompt_ids=None,
                      frozen: torch.Tensor | None = None,
                      remask_ratio: float = 0.0,
                      return_trajectory: bool = False,
                      order: str = "confidence",
                      softmask_k: int = 0,
                      constraints: dict | None = None,
                      image_embeds: torch.Tensor | None = None,
                      allow_frozen_masks: bool = False,
                      device="cpu"):
    """Parallel de-masking generation.

    The output is ALWAYS exactly ``length`` TOKENS — "exact length" is a
    token count. Decoded character/word length is not controlled and is
    reported separately if needed. EOS is an ordinary vocabulary token:
    the sampler never truncates at EOS and keeps generating to full length.

    Args:
        length: total sequence length to generate (token count).
        prompt_ids: optional 1-D tensor placed at the start (never re-masked).
        frozen: optional bool mask (length,) — True positions are never changed
            (used for infilling: keep the context, fill only [MASK] spans).
        remask_ratio: fraction of lowest-confidence unmasked (non-frozen)
            tokens to re-mask each step for iterative refinement (0 = off).
        return_trajectory: if True, also return the per-step token-id matrix
            for visualization.
        order: unmasking order preset — "confidence" (default),
            "left-to-right", "right-to-left", "ends-first", "middle-last",
            "random". Implemented as priority-weighted top-k selection:
            score = priority * 1e6 + confidence, so priority tiers dominate
            and confidence breaks ties WITHIN a tier. left-to-right and
            right-to-left are therefore fully positional; the tiered presets
            (ends-first, middle-last) order groups, not individual positions.
            "random" draws its permutation from the seeded generator, so
            seed -> identical output for every preset.
        constraints: optional dict with hard constraints enforced DURING
            unmasking (not hoped for afterward):
              forbidden_ids: list[int] — logits forced to -inf every step,
                  so these tokens can never be model-sampled. Structural
                  exclusion. Applies to generated positions only;
                  prompt/frozen context is the caller's responsibility.
                  Constraint sets that leave no valid token raise ValueError
                  instead of producing NaN.
              required_ids: list[int] — SET semantics: each id is guaranteed
                  to occur >= 1 in the final output. Ids already present in
                  prompt/frozen context count; duplicates are dropped. Any
                  still missing on the final step are force-placed into the
                  most confident remaining masked positions (slot-reservation
                  invariant — see DenoisingState). (Implies remask_ratio=0.)
              anchors: dict[pos, id] — pre-placed and locked, e.g. required
                  words at exact positions ("structure priors"). The model
                  writes around them.

    Returns (ids, confidences) or (ids, confidences, trajectory).
    Deterministic given `seed`. If nothing is masked (e.g. a fully frozen
    canvas), returns immediately with zero confidences.
    """
    vocab_size = getattr(getattr(model, "cfg", None), "vocab_size", None)
    st = DenoisingState(
        length=length, mask_id=mask_id, eos_id=eos_id, steps=steps,
        temperature=temperature, top_p=top_p, seed=seed,
        prompt_ids=prompt_ids, frozen=frozen, remask_ratio=remask_ratio,
        order=order, constraints=constraints, device=device,
        vocab_size=vocab_size, return_trajectory=return_trajectory,
        image_embeds=image_embeds, allow_frozen_masks=allow_frozen_masks,
        softmask_k=softmask_k)
    while st.step(model):
        pass
    out = (st.x.cpu(), st.conf.cpu())
    if return_trajectory:
        out += (torch.stack(st.trajectory).cpu(),)
    return out


@torch.no_grad()
def sample_trajectory_stateful(model, length: int, mask_id: int, eos_id: int,
                               steps: int = 16, temperature: float = 1.0,
                               top_p: float = 0.95, seed: int | None = None,
                               prompt_ids=None,
                               frozen: torch.Tensor | None = None,
                               remask_ratio: float = 0.0,
                               return_trajectory: bool = False,
                               order: str = "confidence",
                               constraints: dict | None = None,
                               image_embeds: torch.Tensor | None = None,
                               allow_frozen_masks: bool = False,
                               softmask_k: int = 0,
                               device="cpu"):
    """Same algorithm as sample_trajectory, but also exposes the
    DenoisingState so callers can pause, snapshot, pin, or otherwise
    intervene mid-process.

    Returns (ids, confidences, state), or
    (ids, confidences, trajectory, state) if return_trajectory=True.
    Deterministic given `seed`.
    """
    vocab_size = getattr(getattr(model, "cfg", None), "vocab_size", None)
    st = DenoisingState(
        length=length, mask_id=mask_id, eos_id=eos_id, steps=steps,
        temperature=temperature, top_p=top_p, seed=seed,
        prompt_ids=prompt_ids, frozen=frozen, remask_ratio=remask_ratio,
        order=order, constraints=constraints, device=device,
        vocab_size=vocab_size, return_trajectory=return_trajectory,
        image_embeds=image_embeds, allow_frozen_masks=allow_frozen_masks,
        softmask_k=softmask_k)
    while st.step(model):
        pass
    out = (st.x.cpu(), st.conf.cpu())
    if return_trajectory:
        out += (torch.stack(st.trajectory).cpu(),)
    return out + (st,)


@torch.no_grad()
def generate_with_scratchpad(model, length: int, mask_id: int, eos_id: int,
                             scratch_len: int, steps: int = 16,
                             temperature: float = 1.0, top_p: float = 0.95,
                             seed: int | None = None, prompt_ids=None,
                             constraints: dict | None = None,
                             image_embeds: torch.Tensor | None = None,
                             device="cpu"):
    """Two-phase reasoning: denoise a scratchpad, freeze it, then denoise the
    answer conditioned on the full trace.

    This is the diffusion-native chain-of-thought. Unlike autoregressive
    models — where each reasoning token sees only the past — the answer
    phase attends BIDIRECTIONALLY to the entire finished reasoning trace,
    and the trace itself stays revisable afterwards (see self_refine).
    Layout: [prompt][scratchpad: scratch_len][answer: length - scratch_len].

    Constraints apply to the answer phase only (the phase whose output the
    caller presents). Deterministic given `seed` (phase 2 uses seed+1).

    Returns (final_ids, final_conf, scratchpad_ids).
    """
    if not 0 < scratch_len < length:
        raise ValueError(
            f"scratch_len must satisfy 0 < scratch_len < length, got {scratch_len}/{length}")
    plen = 0 if prompt_ids is None else len(prompt_ids)
    total = plen + length

    # Phase 1: denoise the scratchpad; the answer region stays frozen masks.
    p1 = torch.full((total,), mask_id, dtype=torch.long)
    if prompt_ids is not None:
        p1[:plen] = torch.as_tensor(prompt_ids, dtype=torch.long)
    frozen1 = torch.zeros(total, dtype=torch.bool)
    frozen1[:plen] = True
    frozen1[plen + scratch_len:] = True
    s_ids, _ = sample_trajectory(
        model, total, mask_id, eos_id, steps=steps, temperature=temperature,
        top_p=top_p, seed=seed, prompt_ids=p1, frozen=frozen1,
        image_embeds=image_embeds, allow_frozen_masks=True, device=device)

    # Phase 2: freeze prompt + finished scratchpad; denoise the answer.
    frozen2 = torch.zeros(total, dtype=torch.bool)
    frozen2[:plen + scratch_len] = True
    a_ids, a_conf = sample_trajectory(
        model, total, mask_id, eos_id, steps=steps, temperature=temperature,
        top_p=top_p, seed=None if seed is None else seed + 1,
        prompt_ids=s_ids, frozen=frozen2, constraints=constraints,
        image_embeds=image_embeds, device=device)
    return a_ids, a_conf, s_ids[plen:plen + scratch_len].clone()


@torch.no_grad()
def self_refine(model, length: int, mask_id: int, eos_id: int,
                steps: int = 16, refine_rounds: int = 2,
                remask_frac: float = 0.15, refine_steps: int = 8,
                temperature: float = 1.0, top_p: float = 0.95,
                seed: int | None = None, prompt_ids=None,
                frozen: torch.Tensor | None = None,
                constraints: dict | None = None,
                image_embeds: torch.Tensor | None = None,
                device="cpu"):
    """Diffusion-native self-correction for long-horizon reasoning.

    The core failure of long chains in autoregressive models is
    compounding error: once a reasoning step is emitted it is FIXED and
    every later step builds on it. Here, after an initial draft, the
    lowest-confidence spans are remasked and re-denoised in place with
    full bidirectional context — the model revises its own trace instead
    of being stuck with it.

    Invariants:
      * prompt prefix, caller-frozen positions, and required-id positions
        are NEVER remasked (the structural guarantees survive refinement).
      * forbidden_ids stay excluded from model sampling in every round.
      * output length is still exactly `length` tokens.
      * deterministic given `seed` (round r uses seed + 1001 + r).

    Returns (ids, confidences).
    """
    ids, conf = sample_trajectory(
        model, length, mask_id, eos_id, steps=steps, temperature=temperature,
        top_p=top_p, seed=seed, prompt_ids=prompt_ids, frozen=frozen,
        constraints=constraints, image_embeds=image_embeds, device=device)

    plen = 0 if prompt_ids is None else len(prompt_ids)
    lock = torch.zeros(length, dtype=torch.bool)
    lock[:plen] = True
    if frozen is not None:
        lock |= frozen.to(torch.bool)
    # Required ids were structurally guaranteed in the draft; keep them
    # out of remasking so the guarantee survives refinement.
    for t in (constraints or {}).get("required_ids", []):
        lock |= (ids == int(t))
    # Forbidden ids must stay excluded in every refine round too, as must
    # the v2 constraints (counts, caps, phrases, order).
    refine_constraints = None
    _forbidden = list((constraints or {}).get("forbidden_ids", []))
    _v2 = {}
    for k in ("required_counts", "at_most", "forbidden_phrases",
              "order_pairs"):
        v = (constraints or {}).get(k)
        if v:
            _v2[k] = v
    if _forbidden or _v2:
        refine_constraints = {"forbidden_ids": _forbidden, **_v2}

    for r in range(refine_rounds):
        if bool(lock.all()):
            break
        open_n = int((~lock).sum())
        k = max(1, min(open_n, int(remask_frac * open_n)))
        cand = conf.clone()
        cand[lock] = float("inf")          # never remask locked positions
        remask_pos = torch.argsort(cand)[:k]
        noisy = ids.clone()
        noisy[remask_pos] = mask_id
        fz = torch.ones(length, dtype=torch.bool)
        fz[remask_pos] = False             # denoise ONLY the remasked spans
        ids, conf = sample_trajectory(
            model, length, mask_id, eos_id, steps=refine_steps,
            temperature=temperature, top_p=top_p,
            seed=None if seed is None else seed + 1001 + r,
            prompt_ids=noisy, frozen=fz, constraints=refine_constraints,
            image_embeds=image_embeds, device=device)
    return ids, conf


@torch.no_grad()
def generate_long(model, total_length: int, chunk_length: int,
                  mask_id: int, eos_id: int, seq_len: int = 128,
                  steps: int = 16, temperature: float = 1.0,
                  top_p: float = 0.95, seed: int | None = None,
                  device="cpu"):
    """EXPERIMENTAL: extend generation past the training window by chaining.

    Generates `chunk_length` tokens at a time; each new chunk is denoised
    with the trailing (seq_len - chunk_length) tokens frozen as context,
    so every forward pass stays inside the trained window. Within-window
    behavior is identical to sample_trajectory; CROSS-chunk coherence is
    NOT guaranteed by training (the model only ever saw seq_len windows)
    — validate per use-case. No constraints (required_ids per chunk would
    be meaningless). Deterministic given `seed` (chunk c uses seed + c).

    Returns (ids, confidences), each of exactly total_length.
    """
    if not 0 < chunk_length < seq_len:
        raise ValueError(
            f"need 0 < chunk_length < seq_len, got {chunk_length}/{seq_len}")
    out, conf_out = [], []
    context = None
    c = 0
    remaining = total_length
    while remaining > 0:
        cur = min(chunk_length, remaining)
        ctx_len = 0 if context is None else len(context)
        canvas_len = ctx_len + cur
        p = torch.full((canvas_len,), mask_id, dtype=torch.long)
        fz = torch.zeros(canvas_len, dtype=torch.bool)
        if context is not None:
            p[:ctx_len] = context
            fz[:ctx_len] = True
        ids, cf = sample_trajectory(
            model, canvas_len, mask_id, eos_id, steps=steps,
            temperature=temperature, top_p=top_p,
            seed=None if seed is None else seed + c,
            prompt_ids=p, frozen=fz, device=device)
        out.append(ids[ctx_len:])
        conf_out.append(cf[ctx_len:])
        keep = seq_len - chunk_length
        context = ids[max(0, canvas_len - keep):].clone()
        remaining -= cur
        c += 1
    return torch.cat(out), torch.cat(conf_out)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    keep = cum <= top_p
    keep[..., 0] = True  # always keep the top token
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_idx, torch.where(keep, sorted_logits,
                                                  float("-inf")))
    return filtered


@torch.no_grad()
@torch.no_grad()
def evaluate_nll_bound(model, batches, mask_id: int,
                       t_min: float = 0.15, t_max: float = 1.0, device="cpu",
                       generator=None):
    """Held-out masked-LM negative log-likelihood (nats/token, an UPPER BOUND
    on the true sequence NLL — reported as a bound, not as perplexity).

    Pass a fixed-seed ``generator`` to use identical corruption masks at
    every validation: otherwise val-to-val comparisons measure mask luck,
    not model improvement (the bound's variance across mask samples is
    ~0.5 nats — larger than real step-to-step gains).
    """
    model.eval()
    total_ll, total_n = 0.0, 0
    for ids in batches:
        ids = ids.to(device)
        noisy, targets, mask = mask_batch(ids, mask_id, t_min, t_max,
                                          generator=generator)
        logits = model(noisy)
        nll = F.cross_entropy(logits[mask], targets[mask], reduction="sum")
        total_ll += nll.item()
        total_n += int(mask.sum())
    return total_ll / max(total_n, 1)
