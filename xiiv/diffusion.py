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
    Occurrences already present in prompt/frozen context count. Missing ids
    are force-placed into reserved masked slots (see the slot-reservation
    invariant in DenoisingState.step).
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
               t_max: float = 1.0, generator=None):
    """Apply the diffusion forward process to a batch of token ids.

    Returns (noisy_ids, targets, mask) where mask marks corrupted positions.
    """
    B = ids.shape[0]
    t = torch.empty(B, device=ids.device).uniform_(t_min, t_max, generator=generator)
    rand = torch.rand(ids.shape, device=ids.device, generator=generator)
    mask = rand < t.view(B, *([1] * (ids.dim() - 1)))
    noisy = ids.clone()
    noisy[mask] = mask_id
    return noisy, ids, mask


def diffusion_loss(model, ids: torch.Tensor, mask_id: int,
                   t_min: float = 0.15, t_max: float = 1.0, generator=None,
                   image_embeds: torch.Tensor | None = None):
    """Mean cross-entropy over masked positions only.

    image_embeds: optional (B, N, d) visual prefix for multimodal training
    (substituted for the first N embeddings by the model; those positions
    should be excluded from masking by the caller).
    """
    noisy, targets, mask = mask_batch(ids, mask_id, t_min, t_max, generator)
    if image_embeds is not None:
        logits = model(noisy, image_embeds)
    else:
        logits = model(noisy)
    n_masked = mask.sum()
    if n_masked == 0:
        return logits.sum() * 0.0  # degenerate guard; keeps graph alive
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


def check_constraints(ids: torch.Tensor, constraints: dict | None,
                      mask_id: int) -> dict:
    """Verify hard constraints on a finished sequence. All structural
    guarantees below are enforced DURING unmasking (see DenoisingState),
    so this is a verification, not a filter."""
    c = constraints or {}
    required = c.get("required_ids", [])
    forbidden = c.get("forbidden_ids", [])
    anchors = c.get("anchors", {})
    ids = ids.cpu()
    return {
        "required_present": all(bool((ids == t).any()) for t in required),
        "forbidden_absent": not any(bool((ids == t).any()) for t in forbidden),
        "anchors_intact": all(ids[p] == t for p, t in anchors.items()
                              if 0 <= p < len(ids)),
        "complete": not bool((ids == mask_id).any()),
    }


class DenoisingState:
    """Full resumable state of one constrained denoising run.

    Captures everything needed to pause, inspect, intervene, and resume:
    the token canvas ``x``, the ``frozen`` mask, the ``masked`` set, the RNG
    ``generator`` state, ``steps_remaining``, the priority/order policy, the
    constraint sets, and per-position ``confidences``.

    Required-token guarantee — slot-reservation invariant:
      Let m_s = #masked positions and q_s = #missing required ids at the
      start of iteration s. INVARIANT: q_s <= m_s.
      * Base: __init__ raises unless m_0 >= len(required_ids) >= q_0.
      * Step (not final): normal top-k fill is capped at (m_s - q_s)
        positions, so at least q_s masked slots survive; force-placement
        adds none. missing only shrinks as tokens appear, so
        m_{s+1} = m_s - filled >= q_s >= q_{s+1}.
      * Final step (steps_remaining == 1): ALL missing ids are force-placed
        into the most-confident remaining masked positions (possible by the
        invariant), so q_end = 0.
      Manual ops (pin/unmask/remask) re-validate the invariant and raise
      instead of breaking it. Hence: every required id occurs >= 1 in the
      final canvas. QED.
    """

    def __init__(self, *, length: int, mask_id: int, eos_id: int,
                 steps: int = 16, temperature: float = 1.0,
                 top_p: float = 0.95, seed: int | None = None,
                 prompt_ids=None, frozen: torch.Tensor | None = None,
                 remask_ratio: float = 0.0, order: str = "confidence",
                 constraints: dict | None = None, device="cpu",
                 vocab_size: int | None = None,
                 return_trajectory: bool = False,
                 image_embeds: torch.Tensor | None = None,
                 allow_frozen_masks: bool = False):
        self.length = int(length)
        self.mask_id = int(mask_id)
        self.eos_id = int(eos_id)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.order = order
        self.remask_ratio = float(remask_ratio)
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
        if vocab_size is not None:
            for t in self.forbidden_ids + self.required_ids:
                if not (0 <= t < vocab_size):
                    raise ValueError(
                        f"constraint token id {t} out of vocabulary "
                        f"range [0, {vocab_size})")
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
        if self.required_ids and int(masked0.sum()) < len(self.required_ids):
            raise ValueError(
                f"unsatisfiable: {len(self.required_ids)} required tokens "
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

    # -- core iteration -------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.steps_remaining <= 0 or not bool(self.masked.any())

    @torch.no_grad()
    def step(self, model) -> bool:
        """Run one denoise iteration. Returns True if more work remains."""
        if self.finished:
            return False
        model.eval()
        xb = self.x.unsqueeze(0)                              # (1, L)
        if self.image_embeds is not None:
            logits = model(xb, self.image_embeds).squeeze(0)   # (L, V)
        else:
            logits = model(xb).squeeze(0)                     # (L, V)
        if self.forbidden_ids:
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

        # required ids absent from the whole canvas (prompt/frozen count).
        missing = [t for t in self.required_ids if not (self.x == t).any()]

        # ENDGAME (invariant: len(missing) <= #masked, proved in class
        # docstring): on the final step, force-place ALL missing ids into
        # the most-confident remaining masked positions.
        if missing and self.steps_remaining == 1:
            mconf = torch.where(self.masked, conf,
                                torch.full_like(conf, -1.0))
            _, fidx = torch.topk(mconf, len(missing))
            for pos_i, tok_i in zip(fidx.tolist(), missing):
                self.x[pos_i] = tok_i
                self.conf[pos_i] = 1.0  # placed deterministically
            self.masked = (self.x == self.mask_id) & ~self.frozen
            missing = []

        # normal parallel fill, reserving one masked slot per missing id.
        m = int(self.masked.sum())
        k = 0
        if m > 0:
            k = max(1, math.ceil(m / self.steps_remaining))
            if missing:
                # SLOT RESERVATION: never consume a slot a missing required
                # token still needs. This is what keeps q_s <= m_s.
                k = min(k, max(0, m - len(missing)))
        if k > 0:
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

    # -- mid-process intervention ----------------------------------------
    def _check_slot_invariant(self, x, masked, context: str):
        missing = sum(1 for t in self.required_ids if not (x == t).any())
        if missing > int(masked.sum()):
            raise ValueError(
                f"{context} would break the required-token slot invariant "
                f"({missing} missing required ids, "
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
            "anchors": dict(self.anchors),
            "trajectory": ([t.cpu().clone() for t in self.trajectory]
                           if self.trajectory is not None else None),
            "image_embeds": (self.image_embeds.cpu().clone()
                             if self.image_embeds is not None else None),
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
        image_embeds=image_embeds, allow_frozen_masks=allow_frozen_masks)
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
        image_embeds=image_embeds, allow_frozen_masks=allow_frozen_masks)
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
    # Forbidden ids must stay excluded in every refine round too.
    refine_constraints = None
    _forbidden = list((constraints or {}).get("forbidden_ids", []))
    if _forbidden:
        refine_constraints = {"forbidden_ids": _forbidden}

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
def evaluate_nll_bound(model, batches, mask_id: int,
                       t_min: float = 0.15, t_max: float = 1.0, device="cpu"):
    """Held-out masked-LM negative log-likelihood (nats/token, an UPPER BOUND
    on the true sequence NLL — reported as a bound, not as perplexity)."""
    model.eval()
    total_ll, total_n = 0.0, 0
    for ids in batches:
        ids = ids.to(device)
        noisy, targets, mask = mask_batch(ids, mask_id, t_min, t_max)
        logits = model(noisy)
        nll = F.cross_entropy(logits[mask], targets[mask], reduction="sum")
        total_ll += nll.item()
        total_n += int(mask.sum())
    return total_ll / max(total_n, 1)
