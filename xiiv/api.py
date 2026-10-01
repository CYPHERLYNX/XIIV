"""XIIV Python API: generate and infill in a few lines.

Example:
    from xiiv import XIIV

    m = XIIV("weights/xiiv.safetensors")
    print(m.infill("The <mask> sat on the <mask>.", seed=3))
    print(m.generate(length=128, steps=16, seed=7))

Zero network calls. Deterministic given ``seed``.
"""
import os

import torch

from .diffusion import sample_trajectory
from .sample import load_model
from .tokenizer import XIIVTokenizer


class XIIV:
    """A loaded XIIV masked-diffusion model.

    Args:
        weights: path to a ``.safetensors`` weight file. The sibling
            ``config.json`` and the tokenizer are resolved automatically:
            ``tokenizer`` defaults to ``weights/tokenizer.json``.
        tokenizer: path to the tokenizer JSON (default: next to weights).
        device: torch device, default ``"cpu"``.
    """

    def __init__(self, weights="weights/xiiv.safetensors",
                 tokenizer=None, device="cpu"):
        self.device = device
        tok_path = tokenizer or os.path.join(
            os.path.dirname(os.path.abspath(weights)), "tokenizer.json")
        self.tok = XIIVTokenizer(tok_path)
        self.model, self.cfg = load_model(weights, device=device)
        assert self.tok.mask_id == self.cfg.mask_id \
            and self.tok.eos_id == self.cfg.eos_id, \
            "tokenizer/weights special-token mismatch"

    def generate(self, length=128, prompt="", steps=16, temperature=1.0,
                 top_p=0.95, remask_ratio=0.0, seed=7):
        """Generate ``length`` tokens from noise (or from ``prompt``).

        Every generation starts as a blank field of ``[MASK]`` tokens and
        resolves the most confident positions in parallel over ``steps``
        denoising passes. Returns the decoded string.
        """
        prompt_ids = None
        if prompt:
            prompt_ids = torch.tensor(self.tok.encode(prompt),
                                      dtype=torch.long)
        ids, _conf = sample_trajectory(
            self.model, length, self.tok.mask_id, self.tok.eos_id,
            steps=steps, temperature=temperature, top_p=top_p, seed=seed,
            prompt_ids=prompt_ids, remask_ratio=remask_ratio,
            device=self.device)
        return self.tok.decode(ids.tolist())

    def infill(self, text, steps=16, temperature=1.0, top_p=0.95,
               remask_ratio=0.0, seed=7):
        """Fill ``<mask>`` / ``<mask:N>`` spans inside ``text``.

        Locked (unmasked) context is preserved exactly; only masked
        positions are denoised. Returns the decoded string.
        """
        ids, frozen = self.tok.encode_with_masks(text)
        length = len(ids)
        out_ids, _conf = sample_trajectory(
            self.model, length, self.tok.mask_id, self.tok.eos_id,
            steps=steps, temperature=temperature, top_p=top_p, seed=seed,
            prompt_ids=ids, frozen=frozen, remask_ratio=remask_ratio,
            device=self.device)
        # Belt-and-braces: the sampler never touches frozen positions,
        # but the contract is exact preservation.
        out_ids[frozen] = ids[frozen]
        return self.tok.decode(out_ids.tolist())

    @property
    def n_params(self):
        """Exact parameter count."""
        return sum(p.numel() for p in self.model.parameters())
