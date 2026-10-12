"""lm-eval wrapper for XIIV's masked-diffusion architecture.

WHY THIS EXISTS: stock lm-eval scores multiple-choice options with
autoregressive teacher-forcing (next-token logprobs). XIIV is a masked
diffusion LM — it has no next-token distribution. Scoring our model with
the stock harness would produce numbers that are not just wrong but
meaninglessly wrong.

THE METHOD (xiiv/diffusion_scorer.py): score each (context, continuation)
by leave-one-out diffusion pseudo-likelihood — each continuation token is
masked individually with all other tokens visible; mean NLL. Lower = more
likely. Deterministic, no random masking samples.

METHODOLOGY CAVEAT: diffusion NLL is NOT comparable to AR logprobs
(bidirectional attention sees future continuation tokens). Rankings within
diffusion models are valid; cross-architecture numbers are directional only.
See COMPARISON_TABLE.md and xiiv/diffusion_scorer.py.

Usage (on the GPU box):
    lm_eval --model xiiv_diffusion --model_args weights=checkpoints_500m_gn/ckpt_8218.pt,tokenizer=weights/tokenizer_32k.json \\
        --tasks arc_easy,hellaswag --device cuda --batch_size 4
"""

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.tokenizer import XIIVTokenizer
from xiiv.diffusion_scorer import score_continuation


@register_model("xiiv_diffusion")
class XIIVDiffusionLM(LM):
    """lm-eval interface over a masked-diffusion LM via diffusion-NLL scoring."""

    def __init__(self, weights, tokenizer, device="cuda", n_mask_samples=4,
                 seed=0, max_length=512, batch_size=4, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self._device = torch.device(device)
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        cfg = ModelConfig.from_json(ck["config"])
        self.cfg = cfg
        self.model = XIIVModel(cfg)
        self.model.load_state_dict(ck["model"], strict=False)
        self.model.to(self.device).eval()
        self.tok = XIIVTokenizer(tokenizer)
        assert self.tok.vocab_size == cfg.vocab_size, \
            "tokenizer/model vocab mismatch"
        self.n_mask_samples = n_mask_samples
        self.seed = seed
        self.max_length = max_length

    def _score(self, context_ids, cont_ids):
        """Leave-one-out diffusion NLL of continuation given context.

        Masks ONLY continuation positions (one at a time); context is
        never masked. Deterministic. See xiiv/diffusion_scorer.py.
        """
        nll, n = score_continuation(
            self.model, context_ids, cont_ids,
            self.cfg.mask_id, device=self._device)
        # empty continuation: fall back to 0 (never outranks real scores)
        return nll if n > 0 else 0.0

    def loglikelihood(self, requests):
        # requests: list of Instance objects; .args = (context, continuation)
        out = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok.encode(context)
            cont_ids = self.tok.encode(continuation)
            nll = self._score(ctx_ids, cont_ids)
            out.append((-nll, False))
        return out

    def loglikelihood_rolling(self, requests):
        # Rolling NLL over full strings (for perplexity-style tasks).
        out = []
        for (text,) in requests:
            ids = self.tok.encode(text)
            nll = self._score([], ids)
            out.append(-nll)
        return out

    def generate_until(self, requests):
        raise NotImplementedError(
            "generate_until not implemented for the diffusion wrapper; "
            "use xiiv.sample for generation tasks.")
