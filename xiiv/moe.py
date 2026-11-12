"""Sparse MoE expansion for XIIV (4B side work, target Dec 24).

Replaces each Block's SwiGLU FFN with a Mixture-of-Experts FFN:
N experts (each a SwiGLU), top-k routing, one router per layer.

Growth discipline (mirrors the SGBD/exact-1B identity gates):
  - Every expert is initialized as an EXACT COPY of the backbone FFN
    it replaces; the router is zero-initialized (uniform routing).
  - Day-0 invariant: the converted model produces character-identical
    outputs to the backbone. Verified by test_moe.py (identity gate).
  - Training: backbone (attention, norms, embeddings) FROZEN; only
    experts + routers train, one layer-group at a time. A Switch-style
    auxiliary load-balancing loss keeps experts from collapsing back
    onto the identity copy.

Sizing (see research/expansion_4b_moe.md): 8 experts/layer, top-2
routing. Headline ~4B total, ~1.59B active/token.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from xiiv.model import SwiGLU


class MoESwiGLU(nn.Module):
    def __init__(self, cfg, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(SwiGLU(cfg) for _ in range(n_experts))
        # Zero-init router -> uniform routing at step 0 (identity gate).
        self.router = nn.Linear(cfg.d_model, n_experts, bias=False)
        nn.init.zeros_(self.router.weight)

    @classmethod
    def from_backbone_ffn(cls, ffn: SwiGLU, cfg, n_experts: int = 8,
                          top_k: int = 2):
        """Identity init: every expert = exact copy of the backbone FFN."""
        moe = cls(cfg, n_experts=n_experts, top_k=top_k)
        for expert in moe.experts:
            expert.load_state_dict(copy.deepcopy(ffn.state_dict()))
        return moe

    def forward(self, x):
        # x: (B, T, D) — already normed by the Block.
        B, T, D = x.shape
        flat = x.reshape(-1, D)                       # (B*T, D)
        scores = F.softmax(self.router(flat), dim=-1)  # (B*T, N)
        top_scores, top_idx = scores.topk(self.top_k, dim=-1)
        top_scores = top_scores / top_scores.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        # Per-expert dispatch (simple loop; batch sizes here are small).
        for k in range(self.top_k):
            idx = top_idx[:, k]                        # (B*T,)
            w = top_scores[:, k].unsqueeze(-1)          # (B*T, 1)
            for e, expert in enumerate(self.experts):
                m = (idx == e)
                if m.any():
                    out[m] += w[m] * expert(flat[m])
        self.last_aux = self._load_balance_loss(scores, top_idx)
        return out.reshape(B, T, D)

    # NOTE: forward returns ONLY the output tensor (same contract as
    # SwiGLU) so Block.forward needs no changes — drop-in. The
    # auxiliary load-balancing loss is stashed on self.last_aux for
    # the training loop to collect via collect_aux_loss(model).

    def _load_balance_loss(self, scores, top_idx):
        # Switch-Transformer style: N * sum_i f_i * P_i, where f_i is the
        # fraction of tokens routed to expert i and P_i the mean router
        # probability for expert i. Equals 1.0 under perfectly uniform
        # routing; larger when imbalanced. (At init with a zero router,
        # top-k ties break deterministically, so expect > 1.)
        tokens = scores.shape[0]
        with torch.no_grad():
            routed = torch.zeros_like(scores)
            routed.scatter_(1, top_idx, 1.0)
            f_i = routed.float().mean(0)
        p_i = scores.mean(0)
        return self.n_experts * (f_i * p_i).sum()


def convert_to_moe(model, n_experts: int = 8, top_k: int = 2):
    """Swap every Block's mlp with an identity-initialized MoESwiGLU.

    Drop-in: MoESwiGLU.forward returns a plain tensor, so Block.forward
    and the sampler need NO changes. Returns the list of converted
    (layer_index, moe) pairs.
    """
    converted = []
    for i, block in enumerate(model.layers):
        moe = MoESwiGLU.from_backbone_ffn(block.mlp, model.cfg,
                                          n_experts=n_experts, top_k=top_k)
        block.mlp = moe
        converted.append((i, moe))
    return converted


def collect_aux_loss(model):
    """Sum the stashed load-balancing aux losses over all MoE layers."""
    total = None
    for block in model.layers:
        aux = getattr(block.mlp, "last_aux", None)
        if aux is not None:
            total = aux if total is None else total + aux
    return total


def freeze_backbone_except_moe(model):
    """Freeze everything except MoE experts + routers (expert-training
    phase). Returns the list of trainable parameters."""
    trainable = []
    for name, p in model.named_parameters():
        if ".mlp.experts." in name or ".mlp.router." in name:
            p.requires_grad = True
            trainable.append(p)
        else:
            p.requires_grad = False
    return trainable
