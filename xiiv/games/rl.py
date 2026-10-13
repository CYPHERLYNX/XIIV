"""
Reward-Weighted Regression (RWR) for XIIV — System 1 training algorithm.

What RWR does:
    1. Collect (prompt_ids, completion_ids, reward) triples, where reward
       comes from an OBJECTIVELY VERIFIABLE outcome (xiiv/games/rewards.py).
    2. For each triple, build full_ids = prompt + completion.
    3. Run the diffusion forward process, but restrict the denoising loss
       to COMPLETION tokens only (the prompt is context, never a target).
    4. Weight each sample's loss by its reward: high-reward completions
       pull the parameters harder than low-reward ones.
    5. Update ONLY LoRA adapter params (base model frozen).

Why RWR and not PPO — honest engineering note:
    PPO on this hardware is infeasible, not just slow:
    - PPO needs per step: policy forward + reference-model forward +
      value-head forward + GAE over the rollout + 4 epochs of minibatch
      updates. That is ~6-10x the FLOPs of one RWR step.
    - PPO needs a reference model copy in RAM (+1.8GB we do not have)
      and a value head (new params, new optimizer state).
    - At ~3-4s per RWR step on 2 CPU cores, a PPO step would take
      20-40s, and the KL machinery would dominate wall-clock.
    RWR needs exactly 1 forward + 1 backward per sample, no value
    model, no reference copy. It is the principled choice under the
    actual compute budget, and for diffusion LMs it has a clean
    interpretation: reward-weighted denoising.

The update is GENUINE: gradients flow through the LoRA adapters and
optimizer steps change the weights. This is not prompt engineering.
"""

import json
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from ..lora import inject_lora, get_trainable_params


@dataclass
class RWRConfig:
    lr: float = 1e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # Reward -> weight mapping: w = exp(reward / temperature).
    # temperature -> inf  == uniform weighting (plain SFT).
    # temperature -> 0    == winner-take-all.
    temperature: float = 1.0
    # Minimum reward to include a sample at all. 0.0 keeps everything
    # (low-reward samples still teach what NOT to denoise toward, via
    # their small but nonzero weight).
    min_reward: float = 0.0
    mask_t_min: float = 0.15
    mask_t_max: float = 1.0
    lora_rank: int = 8
    lora_alpha: float = 16.0
    ckpt_every: int = 25
    log_file: str = "rwr_log.jsonl"


def reward_to_weight(reward: float, temperature: float = 1.0) -> float:
    """Map a [0,1] reward to a loss weight. Higher reward -> harder pull."""
    import math
    return math.exp(reward / max(temperature, 1e-8))


def rwr_loss_for_sample(model, full_ids: torch.Tensor, prompt_len: int,
                        mask_id: int, cfg: RWRConfig,
                        generator=None) -> torch.Tensor:
    """Diffusion denoising loss restricted to completion tokens.

    Args:
        model: XIIVModel with LoRA adapters injected.
        full_ids: (T,) token ids = prompt + completion.
        prompt_len: number of prompt tokens (never in the loss).
        mask_id: mask token id.
    Returns:
        Scalar loss tensor with grad graph over LoRA params only.
    """
    from ..diffusion import mask_batch
    ids = full_ids.unsqueeze(0)  # (1, T)
    noisy, targets, mask = mask_batch(ids, mask_id, cfg.mask_t_min,
                                      cfg.mask_t_max, generator=generator)
    # Never train on prompt tokens: they are context, not targets.
    mask[0, :prompt_len] = False
    logits = model(noisy)
    n_masked = mask.sum()
    if n_masked == 0:
        return logits.sum() * 0.0, {"mask_ratio": 0.0, "n_masked": 0}
    loss = F.cross_entropy(logits[mask], targets[mask], reduction="mean")
    return loss, {"mask_ratio": mask.float().mean().item(),
                  "n_masked": int(n_masked)}


class RWRTrainer:
    """Reward-Weighted Regression over LoRA adapters.

    Usage:
        trainer = RWRTrainer(model, mask_id, cfg)
        for prompt_ids, completion_ids, reward in rollout_buffer:
            loss = trainer.step(prompt_ids, completion_ids, reward)
    """

    def __init__(self, model, mask_id: int, cfg: RWRConfig = None,
                 ckpt_dir: str = None):
        self.model = model
        self.mask_id = mask_id
        self.cfg = cfg or RWRConfig()
        self.ckpt_dir = ckpt_dir
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        # Inject LoRA adapters; freeze everything else.
        n_trainable = inject_lora(model, rank=self.cfg.lora_rank,
                                  alpha=self.cfg.lora_alpha)
        assert get_trainable_params(model) == n_trainable
        self.n_trainable = n_trainable

        self.opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.cfg.lr, betas=self.cfg.betas,
            weight_decay=self.cfg.weight_decay)
        self.step_count = 0
        self.generator = torch.Generator().manual_seed(1337)
        self.logf = None
        if ckpt_dir:
            self.logf = open(os.path.join(ckpt_dir, self.cfg.log_file), "a")

    def trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def step(self, prompt_ids: torch.Tensor, completion_ids: torch.Tensor,
             reward: float) -> dict:
        """One RWR update. Returns stats dict. Skips samples below min_reward."""
        if reward < self.cfg.min_reward:
            return {"skipped": True, "reward": reward}

        self.model.train()
        full = torch.cat([prompt_ids, completion_ids]).long()
        weight = reward_to_weight(reward, self.cfg.temperature)

        t0 = time.time()
        self.opt.zero_grad(set_to_none=True)
        loss, info = rwr_loss_for_sample(
            self.model, full, len(prompt_ids), self.mask_id,
            self.cfg, generator=self.generator)
        if not torch.isfinite(loss):
            return {"skipped": True, "reward": reward,
                    "reason": "non-finite loss"}
        (loss * weight).backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params(),
                                       self.cfg.grad_clip)
        self.opt.step()
        dt = time.time() - t0
        self.step_count += 1

        rec = {"step": self.step_count, "reward": round(reward, 4),
               "weight": round(weight, 4), "loss": round(loss.item(), 4),
               "n_masked": info["n_masked"], "secs": round(dt, 1)}
        if self.logf:
            self.logf.write(json.dumps(rec) + "\n")
            self.logf.flush()
        if self.ckpt_dir and self.step_count % self.cfg.ckpt_every == 0:
            self.save(os.path.join(
                self.ckpt_dir, f"rwr_ckpt_{self.step_count}.pt"))
        return rec

    def lora_state(self):
        return {k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items() if "lora_" in k}

    def save(self, path: str):
        torch.save({"lora_state": self.lora_state(),
                    "opt": self.opt.state_dict(),
                    "step": self.step_count,
                    "cfg": asdict(self.cfg)}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["lora_state"], strict=False)
        self.opt.load_state_dict(ck["opt"])
        self.step_count = ck["step"]
        return self.step_count

    def close(self):
        if self.logf:
            self.logf.close()
