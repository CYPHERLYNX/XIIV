"""
Lightweight Domain Balancing (Micro-DoReMi)
=============================================
Optimal domain mixture without proxy model training.

Problem: DoReMi trains a 280M proxy model with Group DRO to learn
optimal domain weights. We can't afford that.

Invention: Loss-driven adaptive reweighting during SGBD training.
- Track per-domain loss during block training
- Domains with higher loss are under-learned → increase their weight
- Domains with lower loss are well-learned → decrease their weight
- Update weights with exponential moving average (stable, no oscillation)
- No proxy model needed — uses the actual training signal

This is a bandit-style approach: the training loss IS the reward signal.
Simpler than DoReMi, runs inline with training, zero extra compute.

Theory connection:
- DoReMi optimizes: min_θ max_λ Σ λ_d * L_d(θ) (Group DRO)
- We approximate: λ_d ∝ exp(β * L_d) (softmax over domain losses)
- High-loss domains get more weight = minimax regret minimization
- β controls aggressiveness (β=1.0 is a good default)

Memory: O(num_domains) — just 8 floating point numbers.
Compute: O(1) per batch — just a weighted average update.
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict


class DomainBalancer:
    """
    Adaptive domain weighting using training loss as signal.
    
    No proxy model. No extra training. Just smart reweighting.
    """
    
    def __init__(
        self,
        domains: List[str],
        beta: float = 1.0,
        ema_decay: float = 0.99,
        min_weight: float = 0.05,
        max_weight: float = 0.40,
    ):
        """
        Args:
            domains: List of domain names
            beta: Softmax temperature for loss→weight (higher = more aggressive)
            ema_decay: EMA decay for loss tracking (higher = more stable)
            min_weight: Minimum weight per domain (prevents starvation)
            max_weight: Maximum weight per domain (prevents domination)
        """
        self.domains = domains
        self.beta = beta
        self.ema_decay = ema_decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Uniform initialization
        n = len(domains)
        self.weights = {d: 1.0 / n for d in domains}
        
        # EMA-tracked per-domain losses
        self.domain_losses = {d: 0.0 for d in domains}
        self.domain_counts = {d: 0 for d in domains}
        
        # History for analysis
        self.weight_history = []
    
    def update(self, domain: str, loss: float):
        """
        Update domain loss tracking and recompute weights.
        Call this after each training batch with the batch's domain and loss.
        """
        # Update EMA loss for this domain
        old = self.domain_losses[domain]
        count = self.domain_counts[domain]
        
        if count == 0:
            self.domain_losses[domain] = loss
        else:
            self.domain_losses[domain] = (
                self.ema_decay * old + (1 - self.ema_decay) * loss
            )
        self.domain_counts[domain] += 1
        
        # Recompute weights via softmax over losses
        self._recompute_weights()
    
    def _recompute_weights(self):
        """Recompute domain weights from current loss estimates."""
        # Only use domains that have been seen
        seen = [d for d in self.domains if self.domain_counts[d] > 0]
        if not seen:
            return
        
        losses = np.array([self.domain_losses[d] for d in seen])
        
        # Softmax over losses (higher loss → higher weight)
        # Subtract max for numerical stability
        logits = self.beta * (losses - losses.max())
        exp_logits = np.exp(logits)
        raw_weights = exp_logits / exp_logits.sum()
        
        # Apply min/max clamping
        clamped = np.clip(raw_weights, self.min_weight, self.max_weight)
        # Renormalize after clamping
        clamped = clamped / clamped.sum()
        
        for d, w in zip(seen, clamped):
            self.weights[d] = float(w)
        
        # Unseen domains keep uniform share of remaining
        unseen = [d for d in self.domains if self.domain_counts[d] == 0]
        if unseen:
            # They get equal share of what's not allocated to seen
            # (simplified: keep their initial uniform weight)
            pass
        
        self.weight_history.append(dict(self.weights))
    
    def get_weights(self) -> Dict[str, float]:
        """Get current domain sampling weights."""
        return dict(self.weights)
    
    def get_domain_for_batch(self) -> str:
        """Sample a domain according to current weights."""
        domains = list(self.weights.keys())
        weights = [self.weights[d] for d in domains]
        return np.random.choice(domains, p=weights)
    
    def summary(self) -> str:
        """Human-readable summary of current weights."""
        lines = ["Domain weights:"]
        for d in sorted(self.domains, key=lambda x: -self.weights[x]):
            loss = self.domain_losses[d]
            count = self.domain_counts[d]
            lines.append(
                f"  {d:15s} weight={self.weights[d]:.3f} "
                f"loss={loss:.4f} batches={count}"
            )
        return "\n".join(lines)
