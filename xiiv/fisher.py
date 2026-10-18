"""
Fisher Information for Elastic Weight Consolidation
====================================================
Computes compressed Fisher Information Matrix diagonal for protecting
critical weights during sequential dataset training.

When training on dataset B after dataset A, we add a regularization term:
    L_ewc = λ/2 * Σ_i F_i * (θ_i - θ*_A,i)^2

Where F_i is the Fisher information for parameter i, and θ*_A are the
optimal parameters after training on A. This penalizes changes to weights
that were important for A.

Memory optimization: Instead of storing full Fisher (916M × 4 bytes = 3.6GB),
we store only for critical layers (attention + early/late blocks), compressed
to ~2GB total across all datasets.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import os


# Layers to track for Fisher (most critical for knowledge retention)
CRITICAL_LAYER_PATTERNS = [
    "attn",       # Attention layers (relational knowledge)
    "mlp",        # MLP layers (factual knowledge)
    "embed",      # Embeddings (vocabulary)
    "lm_head",    # Output head
]

# Skip these to save memory (less critical for forgetting)
SKIP_PATTERNS = [
    "norm",       # Layer norms adapt quickly anyway
    "bias",       # Biases are less critical
]


def should_track_param(name: str) -> bool:
    """Determine if a parameter should have Fisher computed."""
    name_lower = name.lower()
    
    # Skip non-critical
    for skip in SKIP_PATTERNS:
        if skip in name_lower:
            return False
    
    # Track critical
    for pattern in CRITICAL_LAYER_PATTERNS:
        if pattern in name_lower:
            return True
    
    # Default: track transformer block weights
    if "block" in name_lower or "layer" in name_lower:
        if "weight" in name_lower:
            return True
    
    return False


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader,
    num_samples: int = 1000,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher Information Matrix.
    
    F_i = E[(∂log p(x|θ)/∂θ_i)^2]
    
    Approximated by average squared gradients over samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader yielding (input_ids,) batches
        num_samples: Number of samples to use for estimation
        device: Device to compute on
    
    Returns:
        Dict mapping param name -> Fisher diagonal tensor
    """
    print(f"[fisher] Computing Fisher diagonal over {num_samples} samples...")
    
    model.eval()
    model.to(device)
    
    # Initialize Fisher accumulators
    fisher = {}
    for name, param in model.named_parameters():
        if should_track_param(name):
            fisher[name] = torch.zeros_like(param.data, dtype=torch.float32)
    
    print(f"[fisher] Tracking {len(fisher)} parameter tensors "
          f"({sum(f.numel() for f in fisher.values()) / 1e6:.1f}M params)")
    
    samples_processed = 0
    for batch in dataloader:
        if samples_processed >= num_samples:
            break
        
        input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
        input_ids = input_ids.to(device)
        
        model.zero_grad()
        
        # Forward
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Use model's own predictions as targets (empirical Fisher)
        # This avoids needing labels
        with torch.no_grad():
            targets = logits.argmax(dim=-1)
        
        # Compute log-prob of predicted tokens
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather log-prob of predicted token at each position
        selected_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        # Sum log-probs and backward
        loss = -selected_log_probs.sum()
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += (param.grad.data.cpu() ** 2)
        
        samples_processed += input_ids.size(0)
        
        if samples_processed % 200 == 0:
            print(f"[fisher] Processed {samples_processed}/{num_samples}")
    
    # Average
    for name in fisher:
        fisher[name] /= samples_processed
    
    total_size_mb = sum(f.numel() * 4 for f in fisher.values()) / 1e6
    print(f"[fisher] Done. Total size: {total_size_mb:.1f}MB")
    
    return fisher


def save_fisher(fisher: Dict[str, torch.Tensor], path: str, dataset_name: str):
    """Save Fisher diagonal to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "fisher": fisher,
        "dataset": dataset_name,
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[fisher] Saved to {path} ({size_mb:.1f}MB)")


def load_fisher(path: str) -> Dict[str, torch.Tensor]:
    """Load Fisher diagonal from disk."""
    data = torch.load(path, map_location="cpu")
    return data["fisher"]


def ewc_loss(
    model: nn.Module,
    fisher_list: List[Dict[str, torch.Tensor]],
    optimal_params_list: List[Dict[str, torch.Tensor]],
    lambda_ewc: float = 1000.0,
) -> torch.Tensor:
    """
    Compute EWC regularization loss.
    
    Args:
        model: Current model
        fisher_list: List of Fisher dicts, one per previous dataset
        optimal_params_list: List of optimal param dicts, one per previous dataset
        lambda_ewc: Regularization strength
    
    Returns:
        Scalar EWC loss tensor
    """
    loss = 0.0
    for fisher, opt_params in zip(fisher_list, optimal_params_list):
        for name, param in model.named_parameters():
            if name in fisher:
                fisher_diag = fisher[name].to(param.device)
                opt_param = opt_params[name].to(param.device)
                loss += (fisher_diag * (param - opt_param) ** 2).sum()
    
    return (lambda_ewc / 2) * loss


def save_optimal_params(model: nn.Module, path: str, dataset_name: str):
    """Save optimal parameters after training on a dataset."""
    opt_params = {}
    for name, param in model.named_parameters():
        if should_track_param(name):
            opt_params[name] = param.data.cpu().clone()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "params": opt_params,
        "dataset": dataset_name,
    }, path)
    print(f"[fisher] Saved optimal params to {path}")
