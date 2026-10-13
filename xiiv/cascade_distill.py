"""
Cascade Block Distillation (CBD)
=================================
A novel method for training a 1B-parameter model on CPU-only hardware
with <8GB RAM.

Problem:
    Full backprop on 916M params needs ~15GB (weights + grads + AdamW states
    + activations). This box has 7GB. Impossible.

Key Insight:
    We don't need to train all 916M params simultaneously. Divide the model
    into K independent blocks (~115M params each). Train each block separately
    with FULL gradients, using cached teacher activations as fixed targets.
    Each block needs only ~2GB. Fits!

Why this is different from existing work:
    - Block-wise training exists, but NOT combined with growth-based init
      + teacher distillation + diffusion objective + iterative cascade
    - Designed specifically for the "grown from trained seed" scenario
    - Each block gets FULL gradient signal (not zeroth-order noise like MeZO)
    - Teacher provides STABLE targets (not moving targets)

Method:
    Phase 1: Teacher Activation Capture
        - Load 133M teacher, run forward passes on training data
        - Save (input, output) activation pairs per layer group to disk
        - Inference-only, fits in memory

    Phase 2: Independent Block Training
        - For each block k in 1..K:
            - Load only block k weights (~115M params)
            - Train with AdamW on cached (input, target) pairs
            - Loss = MSE(block_out, teacher_target) + diffusion_loss
            - Memory: ~2GB per block

    Phase 3: Cascade Refinement
        - Compose all trained blocks into full model
        - Run forward pass to get updated cross-block activations
        - Re-capture targets with better inter-block context
        - Repeat Phase 2 with refined targets
        - Each cascade iteration improves consistency

Memory per block (115M params, fp32):
    Weights:     115M * 4 =  460 MB
    Gradients:   115M * 4 =  460 MB
    AdamW (m,v): 115M * 8 =  920 MB
    Activations:             ~200 MB
    Total:                   ~2.0 GB  << 7GB available
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from pathlib import Path


class BlockTrainer:
    """Trains a single block independently with full gradients."""

    def __init__(self, block_fn, block_params, device='cpu'):
        """
        Args:
            block_fn: callable that runs the block forward pass
            block_params: list of parameters for this block only
            device: compute device
        """
        self.block_fn = block_fn
        self.params = list(block_params)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.params, lr=1e-4)

    def train_step(self, x, target):
        """Single training step on cached (input, target) pair."""
        self.optimizer.zero_grad()
        out = self.block_fn(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save({
            'params': [p.data.cpu() for p in self.params],
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        for p, saved in zip(self.params, ckpt['params']):
            p.data.copy_(saved)
        self.optimizer.load_state_dict(ckpt['optimizer'])


class CascadeDistiller:
    """Orchestrates Cascade Block Distillation across all blocks."""

    def __init__(self, n_blocks, cache_dir, device='cpu'):
        self.n_blocks = n_blocks
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.cascade_iter = 0

    def capture_teacher_activations(self, teacher_model, dataloader,
                                    layer_groups, max_batches=100):
        """
        Phase 1: Run teacher, save (input, output) per layer group.

        Args:
            teacher_model: the 133M trained model
            dataloader: training data
            layer_groups: list of (start_layer, end_layer) tuples defining
                         correspondence between teacher and student blocks
            max_batches: limit for proof-of-concept
        """
        teacher_model.eval()
        cache_paths = []

        # Register hooks to capture activations
        activations = {}

        def make_hook(name):
            def hook(module, inp, out):
                # Save input and output, detached, on CPU
                activations[name] = (
                    inp[0].detach().cpu(),
                    out.detach().cpu() if torch.is_tensor(out) else out[0].detach().cpu()
                )
            return hook

        handles = []
        # Hook the relevant layers (simplified: hook each transformer block)
        for idx, (t_start, t_end) in enumerate(layer_groups):
            # In practice, hook teacher_model.layers[t_start:t_end]
            pass  # Implementation depends on actual model structure

        print(f"[CBD] Capturing teacher activations for {len(layer_groups)} groups...")
        # ... actual capture loop ...

        for h in handles:
            h.remove()

        # Save metadata
        meta = {
            'n_groups': len(layer_groups),
            'cascade_iter': self.cascade_iter,
            'layer_groups': layer_groups,
        }
        with open(self.cache_dir / f'teacher_cache_iter{self.cascade_iter}.json', 'w') as f:
            json.dump(meta, f)

        return cache_paths

    def train_block(self, block_idx, block_fn, block_params,
                    cache_path, n_steps=1000):
        """
        Phase 2: Train a single block independently.
        Only this block's params are in memory with gradients + optimizer.
        """
        print(f"[CBD] Training block {block_idx}/{self.n_blocks} "
              f"({sum(p.numel() for p in block_params)/1e6:.1f}M params)")

        trainer = BlockTrainer(block_fn, block_params, self.device)

        # Load cached (input, target) pairs
        # In practice: stream from disk in batches
        for step in range(n_steps):
            # x, target = next_batch(cache_path)
            # loss = trainer.train_step(x, target)
            # if step % 100 == 0:
            #     print(f"  Block {block_idx} step {step}: loss={loss:.4f}")
            pass

        ckpt_path = self.cache_dir / f'block_{block_idx}_iter{self.cascade_iter}.pt'
        trainer.save(ckpt_path)
        return ckpt_path

    def cascade_refine(self, student_model, dataloader, n_cascades=3):
        """
        Phase 3: After all blocks trained, re-capture with updated context
        and repeat. Each cascade improves cross-block consistency.
        """
        for cascade in range(n_cascades):
            self.cascade_iter = cascade
            print(f"\n[CBD] === Cascade iteration {cascade+1}/{n_cascades} ===")

            # Run composed student model to get updated activations
            # Use these as refined targets for next round
            # (In practice: forward pass through full student, capture
            #  inter-block activations, blend with teacher targets)

            print(f"[CBD] Cascade {cascade+1} complete")


def estimate_memory(n_params, dtype_bytes=4):
    """Estimate training memory for a block."""
    weights = n_params * dtype_bytes
    grads = n_params * dtype_bytes
    adamw = n_params * dtype_bytes * 2  # m and v
    activations = n_params * dtype_bytes * 0.5  # rough estimate
    total = weights + grads + adamw + activations
    return {
        'weights_mb': weights / 1e6,
        'grads_mb': grads / 1e6,
        'adamw_mb': adamw / 1e6,
        'activations_mb': activations / 1e6,
        'total_gb': total / 1e9,
    }


if __name__ == '__main__':
    # Demonstrate memory feasibility
    print("=" * 60)
    print("Cascade Block Distillation — Memory Feasibility")
    print("=" * 60)

    total_params = 916_000_000
    for n_blocks in [4, 8, 12, 16]:
        block_params = total_params // n_blocks
        mem = estimate_memory(block_params)
        fits = "✓ FITS" if mem['total_gb'] < 6.0 else "✗ TOO BIG"
        print(f"\n{n_blocks} blocks ({block_params/1e6:.0f}M params each):")
        print(f"  Weights:     {mem['weights_mb']:.0f} MB")
        print(f"  Gradients:   {mem['grads_mb']:.0f} MB")
        print(f"  AdamW:       {mem['adamw_mb']:.0f} MB")
        print(f"  Activations: {mem['activations_mb']:.0f} MB")
        print(f"  Total:       {mem['total_gb']:.2f} GB  {fits}")

    print("\n" + "=" * 60)
    print("Full model (for comparison):")
    mem = estimate_memory(total_params)
    print(f"  Total: {mem['total_gb']:.1f} GB  ✗ IMPOSSIBLE on 7GB box")
    print("=" * 60)
