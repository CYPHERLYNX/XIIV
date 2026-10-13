"""
Cascade Block Distillation — Proof of Concept
==============================================
Demonstrates the core mechanism on small models:
1. Train a small "teacher" model
2. Create a larger "student" by growing (duplicating layers)
3. Train student blocks independently using cached teacher activations
4. Show student improves without ever loading full student with gradients

This validates the MECHANISM. Scaling to 1B is an engineering step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TinyBlock(nn.Module):
    """A single transformer-like block for PoC."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyModel(nn.Module):
    """Stack of blocks."""
    def __init__(self, dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def train_teacher():
    """Train a small teacher model on a synthetic task."""
    print("Phase 0: Training teacher (4 layers, dim=64)...")
    torch.manual_seed(42)
    teacher = TinyModel(dim=64, n_layers=4)

    # Synthetic task: learn a fixed random projection
    target_proj = torch.randn(64, 64)

    opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    for step in range(500):
        x = torch.randn(32, 16, 64)  # batch, seq, dim
        out = teacher(x)
        # Target: teacher should learn to apply target_proj
        target = x @ target_proj
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    with torch.no_grad():
        x = torch.randn(32, 16, 64)
        final_loss = F.mse_loss(teacher(x), x @ target_proj).item()
    print(f"  Teacher final loss: {final_loss:.4f}")
    return teacher, target_proj


def grow_student(teacher, growth_factor=2):
    """Grow student by duplicating teacher layers."""
    print(f"\nGrowing student ({growth_factor}x layers)...")
    teacher_layers = len(teacher.layers)
    student_layers = teacher_layers * growth_factor

    student = TinyModel(dim=64, n_layers=student_layers)

    # Initialize: copy teacher layers, duplicate for new ones
    with torch.no_grad():
        for i in range(student_layers):
            teacher_idx = i % teacher_layers
            student.layers[i].load_state_dict(
                teacher.layers[teacher_idx].state_dict()
            )
        student.head.load_state_dict(teacher.head.state_dict())

    print(f"  Student: {student_layers} layers "
          f"({sum(p.numel() for p in student.parameters())} params)")
    return student


def cascade_block_distill(teacher, student, target_proj, n_blocks=4, steps_per_block=200):
    """
    Core CBD algorithm:
    1. Capture teacher activations (inference only)
    2. Divide student into blocks, train each independently
    3. Each block only needs its own params + grads + optimizer in memory
    """
    print(f"\nPhase 1: Capturing teacher activations...")
    teacher.eval()

    # Capture: for each teacher layer, save (input, output)
    teacher_cache = []
    x_sample = torch.randn(64, 16, 64)
    with torch.no_grad():
        x = x_sample.clone()
        for layer in teacher.layers:
            x_in = x.clone()
            x = layer(x)
            teacher_cache.append((x_in, x.clone()))

    print(f"  Captured {len(teacher_cache)} layer activations")

    print(f"\nPhase 2: Training {n_blocks} student blocks independently...")
    student_layers = len(student.layers)
    layers_per_block = student_layers // n_blocks

    # Map student blocks to teacher layers
    # Student has 2x layers, so each teacher layer maps to 2 student layers
    # Each block trains to match the teacher's transformation

    for b in range(n_blocks):
        start = b * layers_per_block
        end = start + layers_per_block
        block_layers = list(student.layers[start:end])
        block_params = [p for layer in block_layers for p in layer.parameters()]

        # Only this block's params get gradients — rest are frozen/detached
        # In a real implementation, other blocks wouldn't even be in memory
        n_block_params = sum(p.numel() for p in block_params)
        print(f"\n  Block {b+1}/{n_blocks}: layers {start}-{end-1} "
              f"({n_block_params} params in memory)")

        # Teacher target for this block: which teacher layers correspond?
        # Student layers [start:end] correspond to teacher layers
        # [start//2 : end//2] (since student is 2x)
        t_start = start // 2
        t_end = end // 2

        opt = torch.optim.AdamW(block_params, lr=1e-3)

        for step in range(steps_per_block):
            # Get input: either from previous block's output or teacher cache
            # For PoC: use teacher's input to corresponding layer
            x_in, _ = teacher_cache[t_start]
            # Target: teacher's output after corresponding layers
            _, x_target = teacher_cache[t_end - 1]

            # Forward through ONLY this block
            x_out = x_in.clone()
            for layer in block_layers:
                x_out = layer(x_out)

            loss = F.mse_loss(x_out, x_target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f"    step {step}: loss={loss.item():.4f}")

    print("\nPhase 3: Evaluating distilled student...")
    student.eval()
    with torch.no_grad():
        x = torch.randn(32, 16, 64)
        student_loss = F.mse_loss(student(x), x @ target_proj).item()

    # Compare to student WITHOUT distillation (just grown init)
    student_untrained = grow_student(teacher, growth_factor=2)
    student_untrained.eval()
    with torch.no_grad():
        x = torch.randn(32, 16, 64)
        init_loss = F.mse_loss(student_untrained(x), x @ target_proj).item()

    print(f"\n  Grown init (no CBD):     loss = {init_loss:.4f}")
    print(f"  After CBD:               loss = {student_loss:.4f}")
    print(f"  Teacher reference:       loss = {F.mse_loss(teacher(torch.randn(32,16,64)), torch.randn(32,16,64) @ target_proj).item():.4f}")

    improvement = (init_loss - student_loss) / init_loss * 100
    print(f"\n  Improvement from CBD: {improvement:.1f}%")
    return student_loss < init_loss


if __name__ == '__main__':
    print("=" * 60)
    print("Cascade Block Distillation — Proof of Concept")
    print("=" * 60)

    teacher, target_proj = train_teacher()
    student = grow_student(teacher, growth_factor=2)
    success = cascade_block_distill(teacher, student, target_proj)

    print("\n" + "=" * 60)
    if success:
        print("✓ PoC SUCCESS: Block-wise distillation improved the student")
        print("  without ever loading the full student with gradients.")
    else:
        print("✗ PoC FAILED: Student did not improve")
    print("=" * 60)
