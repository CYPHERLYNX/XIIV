"""XIIV model configuration."""
from dataclasses import dataclass, asdict
import json


@dataclass
class ModelConfig:
    # --- vocabulary ---
    # NOTE: mask_id/eos_id MUST match the trained tokenizer (see
    # meta.json written by scripts/prepare_data.py). The BPE trainer
    # assigns special-token ids, so never rely on these defaults —
    # train.py always overrides them from meta.json, and sample.py
    # asserts tokenizer/config agreement at load time.
    vocab_size: int = 8194        # placeholder; overridden from meta.json
    mask_id: int = 8192           # placeholder; overridden from meta.json
    eos_id: int = 8193            # placeholder; overridden from meta.json

    # --- architecture: bidirectional transformer (no causal mask) ---
    d_model: int = 320
    n_layers: int = 6
    n_heads: int = 5              # head_dim = 64
    ffn_dim: int = 864            # SwiGLU intermediate dim
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True   # LM head shares the token embedding matrix

    # --- diffusion training ---
    seq_len: int = 128
    t_min: float = 0.15           # mask-ratio sampled ~ Uniform(t_min, 1.0)
    t_max: float = 1.0

    # --- multimodal scaffold (architecture only; untrained unless enabled) ---
    # When image_enabled, the model prepends n_image_tokens visual prefix
    # positions whose embeddings are SUBSTITUTED by PatchEmbed output
    # (see XIIVModel.forward). The diffusion sampler treats the visual
    # prefix as an ordinary frozen span: multimodal generation is just
    # infilling conditioned on a frozen image prefix — no sampler fork.
    # Default False: existing text-only checkpoints are bit-unaffected.
    image_enabled: bool = False
    image_size: int = 64            # input images resized to SxS
    image_patch_size: int = 16      # -> (image_size/patch_size)^2 tokens

    @property
    def n_image_tokens(self) -> int:
        if not self.image_enabled:
            return 0
        return (self.image_size // self.image_patch_size) ** 2

    def param_count_estimate(self) -> int:
        d, L, h, f, v = (self.d_model, self.n_layers, self.n_heads,
                         self.ffn_dim, self.vocab_size)
        per_layer = 4 * d * d + 3 * d * f + 2 * d   # attn qkvo + swiglu + norms
        total = v * d + L * per_layer + d           # embeddings + layers + final norm
        if self.image_enabled:
            n = self.n_image_tokens
            # PatchEmbed: conv 3->d over patch^2 + pos embeddings + norm
            total += 3 * d * self.image_patch_size ** 2 + n * d + d
        return total  # tied embeddings: no separate LM head

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ModelConfig":
        return cls(**json.loads(s))
