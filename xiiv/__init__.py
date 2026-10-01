"""XIIV — a tiny masked-diffusion language model, trained from scratch."""
import torch

from .config import ModelConfig
from .model import XIIVModel

__version__ = "0.1.0"
__all__ = ["ModelConfig", "XIIVModel", "__version__", "load_ckpt_trusted"]


def load_ckpt_trusted(path, map_location="cpu"):
    """Load a XIIV training checkpoint.

    TRUST WARNING: checkpoints are Python pickles and can execute arbitrary
    code on load. Only load checkpoint files produced by your own training
    runs (or from sources you trust completely). Never load a checkpoint
    downloaded from an untrusted third party with this function.
    ``weights_only=False`` is required because resume needs the optimizer
    and RNG states, which ``weights_only=True`` cannot deserialize.
    """
    return torch.load(path, map_location=map_location, weights_only=False)
