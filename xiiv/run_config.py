"""Typed, frozen training-run configuration (ecc_mle_workflow §3).

Every hyperparameter, path, seed, dataset snapshot, and code hash for a
training run lives in ONE frozen dataclass. No hidden notebook state, no
argparse-only configs that evaporate after the run. train.py builds this
at startup, logs its hash, and writes run_config.json next to checkpoints.
"""
import hashlib
import json
import os
from dataclasses import dataclass, asdict, field


def code_sha(source_dir: str) -> str:
    """SHA256 over all .py sources (sorted) — the 'code version' for a run.

    The repo is not necessarily a git checkout, so we hash source text
    directly instead of pretending a git SHA exists.
    """
    h = hashlib.sha256()
    for root, _, files in os.walk(source_dir):
        if "__pycache__" in root or ".venv" in root:
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                with open(os.path.join(root, f), "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


@dataclass(frozen=True)
class RunConfig:
    # identity
    run_name: str = "xiiv-10m-tinystories"
    model_version: str = "0.1.0"
    objective: str = "masked-diffusion"   # or "autoregressive" (baseline)
    # data
    dataset_snapshot_id: str = ""          # e.g. tinystories-v2-gpt4-20261001
    data_dir: str = "data"
    train_blocks: int = 0
    valid_blocks: int = 0
    seq_len: int = 128
    # model
    d_model: int = 320
    n_layers: int = 6
    n_heads: int = 5
    ffn_dim: int = 864
    vocab_size: int = 8192
    mask_id: int = 2
    eos_id: int = 3
    tie_embeddings: bool = True
    # optimization
    batch_size: int = 48
    total_steps: int = 5000
    lr: float = 3e-4
    lr_min: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    bf16: bool = True
    t_min: float = 0.15                   # mask-ratio schedule ~ U(t_min, 1)
    t_max: float = 1.0
    # reproducibility
    seed: int = 1337
    threads: int = 2
    torch_version: str = ""
    # provenance (filled at run start)
    code_sha256: str = ""
    config_hash: str = field(default="", compare=False)

    def with_provenance(self, source_dir: str,
                        torch_version: str) -> "RunConfig":
        import dataclasses
        sha = code_sha(source_dir)
        # config hash covers everything EXCEPT provenance fields; the torch
        # version IS part of the config identity (different torch builds can
        # produce different numerics).
        d = asdict(self)
        d.pop("config_hash", None)
        d.pop("code_sha256", None)
        d["torch_version"] = torch_version
        cfg_hash = hashlib.sha256(
            json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
        return dataclasses.replace(self, code_sha256=sha,
                                   torch_version=torch_version,
                                   config_hash=cfg_hash)

    def save(self, path: str):
        d = asdict(self)
        d["betas"] = list(d["betas"])
        with open(path, "w") as f:
            json.dump(d, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        d = json.load(open(path))
        d["betas"] = tuple(d["betas"])
        return cls(**d)
