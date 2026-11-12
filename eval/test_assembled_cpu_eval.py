"""Tests for eval/assembled_cpu_eval.py helpers (no model load)."""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from eval.assembled_cpu_eval import (
    LENGTH,
    PROMPT,
    SEED,
    STEPS,
    TEMPERATURE,
    TOP_P,
    build_record,
    sha256_file,
    tokens_per_second,
)


class _Cfg:
    d_model = 1280
    n_layers = 36
    vocab_size = 32768
    contract_enabled = False
    tie_embeddings = True


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"xiiv-test-bytes")
    assert sha256_file(str(p)) == hashlib.sha256(b"xiiv-test-bytes").hexdigest()


def test_tokens_per_second():
    assert tokens_per_second(64, 32.0) == 2.0
    assert tokens_per_second(64, 0.0) == 0.0


def test_spec_is_the_gpu_v1_spec():
    # The CPU/GPU comparison is only valid if this spec never drifts
    # from the GPU job's spec without a matching GPU change.
    assert PROMPT == "What is quantum computing vs quantum mechanics."
    assert (LENGTH, STEPS, SEED, TEMPERATURE, TOP_P) == (64, 16, 7, 0.8, 0.9)


def test_build_record_counts_non_mask_tokens(tmp_path):
    w = tmp_path / "w.bin"
    w.write_bytes(b"fake-weights")
    ids = torch.tensor([10, 11, 2, 12])  # one mask token (id 2)
    rec = build_record(str(w), "hello", ids, 2.0, _Cfg(), mask_id=2)
    assert rec["output"]["n_tokens_non_mask"] == 3
    assert rec["timing"]["tokens_per_second"] == 1.5
    assert rec["spec"]["device"] == "cpu"
    assert rec["weights"]["bytes"] == len(b"fake-weights")
    assert "Not a benchmark score" in rec["disclaimer"]
