"""Unit tests for eval/adapters/hf_causal_adapter.py.

No GPU, no network, no `transformers`: _load is monkeypatched to return
fakes, so these tests exercise the adapter's own logic only —
contract shape, verbatim prompt, max_new_tokens derivation, refusal
paths, and the honesty note.
"""

import contextlib
import importlib.util
import os
import sys
import types

ADAPTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hf_causal_adapter.py")


def load_adapter():
    spec = importlib.util.spec_from_file_location("hf_causal_adapter",
                                                  ADAPTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeTensor(list):
    @property
    def shape(self):
        return (1, len(self))

    def to(self, device):
        return self


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeTok:
    def __init__(self):
        self.pad_token_id = None
        self.eos_token_id = 2
        self.last_prompt = None

    def __call__(self, prompt, return_tensors=None):
        self.last_prompt = prompt
        return FakeInputs({"input_ids": FakeTensor([1, 2, 3])})

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


class FakeModel:
    def __init__(self):
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        # echoes input ids [1,2,3] + 5 new ids
        return [FakeTensor([1, 2, 3, 10, 11, 12, 13, 14])]


class FakeTorch:
    def __init__(self):
        self.seeds = []

    def manual_seed(self, s):
        self.seeds.append(s)

    def no_grad(self):
        return contextlib.nullcontext()


SPEC = {"length": 64, "steps": 16, "seed": 7, "temperature": 0.8,
        "top_p": 0.9, "device": "cpu", "spec_version": "1.0"}


def with_fakes(mod, model=None, tok=None, torch_mod=None):
    model = model or FakeModel()
    tok = tok or FakeTok()
    torch_mod = torch_mod or FakeTorch()

    def fake_load(repo, device):
        return model, tok, 1_234_567, torch_mod

    mod._load = fake_load
    return model, tok, torch_mod


def _set_env(name):
    os.environ["XIIV_COMPETITOR"] = name


def test_known_models_lists_all_six():
    mod = load_adapter()
    known = mod.known_models()
    assert len(known) == 6, f"expected 6 competitors, got {len(known)}"
    assert known["Qwen3-0.6B"] == "Qwen/Qwen3-0.6B"


def test_refuses_missing_env():
    mod = load_adapter()
    os.environ.pop("XIIV_COMPETITOR", None)
    try:
        mod.generate("hi", SPEC)
    except RuntimeError as e:
        assert "XIIV_COMPETITOR" in str(e)
    else:
        raise AssertionError("missing env var must refuse")


def test_refuses_unknown_competitor():
    mod = load_adapter()
    _set_env("Not-A-Real-Model-1B")
    try:
        mod.generate("hi", SPEC)
    except RuntimeError as e:
        assert "Unknown competitor" in str(e)
    else:
        raise AssertionError("unknown competitor must refuse")
    finally:
        os.environ.pop("XIIV_COMPETITOR", None)


def test_generate_contract_with_fakes():
    mod = load_adapter()
    _set_env("Qwen3-0.6B")
    model, tok, torch_mod = with_fakes(mod)
    try:
        out = mod.generate("hello world", SPEC)
    finally:
        os.environ.pop("XIIV_COMPETITOR", None)

    assert isinstance(out["text"], str) and out["text"]
    assert isinstance(out["token_ids"], list)
    assert all(isinstance(x, int) for x in out["token_ids"])
    # only the NEW ids: input was [1,2,3], fake emitted +[10..14]
    assert out["token_ids"] == [10, 11, 12, 13, 14]
    assert isinstance(out["generation_seconds"], float)
    assert out["generation_seconds"] >= 0.0
    assert isinstance(out["tokens_per_second"], float)
    # prompt verbatim, no template wrapping
    assert tok.last_prompt == "hello world"
    # seed from spec applied
    assert torch_mod.seeds == [7]
    # max_new_tokens = length(64) - prompt budget(24) = 40
    assert model.last_kwargs["max_new_tokens"] == 40
    assert model.last_kwargs["temperature"] == 0.8
    assert model.last_kwargs["top_p"] == 0.9
    assert model.last_kwargs["do_sample"] is True
    # honesty note: steps N/A for causal, repo named
    assert "N/A" in out["note"] and "steps" in out["note"]
    assert "Qwen/Qwen3-0.6B" in out["note"]
    assert "1,234,567" in out["note"]
    assert out["competitor"] == "Qwen3-0.6B"


def test_loader_singleton_per_device():
    mod = load_adapter()
    _set_env("Qwen3-0.6B")
    calls = []

    def fake_load(repo, device):
        calls.append((repo, device))
        return FakeModel(), FakeTok(), 9, FakeTorch()

    mod._load = fake_load
    try:
        mod.generate("a", SPEC)
        mod.generate("b", SPEC)
    finally:
        os.environ.pop("XIIV_COMPETITOR", None)
    assert len(calls) == 1, "model must load once per (name, device)"


if __name__ == "__main__":
    test_known_models_lists_all_six()
    test_refuses_missing_env()
    test_refuses_unknown_competitor()
    test_generate_contract_with_fakes()
    test_loader_singleton_per_device()
    print("5/5 adapter tests passed")
