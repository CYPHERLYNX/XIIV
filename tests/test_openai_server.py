"""Tests for the OpenAI-compatible server's model identity.

No model is loaded and no socket is opened: the handler is built with
object.__new__ and _send is captured. This guards the Oct-31 delivery
package: a served 1B checkpoint must never be reported as xiiv-52m.
"""

from xiiv.openai_server import MODEL_ID, Handler
from xiiv.sample import resolve_device


def _handler(path):
    h = object.__new__(Handler)
    h.path = path
    captured = {}
    h._send = lambda code, obj: captured.setdefault("res", (code, obj))  # noqa: E731
    return h, captured


def test_default_model_id_is_52m():
    assert MODEL_ID == "xiiv-52m"
    assert Handler.model_id == "xiiv-52m"


def test_models_endpoint_reports_overridden_model_id():
    old = Handler.model_id
    try:
        Handler.model_id = "xiiv-1b"
        h, captured = _handler("/v1/models")
        h.do_GET()
        code, obj = captured["res"]
        assert code == 200
        assert obj["data"][0]["id"] == "xiiv-1b"
    finally:
        Handler.model_id = old


def test_models_endpoint_unknown_path_404():
    h, captured = _handler("/v1/unknown")
    h.do_GET()
    assert captured["res"][0] == 404


def test_resolve_device_cpu_and_auto():
    assert resolve_device("cpu") == "cpu"
    # This box has CPU-only torch: auto must resolve to cpu, never cuda.
    assert resolve_device("auto") == "cpu"


def test_resolve_device_cuda_refuses_without_cuda():
    import pytest

    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present; refusal path not exercised here")
    with pytest.raises(SystemExit):
        resolve_device("cuda")
