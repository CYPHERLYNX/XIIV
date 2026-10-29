"""OpenAI-compatible server for XIIV: plug it into OpenCode, or anything
that speaks the OpenAI API.

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions  {"model","messages":[{"role","content"}],
        "max_tokens","temperature","seed",
        "xiiv_contract": {"lock": str, "require": [str], "forbid": [str]},
        "xiiv_steps": int}

The optional xiiv_contract carries the guarantee-native features through
the standard API: locked/forbidden/required terms compiled to sampler
constraints, verified before the response is returned.

Run:
    .venv/bin/python -m xiiv.openai_server --weights weights_52m/xiiv-52m-fp16.safetensors --port 11434

For the 1B model, pass --model-id explicitly so /v1/models reports
the model actually being served (the default "xiiv-52m" belongs to
the default 52M weights, not to any weights file):
    .venv/bin/python -m xiiv.openai_server \
        --weights checkpoints_1b_sgbd/xiiv_1b_sgbd.safetensors \
        --tokenizer weights/tokenizer_32k.json \
        --model-id xiiv-1b --port 11434

OpenCode config (~/.config/opencode/opencode.json):
    {"provider": {"xiiv": {"npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://localhost:11434/v1"}}},
     "model": "xiiv/xiiv-52m"}
    (for the 1B package, use "model": "xiiv/xiiv-1b" to match --model-id)

Stdlib only. Localhost only.
"""
import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .sample import load_model
from .tokenizer import XIIVTokenizer
from .diffusion import sample_trajectory
from .contracts import Contract

MODEL_ID = "xiiv-52m"


class Handler(BaseHTTPRequestHandler):
    model = None
    tok = None
    # Reported by GET /v1/models and echoed as the default in chat
    # completions. Defaults to MODEL_ID (the 52M weights); main() sets
    # this from --model-id so a served 1B checkpoint is never reported
    # under the 52M name.
    model_id = MODEL_ID
    # When True, every generation forbids token ids >= parent_vocab_size.
    # Used for grown checkpoints whose expanded-vocabulary embeddings are
    # still untrained: the model then samples only from the trained parent
    # vocabulary (e.g. 500M grown from the 8k-vocab 133M parent).
    parent_vocab_only = False
    parent_vocab_size = 8192

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._send(200, {"object": "list", "data": [{
                "id": self.model_id, "object": "model",
                "created": int(time.time()), "owned_by": "xiiv"}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        if n > 1 << 20:
            return self._send(413, {"error": "request too large"})
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        try:
            return self._send(200, self.chat(req))
        except Exception as e:  # never leak tracebacks
            return self._send(500, {"error": f"generation failed: {type(e).__name__}"})

    def chat(self, req):
        tok, model = self.tok, self.model
        messages = req.get("messages", [])
        if not messages:
            return {"error": "no messages"}
        # flatten the conversation into a prompt; the final user message
        # is what gets answered
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            parts.append(f"{role}: {content}" if role != "user" else content)
        prompt_text = "\n".join(parts) + "\nassistant:"
        prompt_ids = torch.tensor(tok.encode(prompt_text[:800]),
                                  dtype=torch.long)

        max_tokens = max(1, min(int(req.get("max_tokens", 128)), 256))
        total = len(prompt_ids) + max_tokens
        constraints = None
        guaranteed = False
        spec = req.get("xiiv_contract") or {}
        if spec:
            c = Contract()
            if spec.get("lock"):
                c.lock(spec["lock"])
            c.require(*(spec.get("require") or []))
            c.forbid(*(spec.get("forbid") or []))
            c.exact_length(total)
            constraints, _ = c.compile(tok, prompt_len=len(prompt_ids))

        steps = max(1, min(int(req.get("xiiv_steps",
                                       req.get("steps", 12))), 32))
        seed = req.get("seed", 7)
        if self.parent_vocab_only:
            # Structural vocab restriction: grown model, untrained new
            # embeddings — sample only from the trained parent vocabulary.
            pv_forbid = set(range(self.parent_vocab_size, tok.vocab_size))
            constraints = dict(constraints or {})
            existing = set(constraints.get("forbidden_ids") or ())
            constraints["forbidden_ids"] = sorted(existing | pv_forbid)
        t0 = time.time()
        ids, _ = sample_trajectory(
            model, total, tok.mask_id, tok.eos_id, steps=steps, seed=seed,
            temperature=float(req.get("temperature", 1.0)),
            prompt_ids=prompt_ids, constraints=constraints,
            device=next(model.parameters()).device)
        dt = time.time() - t0
        if constraints:
            check = Contract.verify(ids, constraints, tok.mask_id)
            guaranteed = Contract.holds(check)
        text = tok.decode(ids.tolist()[len(prompt_ids):])
        # strip padding: cut at the first EOS-ish run or trailing masks
        text = text.replace(tok.decode([tok.mask_id]), "").strip()
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", self.model_id),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt_ids),
                      "completion_tokens": max_tokens,
                      "total_tokens": len(prompt_ids) + max_tokens},
            "xiiv": {"guaranteed": guaranteed,
                     "seconds": round(dt, 2)},
        }
        return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights_52m/xiiv-52m-fp16.safetensors")
    ap.add_argument("--tokenizer", default="weights_52m/tokenizer.json")
    ap.add_argument("--model-id", default=None,
                    help="model id reported by GET /v1/models "
                         "(default: xiiv-52m, matching the default weights; "
                         "pass xiiv-1b when serving a 1B checkpoint)")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--parent-vocab-only", action="store_true",
                    help="restrict sampling to the first 8192 token ids "
                         "(trained parent vocab of grown checkpoints)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    Handler.tok = XIIVTokenizer(args.tokenizer)
    Handler.model, _ = load_model(args.weights)
    Handler.parent_vocab_only = args.parent_vocab_only
    if args.model_id:
        Handler.model_id = args.model_id
    print(f"XIIV OpenAI-compatible API at http://127.0.0.1:{args.port}/v1 "
          f"({Handler.model.count_parameters()/1e6:.1f}M params, "
          f"model={Handler.model_id})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
