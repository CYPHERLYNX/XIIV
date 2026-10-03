"""XIIV guarantee-demo server: ask anything, watch verification happen live.

Serves demo/index.html and two JSON endpoints:
  POST /api/ask      {"question": str} -> answer + verification badges
  POST /api/contract {"task": str, "policy": str, "must_do": [...],
                      "never_do": [...], "enforce": bool}

Run:  .venv/bin/python demo/server.py  (then open http://localhost:8099)
"""
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL = None
ROOT = os.path.dirname(os.path.abspath(__file__))


def get_model():
    global MODEL
    if MODEL is None:
        from xiiv import XIIV
        weights = os.path.join(ROOT, "..", "weights_52m",
                               "xiiv-52m-fp16.safetensors")
        MODEL = XIIV(os.path.abspath(weights))
    return MODEL


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "index.html"), "rb") as f:
                self._send(200, f.read().decode(), "text/html")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "invalid JSON"})
        try:
            if self.path == "/api/ask":
                return self._send(200, self.api_ask(data))
            if self.path == "/api/contract":
                return self._send(200, self.api_contract(data))
            return self._send(404, {"error": "unknown endpoint"})
        except Exception:
            traceback.print_exc()
            return self._send(200, {"error":
                "Something went wrong on the model side. Try again."})

    def api_ask(self, data):
        from xiiv.knowledge import pipeline
        q = (data.get("question") or "").strip()
        if not q:
            return {"error": "Type a question first."}
        model = get_model()
        enforce = bool(data.get("enforce", True))
        out = pipeline.answer(q, model, steps=12, seed=7, enforce=enforce)
        return {
            "answer": out["answer"],
            "facts": out["facts"],
            "sources": out["sources"],
            "key_terms": out["key_terms"],
            "guaranteed": out["guaranteed"],
            "enforce": enforce,
        }

    def api_contract(self, data):
        from xiiv.agents import generate_plan
        task = (data.get("task") or "").strip()
        if not task:
            return {"error": "Describe the task first."}
        model = get_model()
        out = generate_plan(
            model, task,
            policy=data.get("policy", ""),
            must_do=data.get("must_do", []),
            never_do=data.get("never_do", []),
            enforce=bool(data.get("enforce", True)),
            steps=12, seed=7)
        return {
            "plan": out["plan"],
            "guaranteed": out["guaranteed"],
            "check": out["check"],
            "enforce": bool(data.get("enforce", True)),
        }


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8099), Handler)
    print("XIIV demo at http://127.0.0.1:8099  (Ctrl-C to stop)")
    srv.serve_forever()
