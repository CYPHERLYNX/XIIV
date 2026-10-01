"""Tiny stdlib-only HTTP server for the XIIV demo page.

Serves demo/ statically and exposes:
    POST /api/generate  {"prompt","length","steps","seed","temperature"}
                        -> {"text","trajectory":[[ids...],...],"tok_s"}
    POST /api/infill    {"text","steps","seed"} -> {"text","tok_s"}

The trajectory lets the page animate parallel de-masking step by step.
No external requests are made, ever.
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch

from .sample import load_model
from .tokenizer import XIIVTokenizer
from .diffusion import sample_trajectory

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(os.path.dirname(HERE), "demo")
MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
        ".json": "application/json", ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    model = None
    cfg = None
    tok = None

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"
        # containment: resolve and refuse anything escaping DEMO_DIR
        fpath = os.path.realpath(os.path.join(DEMO_DIR, path.lstrip("/")))
        if not fpath.startswith(os.path.realpath(DEMO_DIR) + os.sep):
            return self._send(403, b'{"error":"forbidden"}')
        if os.path.isfile(fpath):
            ext = os.path.splitext(fpath)[1]
            with open(fpath, "rb") as f:
                self._send(200, f.read(), MIME.get(ext, "application/octet-stream"))
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0))
        if n > 65536:  # never read an unbounded body into memory
            return self._send(413, b'{"error":"request too large"}')
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, b'{"error":"bad json"}')
        try:
            if path == "/api/generate":
                body = self.api_generate(req)
            elif path == "/api/infill":
                body = self.api_infill(req)
            else:
                return self._send(404, b'{"error":"unknown endpoint"}')
        except Exception as e:  # never leak tracebacks to the page
            return self._send(500, json.dumps(
                {"error": f"generation failed: {type(e).__name__}"}).encode())
        self._send(200, json.dumps(body).encode())

    def api_generate(self, req):
        tok, model, cfg = self.tok, self.model, self.cfg
        length = max(8, min(int(req.get("length", 128)), 256))
        steps = max(1, min(int(req.get("steps", 16)), 64))
        seed = req.get("seed", 7)
        prompt_ids = None
        if req.get("prompt"):
            prompt_ids = torch.tensor(tok.encode(req["prompt"][:500]),
                                      dtype=torch.long)
        t0 = time.time()
        ids, conf, traj = sample_trajectory(
            model, length, tok.mask_id, tok.eos_id, steps=steps, seed=seed,
            temperature=float(req.get("temperature", 1.0)),
            prompt_ids=prompt_ids, return_trajectory=True)
        dt = time.time() - t0
        texts = [tok.decode(t.tolist()) for t in traj]
        n_tok = int((ids != tok.mask_id).sum())
        return {"text": tok.decode(ids.tolist()), "trajectory": texts,
                "steps": steps, "tok_s": round(n_tok / max(dt, 1e-6), 1),
                "seconds": round(dt, 2)}

    def api_infill(self, req):
        tok, model = self.tok, self.model
        text = req.get("text", "")[:1000]
        if "<mask" not in text:
            return {"error": "include at least one <mask> or <mask:N> marker"}
        ids, frozen = tok.encode_with_masks(text)
        length = len(ids)
        t0 = time.time()
        out_ids, conf, traj = sample_trajectory(
            model, length, tok.mask_id, tok.eos_id,
            steps=max(1, min(int(req.get("steps", 16)), 64)),
            seed=req.get("seed", 7), prompt_ids=ids, frozen=frozen,
            return_trajectory=True)
        out_ids[frozen] = ids[frozen]  # belt-and-braces: context immutable
        dt = time.time() - t0
        texts = [tok.decode(t.tolist()) for t in traj]
        return {"text": tok.decode(out_ids.tolist()), "trajectory": texts,
                "tok_s": round(length / max(dt, 1e-6), 1),
                "seconds": round(dt, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/xiiv.safetensors")
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    Handler.tok = XIIVTokenizer(args.tokenizer)
    Handler.model, Handler.cfg = load_model(args.weights)
    print(f"xiiv demo: http://localhost:{args.port}/ "
          f"({Handler.model.count_parameters()/1e6:.1f}M params)")
    # NOTE: ThreadingHTTPServer spawns an unbounded thread per connection —
    # fine on localhost, but add a thread pool / reverse proxy if ever exposed.
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
