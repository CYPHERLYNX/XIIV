"""Build XIIV frontier training dataset — multi-source streaming tokenizer.

Streams documents from HuggingFace (no full download), tokenizes with the
XIIV 32k tokenizer, writes packed uint16 memmap blocks (seq_len=128,
EOS-separated) + meta.json — the exact format xiiv.train expects.

Sources (all openly licensed):
  fineweb-edu : HuggingFaceFW/fineweb-edu (sample-10BT) — educational web
  dclm        : mlfoundations/dclm-baseline-1.0 — DCLM-filtered web
  starcoder   : bigcode/starcoderdata — permissively licensed code
  finemath    : HuggingFaceTB/finemath (finemath-4plus) — math web
  openwebmath : open-web-math/open-web-math — math web

Usage:
  python scripts/build_frontier_data.py --source fineweb-edu \\
      --out-dir data_frontier/fineweb_edu --max-tokens 3000000000 --resume
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

for _v in ("no_proxy", "NO_PROXY"):
    os.environ.pop(_v, None)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
import numpy as np

from xiiv.tokenizer import XIIVTokenizer

_SOURCES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "dclm": ("mlfoundations/dclm-baseline-1.0", None, "text"),
    "starcoder": ("bigcode/starcoderdata", "default", "content"),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus", "text"),
    "openwebmath": ("open-web-math/open-web-math", None, "text"),
}

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data_frontier", "build.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _doc_ok(text, min_chars, max_chars, seen):
    if not text or not (min_chars <= len(text) <= max_chars):
        return False
    h = hashlib.blake2b(text.encode("utf-8", "ignore"),
                         digest_size=16).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    return True


def disk_ok(path, min_free_gb=5.0):
    """Stop before filling the disk."""
    free_gb = shutil.disk_usage(path).free / 1e9
    return free_gb >= min_free_gb, free_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(_SOURCES), required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=100_000)
    ap.add_argument("--valid-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    args = ap.parse_args()

    tok = XIIVTokenizer(args.tokenizer)
    repo, subset, field = _SOURCES[args.source]
    log(f"=== source={args.source} repo={repo} max_tokens={args.max_tokens/1e9:.1f}B ===")

    from datasets import load_dataset
    kw = dict(split="train", streaming=True)
    if subset:
        kw["name"] = subset
    ds = load_dataset(repo, **kw)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.bin")
    valid_path = os.path.join(args.out_dir, "valid.bin")
    mode = "ab" if args.resume else "wb"
    train_f = open(train_path, mode)
    valid_f = open(valid_path, mode)

    rng = np.random.default_rng(args.seed)
    seen = set()
    buf = []
    n_train_blocks = n_valid_blocks = 0
    n_docs = n_dropped = 0
    if args.resume:
        for _name, _path in (("train", train_path), ("valid", valid_path)):
            if os.path.exists(_path):
                _blocks = os.path.getsize(_path) // (args.seq_len * 2)
                if _name == "train":
                    n_train_blocks = _blocks
                else:
                    n_valid_blocks = _blocks
        log(f"resuming: {n_train_blocks} train + {n_valid_blocks} valid blocks")
    n_train_tok = n_train_blocks * args.seq_len
    n_valid_tok = n_valid_blocks * args.seq_len
    t_start = time.time()

    def flush(buf):
        nonlocal n_train_tok, n_valid_tok, n_train_blocks, n_valid_blocks
        ids = []
        for doc_ids in buf:
            ids.extend(doc_ids)
            ids.append(tok.eos_id)
        n_blocks = len(ids) // args.seq_len
        if n_blocks == 0:
            return buf
        arr = np.array(ids[:n_blocks * args.seq_len], dtype=np.uint16)
        if rng.random() < args.valid_frac:
            valid_f.write(arr.tobytes())
            n_valid_blocks += n_blocks
            n_valid_tok += n_blocks * args.seq_len
        else:
            train_f.write(arr.tobytes())
            n_train_blocks += n_blocks
            n_train_tok += n_blocks * args.seq_len
        rest = ids[n_blocks * args.seq_len:]
        return [rest] if rest else []

    try:
        for ex in ds:
            total = n_train_tok + n_valid_tok
            if total >= args.max_tokens:
                break
            ok, free_gb = disk_ok(args.out_dir, args.min_free_gb)
            if not ok:
                log(f"DISK LOW ({free_gb:.1f}GB free) — stopping {args.source}")
                break
            text = ex.get(field) or ""
            n_docs += 1
            if not _doc_ok(text, args.min_chars, args.max_chars, seen):
                n_dropped += 1
                continue
            ids = [i for i in tok.encode(text)
                   if i not in (tok.mask_id, tok.eos_id)]
            if not ids:
                n_dropped += 1
                continue
            buf.append(ids)
            if sum(map(len, buf)) > 100_000:
                buf = flush(buf)
            if n_docs % 5000 == 0:
                el = time.time() - t_start
                rate = total / max(el, 1) / 1e6
                log(f"  {args.source}: docs={n_docs} tok={total/1e6:.1f}M "
                    f"dropped={n_dropped} rate={rate:.2f}Mtok/s")
    except Exception as e:
        log(f"ERROR in {args.source}: {type(e).__name__}: {str(e)[:200]}")
    finally:
        flush(buf)
        train_f.close()
        valid_f.close()

    meta = {
        "vocab_size": tok.vocab_size,
        "mask_id": tok.mask_id,
        "eos_id": tok.eos_id,
        "seq_len": args.seq_len,
        "train_blocks": n_train_blocks,
        "train_tokens": n_train_blocks * args.seq_len,
        "valid_blocks": n_valid_blocks,
        "valid_tokens": n_valid_blocks * args.seq_len,
        "source": f"{repo}:{subset or 'default'}",
        "docs_seen": n_docs,
        "docs_dropped": n_dropped,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"),
              indent=1)
    log(f"DONE {args.source}: {n_train_blocks*args.seq_len/1e6:.1f}M train tok, "
        f"docs={n_docs} dropped={n_dropped}")


if __name__ == "__main__":
    main()
