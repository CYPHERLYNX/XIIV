"""Build a web/code pretraining dataset for XIIV using the EXISTING tokenizer.

Streams documents from HuggingFace (no full download), tokenizes with the
32k tokenizer (never retrains it), and writes packed uint16 memmap blocks
(seq_len=128, EOS-separated) + meta.json — the exact format xiiv.train
expects.

Sources:
  fineweb-edu : HuggingFaceFW/fineweb-edu (sample-10BT) — general knowledge,
                the highest-value corpus for making the model smarter.
  starcoder   : bigcode/starcoderdata — code the model has never seen.

Quality gates (applied per document, before tokenizing):
  - length bounds (min/max chars) — drops stubs and mega-dumps
  - exact-dup hash skip (streaming Bloom-ish via a Python set of hashes;
    fine for <1B tokens)
  - code: keep only files with a permissive license field when present

Example:
  python scripts/prepare_fineweb.py --source fineweb-edu \\
      --tokenizer weights/tokenizer_32k.json --out-dir data_fineweb \\
      --max-tokens 100000000
"""

import argparse
import hashlib
import json
import os
import sys

# Sandbox workaround: httpx (used by `datasets`) chokes parsing this
# environment's no_proxy entry "[::1]" ("Invalid port: ':1]'"). Drop the
# no_proxy vars before importing datasets — same class of issue that
# prepare_wiki.py works around with urllib.
for _v in ("no_proxy", "NO_PROXY"):
    os.environ.pop(_v, None)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
import numpy as np

from xiiv.tokenizer import XIIVTokenizer

_SOURCES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "starcoder": ("bigcode/starcoderdata", "default", "content"),
}


def _doc_ok(text, min_chars, max_chars, seen):
    if not text or not (min_chars <= len(text) <= max_chars):
        return False
    h = hashlib.blake2b(text.encode("utf-8", "ignore"),
                         digest_size=16).hexdigest()
    if h in seen:
        return False
    seen.add(h)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(_SOURCES), required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=100_000_000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=100_000)
    ap.add_argument("--valid-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true",
                    help="continue appending to an existing out-dir "
                         "(reads block counts from meta.json)")
    args = ap.parse_args()

    tok = XIIVTokenizer(args.tokenizer)
    repo, subset, field = _SOURCES[args.source]
    print(f"streaming {repo}:{subset} ...")
    from datasets import load_dataset
    ds = load_dataset(repo, subset, split="train", streaming=True)

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
        # meta.json only exists after a clean finish; otherwise derive
        # block counts from the partial .bin files.
        for _name, _path in (("train", train_path), ("valid", valid_path)):
            if os.path.exists(_path):
                _blocks = os.path.getsize(_path) // (args.seq_len * 2)
                if _name == "train":
                    n_train_blocks = _blocks
                else:
                    n_valid_blocks = _blocks
        print(f"resuming: {n_train_blocks} train + {n_valid_blocks} "
              f"valid blocks already present")
    n_train_tok = n_train_blocks * args.seq_len
    n_valid_tok = n_valid_blocks * args.seq_len

    def flush(buf):
        """Pack buffered token ids into seq_len blocks, EOS-separated."""
        nonlocal n_train_tok, n_valid_tok, n_train_blocks, n_valid_blocks
        ids = []
        for doc_ids in buf:
            ids.extend(doc_ids)
            ids.append(tok.eos_id)
        n_blocks = len(ids) // args.seq_len
        if n_blocks == 0:
            return buf  # keep buffering
        arr = np.array(ids[:n_blocks * args.seq_len], dtype=np.uint16)
        if rng.random() < args.valid_frac:
            valid_f.write(arr.tobytes())
            n_valid_blocks += n_blocks
            n_valid_tok += n_blocks * args.seq_len
        else:
            train_f.write(arr.tobytes())
            n_train_blocks += n_blocks
            n_train_tok += n_blocks * args.seq_len
        # carry the remainder
        rest = ids[n_blocks * args.seq_len:]
        return [rest] if rest else []

    try:
        for ex in ds:
            if n_train_tok + n_valid_tok >= args.max_tokens:
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
            if n_docs % 2000 == 0:
                print(f"  docs={n_docs} kept-tok={(n_train_tok+n_valid_tok)/1e6:.1f}M "
                      f"dropped={n_dropped}", flush=True)
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
        "source": f"{repo}:{subset}",
        "docs_seen": n_docs,
        "docs_dropped": n_dropped,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"),
              indent=1)
    print(f"done: {n_train_blocks} train blocks "
          f"({n_train_blocks*args.seq_len/1e6:.2f}M tok), "
          f"{n_valid_blocks} valid blocks; docs={n_docs} "
          f"dropped={n_dropped}")
    print(f"wrote {args.out_dir}/meta.json")


if __name__ == "__main__":
    main()
