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
    "stackv2": ("bigcode/the-stack-v2", "default", "content"),
    "pes2o": ("scientifi-papers/scientific-papers", "pes2o", "raw_fulltext"),
    "stackexchange": ("donfu/oa-stackexchange", None, None),  # no single text column;
    # INSTRUCTION+RESPONSE combined via _TRANSFORMS (see _transform_stackexchange)
    # NOTE (2026-10-17): canonical `pg19` repo is a dataset script, which modern
    # `datasets` no longer supports (RuntimeError). Load via the emozilla/pg19
    # parquet mirror (byte-identical, `text` field) in _load_pg19().
    "pg19": ("emozilla/pg19", None, "text"),
    "openthoughts": ("open-thoughts/OpenThoughts3-1.2M", None, "conversations"),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus", "text"),
    # NOTE (2026-10-16): LLM360/MegaMath moved to IFM/MegaMath and is a
    # script-less multi-subset parquet repo (megamath-web/code/qa/...).
    # Generic auto-discovery resolves a stale builder schema and dies with
    # CastError (13.3M docs streamed, 0 tokens written) — so megamath is
    # loaded via _load_megamath() below, not the generic path in main().
    "megamath": ("IFM/MegaMath", None, "text"),
    "openwebmath": ("open-web-math/open-web-math", None, "text"),
}

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data_frontier", "build.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _transform_openthoughts(ex):
    """Convert OpenThoughts3 conversations list to plain text."""
    value = ex.get("conversations")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = []
        for m in value:
            if isinstance(m, dict):
                role = m.get("from", "")
                content = m.get("value", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(m))
        return "\n\n".join(parts)
    return str(value) if value else ""


def _transform_stackexchange(ex):
    """Combine donfu/oa-stackexchange Q/A pairs into one training doc.

    The repo exposes INSTRUCTION / RESPONSE / SOURCE / METADATA — there is
    no TEXT column (2026-10-16 the script read "TEXT", so every doc was
    empty and 6.33M docs were dropped with 0 tokens written). The raw Q/A
    split is preserved: instruction first, answer second.
    """
    instr = ex.get("INSTRUCTION") or ""
    resp = ex.get("RESPONSE") or ""
    if not instr and not resp:
        return ""
    return f"Question: {instr}\n\nAnswer: {resp}".strip()


# Per-source transforms applied to the raw stream example.
# A transform receives the full example dict and returns the training text;
# used when the text is not a single plain-string field (e.g. list of
# messages, or multi-field Q/A pairs).
_TRANSFORMS = {
    "openthoughts": _transform_openthoughts,
    "stackexchange": _transform_stackexchange,
}


# MegaMath subset dirs on IFM/MegaMath, curated-first so the 10B token cap
# lands on the highest-quality math data before the raw web crawl.
_MEGAMATH_SUBSETS = [
    "megamath-web-pro",
    "megamath-qa",
    "megamath-text-code-block",
    "megamath-code",
    "megamath-translated-code",
    "megamath-web",
]
# Verified 2026-10-16 via the Hub API: 5-digit zero-padded 00000.parquet
# sequences per subset. Used only if the API file-list fetch fails.
_MEGAMATH_FALLBACK_COUNTS = {
    "megamath-web-pro": 100,
    "megamath-qa": 32,
    "megamath-text-code-block": 200,
    "megamath-code": 16,
    "megamath-translated-code": 16,
    "megamath-web": 4950,
}


def _load_megamath():
    """Load IFM/MegaMath via the explicit parquet builder.

    The repo has no dataset script and several subset dirs; letting
    load_dataset() auto-discover resolves a stale builder schema and
    raises CastError on the real shards (text/domain/url). Passing the
    exact file list avoids the schema cast entirely.
    """
    from datasets import load_dataset
    base = "https://huggingface.co/datasets/IFM/MegaMath/resolve/main"
    ordered = []
    try:
        import urllib.request
        with urllib.request.urlopen(
                "https://huggingface.co/api/datasets/IFM/MegaMath",
                timeout=60) as r:
            info = json.load(r)
        by_sub = {}
        for s in info.get("siblings", []):
            fn = s.get("rfilename", "")
            if fn.endswith(".parquet") and "/" in fn:
                by_sub.setdefault(fn.split("/")[0], []).append(fn)
        for sub in _MEGAMATH_SUBSETS:
            ordered += sorted(by_sub.get(sub, []))
        log(f"megamath: file list from Hub API "
            f"({len(ordered)} parquet files)")
    except Exception as e:
        log(f"megamath: Hub API file list failed ({e}); "
            f"using verified fallback counts")
        for sub in _MEGAMATH_SUBSETS:
            n = _MEGAMATH_FALLBACK_COUNTS[sub]
            ordered += [f"{sub}/{i:05d}.parquet" for i in range(n)]
    if not ordered:
        raise RuntimeError("megamath: no parquet files resolved")
    data_files = [f"{base}/{f}" for f in ordered]
    return load_dataset("parquet", data_files={"train": data_files},
                        split="train", streaming=True)


# Verified 2026-10-17 via the Hub API: 23 zero-padded data/train-*-of-00023
# shards on the emozilla/pg19 parquet mirror. NOTE (2026-10-17): the
# canonical `pg19` repo is a dataset *script*, which modern `datasets` no
# longer supports (RuntimeError: "Dataset scripts are no longer supported,
# but found pg19.py") -- the canonical repo is unreachable. emozilla/pg19 is
# a byte-identical parquet mirror (card stats: 11,425,076,324 total chars,
# 28,602 train docs), with the same `text` field.
_PG19_FALLBACK_FILES = [
    "data/train-00000-of-00023-5263dc4323e881d2.parquet",
    "data/train-00001-of-00023-2b13c68dbeba63b5.parquet",
    "data/train-00002-of-00023-8e4dd6b70d51dfcf.parquet",
    "data/train-00003-of-00023-e7f0ba0d3c2eb6c0.parquet",
    "data/train-00004-of-00023-87dece6a0ab708e8.parquet",
    "data/train-00005-of-00023-6c0bc91e6df2e661.parquet",
    "data/train-00006-of-00023-c50fea56d0518a87.parquet",
    "data/train-00007-of-00023-e240b232c0daddca.parquet",
    "data/train-00008-of-00023-275017526d706c9d.parquet",
    "data/train-00009-of-00023-f09a29af90b71085.parquet",
    "data/train-00010-of-00023-dd5d8583a9f6a549.parquet",
    "data/train-00011-of-00023-6c0bc91e6df2e661.parquet",
    "data/train-00012-of-00023-78d820574bddc6bb.parquet",
    "data/train-00013-of-00023-5b853732c713dded.parquet",
    "data/train-00014-of-00023-54b567998cd5eb4b.parquet",
    "data/train-00015-of-00023-7a78b1e66cfe52de.parquet",
    "data/train-00016-of-00023-8269d10108995a28.parquet",
    "data/train-00017-of-00023-e240b232c0daddca.parquet",
    "data/train-00018-of-00023-f59d477d80180c78.parquet",
    "data/train-00019-of-00023-198af082bd917d87.parquet",
    "data/train-00020-of-00023-f9b058c5e9338514.parquet",
    "data/train-00021-of-00023-e221306b7359a366.parquet",
    "data/train-00022-of-00023-5a956eb2a5d6cab5.parquet",
]


def _load_pg19():
    """Load pg19 via the emozilla/pg19 parquet mirror.

    The canonical `pg19` repo uses a dataset script that modern `datasets`
    refuses to run, so pass the mirror's exact file list to the explicit
    parquet builder (same approach as _load_megamath).
    """
    from datasets import load_dataset
    base = "https://huggingface.co/datasets/emozilla/pg19/resolve/main"
    ordered = []
    try:
        import urllib.request
        with urllib.request.urlopen(
                "https://huggingface.co/api/datasets/emozilla/pg19",
                timeout=60) as r:
            info = json.load(r)
        for s in info.get("siblings", []):
            fn = s.get("rfilename", "")
            if fn.startswith("data/train-") and fn.endswith(".parquet"):
                ordered.append(fn)
        ordered.sort()
        log(f"pg19: file list from Hub API ({len(ordered)} parquet files)")
    except Exception as e:
        log(f"pg19: Hub API file list failed ({e}); using verified fallback")
        ordered = list(_PG19_FALLBACK_FILES)
    if not ordered:
        raise RuntimeError("pg19: no parquet files resolved")
    data_files = [f"{base}/{f}" for f in ordered]
    return load_dataset("parquet", data_files={"train": data_files},
                        split="train", streaming=True)


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

    if args.source == "megamath":
        ds = _load_megamath()
    elif args.source == "pg19":
        ds = _load_pg19()
    else:
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
    # Resume state: raw stream docs already tokenized in prior runs.
    # Without this, a restart re-streams from doc 0 with an empty `seen`
    # set and appends duplicates of everything processed before.
    state_path = os.path.join(args.out_dir, "resume_state.json")
    skip_docs = 0
    if args.resume:
        for _name, _path in (("train", train_path), ("valid", valid_path)):
            if os.path.exists(_path):
                _blocks = os.path.getsize(_path) // (args.seq_len * 2)
                if _name == "train":
                    n_train_blocks = _blocks
                else:
                    n_valid_blocks = _blocks
        log(f"resuming: {n_train_blocks} train + {n_valid_blocks} valid blocks")
        if os.path.exists(state_path):
            try:
                skip_docs = int(
                    json.load(open(state_path)).get("docs_seen", 0))
            except Exception:
                skip_docs = 0
            if skip_docs > 0:
                log(f"resuming: skipping first {skip_docs} stream docs "
                    f"(already tokenized in prior runs)")

    def save_state():
        # Stream position of data actually tokenized: docs 1..skip_docs were
        # tokenized in prior runs; docs skip_docs+1..n_docs were tokenized
        # this run. During the skip phase n_docs <= skip_docs, so the saved
        # position must NOT advance past skip_docs — writing skip_docs+n_docs
        # double-counts the skipped docs, the skip target grows on every
        # reboot, and the build never resumes tokenizing (2026-10-15: target
        # ballooned 11.6M -> 54.1M docs with zero new tokens in ~28h).
        try:
            with open(state_path, "w") as f:
                json.dump({"docs_seen": max(skip_docs, n_docs)}, f)
        except Exception:
            pass
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
            _transform = _TRANSFORMS.get(args.source)
            text = _transform(ex) if _transform is not None else (
                ex.get(field) or "")
            n_docs += 1
            if n_docs <= skip_docs:
                # Box reboots are frequent (6x in one day). Checkpoint skip
                # progress so an interrupted skip resumes from here, not doc 0,
                # and log a heartbeat so a stalled skip is visible.
                if n_docs % 250000 == 0:
                    save_state()
                    log(f"  {args.source}: skipping... docs={n_docs}/{skip_docs}")
                continue  # processed in a prior run — skip, no duplicates
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
                save_state()
    except Exception as e:
        log(f"ERROR in {args.source}: {type(e).__name__}: {str(e)[:200]}")
    finally:
        save_state()
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
        "docs_seen": max(skip_docs, n_docs),
        "docs_dropped": n_dropped,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"),
              indent=1)
    log(f"DONE {args.source}: {n_train_blocks*args.seq_len/1e6:.1f}M train tok, "
        f"docs={n_docs} dropped={n_dropped}")


if __name__ == "__main__":
    main()
