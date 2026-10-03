"""Build a Wikipedia fine-tune dataset for XIIV using the EXISTING tokenizer.

Collects article text via the Wikipedia API (urllib — works in this
sandbox where HF's httpx chokes on the proxy), tokenizes with
weights_52m/tokenizer.json (never retrains it), and writes packed
uint16 memmap blocks (seq_len=128, EOS-separated) + meta.json into
--out-dir. Target ~4M tokens: enough to shift the model from
children's-story prose toward adult expository writing.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
import numpy as np

from xiiv.tokenizer import XIIVTokenizer

_API = "https://en.wikipedia.org/w/api.php"
_UA = {"User-Agent": "XIIV-research/0.1 (contact: research@localhost)"}


def _get(params):
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(_API + "?" + qs, headers=_UA)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 + attempt * 2)
    raise RuntimeError("wikipedia api failed")


def collect_titles(n):
    """Featured-article titles via their talk pages (articles are
    categorized through Talk:). Featured = long, high-quality
    expository prose across all topics."""
    titles = []
    cmcontinue = None
    while len(titles) < n:
        p = {"action": "query", "format": "json", "list": "categorymembers",
             "cmtitle": "Category:Wikipedia featured articles",
             "cmtype": "page", "cmlimit": "500"}
        if cmcontinue:
            p["cmcontinue"] = cmcontinue
        d = _get(p)
        for pg in d["query"]["categorymembers"]:
            t = pg["title"]
            if pg["ns"] == 1 and t.startswith("Talk:"):
                t = t[len("Talk:"):]
            elif pg["ns"] != 0:
                continue
            titles.append(t)
            if len(titles) >= n:
                break
        cmcontinue = d.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.2)
    return titles


def fetch_extracts(titles):
    """Plain-text extracts, 20 titles per request."""
    texts = []
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        d = _get({"action": "query", "format": "json",
                  "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
                  "titles": "|".join(chunk)})
        for pg in d["query"]["pages"].values():
            t = (pg.get("extract") or "").strip()
            if len(t) >= 2000:
                texts.append(t)
        time.sleep(0.3)
        if (i // 20) % 25 == 0:
            print(f"  {i}/{len(titles)} titles, {len(texts)} kept")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_wiki")
    ap.add_argument("--tokenizer", default="weights_52m/tokenizer.json")
    ap.add_argument("--max-articles", type=int, default=5000)
    ap.add_argument("--target-tokens", type=int, default=4_000_000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tok = XIIVTokenizer(args.tokenizer)
    print("collecting wikipedia article titles ...")
    titles = collect_titles(args.max_articles)
    print(f"{len(titles)} titles; fetching extracts ...")
    texts = fetch_extracts(titles)

    ids_all = []
    for text in texts:
        ids = tok.encode(text)
        ids = [i for i in ids if i < 65536]
        ids_all.extend(ids)
        ids_all.append(tok.eos_id)
        if len(ids_all) >= args.target_tokens:
            break
    print(f"collected {len(texts)} articles, {len(ids_all)/1e6:.2f}M tokens")

    rng = np.random.RandomState(args.seed)
    arr = np.array(ids_all, dtype=np.uint16)
    n_blocks = len(arr) // args.seq_len
    arr = arr[:n_blocks * args.seq_len].reshape(n_blocks, args.seq_len)
    perm = rng.permutation(n_blocks)
    n_valid = max(16, int(n_blocks * args.valid_frac))
    valid = arr[perm[:n_valid]]
    train = arr[perm[n_valid:]]

    os.makedirs(args.out_dir, exist_ok=True)
    train.tofile(os.path.join(args.out_dir, "train.bin"))
    valid.tofile(os.path.join(args.out_dir, "valid.bin"))
    meta = {"vocab_size": 8192, "mask_id": tok.mask_id, "eos_id": tok.eos_id,
            "seq_len": args.seq_len,
            "train_blocks": int(train.shape[0]),
            "valid_blocks": int(valid.shape[0]),
            "train_tokens": int(train.size),
            "valid_tokens": int(valid.size)}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"),
              indent=2)
    print("wrote", args.out_dir, json.dumps(meta))


if __name__ == "__main__":
    main()
