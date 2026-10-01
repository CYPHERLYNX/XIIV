"""Prepare XIIV training data.

1. Downloads TinyStories (train/valid) from Hugging Face.
2. Trains a byte-level BPE tokenizer (default 8192 pieces) on a sample.
3. Tokenizes a text shard and writes packed uint16 memmap blocks of seq_len,
   with [EOS] separators between stories. A tail slice becomes validation.

Outputs (under --out-dir):
    tokenizer.json, train.bin (uint16 memmap), valid.bin, meta.json
"""
import argparse
import json
import os
import sys

# allow running as `python scripts/prepare_data.py` from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import numpy as np


def download_tinystories(cache_dir: str):
    # Downloaded with curl (huggingface_hub's httpx chokes on this sandbox's
    # proxy env vars). Files land next to this script's --out-dir.
    here = os.path.dirname(os.path.abspath(__file__))
    train = os.path.join(here, "..", "data", "TinyStoriesV2-GPT4-train.txt")
    valid = os.path.join(here, "..", "data", "TinyStoriesV2-GPT4-valid.txt")
    for p in (train, valid):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing {p}: download it first, e.g.\n"
                "curl -sL -o data/TinyStoriesV2-GPT4-train.txt "
                "https://huggingface.co/datasets/roneneldan/TinyStories/"
                "resolve/main/TinyStoriesV2-GPT4-train.txt")
    return train, valid


def read_shard(path: str, max_chars: int) -> str:
    # TinyStories stories are separated by <|endoftext|>
    buf = []
    total = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            total += len(line)
            if total >= max_chars:
                break
    return "".join(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--cache-dir", default="data/hf_cache")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--train-chars", type=int, default=400_000_000,
                    help="how much raw train text to tokenize (~chars)")
    ap.add_argument("--valid-chars", type=int, default=5_000_000)
    ap.add_argument("--tok-train-chars", type=int, default=50_000_000,
                    help="how much text to train the BPE tokenizer on")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_txt, valid_txt = download_tinystories(args.cache_dir)
    print(f"train txt: {train_txt} ({os.path.getsize(train_txt)/1e9:.2f} GB)")
    print(f"valid txt: {valid_txt} ({os.path.getsize(valid_txt)/1e9:.2f} GB)")

    # 1) tokenizer
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    tok_sample = os.path.join(args.out_dir, "_tok_sample.txt")
    with open(tok_sample, "w", encoding="utf-8") as f:
        f.write(read_shard(train_txt, args.tok_train_chars))
    if not os.path.exists(tok_path):
        from xiiv.tokenizer import XIIVTokenizer
        print("training BPE tokenizer ...")
        XIIVTokenizer.train([tok_sample], tok_path,
                              vocab_size=args.vocab_size)
    os.remove(tok_sample)

    from xiiv.tokenizer import XIIVTokenizer
    tok = XIIVTokenizer(tok_path)
    print(f"tokenizer ready: vocab={tok.vocab_size} mask_id={tok.mask_id} "
          f"eos_id={tok.eos_id}")

    # 2) tokenize shards -> packed blocks (in chunks: encode_batch on the
    #    full story list at once can OOM small machines)
    def process(txt_path, max_chars, name, chunk=5000):
        text = read_shard(txt_path, max_chars)
        stories = [s for s in text.split("<|endoftext|>") if s.strip()]
        print(f"{name}: {len(stories)} stories, {len(text)/1e6:.1f} MB text",
              flush=True)
        ids = []
        for i in range(0, len(stories), chunk):
            enc = tok._tok.encode_batch(stories[i:i + chunk])
            for e in enc:
                ids.extend(e.ids)
                ids.append(tok.eos_id)
            if (i // chunk) % 10 == 0:
                print(f"  ... {min(i + chunk, len(stories))}/{len(stories)} "
                      f"stories", flush=True)
        arr = np.array(ids, dtype=np.uint16)
        assert arr.max() < 2**16, "vocab exceeds uint16"
        n_blocks = len(arr) // args.seq_len
        arr = arr[: n_blocks * args.seq_len].reshape(n_blocks, args.seq_len)
        out = os.path.join(args.out_dir, f"{name}.bin")
        fp = np.memmap(out, dtype=np.uint16, mode="w+",
                       shape=(n_blocks, args.seq_len))
        fp[:] = arr
        fp.flush()
        del fp
        print(f"{name}: {n_blocks} blocks x {args.seq_len} "
              f"= {n_blocks * args.seq_len / 1e6:.1f}M tokens -> {out}")
        return n_blocks

    n_train = process(train_txt, args.train_chars, "train")
    n_valid = process(valid_txt, args.valid_chars, "valid")

    meta = {"vocab_size": tok.vocab_size, "mask_id": tok.mask_id,
            "eos_id": tok.eos_id, "seq_len": args.seq_len,
            "train_blocks": n_train, "valid_blocks": n_valid,
            "train_tokens": n_train * args.seq_len,
            "valid_tokens": n_valid * args.seq_len}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("meta:", json.dumps(meta))


if __name__ == "__main__":
    main()
