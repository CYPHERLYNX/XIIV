#!/usr/bin/env python3
"""Expand the tokenizer vocabulary for a growth rung: 8k -> 32k.

Frontier labs train with 50k-128k vocabularies; our 8k BPE is a toy by
comparison. At our scale and token budget, 32k is the honest choice:
100k would leave most embedding rows starved of updates on a ~20M-token
budget, while 32k keeps the embedding fraction sane (~12% at the 200M
rung) and every row trainable.

Lineage-safe procedure (run BEFORE grow_checkpoint.py --vocab-map at the
growth event):
  1. Decode data_all/train.bin + valid.bin with the OLD tokenizer back to
     text (byte-level BPE decode is lossless).
  2. Train a new byte-level BPE (32768 pieces) on a 50M-char sample
     -> weights/tokenizer_32k.json
  3. Build weights/vocab_map_8k_32k.json: for each new id, either
     {"old_id": i} when the surface form survives, or
     {"pieces": [old_ids]} for genuinely new tokens (initialized from the
     mean of their old-token pieces by grow_checkpoint.py).
  4. Re-tokenize the full corpus with the new tokenizer
     -> data_32k/{train,valid}.bin + meta.json

Example:
  .venv/bin/python scripts/expand_vocab.py \
      --old-tokenizer weights/tokenizer.json \
      --old-data data_all \
      --new-tokenizer weights/tokenizer_32k.json \
      --vocab-map weights/vocab_map_8k_32k.json \
      --out-data data_32k \
      --vocab-size 32768
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiiv.tokenizer import XIIVTokenizer

SEQ_LEN = 128
SPECIALS = ["[UNK]", "[PAD]"]  # + mask/eos read from the tokenizer module


def _read_bins(data_dir):
    out = {}
    for name in ("train", "valid"):
        p = os.path.join(data_dir, f"{name}.bin")
        arr = np.memmap(p, dtype=np.uint16, mode="r")
        out[name] = np.asarray(arr).reshape(-1, SEQ_LEN)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-tokenizer", required=True)
    ap.add_argument("--old-data", required=True)
    ap.add_argument("--new-tokenizer", required=True)
    ap.add_argument("--vocab-map", required=True)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--tok-train-chars", type=int, default=50_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    old_tok = XIIVTokenizer(args.old_tokenizer)
    print(f"old tokenizer: vocab={old_tok.vocab_size} "
          f"mask_id={old_tok.mask_id} eos_id={old_tok.eos_id}", flush=True)

    # 1) split the id stream on eos_id -> docs, then decode each doc.
    # Splitting on ids (not the decoded EOS surface string) is robust to
    # the HF decoder's skip_special_tokens behaviour. Chunked so no giant
    # Python int list is ever materialised.
    texts = {}
    for name, blocks in _read_bins(args.old_data).items():
        ids = blocks.reshape(-1)
        docs = []
        cur = []          # list of id-chunks for the doc in progress
        CH = 2_000_000
        n = len(ids)
        for s in range(0, n, CH):
            chunk = ids[s:s + CH]
            pos = np.flatnonzero(chunk == old_tok.eos_id)
            prev = 0
            for p in pos:
                if p > prev:
                    cur.append(chunk[prev:p])
                if cur:
                    doc = np.concatenate(cur) if len(cur) > 1 else cur[0]
                    if len(doc):
                        docs.append(old_tok.decode(doc.tolist()))
                    cur = []
                prev = int(p) + 1
            if prev < len(chunk):
                cur.append(chunk[prev:])
            print(f"  {name}: split {min(s+CH, n)/1e6:.0f}M/"
                  f"{n/1e6:.0f}M tokens", flush=True)
        if cur:  # trailing doc without a final eos
            doc = np.concatenate(cur) if len(cur) > 1 else cur[0]
            if len(doc):
                docs.append(old_tok.decode(doc.tolist()))
        texts[name] = docs
        print(f"{name}: {len(docs)} docs, "
              f"{sum(len(d) for d in docs)/1e6:.1f} MB text", flush=True)

    # 2) train the new BPE on a sample of train text
    sample_p = os.path.join(args.out_data, "_tok_sample.txt")
    os.makedirs(args.out_data, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(texts["train"]))
    with open(sample_p, "w", encoding="utf-8") as f:
        n = 0
        for i in order:
            d = texts["train"][i]
            f.write(d + "\n")
            n += len(d) + 1
            if n >= args.tok_train_chars:
                break
    print(f"training {args.vocab_size}-piece BPE ...", flush=True)
    new_tok = XIIVTokenizer.train([sample_p], args.new_tokenizer,
                                  vocab_size=args.vocab_size)
    os.remove(sample_p)
    print(f"new tokenizer: vocab={new_tok.vocab_size} "
          f"mask_id={new_tok.mask_id} eos_id={new_tok.eos_id}", flush=True)
    assert new_tok.mask_id == old_tok.mask_id
    assert new_tok.eos_id == old_tok.eos_id

    # 3) surface-form map old -> new
    old_surf_to_id = {}
    for i in range(old_tok.vocab_size):
        s = old_tok._tok.id_to_token(i)
        old_surf_to_id[s] = i
    vmap = {}
    n_survived, n_new = 0, 0
    for nid in range(new_tok.vocab_size):
        surf = new_tok._tok.id_to_token(nid)
        if surf in old_surf_to_id:
            vmap[str(nid)] = {"old_id": old_surf_to_id[surf]}
            n_survived += 1
        else:
            text = new_tok._tok.decode([nid])
            pieces = [t for t in old_tok.encode(text)
                      if t not in (old_tok.mask_id, old_tok.eos_id)]
            assert pieces, f"new token {nid} has no old pieces"
            vmap[str(nid)] = {"pieces": pieces}
            n_new += 1
    with open(args.vocab_map, "w") as f:
        json.dump(vmap, f)
    print(f"vocab map: {n_survived} survived, {n_new} new "
          f"-> {args.vocab_map}", flush=True)

    # 4) re-tokenize with the new tokenizer, repack blocks.
    # Stream chunk arrays (uint16) instead of one giant Python int list.
    meta = {"vocab_size": args.vocab_size, "mask_id": new_tok.mask_id,
            "eos_id": new_tok.eos_id, "seq_len": SEQ_LEN}
    for name, docs in texts.items():
        chunks = []
        for i in range(0, len(docs), 5000):
            enc = new_tok._tok.encode_batch(docs[i:i + 5000])
            buf = []
            for e in enc:
                buf.extend(e.ids)
                buf.append(new_tok.eos_id)
            chunks.append(np.array(buf, dtype=np.uint16))
            del buf
            if (i // 5000) % 10 == 0:
                print(f"  {name}: {min(i+5000, len(docs))}/{len(docs)} docs",
                      flush=True)
        arr = np.concatenate(chunks)
        del chunks
        assert arr.max() < 2 ** 16, "vocab exceeds uint16"
        n_blocks = len(arr) // SEQ_LEN
        arr = arr[:n_blocks * SEQ_LEN].reshape(n_blocks, SEQ_LEN)
        out_p = os.path.join(args.out_data, f"{name}.bin")
        fp = np.memmap(out_p, dtype=np.uint16, mode="w+",
                       shape=(n_blocks, SEQ_LEN))
        fp[:] = arr
        fp.flush()
        meta[f"{name}_blocks"] = n_blocks
        meta[f"{name}_tokens"] = int(n_blocks * SEQ_LEN)
        print(f"{name}: {n_blocks} blocks, "
              f"{n_blocks * SEQ_LEN / 1e6:.1f}M tokens", flush=True)
    with open(os.path.join(args.out_data, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("done.")


if __name__ == "__main__":
    main()
