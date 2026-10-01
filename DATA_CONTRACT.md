# XIIV Data Contract — snapshot 2026-10-01

## Source
- Corpus: TinyStoriesV2-GPT4 (train + valid .txt as shipped upstream).
- This contract does NOT re-assert upstream licensing; it pins the exact bytes used.

## Pinned files (SHA256, first 16 hex)
| file | bytes | sha16 |
|---|---|---|
| TinyStoriesV2-GPT4-train.txt | 2,227,753,166 | ab10e146618028c2 |
| TinyStoriesV2-GPT4-valid.txt | 22,502,601 | 6874bae9a4c1a4e7 |
| train.bin | 73,099,520 | bdd97b2246fb2b94 |
| valid.bin | 2,437,632 | 3d9c58dc019eb1dc |
| tokenizer.json | 560,958 | bc2420b75b730880 |

Full hashes: `data/data_provenance.json`.

## Tokenizer
Byte-level BPE, vocab 8192, trained on the train split (see `scripts/prepare_data.py`).
Special ids: mask=2, eos=3 (recorded in meta.json; model code never assumes defaults).

## Blocking / split policy
- Fixed 128-token blocks, row-major from the token stream. No shuffling across
  block boundaries at rest; the training RNG draws block indices per batch.
- Split is upstream's train/valid file split (not random). No test set is claimed;
  "valid" is used for BOTH early stopping and reporting — documented limitation.
- Leakage check (2026-10-01): SHA256 of every 7th 128-token block, train vs valid →
  **0 overlaps** in 40,793 train / 1,361 valid sampled blocks.

## Preprocessing (must match to reproduce)
`scripts/prepare_data.py`: read txt → tokenizer encode → pack into seq_len=128
blocks (truncate tail) → write uint16 memmap bins + meta.json.
Determinism: tokenizer training seed fixed in script; bin packing is order-stable.

## Known limitations
- Single-domain (children's stories) — claims about general language ability are
  out of scope; the model's defensible claim is structural control, not knowledge.
- No decontamination against public benchmarks was needed (no benchmark claims made
  on this data); if benchmark numbers are ever reported, re-run decontamination.
- Valid split doubles as selection + report split (see above).

## Reproduce
```bash
cd ~/workspace/ai-tools/xiiv
.venv/bin/python scripts/prepare_data.py   # regenerates bins from pinned txt
.venv/bin/python -m xiiv.train --data data --ckpt-dir checkpoints \
    --total-steps 5000 --batch-size 48 --bf16 --threads 2
```
Expected: run_config.json hash must match the compact for a like-for-like run.
