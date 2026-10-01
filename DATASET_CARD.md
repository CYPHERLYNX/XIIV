# XIIV dataset card (data contract — ecc_mle_workflow §2)

## Snapshot

| field | value |
|---|---|
| snapshot id | `tinystories-v2-gpt4-20261001` (hashes in `data/dataset_snapshot.json`) |
| source | `roneneldan/TinyStories` — `TinyStoriesV2-GPT4-train.txt` (2.2 GB) and `TinyStoriesV2-GPT4-valid.txt` (22.5 MB), CC-BY |
| tokenizer | byte-level BPE, vocab 8192, trained on a 50 MB sample of the train file; `[MASK]`=2, `[EOS]`=3 (`data/tokenizer.json`) |
| license | TinyStories is CC-BY; the tokenized derivatives inherit it |

## Split policy (explicit)

| split | source | content | blocks × 128 | tokens | used for |
|---|---|---|---|---|---|
| train | train file, chars [0, 150M) | `data/train.bin` | 285,545 | 36.5M | training only |
| valid | valid file, chars [0, 5M) | `data/valid.bin` | 9,522 | 1.2M | checkpoint selection (`best.pt`), LR/mask-schedule decisions |
| test | valid file, chars [10M, 12M) | `data/test.bin` | ~15,600 | ~2.0M | FINAL reporting only — never used for any training or selection decision |

Entity grain: one 128-token block (story fragments; blocks are NOT story-aligned,
so no story can straddle splits — see leakage note).

## Leakage analysis (checked, not assumed)

1. **Upstream split integrity.** The TinyStoriesV2 train/valid files are disjoint
   story sets by construction (separate generation runs). Verified empirically:
   `scripts/dataset_audit.py` hashes every 128-token block in train.bin and
   valid.bin and reports exact block overlap — **0 overlapping blocks**.
2. **Valid/test separation.** test.bin is carved from a disjoint char range of
   the valid file ([10M, 12M) vs valid's [0, 5M)), with a 5 MB gap, so no
   128-token block can appear in both. The audit script re-verifies this.
3. **Selection-bias note (honest).** `best.pt` was selected on valid.bin, so
   valid numbers are selection-biased (optimistic). All headline numbers are
   reported on **test.bin**, which no training or selection decision ever saw.

## Reproduction

```bash
.venv/bin/python scripts/prepare_data.py --out-dir data \
    --train-chars 150000000 --valid-chars 5000000
.venv/bin/python scripts/dataset_audit.py --data data   # hashes + leakage check
```

Raw text files (`TinyStoriesV2-GPT4-*.txt`, 2.2 GB) are NOT committed to the
repo; the snapshot hashes let anyone verify a re-download is bit-identical.
