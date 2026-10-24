<p align="center">
  <img src="assets/banner.png" alt="XIIV" width="100%"/>
</p>

# XIIV

A language model I'm building from scratch using masked diffusion
instead of autoregressive generation. The idea I'm testing is that
if generation works on a whole sequence at once, you can check the
output against constraints (I call them contracts in the code) with
a separate verifier before you accept it, instead of hoping the
model followed the instruction.

It's a personal research project. It's not finished.

<p align="center">
  <img src="assets/how-it-works.png" alt="How XIIV generates — a masked canvas is filled in in parallel and checked against its contracts" width="100%"/>
</p>

## Status (Oct 2026)

- 133M — trained first, used as the teacher / starting point
- 500M — grown from the 133M checkpoint
- 1B — grown from the 500M checkpoint, training in blocks:
  blocks 0–4 done (622.5M parameters), blocks 5–7 still to go.
  The assembled 1B hasn't been evaluated end to end yet.

Earlier README versions claimed more than this. This is what's
actually true today.

## Layout

- `xiiv/model.py` — transformer model
- `xiiv/diffusion.py` — sampling / denoising loop
- `xiiv/contracts.py` — contract definitions and verification
- `xiiv/train.py` — normal training loop
- `xiiv/sgbd_train_block.py` — block-wise training for the 1B
- `tests/` — tests for the sampler, contracts, growth and repair code
- `docs/` — model cards and training notes (some are out of date,
  the dates on them are the ones to trust)
- `research/` — experiments, most of them small probes with a
  pass/fail written down before running them

## Model specs (1B)

- 36 layers, d=1280, 20 heads
- SwiGLU feed-forward, 3584 wide
- Vocab 32,768, tied input/output embeddings
- Bidirectional attention with RoPE
- Context 128 tokens (small — known limitation)

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

Loading a checkpoint:

```python
from xiiv import XIIVModel

model = XIIVModel.from_checkpoint("checkpoints_1b_gn/ckpt_8218.pt")
```

## Training notes

The 1B is trained in 8 blocks (SGBD in the code) against cached
activations from the 133M model, because I don't have a GPU that
can train it in one go. Blocks 3 and 4 were done on Kaggle T4s.
Logs are in `sgbd_blocks/` next to the checkpoints.

Block 2 failed four times before finishing. Most failures were
memory related.

## Known issues

- 128 token context is too small for anything real
- No instruction tuning or chat template
- Some docs in `docs/` describe older checkpoints — check dates
- Growth code for the 1B had a bug in the key/value widening
  (fixed Oct 2026, see git history). The checkpoint produced by
  the old script needs to be regenerated.

## License

Apache-2.0
