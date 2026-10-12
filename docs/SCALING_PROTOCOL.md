# XIIV Scaling Protocol

How every model scale-up is built: to the standard of a frontier-lab
training run (Anthropic/OpenAI methodology), not a bigger guess.
This document governs every rung of the ladder. Standing rule per
Fahad, 2026-10-07.

## The ladder

| Rung | Params (exact) | Arch | Status |
|------|---------------|------|--------|
| 52.5M baseline | 52,508,736 | d=576, L=12, h=9, ffn=1536 | done 2026-10-02 |
| 133M guarantee-native | 133,520,000 (~133.52M) | d=704, L=15, h=11, ffn=2048 | trained to step 8218, then superseded |
| 200M guarantee-native | ~204.7M | d=768, L=18, h=12, ffn=2048, vocab 32768 | SKIPPED per Fahad 2026-10-08: direct 133M→700M |
| 500M guarantee-native | 503,450,000 (~503.45M) | d=1024, L=27, h=16, ffn=2816, vocab 32768 | GROWN + VERIFIED 2026-10-09 on this box (drift ratio 0.113 < 0.6; forward smoke test finite); not yet trained at 500M — training is the next step |
| 700M guarantee-native | 697,930,000 (~697.93M) | d=1152, L=30, h=18, ffn=3200, vocab 32768 | growth staged 2026-10-08 (script verified 7/7, full-scale growth never completed — this box lacks the RAM); grown-ckpt build runs on his PC via `grow_xiiv_700m.ps1`, then RTX 4050 training |

**One lineage, not separate models.** Each rung GROWS the previous rung's
weights and continues training — never a from-scratch restart. Growth is
approximately function-preserving: Net2Net-style widening (new dims/heads
replicate existing ones with split outgoing weights, plus small
symmetry-breaking noise so replicas don't stay redundant forever) and
identity depth insertion (new blocks zero-initialized: exact identity).
Exact preservation is impossible through RMSNorm (the rms statistic
changes when dims are added) — the honest claim is *close*: the grown
model starts far nearer the parent than a random init (measured), and the
growth script aborts past its drift tolerance. The optimizer always
restarts fresh at a growth event (shapes change); step count and RNG
state carry forward. Growth is performed by `scripts/grow_checkpoint.py`,
verified by `tests/test_grow.py`.

**Vocabulary grows with the model.** Frontier labs train with 50k-128k
vocabularies; XIIV's 8k BPE grows to 32k at the 700M growth event. 100k
would starve most embedding rows on a ~20M-token budget; 32k keeps
embeddings ~6% of params with every row trainable. Lineage-safe
procedure (`scripts/expand_vocab.py`): split the old corpus on eos_id
and decode (byte-level BPE decode is lossless), train the new BPE on it,
keep trained rows for surviving surface forms, initialize genuinely new
tokens from the mean of their old-token pieces, re-tokenize the corpus
(`data_32k/`). The contract encoder's token embedding follows the same
map.

## Rules for every rung

1. **Frozen config, recorded provenance.** Architecture, dataset version
   (hash), code commit, seed, and full CLI args are frozen before launch.
   A run-config hash is written to the train log at startup. Nothing
   changes mid-run; a change means a new rung or a restarted run, named
   as such. For growth rungs, provenance includes the parent checkpoint
   (rung, step, hash) and the growth script's verification result.

2. **Hyperparameters continue by rule, not by feel.** The optimizer
   always restarts fresh at a growth event (tensor shapes change), so the
   new rung gets a short warmup (500 steps). The peak LR follows the
   inverse-width rule from the parent rung's LR: lr_new = lr_parent ×
   d_parent/d_new (wider model → smaller step, µP-style). The cosine
   schedule continues from the growth step over the new total — no
   schedule restart. Batch tokens/step stay fixed unless memory forces a
   change; any change is recorded with the reason. The RNG state carries
   forward in the grown checkpoint, but the data stream is NOT
   bit-identical when the vocabulary changed: the corpus was retokenized
   (same documents, same order, new tokenization), so the token stream
   differs by construction. Document order is preserved; token identity
   is not.

3. **Fixed token budget per growth phase, honestly labeled.** Each growth
   phase trains another 40,000 steps x 512 tokens = 20.48M tokens on top
   of its parent (700M phase: steps 8218→48218 of the lineage). This is
   far below Chinchilla-optimal (~20 tok/param) and is documented as a
   hardware-constrained budget, not presented as compute-optimal.

4. **Identical eval gates every rung.** Same measurements, same code:
   - Validation NLL every 500 steps (8 val batches), full curve kept.
   - Intervention-rate benchmark, identical to the 52.5M baseline
     (raw success 0.00, bricked success 1.00, intervention rate 1.00).
     Promotion requires the intervention rate to move DOWN from 1.00
     while val NLL trends down — that is the guarantee-native thesis
     being tested.
   - Sampler/lock-span regression tests (byte-exact locked-span infill).

5. **Promotion criteria.** A rung is "good" and unlocks the next only if:
   val NLL is still trending down or plateaued at/below the previous
   rung's trajectory, AND the intervention rate improves on the previous
   rung. A rung that only memorizes (low NLL, intervention stuck at
   1.00) does not promote.

6. **Model card per rung.** Architecture, exact param count, data version,
   token budget, eval numbers (final AND best-val), code commit,
   checkpoint location, known limitations. No rung ships without one.

7. **No cherry-picking.** Report final-checkpoint and best-val numbers
   side by side. Every val reading stays in the log. Negative results
   are recorded, not buried.

8. **Guarantee machinery re-verified per rung.** Contract v2 checks,
   brick certificates, and the independent verifier are re-run against
   each trained rung; structural guarantees are re-proven, not assumed
   to transfer.

## Uniqueness guardrails (standing)

XIIV's thesis — generation under proof: contract-native computation,
deterministic guarantee bricks, proof-carrying certificates, independent
verification, intervention rate as the training metric — is the
differentiator at every scale. Scaling must never dilute it:

- The contract path (encoder + per-block cross-attention) grows WITH the
  model at every rung, function-preserving, never dropped or shrunk to
  save parameters.
- Bricks and the independent verifier are re-proven per rung (rule 8).
- Intervention-rate improvement is a promotion criterion (rule 5): a
  bigger model that doesn't improve the guarantee-native metric doesn't
  promote, no matter its NLL.
- Model cards state guarantee capabilities honestly: structural
  compliance only — never claim truth, factuality, or safety beyond what
  the certificates prove.
- Build quality at every rung must stand next to the world's best:
  reproducible, documented, honest numbers. The aim is a model family
  that can be talked about alongside the most powerful in the world —
  on methodology, not hype.
