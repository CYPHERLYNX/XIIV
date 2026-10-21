"""Invention #1 discriminating experiment.

Invention #1: proof-difficulty-adaptive diffusion sampling — allocate
denoising compute by predicted certificate-construction difficulty.
Prior-art verdict (research/invention01_prior_art.md): NOVEL
conditionally. The kill condition: if certificate difficulty is just
token entropy in disguise, the invention collapses to 'CadLLM with
extra steps.'

This experiment is the FIRST gate: do positions exist that are
  (a) high model confidence (low entropy — cheap by confidence), AND
  (b) high certificate difficulty (expensive by guarantee load)?
If the confident-but-hard quadrant is substantially populated, the two
signals are genuinely distinct and the invention survives the gate.
If confidence and guarantee load are near-perfectly correlated, KILL.

Operationalization:
  - Model: tiny XIIV trained briefly on template-structured data so its
    confidence carries real signal (an untrained model's entropy is
    uniform noise and the test would be meaningless).
  - Confidence: model's top-1 probability at each masked position on a
    fully-masked canvas under constraints.
  - Certificate difficulty proxy: GLOD's guarantee_load_priority —
    constraint-interaction density per position (frozen proximity,
    required-count tightness, forbidden-phrase density, at_most
    pressure). This is a static function of the constraint structure,
    computed WITHOUT the model — exactly the independence the invention
    claims.
  - Stats: Spearman correlation over positions; population of the
    top-quartile-confidence AND top-quartile-load quadrant (chance =
    6.25% under independence; near 0% under redundancy).
"""
import torch
import torch.nn.functional as F

from xiiv.config import ModelConfig
from xiiv.diffusion import guarantee_load_priority
from xiiv.model import XIIVModel

torch.manual_seed(0)

VOCAB = 256
LEN = 48


def tiny_cfg():
    return ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2,
                       n_heads=4, ffn_dim=128, mask_id=254, eos_id=255,
                       seq_len=64)


def make_data(n, gen):
    """Template-structured sequences: 6 blocks of 8 from 10 templates."""
    templates = torch.randint(0, 200, (10, 8), generator=gen)
    bids = torch.randint(0, 10, (n, 6), generator=gen)
    x = templates[bids].reshape(n, LEN).clone()
    noise = torch.rand(n, LEN, generator=gen) < 0.1
    x = torch.where(noise, torch.randint(0, 200, (n, LEN), generator=gen), x)
    return x


def train_briefly(m, steps=250, bs=64):
    gen = torch.Generator().manual_seed(7)
    data = make_data(4096, gen)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for s in range(steps):
        idx = torch.randint(0, 4096, (bs,))
        x = data[idx]
        mask = torch.rand(bs, LEN) < 0.5
        xm = torch.where(mask, torch.full_like(x, 254), x)
        logits = m(xm)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


def spearman(a, b):
    """Rank correlation without scipy."""
    def ranks(v):
        order = v.argsort()
        r = torch.empty_like(order, dtype=torch.float)
        r[order] = torch.arange(len(v), dtype=torch.float)
        return r
    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm())).item()


def main():
    cfg = tiny_cfg()
    m = XIIVModel(cfg)
    m.init_weights()
    m = train_briefly(m)
    m.eval()

    confs, loads = [], []
    gen = torch.Generator().manual_seed(123)
    frozen = torch.zeros(LEN, dtype=torch.bool)
    frozen[::12] = True  # anchor positions every 12
    with torch.no_grad():
        for trial in range(40):
            # randomized constraint sets each trial
            req = torch.randint(0, 200, (4,), generator=gen).tolist()
            forb = torch.randint(0, 200, (2,), generator=gen).tolist()
            constraints = {"required_ids": req, "forbidden_ids": forb,
                           "at_most": {50: 2}}
            load = guarantee_load_priority(
                LEN, frozen, {}, {t: 1 for t in req},
                {50: 2}, [[f] for f in forb], device="cpu")
            x = torch.full((1, LEN), 254, dtype=torch.long)
            x[0, frozen] = torch.randint(0, 200, (int(frozen.sum()),),
                                         generator=gen)
            logits = m(x)
            probs = logits[0].softmax(-1)
            top1 = probs.max(-1).values
            masked = ~frozen
            confs.append(top1[masked])
            loads.append(load[masked])
    confs = torch.cat(confs)
    loads = torch.cat(loads)
    rho = spearman(confs, loads)
    cq = confs.quantile(0.75)
    lq = loads.quantile(0.75)
    quad = ((confs >= cq) & (loads >= lq)).float().mean().item()
    hard_only = ((confs < cq) & (loads >= lq)).float().mean().item()
    print(f"positions sampled: {len(confs)}")
    print(f"Spearman(confidence, guarantee_load) = {rho:.3f}")
    print(f"confident-but-hard quadrant: {quad * 100:.1f}% of positions "
          f"(independence baseline 6.25%, redundancy ~0%)")
    print(f"hard-but-unconfident quadrant: {hard_only * 100:.1f}%")
    verdict = ("SIGNALS DISTINCT — invention #1 survives gate 1"
               if quad > 0.04 and abs(rho) < 0.85 else
               "SIGNALS REDUNDANT — invention #1 collapses toward CadLLM")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
