"""Quality measurement: books fine-tune (ckpt_2000) vs 52.5M base.
Same prompts, same seeds, both models. Writes markdown comparison."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from xiiv import load_ckpt_trusted
from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.tokenizer import XIIVTokenizer
from xiiv.diffusion import sample_trajectory

def load(ckpt_path):
    ck = load_ckpt_trusted(ckpt_path)
    cfg_d = ck["config"]
    cfg = ModelConfig.from_json(cfg_d) if isinstance(cfg_d, str) else ModelConfig(**cfg_d)
    m = XIIVModel(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    m.eval()
    return m, cfg

tok = XIIVTokenizer("data/tokenizer.json")
PROMPTS = [
    ("descriptive-tech", "A convolutional neural network processes images by"),
    ("analytical", "The main advantage of open-source software over proprietary alternatives is"),
    ("story-control", "Once upon a time in a small village"),
    ("descriptive-nature", "The old lighthouse stood on the cliff, and"),
]
SEEDS = [7, 7, 7, 7]
LENGTH, STEPS = 96, 16

base, _ = load("checkpoints_50m/best.pt")
books, _ = load("checkpoints_books/ckpt_2000.pt")

out = ["# Books fine-tune vs 52.5M base — quality comparison",
       f"_generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, length={LENGTH} tok, steps={STEPS}_", ""]
for (name, prompt), seed in zip(PROMPTS, SEEDS):
    out.append(f"## {name}\n_prompt: \"{prompt}\" seed={seed}_\n")
    for label, model in [("52.5M base (stories)", base), ("books fine-tune (ckpt_2000)", books)]:
        pids = torch.tensor(tok.encode(prompt), dtype=torch.long)
        ids, conf = sample_trajectory(model, LENGTH, tok.mask_id, tok.eos_id,
                                      steps=STEPS, seed=seed, prompt_ids=pids, device="cpu")
        txt = tok.decode(ids.tolist())
        out.append(f"**{label}:**\n> {txt}\n")
    out.append("")

path = "BOOKS_QUALITY_REPORT.md"
open(path, "w").write("\n".join(out))
print("wrote", path)
