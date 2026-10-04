import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from xiiv import load_ckpt_trusted
from xiiv.config import ModelConfig
from xiiv.model import XIIVModel
from xiiv.tokenizer import XIIVTokenizer
from xiiv.diffusion import sample_trajectory

def load(p):
    ck = load_ckpt_trusted(p)
    cfg_d = ck["config"]
    cfg = ModelConfig.from_json(cfg_d) if isinstance(cfg_d, str) else ModelConfig(**cfg_d)
    m = XIIVModel(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    m.load_state_dict(state, strict=False)
    return m.eval()

tok = XIIVTokenizer("data/tokenizer.json")
books = load("checkpoints_books/ckpt_2000.pt")
prompt = "The main advantage of open-source software over proprietary alternatives is"
for seed in [7, 42]:
    for temp in [0.7, 1.0]:
        for steps in [16, 32]:
            pids = torch.tensor(tok.encode(prompt), dtype=torch.long)
            ids, _ = sample_trajectory(books, 64, tok.mask_id, tok.eos_id,
                                       steps=steps, temperature=temp, seed=seed,
                                       prompt_ids=pids, device="cpu")
            print(f"--- seed={seed} temp={temp} steps={steps}")
            print(tok.decode(ids.tolist()))
            print()
