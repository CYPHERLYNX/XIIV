"""Example 2: Natively infill a masked span — no prompting tricks."""
from xiiv import XIIVModel
from xiiv.diffusion import sample_trajectory
from tokenizers import Tokenizer

model = XIIVModel.from_checkpoint("checkpoints_1b_gn/ckpt_8218.pt")
tok = Tokenizer.from_file("weights/tokenizer_32k.json")

# "The [MASK] sat on the mat" — model fills the mask in parallel
prefix = tok.encode("The").ids
suffix = tok.encode("sat on the mat.").ids
mask_id = tok.token_to_id("[MASK]")

prompt = prefix + [mask_id] + suffix
tokens, cert = sample_trajectory(model, prompt_ids=prompt, steps=16)
print(tok.decode(tokens))
