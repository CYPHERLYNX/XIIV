"""Example 1: Generate text with hard contract constraints.

The contract is enforced structurally during denoising — not via prompting.
"""
from xiiv import XIIVModel
from xiiv.diffusion import sample_trajectory
from tokenizers import Tokenizer

model = XIIVModel.from_checkpoint("checkpoints_1b_gn/ckpt_8218.pt")
tok = Tokenizer.from_file("weights/tokenizer_32k.json")

# Contract: must include "quantum", must NOT include "banana"
prompt = tok.encode("The future of computing is").ids
must_include = tok.encode("quantum").ids
forbidden = tok.encode("banana").ids

tokens, cert = sample_trajectory(
    model,
    prompt_ids=prompt,
    contract_ids=must_include,
    forbidden_ids=forbidden,
    steps=32,
)
print(tok.decode(tokens))
assert cert.verify(), "independent verification failed"
print("✓ contract verified by independent checker")
