"""Guarantee Brick demo: generate under proof.

Shows the core XIIV thesis in one run:
  1. A generator left alone FAILS the contract silently (this is what
     every frontier API does -- hope, not proof).
  2. The same generator behind a MustIncludeBrick SATISFIES the contract
     and emits a proof certificate.
  3. The independent verifier confirms the certificate.
  4. Any tampering with output or certificate is REJECTED.

Uses adversarial dummy models (no weights needed): the guarantee holds
by construction, not by luck.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tests.test_bricks import (PhraseSuppressor, clean_ids, has_phrase,
                               TOKENIZER, VOCAB, MASK_ID, EOS_ID)
from xiiv.bricks import MustIncludeBrick
from xiiv.bricks.verify import verify_certificate
from xiiv.diffusion import DenoisingState
from xiiv.tokenizer import XIIVTokenizer


def main():
    tok = XIIVTokenizer(TOKENIZER)
    phrase = "quantum leap"
    pids = clean_ids(tok, phrase)
    contract = {"require_phrase": phrase}

    print("=== 1. Generator alone (the giants' approach: hope) ===")
    prompt = torch.randint(100, 1000, (8,))
    st = DenoisingState(length=64, mask_id=MASK_ID, eos_id=EOS_ID,
                        steps=8, seed=0, prompt_ids=prompt,
                        constraints=None, vocab_size=VOCAB, device="cpu")
    model = PhraseSuppressor(pids, seed=1)
    while st.step(model):
        pass
    ids = st.x.tolist()
    print(f"phrase present: {has_phrase(ids, pids)}  <- silent failure, "
          f"no proof offered")

    print("\n=== 2. Same generator behind a MustIncludeBrick ===")
    brick = MustIncludeBrick(phrase, pids)
    st = DenoisingState(length=64, mask_id=MASK_ID, eos_id=EOS_ID,
                        steps=8, seed=0, prompt_ids=prompt,
                        constraints=None, vocab_size=VOCAB, device="cpu")
    brick.allocate(64, occupied=set(), prompt_len=8)
    brick.enforce(st)
    while st.step(PhraseSuppressor(pids, seed=1)):
        pass
    ids = st.x.tolist()
    print(f"phrase present: {has_phrase(ids, pids)}")
    print(f"certified text: ...{tok.decode(ids[brick.positions[0]-8: brick.positions[-1]+8])}...")

    print("\n=== 3. Proof certificate ===")
    cert = brick.certify(ids, contract)
    for k in ("brick", "phrase", "positions", "contract_hash",
              "output_hash", "cert_hash"):
        v = cert[k]
        print(f"  {k}: {str(v)[:48]}")

    print("\n=== 4. Independent verifier ===")
    result = verify_certificate(cert, ids, contract)
    print(f"  honest output : {'VERIFIED' if result.ok else 'REJECTED'}")

    attacked = list(ids)
    for pos in cert["positions"]:
        attacked[pos] = 100
    result = verify_certificate(cert, attacked, contract)
    print(f"  tampered output: {'VERIFIED' if result.ok else 'REJECTED'} "
          f"({result.reason[:60]}...)")


if __name__ == "__main__":
    main()
