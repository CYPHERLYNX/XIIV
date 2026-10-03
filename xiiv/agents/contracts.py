"""Agent behavior contracts: one application of the guarantee architecture.

Built on xiiv.contracts.Contract — the unified guarantee path. A
contract declares, for an agent's action plan:
- ``policy``: locked preamble, anchored byte-exact at the plan start.
- ``must_do``: terms that MUST appear (required_ids).
- ``never_do``: terms that MUST NOT appear (forbidden_ids).

This module is an application demo of the core (xiiv/diffusion.py's
DenoisingState), not the core itself.
"""
import torch

from ..contracts import Contract, ContractError
from ..diffusion import sample_trajectory

TOTAL_LENGTH = 128

__all__ = ["ContractError", "compile_contract", "generate_plan"]


def compile_contract(tokenizer, policy="", must_do=(), never_do=()):
    """Compile to a Contract. Raises ContractError on malformed input.

    Validation is eager: multi-token require/forbid terms raise here,
    not at generation time.
    """
    c = Contract()
    if policy:
        c.lock(policy)
    c.require(*must_do)
    c.forbid(*never_do)
    c.exact_length(TOTAL_LENGTH)
    c.compile(tokenizer, prompt_len=0)  # fail fast on malformed terms
    return c


def generate_plan(model, task, policy="", must_do=(), never_do=(),
                  steps=16, seed=7, temperature=1.0, top_p=0.95,
                  enforce=True):
    """Generate an agent action plan under a behavior contract.

    Returns {"plan", "guaranteed", "check"}. With enforce=False the
    contract is pasted as ordinary prompt context (baseline).
    """
    tok = model.tok
    task_ids = tok.encode("Task: " + task.strip() + "\nPlan:")
    prompt_ids = torch.tensor(task_ids, dtype=torch.long)

    constraints, verified = None, False
    if enforce:
        contract = compile_contract(tok, policy=policy, must_do=must_do,
                                    never_do=never_do)
        constraints, _ = contract.compile(tok, prompt_len=len(task_ids))
        verified = True
    else:
        ctx = ("Policy: " + policy + "\nMust do: " + ", ".join(must_do)
               + "\nNever do: " + ", ".join(never_do)
               + "\nTask: " + task.strip() + "\nPlan:")
        prompt_ids = torch.tensor(tok.encode(ctx)[:100], dtype=torch.long)

    ids, _ = sample_trajectory(
        model.model, TOTAL_LENGTH, tok.mask_id, tok.eos_id,
        steps=steps, temperature=temperature, top_p=top_p, seed=seed,
        prompt_ids=prompt_ids,
        constraints=constraints if verified else None,
        device=model.device)
    check = (Contract.verify(ids, constraints, tok.mask_id)
             if verified else {})
    plan = tok.decode(ids.tolist()[len(prompt_ids):])
    guaranteed = verified and Contract.holds(check)
    return {"plan": plan, "guaranteed": guaranteed, "check": check}
