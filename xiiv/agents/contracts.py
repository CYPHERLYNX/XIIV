"""Agent behavior contracts compiled to sampler constraints.

A contract declares, for an agent's action plan:
- ``policy``: locked preamble, anchored byte-exact at the plan start.
- ``must_do``: terms that MUST appear (required_ids — the sampler cannot
  finish without placing each).
- ``never_do``: terms that MUST NOT appear (forbidden_ids — logits at
  -inf every step).

All terms must be single tokens: the sampler enforces token ids, and a
multi-token word's tokens would scatter instead of forming the word.
``compile_contract`` validates this and raises otherwise.

This is the trust-layer primitive for the agentic era: the same
machinery as faithful text generation, applied to agent action plans.
Demonstration scope — a proof of the mechanism, not a production agent.
"""
import torch

from ..diffusion import check_constraints, sample_trajectory

TOTAL_LENGTH = 128
MAX_POLICY_TOKENS = 40


class ContractError(ValueError):
    """Unsatisfiable or malformed contract."""


def compile_contract(tokenizer, policy="", must_do=(), never_do=()):
    """Compile a behavior contract to sampler constraints.

    Returns (prompt_prefix_ids, constraints). Raises ContractError if a
    term is not a single token or the policy is too long.
    """
    p_ids = [i for i in tokenizer.encode(policy)
             if i not in (tokenizer.mask_id, tokenizer.eos_id)]
    if len(p_ids) > MAX_POLICY_TOKENS:
        raise ContractError(
            f"policy too long: {len(p_ids)} > {MAX_POLICY_TOKENS} tokens")
    constraints = {}
    if p_ids:
        # anchor policy verbatim at the plan start (position a0 = 0 of
        # the generated span; prompt sits before it)
        constraints["_policy_ids"] = p_ids
    required, forbidden = [], []
    for term in must_do:
        ids = [i for i in tokenizer.encode(term)
               if i not in (tokenizer.mask_id, tokenizer.eos_id)]
        if len(ids) != 1:
            raise ContractError(
                f"must_do term {term!r} is {len(ids)} tokens, need exactly 1")
        required.append(ids[0])
    for term in never_do:
        ids = [i for i in tokenizer.encode(term)
               if i not in (tokenizer.mask_id, tokenizer.eos_id)]
        if len(ids) != 1:
            raise ContractError(
                f"never_do term {term!r} is {len(ids)} tokens, need exactly 1")
        forbidden.append(ids[0])
    overlap = set(required) & set(forbidden)
    if overlap:
        raise ContractError("term both required and forbidden")
    if required:
        constraints["required_ids"] = sorted(set(required))
    if forbidden:
        constraints["forbidden_ids"] = sorted(set(forbidden))
    return p_ids, constraints


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
        p_ids, constraints = compile_contract(
            tok, policy=policy, must_do=must_do, never_do=never_do)
        # anchor the policy at the start of the generated span
        a0 = len(task_ids)
        anchors = {a0 + p: t for p, t in enumerate(p_ids)}
        constraints = dict(constraints)
        constraints["anchors"] = anchors
        constraints.pop("_policy_ids", None)
        n_masks = TOTAL_LENGTH - len(task_ids) - len(p_ids)
        n_req = len(constraints.get("required_ids", []))
        if n_masks < n_req:
            raise ContractError("not enough room for required terms")
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
    check = (check_constraints(ids, constraints, tok.mask_id)
             if verified else {})
    plan = tok.decode(ids.tolist()[len(prompt_ids):])
    guaranteed = verified and all(check.get(k, False)
                                  for k in ("required_present",
                                            "forbidden_absent",
                                            "anchors_intact"))
    return {"plan": plan, "guaranteed": guaranteed, "check": check}
