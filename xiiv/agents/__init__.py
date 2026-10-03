"""Agent behavior contracts: one application of the guarantee architecture.

This module is NOT the core novelty — it applies the sampler's
constraint machinery (required/forbidden/anchored/frozen) to agent
action plans. The core is xiiv/diffusion.py's DenoisingState.
"""
from .contracts import ContractError, compile_contract, generate_plan

__all__ = ["ContractError", "compile_contract", "generate_plan"]
