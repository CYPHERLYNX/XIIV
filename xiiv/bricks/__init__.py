"""Guarantee Bricks: per-guarantee-type enforcement micro-modules.

Each brick is an independent, deterministic enforcer for ONE contract
primitive (must-include, never-include, ordering, cardinality, ...).
Bricks are generator-agnostic: they sit between any generator and its
output, enforce their guarantee, and emit a proof certificate that a
tiny independent verifier can check without trusting the generator
or the brick.

Design rules:
  * Bricks are deterministic transducers, not neural nets. They are
    small enough to be exhaustively tested and audited.
  * A brick never modifies the sampler: it *drives* it through the
    sampler's own pinning/masking primitives.
  * Every enforcement act produces a certificate. No certificate,
    no claim: verification is fail-closed.
  * The verifier (xiiv.bricks.verify) is independent: it re-derives
    everything from (certificate, output, contract) and shares no
    enforcement code with the bricks.
"""
from .must_include import MustIncludeBrick

BRICKS = {
    "must-include": MustIncludeBrick,
}

__all__ = ["BRICKS", "MustIncludeBrick"]
