"""Tests for GLOD (Guarantee-Load-Ordered Decoding) — XIIV invention #1.

GLOD reorders the denoising fill priority by per-position
constraint-interaction density (MRV heuristic): positions adjacent to
frozen/anchored text, and canvases under tight required-count /
forbidden-phrase / at_most pressure, are filled first while the canvas
holds maximum flexibility.

These tests verify:
  1. Priority correctness: frozen-adjacent positions outrank distant ones.
  2. Guarantee preservation: every structural guarantee holds under
     order="guarantee_load" (exact length, no masks left, required
     tokens present, forbidden ids absent).
  3. Intervention effect: GLOD does not increase the intervention
     count vs confidence order (and typically reduces endgame
     force-placements) on constraint-dense generations.
"""
import torch

from xiiv.config import ModelConfig
from xiiv.diffusion import (DenoisingState, check_constraints,
                            guarantee_load_priority)
from xiiv.model import XIIVModel


def tiny_cfg(**kw):
    d = dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
             ffn_dim=128, mask_id=254, eos_id=255, seq_len=32)
    d.update(kw)
    return ModelConfig(**d)


def _tiny_model(seed=0):
    cfg = tiny_cfg()
    torch.manual_seed(seed)
    m = XIIVModel(cfg)
    m.init_weights()
    m.eval()
    return m, cfg


def test_glod_priority_frozen_proximity():
    # Positions adjacent to frozen text must outrank distant positions.
    length = 20
    frozen = torch.zeros(length, dtype=torch.bool)
    frozen[10] = True  # single frozen position in the middle
    pr = guarantee_load_priority(length, frozen, {}, {}, {}, [],
                                 device="cpu")
    # adjacent (9, 11) > distance-2 (8, 12) > far (0, 19)
    assert pr[9] > pr[8] > pr[0], \
        f"proximity decay broken: {pr[9]}, {pr[8]}, {pr[0]}"
    assert pr[11] > pr[12] > pr[19]
    # frozen position itself gets proximity load too (distance 0)
    assert pr[10] >= pr[9]
    print("PASS test_glod_priority_frozen_proximity")


def test_glod_priority_uniform_terms():
    # Required/phrase/at_most pressure raises the tier uniformly.
    length = 16
    frozen = torch.zeros(length, dtype=torch.bool)
    pr_plain = guarantee_load_priority(length, frozen, {}, {}, {}, [],
                                       device="cpu")
    pr_loaded = guarantee_load_priority(
        length, frozen, {}, {5: 2, 6: 1}, {7: 3}, [[8, 9], [10]],
        device="cpu")
    assert bool((pr_loaded > pr_plain).all()), \
        "constraint pressure did not raise the priority tier"
    # No constraints at all -> all zeros (falls back to pure confidence).
    assert bool((pr_plain == 0).all())
    print("PASS test_glod_priority_uniform_terms")


def test_glod_guarantees_hold():
    # Full generation under GLOD: every guarantee must hold.
    m, cfg = _tiny_model(42)
    required = [10, 20, 30]
    for seed in range(10):
        st = DenoisingState(
            length=24, mask_id=cfg.mask_id, eos_id=cfg.eos_id,
            steps=8, seed=seed, vocab_size=cfg.vocab_size,
            order="guarantee_load",
            constraints={"required_ids": required,
                         "forbidden_ids": [200],
                         "at_most": {50: 2}})
        while st.step(m):
            pass
        out = st.x.cpu()
        assert len(out) == 24, "exact token length violated"
        assert not (out == cfg.mask_id).any(), \
            f"seed {seed}: masks left unfinished"
        for t in required:
            assert bool((out == t).any()), \
                f"seed {seed}: required token {t} missing"
        assert not (out == 200).any(), \
            f"seed {seed}: forbidden token present"
        assert int((out == 50).sum()) <= 2, \
            f"seed {seed}: at_most cap violated"
        rep = check_constraints(out, {"required_ids": required,
                                      "forbidden_ids": [200],
                                      "at_most": {50: 2}}, cfg.mask_id)
        assert all(rep.values()), f"seed {seed}: {rep}"
    print("PASS test_glod_guarantees_hold")


def test_glod_intervention_comparison():
    # Constraint-dense generation: GLOD must not increase interventions
    # vs confidence order. Metric = total intervention-trail entries.
    m, cfg = _tiny_model(7)
    constraints = {"required_ids": list(range(100, 112)),
                   "forbidden_ids": [200, 201],
                   "at_most": {50: 1}}
    totals = {"confidence": 0, "guarantee_load": 0}
    endgames = {"confidence": 0, "guarantee_load": 0}
    for seed in range(15):
        for order in ("confidence", "guarantee_load"):
            st = DenoisingState(
                length=32, mask_id=cfg.mask_id, eos_id=cfg.eos_id,
                steps=10, seed=seed, vocab_size=cfg.vocab_size,
                order=order, constraints=constraints)
            while st.step(m):
                pass
            totals[order] += len(st.interventions)
            endgames[order] += sum(
                1 for iv in st.interventions
                if iv["kind"] == "endgame_forced_placement")
    print(f"  interventions: confidence={totals['confidence']} "
          f"glod={totals['guarantee_load']}")
    print(f"  endgame force-placements: confidence={endgames['confidence']} "
          f"glod={endgames['guarantee_load']}")
    assert totals["guarantee_load"] <= totals["confidence"], \
        "GLOD increased total interventions vs confidence order"
    print("PASS test_glod_intervention_comparison")


if __name__ == "__main__":
    test_glod_priority_frozen_proximity()
    test_glod_priority_uniform_terms()
    test_glod_guarantees_hold()
    test_glod_intervention_comparison()
    print("ALL GLOD TESTS PASSED")
