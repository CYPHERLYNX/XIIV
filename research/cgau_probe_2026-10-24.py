"""CGAU falsifier probe (2026-10-24).

Compares three unmasking policies on synthetic contract tasks with a
deterministic adversarial stub model ("confident but wrong" spikes on
forbidden/capped tokens -- the failure mode certificate gating exists for):

  vanilla : confidence-ordered unmasking, no checks
  gated   : CGMS-4th-gate spirit -- per-placement certificate gate with
            top-k fallback scan, defer on failure (no backtrack/taboo/recovery)
  cgau    : full triad -- gate + certified-state rollback + certificate-margin
            early exit + targeted re-mask backtracking

Pre-registered (research/cgau_design_2026-10-24.md):
  P1 final_valid(cgau) >= final_valid(gated) > final_valid(vanilla)
  P2 gate fires on adversarial instances (gated/cgau gate_rejects > 0)
  P3 forward_passes(cgau) <= forward_passes(gated)
  P4 rollbacks + remasks + early_exits > 0 across the suite
  KILL: final_valid(cgau) <= final_valid(vanilla)

Writes research/cgau_results_2026-10-24.json and prints the verdict table.
CPU-only; the stub is a few KB of tensors.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiiv.cgau import CGAUConfig, cgau_sample  # noqa: E402

VOCAB = 32            # real token ids 0..31
MASK_ID = 32          # mask token id; model vocab size = 33
LENGTH = 24
N_INSTANCES = 60
SUITE_SEED = 20261024


class AdversarialStub(torch.nn.Module):
    """Deterministic stub LM with 'confident but wrong' spikes.

    Base logits are fixed random; a small canvas-dependent term keeps
    successive steps non-identical; required tokens get a modest +1 boost
    (the stub "knows the task", like a real LM); at seeded positions the
    top-1 is a forbidden/capped token with a +4 logit spike -- the exact
    pattern a certificate gate must catch.
    """

    def __init__(self, forbidden, capped, required, seed):
        super().__init__()
        V = VOCAB + 1
        g = torch.Generator().manual_seed(seed)
        base = torch.randn(LENGTH, V, generator=g) * 1.5
        for t in required:
            base[:, int(t)] += 1.0
        self.register_buffer("base", base)
        self.register_buffer("mix", torch.randn(V, V, generator=g) * 0.12)
        pool = sorted(set(forbidden) | set(capped))
        self.spike = {}
        for pos in range(LENGTH):
            if pool and torch.rand(1, generator=g).item() < 0.45:
                self.spike[pos] = pool[torch.randint(len(pool), (1,),
                                                     generator=g).item()]

    def forward(self, x):
        B, L = x.shape
        V = VOCAB + 1
        logits = self.base[:L].unsqueeze(0).expand(B, -1, -1).clone()
        cur = torch.nn.functional.one_hot(x.clamp(min=0), num_classes=V).float()
        logits = logits + cur @ self.mix.t()
        for pos, tok in self.spike.items():
            if pos < L:
                logits[:, pos, tok] += 4.0
        return logits


def gen_contract(rng):
    """Sample a satisfiable contract dict for the verifier's schema."""
    toks = list(range(VOCAB))
    perm = torch.randperm(VOCAB, generator=rng).tolist()
    required = perm[0:2]
    forbidden = perm[2:4]
    capped = perm[4:5]
    rest = perm[5:]
    required_counts = {t: 1 + int(torch.randint(1, 3, (1,), generator=rng).item())
                       for t in required}
    at_most = {capped[0]: 1}
    order_pairs = []
    if torch.rand(1, generator=rng).item() < 0.5 and len(rest) >= 2:
        a, b = rest[0], rest[1]
        order_pairs.append(([a], [b]))
        required_counts[a] = required_counts.get(a, 0) + 1
        required_counts[b] = required_counts.get(b, 0) + 1
    anchors = {}
    if torch.rand(1, generator=rng).item() < 0.3:
        # int keys, matching Contract.compile's real output
        anchors[int(torch.randint(0, LENGTH, (1,),
                                  generator=rng).item())] = rest[2]
    total_need = sum(required_counts.values()) + len(anchors)
    assert total_need <= LENGTH - 2, "contract over-constrained"
    return {
        "anchors": anchors,
        "forbidden_ids": forbidden,
        "required_ids": required,
        "required_counts": required_counts,
        "at_most": at_most,
        "forbidden_phrases": [],
        "order_pairs": order_pairs,
    }


def run_suite(n=N_INSTANCES, seed=SUITE_SEED):
    rng = torch.Generator().manual_seed(seed)
    cfg = CGAUConfig(topk=4, budget=2, max_steps=64, tau_margin=0.35,
                     stall_budget=3, taboo_cooldown=4)
    agg = {p: {"valid": 0, "completed": 0, "passes": 0, "rejects": 0,
               "proposals": 0, "rollbacks": 0, "remasks": 0,
               "early_exits": 0, "steps": 0}
           for p in ("vanilla", "gated", "cgau")}
    per_instance = []
    for i in range(n):
        contract = gen_contract(rng)
        model = AdversarialStub(contract["forbidden_ids"],
                                list(contract["at_most"].keys()),
                                list(contract["required_counts"].keys()),
                                seed=1000 + i)
        row = {"instance": i, "contract": contract}
        for p in ("vanilla", "gated", "cgau"):
            _ids, tr = cgau_sample(model, length=LENGTH, mask_id=MASK_ID,
                                   constraints=contract, config=cfg,
                                   seed=7000 + i, policy=p)
            d = tr.to_dict()
            row[p] = d
            a = agg[p]
            a["valid"] += int(d["final_valid"])
            a["completed"] += int(d["completed"])
            a["passes"] += d["forward_passes"]
            a["rejects"] += d["gate_rejects"]
            a["proposals"] += d["proposals"]
            a["rollbacks"] += d["rollbacks"]
            a["remasks"] += d["remasks"]
            a["early_exits"] += d["early_exits"]
            a["steps"] += d["steps"]
        per_instance.append(row)
    return agg, per_instance, cfg


def verdict(agg, n):
    v = {}
    for p in ("vanilla", "gated", "cgau"):
        a = agg[p]
        v[p] = {
            "final_valid_rate": a["valid"] / n,
            "completed_rate": a["completed"] / n,
            "mean_forward_passes": a["passes"] / n,
            "mean_steps": a["steps"] / n,
            "cert_failure_rate": (a["rejects"] / a["proposals"]
                                  if a["proposals"] else 0.0),
            "total_rollbacks": a["rollbacks"],
            "total_remasks": a["remasks"],
            "total_early_exits": a["early_exits"],
        }
    checks = {
        "P1_validity_ladder":
            v["cgau"]["final_valid_rate"] >= v["gated"]["final_valid_rate"]
            and v["gated"]["final_valid_rate"] > v["vanilla"]["final_valid_rate"],
        "P2_gate_fires":
            agg["gated"]["rejects"] > 0 and agg["cgau"]["rejects"] > 0,
        "P3_cgau_compute_le_gated":
            v["cgau"]["mean_forward_passes"] <= v["gated"]["mean_forward_passes"],
        "P4_mechanisms_fire":
            (agg["cgau"]["rollbacks"] + agg["cgau"]["remasks"]
             + agg["cgau"]["early_exits"]) > 0,
        "KILL_cgau_le_vanilla_validity":
            v["cgau"]["final_valid_rate"] <= v["vanilla"]["final_valid_rate"],
    }
    return v, checks


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_INSTANCES
    agg, per_instance, cfg = run_suite(n=n)
    summary, checks = verdict(agg, n)
    out = {
        "date": "2026-10-24",
        "n_instances": n,
        "suite_seed": SUITE_SEED,
        "vocab": VOCAB,
        "mask_id": MASK_ID,
        "length": LENGTH,
        "config": {k: getattr(cfg, k) for k in
                   ("topk", "budget", "max_steps", "tau_margin",
                    "stall_budget", "taboo_cooldown")},
        "summary": summary,
        "pre_registered_checks": checks,
        "per_instance": per_instance,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "cgau_results_2026-10-24.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'policy':<8} {'valid':>6} {'compl':>6} {'passes':>7} "
          f"{'certfail':>8} {'rb':>4} {'rm':>5} {'ee':>5}")
    for p in ("vanilla", "gated", "cgau"):
        s = summary[p]
        print(f"{p:<8} {s['final_valid_rate']:>6.3f} "
              f"{s['completed_rate']:>6.3f} {s['mean_forward_passes']:>7.2f} "
              f"{s['cert_failure_rate']:>8.3f} {s['total_rollbacks']:>4d} "
              f"{s['total_remasks']:>5d} {s['total_early_exits']:>5d}")
    print("\npre-registered checks:")
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    killed = checks["KILL_cgau_le_vanilla_validity"]
    passed = all(v for k, v in checks.items() if not k.startswith("KILL"))
    print("\nVERDICT:",
          "KILL -- triad adds nothing" if killed
          else "PASS -- all pre-registered checks hold" if passed
          else "MIXED -- some checks failed, see above")


if __name__ == "__main__":
    main()
