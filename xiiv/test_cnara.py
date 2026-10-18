"""Tests for CNaRA (xiiv/lora.py): contract-conditioned noise-aware LoRA.

Run:  cd ~/workspace/ai-tools/corsac && .venv/bin/python -m xiiv.test_cnara
Exit code 0 = all pass. No model weights needed; pure CPU.

What is verified:
  * fourier_features: shape, range, determinism, clamping.
  * Safe rollout: hypernetwork zero-init -> S = I at init; a CNaRA adapter
    with no condition set behaves EXACTLY like static LoRA.
  * One globally-shared controller: set_condition updates every wrapper.
  * The condition path is live: with a non-degenerate hypernetwork, S != I
    and outputs change with (lam, contract).
  * Gradients reach the hypernetwork once B has moved off zero (at step 0
    B = 0 so there is honestly nothing to modulate yet — asserted, not hidden).
  * inject_cnara: trainable count = adapters + hypernetwork; base frozen;
    controller registered once in state_dict (not once per layer).
  * Guards: double injection refused; merge_lora refuses CNaRA models;
    apply_cnara_condition raises without a controller.
  * merge_cnara: folding at a fixed condition reproduces the wrapped output.
  * diffusion_loss(cnara_auto=True): sets the condition from the batch's own
    mask ratio and contract presence.
"""

import torch
import torch.nn as nn

from .lora import (
    CNARAController,
    CNARAHyperNet,
    CNARALinear,
    LoRALinear,
    apply_cnara_condition,
    fourier_features,
    get_trainable_params,
    inject_cnara,
    inject_lora,
    merge_cnara,
    merge_lora,
)
from .diffusion import diffusion_loss

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name)


class FakeAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)


class FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = FakeAttn(d)


class FakeModel(nn.Module):
    def __init__(self, d=16, n=2):
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock(d) for _ in range(n)])


def main():
    torch.manual_seed(0)

    # ---- fourier_features ------------------------------------------------
    f = fourier_features(0.5, k=8)
    check("fourier shape", tuple(f.shape) == (16,))
    check("fourier range", bool(((f >= -1.0) & (f <= 1.0)).all()))
    check("fourier deterministic",
          bool((f - fourier_features(0.5, k=8)).abs().max() == 0))
    check("fourier clamps", tuple(fourier_features(2.0, k=4).shape) == (8,))
    check("fourier varies with lam",
          bool((fourier_features(0.2, k=8) - fourier_features(0.8, k=8))
               .abs().max() > 1e-3))

    # ---- hypernet init: S = I (safe rollout) ------------------------------
    hn = CNARAHyperNet(rank=8)
    S0 = hn(0.37, True)
    check("hypernet S=I at init",
          bool((S0 - torch.eye(8)).abs().max() < 1e-6))
    check("hypernet S=I regardless of condition",
          bool((hn(0.99, False) - torch.eye(8)).abs().max() < 1e-6))
    check("hypernet param count small", hn.param_count() < 20000)

    # ---- controller -------------------------------------------------------
    ctl = CNARAController(rank=8)
    check("controller S=I before set_condition",
          bool((ctl.core() - torch.eye(8)).abs().max() < 1e-6))
    check("controller last_condition None at init", ctl.last_condition is None)
    S = ctl.set_condition(0.6, contract_present=True)
    check("controller caches S", ctl.core() is S)
    lam_rec, c_rec = ctl.last_condition
    check("controller records condition",
          abs(lam_rec - 0.6) < 1e-6 and c_rec is True)
    names = [n for n, _ in ctl.named_parameters()]
    check("controller params registered",
          any("hypernet" in n for n in names))

    # ---- step-0 equivalence: CNaRA (no condition) == static LoRA -----------
    base = nn.Linear(16, 16)
    ctl2 = CNARAController(rank=4)
    cw = CNARALinear(base, ctl2, rank=4, alpha=8.0)
    base2 = nn.Linear(16, 16)
    base2.load_state_dict(base.state_dict())
    lw = LoRALinear(base2, rank=4, alpha=8.0)
    lw.lora_A.data.copy_(cw.lora_A.data)
    lw.lora_B.data.copy_(cw.lora_B.data)
    x = torch.randn(3, 16)
    check("step-0 equivalence (S=I, no condition)",
          bool((cw(x) - lw(x)).abs().max() < 1e-6))

    # ---- one shared controller drives all wrappers ------------------------
    m = FakeModel(d=16, n=2)
    n_train = inject_cnara(m, rank=4, alpha=8.0)
    wrappers = [m.layers[i].attn.qkv for i in range(2)]
    check("wrappers share one controller",
          wrappers[0].controller is wrappers[1].controller
          is m.cnara_controller)
    x = torch.randn(2, 16)
    out_before = [w(x).clone() for w in wrappers]
    # Degenerate hypernet at init: S stays I even after set_condition.
    m.cnara_controller.set_condition(0.9, contract_present=True)
    out_same = [w(x) for w in wrappers]
    check("init hypernet: condition changes nothing (S=I)",
          all(bool((a - b).abs().max() < 1e-6)
              for a, b in zip(out_before, out_same)))
    # Make the hypernetwork non-degenerate -> the path must be live.
    # (Perturb the last layer AND the contract embedding: the embedding is
    # zero-init by design, so the contract flag is inert until it learns.)
    with torch.no_grad():
        m.cnara_controller.hypernet.mlp[-1].weight.fill_(0.01)
        # rows must DIFFER for the flag to have an effect
        m.cnara_controller.hypernet.contract_embed.weight.normal_(0, 0.05)
        # B is zero-init: move it so there is something to modulate.
        for w in wrappers:
            w.lora_B.data.normal_(0, 0.05)
    m.cnara_controller.set_condition(0.9, contract_present=True)
    S_live = m.cnara_controller.core()
    check("non-degenerate hypernet: S != I",
          bool((S_live - torch.eye(4)).abs().max() > 1e-4))
    out_after = [w(x) for w in wrappers]
    check("condition moves every wrapper output",
          all(bool((a - b).abs().max() > 1e-6)
              for a, b in zip(out_before, out_after)))
    # contract flag changes S too
    m.cnara_controller.set_condition(0.9, contract_present=False)
    check("contract flag changes S",
          bool((m.cnara_controller.core() - S_live).abs().max() > 1e-6))
    check("no NaN in S", bool(torch.isfinite(S_live).all()))

    # ---- gradients reach the hypernetwork (once B != 0) --------------------
    m.cnara_controller.set_condition(0.5, contract_present=True)
    loss = sum(w(x).sum() for w in wrappers)
    m.zero_grad()
    loss.backward()
    hg = m.cnara_controller.hypernet.mlp[-1].weight.grad
    check("hypernet gets grads when B != 0",
          hg is not None and bool(hg.abs().max() > 0))
    # Honest counterpart: at true step-0 (B == 0) there is nothing to
    # modulate, so hypernet grads are exactly zero — not a bug.
    m2 = FakeModel(d=16, n=1)
    inject_cnara(m2, rank=4)
    w0 = m2.layers[0].attn.qkv
    m2.cnara_controller.set_condition(0.5, True)
    m2.zero_grad()
    w0(torch.randn(2, 16)).sum().backward()
    hg0 = m2.cnara_controller.hypernet.mlp[-1].weight.grad
    check("hypernet grads zero at true step-0 (B=0, documented)",
          hg0 is not None and bool(hg0.abs().max() == 0))

    # ---- inject_cnara bookkeeping -----------------------------------------
    m3 = FakeModel(d=16, n=2)
    n3 = inject_cnara(m3, rank=4, alpha=8.0)
    # per block: qkv A(4,16)+B(48,4) = 256; out A(4,16)+B(16,4) = 128
    adapter_params = 2 * (256 + 128)
    hyper_params = sum(p.numel() for p in m3.cnara_controller.parameters())
    check("trainable = adapters + hypernet",
          n3 == adapter_params + hyper_params == get_trainable_params(m3))
    check("base params frozen",
          not any(p.requires_grad for n, p in m3.named_parameters()
                  if "lora_" not in n and "cnara_controller" not in n))
    sd_keys = list(m3.state_dict().keys())
    ctl_keys = [k for k in sd_keys if k.startswith("cnara_controller.")]
    leaked = [k for k in sd_keys
              if ".controller." in k and not k.startswith("cnara_controller.")]
    check("controller registered once in state_dict",
          len(ctl_keys) == len(set(ctl_keys)) and len(leaked) == 0
          and len(ctl_keys) == 7)  # embed.w + 3 linear x (w, b)
    # double injection refused
    try:
        inject_cnara(m3, rank=4)
        check("double inject_cnara refused", False)
    except ValueError:
        check("double inject_cnara refused", True)
    m4 = FakeModel(d=16, n=1)
    inject_lora(m4, rank=4)
    try:
        inject_cnara(m4, rank=4)
        check("inject_cnara over inject_lora refused", False)
    except ValueError:
        check("inject_cnara over inject_lora refused", True)
    m5 = FakeModel(d=16, n=1)
    inject_cnara(m5, rank=4)
    try:
        inject_lora(m5, rank=4)
        check("inject_lora over inject_cnara refused", False)
    except ValueError:
        check("inject_lora over inject_cnara refused", True)

    # ---- apply_cnara_condition --------------------------------------------
    try:
        apply_cnara_condition(FakeModel(), 0.5)
        check("apply_cnara_condition raises w/o controller", False)
    except ValueError:
        check("apply_cnara_condition raises w/o controller", True)
    apply_cnara_condition(m3, 0.33, contract_present=False)
    lam_a, c_a = m3.cnara_controller.last_condition
    check("apply_cnara_condition sets condition",
          abs(lam_a - 0.33) < 1e-6 and c_a is False)

    # ---- merge_cnara -------------------------------------------------------
    m6 = FakeModel(d=16, n=1)
    inject_cnara(m6, rank=4, alpha=8.0)
    w6 = m6.layers[0].attn.qkv
    with torch.no_grad():
        w6.lora_B.data.normal_(0, 0.05)
        m6.cnara_controller.hypernet.mlp[-1].weight.fill_(0.01)
    x = torch.randn(2, 16)
    m6.cnara_controller.set_condition(0.7, contract_present=True)
    before = w6(x).clone()
    merge_cnara(m6, lam=0.7, contract_present=True)
    check("controller removed after merge",
          not hasattr(m6, "cnara_controller"))
    check("merged base reproduces conditioned output",
          bool((m6.layers[0].attn.qkv(x) - before).abs().max() < 1e-5))
    m7 = FakeModel(d=16, n=1)
    inject_cnara(m7, rank=4)
    try:
        merge_lora(m7)
        check("merge_lora refuses CNaRA model", False)
    except ValueError:
        check("merge_lora refuses CNaRA model", True)

    # ---- merge_lora correctness (static LoRA, incl. non-square qkv) --------
    m8 = FakeModel(d=16, n=1)
    inject_lora(m8, rank=4, alpha=8.0)
    wq = m8.layers[0].attn.qkv   # Linear(16 -> 48): non-square
    wo = m8.layers[0].attn.out   # Linear(16 -> 16): square
    with torch.no_grad():
        wq.lora_B.data.normal_(0, 0.05)
        wo.lora_B.data.normal_(0, 0.05)
    x = torch.randn(2, 16)
    bq, bo = wq(x).clone(), wo(x).clone()
    merge_lora(m8)
    check("merge_lora folds non-square qkv exactly",
          bool((m8.layers[0].attn.qkv(x) - bq).abs().max() < 1e-5))
    check("merge_lora folds square out exactly",
          bool((m8.layers[0].attn.out(x) - bo).abs().max() < 1e-5))
    check("merge_lora removes wrappers",
          isinstance(m8.layers[0].attn.qkv, nn.Linear))

    # ---- diffusion_loss cnara_auto -----------------------------------------
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 32)

        def forward(self, noisy):
            return self.proj(noisy.float())

    net = TinyNet()
    net.cnara_controller = CNARAController(rank=4)
    ids = torch.randint(0, 32, (4, 8))
    loss, info = diffusion_loss(net, ids, mask_id=31, cnara_auto=True)
    check("cnara_auto sets condition",
          net.cnara_controller.last_condition is not None)
    lam_seen, c_seen = net.cnara_controller.last_condition
    check("cnara_auto lam == batch mask ratio",
          abs(lam_seen - info["mask_ratio"]) < 1e-6)
    check("cnara_auto contract False when no contract_ids", c_seen is False)
    cids = torch.randint(0, 32, (4, 5))
    diffusion_loss(net, ids, mask_id=31, cnara_auto=True, contract_ids=cids)
    check("cnara_auto contract True when contract_ids given",
          net.cnara_controller.last_condition[1] is True)
    # default off: existing callers unaffected
    net2 = TinyNet()
    net2.cnara_controller = CNARAController(rank=4)
    diffusion_loss(net2, ids, mask_id=31)
    check("cnara_auto defaults off",
          net2.cnara_controller.last_condition is None)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
