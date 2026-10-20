#!/usr/bin/env python3
"""Probe PM-2: the tightness loss under INEXACT abstraction.

Kill gate for the Compositional Proof Machine's neural bridge (spec 4.3,
implemented in primitives.py / train_primitives.py).

Claim under test:  L = L_task + lambda * L_tight  with
    L_tight = E_x[ |M(x) - center(M#(alpha(x)))| / width(M#(alpha(x))) ]
trains a network that is BOTH accurate AND tightly abstracted by a FIXED
inexact abstract transformer.

Setup
-----
  Target:      f(x) = |x|,  x in [-2, 2]   (nonlinear kink; Presburger-expressible)
  Transformer: interval domain, |.|#([a,b]) = [0, max(|a|,|b|)].
               This is the standard, best-possible SOUND interval transformer
               for abs -- not a strawman. On point inputs: center = |x|/2,
               width = |x| (inexact: relative width 100%).
  Network:     1 -> 32 -> 32 -> 1 tanh MLP, Adam 1e-3, 4000 full-batch steps,
               2000 uniform training samples. (torch; a few MB RAM.)

Pre-registered predictions (derived BEFORE running)
---------------------------------------------------
  P1 (theta-independence): certificates are M#(alpha(x)) -- a fixed function
     of the input. They do not depend on network weights theta. Therefore
     mean relative certificate width = 100% at EVERY lambda. No training
     loss can tighten certificates; L_tight can only bias concrete outputs
     toward abstract centers.
  P2 (bias shape): the pointwise minimizer of (M-|x|)^2 + lambda*|M-|x|/2|/|x|
     is
         M*(x) = |x|/2               for |x| < sqrt(lambda)
         M*(x) = |x| - lambda/(2|x|) for |x| >= sqrt(lambda)
     The tightness loss drags the network toward the WRONG abstract center,
     and the bias diverges as |x| -> 0 (exactly where precision matters most).

KILL criterion (pre-registered): no lambda in {0, 0.01, 0.1, 1.0} achieves
  BOTH test MSE < 1e-4 AND mean relative certificate width < 50%.
PASS criterion: some lambda achieves both (would falsify P1/P2).

Scope honesty: this kills mechanism (b) -- fixed abstract transformer +
tightness loss -- as specified and implemented. The charitable retreat
("use exact domains only") is a scope-kill on the ASI claim, discussed in
the results memo, not tested here.
"""

import math
import torch
import torch.nn as nn

torch.manual_seed(0)

LAMBDAS = [0.0, 0.01, 0.1, 1.0]
STEPS = 4000
LR = 1e-3
N_TRAIN = 2000
N_TEST = 1000


def make_net():
    return nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )


def target(x):
    return x.abs()


def train(lmbda):
    torch.manual_seed(0)  # identical init & data across lambdas: fair comparison
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    x = torch.empty(N_TRAIN, 1).uniform_(-2, 2)
    y = target(x)
    # fixed interval transformer |.|# on point inputs: center=|x|/2, width=|x|
    center = y / 2.0
    width = y.clamp_min(1e-6)
    mask = (y.squeeze(1) >= 1e-6)  # term undefined at x=0; masked there
    for _ in range(STEPS):
        pred = net(x)
        l_task = (pred - y).pow(2).mean()
        l_tight = ((pred.squeeze(1)[mask] - center.squeeze(1)[mask]).abs()
                   / width.squeeze(1)[mask]).mean()
        loss = l_task + lmbda * l_tight
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def predicted_mstar(x, lmbda):
    ax = x.abs()
    out = torch.where(ax < math.sqrt(lmbda), ax / 2.0, ax - lmbda / (2 * ax.clamp_min(1e-9)))
    return out


def main():
    print("lambda | test MSE  | max|err| | corr(M,M*) | mean rel cert width")
    print("-------+-----------+---------+------------+--------------------")
    results = {}
    for lmbda in LAMBDAS:
        net = train(lmbda)
        net.eval()
        with torch.no_grad():
            xt = torch.empty(N_TEST, 1).uniform_(-2, 2)
            yt = target(xt)
            pred = net(xt)
            mse = (pred - yt).pow(2).mean().item()
            maxerr = (pred - yt).abs().max().item()
            # P2 check: correlation of learned function with predicted M*
            grid = torch.linspace(-2, 2, 401).unsqueeze(1)
            m_learned = net(grid).squeeze(1)
            m_star = predicted_mstar(grid.squeeze(1), lmbda)
            corr = torch.corrcoef(torch.stack([m_learned, m_star]))[0, 1].item()
            # P1 check: certificate width from the FIXED transformer (theta-independent)
            w = yt.squeeze(1).clamp_min(1e-9)          # width = |x|
            rel_w = (w / yt.squeeze(1).clamp_min(1e-9)) # = 1.0 everywhere
            mean_rel_w = rel_w[(yt.squeeze(1) > 0.05)].mean().item()
        results[lmbda] = (mse, maxerr, corr, mean_rel_w)
        print(f"{lmbda:6.2f} | {mse:.2e} | {maxerr:.4f} | {corr:+.4f}     | {mean_rel_w*100:.1f}%")

    print()
    kill = True
    for lmbda, (mse, maxerr, corr, mrw) in results.items():
        if mse < 1e-4 and mrw < 0.50:
            kill = False
    print("KILL criterion (no lambda: MSE<1e-4 AND relwidth<50%):",
          "TRIGGERED -> KILL" if kill else "not triggered -> PASS")
    # P2 shape diagnostics at lambda=1: inside |x|<1 the net should learn ~|x|/2
    net1 = train(1.0)
    with torch.no_grad():
        g = torch.tensor([[0.25], [0.5], [1.0], [1.5]])
        vals = net1(g).squeeze(1).tolist()
    print(f"lambda=1 learned f(0.25)={vals[0]:.3f} (target 0.25, M*=0.125), "
          f"f(0.5)={vals[1]:.3f} (target 0.5, M*=0.25), "
          f"f(1.0)={vals[2]:.3f} (target 1.0, M*=0.5), "
          f"f(1.5)={vals[3]:.3f} (target 1.5, M*=1.333)")


if __name__ == "__main__":
    main()
