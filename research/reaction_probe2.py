#!/usr/bin/env python3
"""Probe 2 — Conservation Reaction Substrate: can library learning DISCOVER reaction rules?

Continuous relaxation of the chemical soup:
  soup            = count-vector x in R^d  (d term types)
  rules           = stoichiometric matrices: A (reactants, d x R),
                    B (products, d x R), k (rates, R)
  discrete steps  = Euler integration of  dx/dt = sum_r rate_r(x) * (B[:,r]-A[:,r])
                    with mass-action rates  rate_r = k_r * prod_i x_i^{A[i,r]}.

Task: discover the catalyzed doubling rule   a + c -> 2a + c
from (soup_before, soup_after) pairs by gradient descent on (A,B,k) with an
L1 sparsity prior (rules should be small). The catalyst c is REQUIRED for the
reaction but CONSERVED by it — the learner must distinguish "consumed" from
"required-but-conserved", which is the actual discovery challenge (a plain
a -> 2a rule would be learnable by mere growth-curve fitting).

Target structure is known, so success is checkable. The PASS/FAIL criteria
below were fixed BEFORE running. Stdlib + numpy only.

KILL conditions (from the probe spec): gradients collapse to trivial rules
(identity / uniform), learned matrix doesn't recover target structure within
tolerance on held-out pairs, or optimization needs more than this box can give
a toy (d <= 20, <= 5k gradient steps).
"""

import numpy as np

# ---------------------------------------------------------------- config
D = 6            # terms: [a, c, d1, d2, d3, d4]  (d1..d4 = inert distractors)
R = 4            # candidate rules (overcomplete: truth is a single rule)
T = 20           # Euler steps per rollout
DT = 0.02
N_TRAIN = 64
N_TEST = 32
L1 = 1e-3        # sparsity prior, chosen a priori (standard small value)
LR = 0.03
MAX_STEPS = 5000
SEED = 7
EPS = 1e-6
XMAX = 50.0

# PASS thresholds (fixed before running)
TOL_TEST_RRMSE = 0.05    # relative RMSE on held-out pairs
TOL_COSINE = 0.85        # best learned rule vs target (A,B) structure
TOL_SPURIOUS_K = 0.10    # other rules' k must be < this fraction of best k
TOL_VFIELD = 0.10        # relative error of learned vector field vs true
TOL_IDENTITY = 0.50      # learned change must be >= this fraction of true change

rng = np.random.default_rng(SEED)
A_IDX = {"a": 0, "c": 1}

# ---------------------------------------------------------------- truth
A_star = np.zeros((D, R)); B_star = np.zeros((D, R)); k_star = np.zeros(R)
A_star[A_IDX["a"], 0] = 1.0; A_star[A_IDX["c"], 0] = 1.0   # consume a, c
B_star[A_IDX["a"], 0] = 2.0; B_star[A_IDX["c"], 0] = 1.0   # produce 2a, c back
k_star[0] = 1.0


def softplus(z):
    return np.logaddexp(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_data(n):
    """Random soups; ~70% reactive (a>=1 and c>=1)."""
    x = rng.integers(0, 5, size=(n, D)).astype(float)
    reactive = rng.random(n) < 0.7
    x[reactive, A_IDX["a"]] = rng.integers(1, 5, size=reactive.sum())
    x[reactive, A_IDX["c"]] = rng.integers(1, 5, size=reactive.sum())
    y = rollout(x, A_star, B_star, k_star)
    return x, y


def rollout(x0, A, B, k):
    """Forward Euler rollout (no caching)."""
    x = x0.copy()
    M = B - A
    for _ in range(T):
        E = np.log(x + EPS) @ A
        rate = k[None, :] * np.exp(np.clip(E, -50, 50))
        x = np.clip(x + DT * (rate @ M.T), 0.0, XMAX)
    return x


def forward_cache(x0, A, B, k):
    xs = [x0]
    rates = []
    x = x0
    M = B - A
    for _ in range(T):
        E = np.log(x + EPS) @ A
        rate = k[None, :] * np.exp(np.clip(E, -50, 50))
        rates.append(rate)
        x = np.clip(x + DT * (rate @ M.T), 0.0, XMAX)
        xs.append(x)
    return xs, rates


def loss_and_grads(x0, y, alpha, beta, kappa):
    """Analytic gradients of  MSE + L1(|A|+|B|+|k|)  w.r.t. pre-softplus params."""
    A, B, k = softplus(alpha), softplus(beta), softplus(kappa)
    N = x0.shape[0]
    xs, rates = forward_cache(x0, A, B, k)
    M = B - A
    delta = 2.0 * (xs[T] - y) / (N * D)  # dL/dx_T ; mean is over N*D entries
    gA = np.zeros_like(A); gB = np.zeros_like(B); gk = np.zeros_like(k)
    for t in range(T - 1, -1, -1):
        x, rate = xs[t], rates[t]          # delta is dL/dx_{t+1} here
        dM = DT * (delta.T @ rate)         # dL/dM  (D,R)
        gB += dM
        gA -= dM                           # M = B - A
        W = delta @ M                      # (N,R)
        G = DT * W                         # dL/drate
        gk += np.sum(G * rate, axis=0) / np.maximum(k, 1e-12)
        logx = np.log(x + EPS)
        gA += ((G * rate).T @ logx).T      # dL/dA via mass-action exponents
        s = np.zeros_like(delta)
        xeps = x + EPS
        for r in range(R):
            w = delta @ M[:, r]            # (N,)
            s += (w * rate[:, r])[:, None] * (A[:, r][None, :] / xeps)
        delta = delta + DT * s             # dL/dx_t
    # L1 subgradients
    gA += L1 * np.sign(A); gB += L1 * np.sign(B); gk += L1 * np.sign(k)
    # chain through softplus
    ga = gA * sigmoid(alpha); gb = gB * sigmoid(beta); gk_ = gk * sigmoid(kappa)
    mse = float(np.mean((xs[T] - y) ** 2))
    l1v = float(L1 * (np.abs(A).sum() + np.abs(B).sum() + np.abs(k).sum()))
    return mse + l1v, mse, ga, gb, gk_


def grad_check():
    """Finite-difference check of analytic gradients at realistic scale.

    Uses the true rule to generate targets (realistic signal, non-flat loss)
    and only asserts on entries with non-tiny analytic gradients.
    """
    r = np.random.default_rng(123)
    n = 4
    x0 = r.uniform(0.5, 4.0, size=(n, D))
    y = rollout(x0, A_star, B_star, k_star)
    alpha = r.uniform(-3, -1, (D, R)); beta = r.uniform(-3, -1, (D, R))
    kappa = r.uniform(-2, -0.5, R)
    _, _, ga, gb, gk_ = loss_and_grads(x0, y, alpha, beta, kappa)
    worst = 0.0
    checked = 0
    for name, p, g in (("alpha", alpha, ga), ("beta", beta, gb), ("kappa", kappa, gk_)):
        it = np.nditer(p, flags=["multi_index"])
        for _ in range(min(8, p.size)):
            idx = it.multi_index
            it.iternext()
            if abs(g[idx]) < 1e-7:
                continue
            h = 1e-5
            pp, pm = p.copy(), p.copy()
            pp[idx] += h; pm[idx] -= h
            args = {"alpha": alpha, "beta": beta, "kappa": kappa}
            args[name] = pp
            lp, _, _, _, _ = loss_and_grads(x0, y, **args)
            args[name] = pm
            lm, _, _, _, _ = loss_and_grads(x0, y, **args)
            num = (lp - lm) / (2 * h)
            den = max(1e-8, abs(num), abs(g[idx]))
            worst = max(worst, abs(num - g[idx]) / den)
            checked += 1
    print(f"grad_check: worst relative error = {worst:.2e} over {checked} entries")
    assert checked > 0 and worst < 5e-3, "analytic gradients disagree with finite differences"
    print("grad_check: PASS")


def cosine(u, v):
    return float(u @ v / max(1e-12, np.linalg.norm(u) * np.linalg.norm(v)))


def evaluate(alpha, beta, kappa, x_test, y_test):
    A, B, k = softplus(alpha), softplus(beta), softplus(kappa)
    yhat = rollout(x_test, A, B, k)
    rrmse = float(np.sqrt(np.mean((yhat - y_test) ** 2)) / np.sqrt(np.mean(y_test ** 2)))
    tgt = np.concatenate([A_star[:, 0], B_star[:, 0]])
    cosines = [cosine(np.concatenate([A[:, r], B[:, r]]), tgt) for r in range(R)]
    best = int(np.argmax(cosines))
    spurious_ok = all(k[r] < TOL_SPURIOUS_K * k[best] for r in range(R) if r != best)
    # identity-collapse check on reactive pairs
    change_true = np.linalg.norm(y_test - x_test, axis=1)
    reactive = change_true > 0.5
    change_hat = np.linalg.norm(yhat - x_test, axis=1)
    ident_ratio = float(np.mean(change_hat[reactive])) / max(1e-12, float(np.mean(change_true[reactive])))
    # vector-field check on random states
    xs = rng.uniform(0, 5, size=(200, D))
    with np.errstate(divide="ignore"):
        logx = np.log(xs + EPS)
    def vfield(A_, B_, k_):
        rate = k_[None, :] * np.exp(np.clip(logx @ A_, -50, 50))
        return rate @ (B_ - A_).T
    vf_true, vf_hat = vfield(A_star, B_star, k_star), vfield(A, B, k)
    vfield_rel = float(np.linalg.norm(vf_hat - vf_true) / max(1e-12, np.linalg.norm(vf_true)))
    return dict(rrmse=rrmse, cosines=cosines, best=best, k=list(k),
                spurious_ok=spurious_ok, ident_ratio=ident_ratio,
                vfield_rel=vfield_rel, A=A, B=B)


def main():
    import sys
    fix_k = "--fix-k" in sys.argv
    if fix_k:
        print("ABLATION: k fixed uniform at 0.5 for all rules; learning A,B only.")
    grad_check()
    x_train, y_train = make_data(N_TRAIN)
    x_test, y_test = make_data(N_TEST)
    r = np.random.default_rng(SEED + 1)
    alpha = r.uniform(-3, -1, (D, R)); beta = r.uniform(-3, -1, (D, R))
    kappa = r.uniform(-2, -0.5, R)
    # Adam state
    m1a, m2a = np.zeros_like(alpha), np.zeros_like(alpha)
    m1b, m2b = np.zeros_like(beta), np.zeros_like(beta)
    m1k, m2k = np.zeros_like(kappa), np.zeros_like(kappa)
    b1, b2, ae = 0.9, 0.999, 1e-8
    verdict, steps_used = "KILL", MAX_STEPS
    ev = None
    KFIX = float(np.log(np.exp(0.5) - 1.0))  # softplus^{-1}(0.5)
    if fix_k:
        kappa[:] = KFIX
    for step in range(1, MAX_STEPS + 1):
        tot, mse, ga, gb, gk_ = loss_and_grads(x_train, y_train, alpha, beta, kappa)
        if not np.isfinite(tot):
            print(f"step {step}: loss non-finite -> DIVERGED")
            break
        # global grad-norm clip
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in (ga, gb, gk_)))
        if gn > 5.0:
            s = 5.0 / gn
            ga, gb, gk_ = ga * s, gb * s, gk_ * s
        for p, g, m1, m2 in ((alpha, ga, m1a, m2a), (beta, gb, m1b, m2b), (kappa, gk_, m1k, m2k)):
            if fix_k and p is kappa:
                continue  # k frozen in ablation
            m1[:] = b1 * m1 + (1 - b1) * g
            m2[:] = b2 * m2 + (1 - b2) * (g * g)
            p -= LR * (m1 / (1 - b1 ** step)) / (np.sqrt(m2 / (1 - b2 ** step)) + ae)
        if step % 100 == 0 or step == 1:
            ev = evaluate(alpha, beta, kappa, x_test, y_test)
            ok = (ev["rrmse"] < TOL_TEST_RRMSE and ev["cosines"][ev["best"]] > TOL_COSINE
                  and ev["spurious_ok"] and ev["ident_ratio"] > TOL_IDENTITY
                  and ev["vfield_rel"] < TOL_VFIELD)
            print(f"step {step:5d} loss={tot:.4f} mse={mse:.4f} "
                  f"rrmse={ev['rrmse']:.4f} cos_best={ev['cosines'][ev['best']]:.3f} "
                  f"k={[f'{v:.3f}' for v in ev['k']]} ident={ev['ident_ratio']:.3f} "
                  f"vf={ev['vfield_rel']:.4f} {'PASS' if ok else ''}")
            if ok:
                verdict, steps_used = "PASS", step
                break
    ev = evaluate(alpha, beta, kappa, x_test, y_test)
    A, B, k = ev["A"], ev["B"], np.array(ev["k"])
    print("\n================ FINAL ================")
    print(f"verdict: {verdict}  (steps used: {steps_used}/{MAX_STEPS})")
    print(f"test relative RMSE : {ev['rrmse']:.4f}  (need < {TOL_TEST_RRMSE})")
    print(f"best rule #{ev['best']}: cosine to target = {ev['cosines'][ev['best']]:.4f} "
          f"(need > {TOL_COSINE}); all cosines = {[f'{c:.3f}' for c in ev['cosines']]}")
    print(f"learned k          : {[f'{v:.4f}' for v in k]}  (target: [1,0,0,0])")
    print(f"spurious suppressed: {ev['spurious_ok']}")
    print(f"identity ratio     : {ev['ident_ratio']:.3f} (need > {TOL_IDENTITY})")
    print(f"vector-field relerr: {ev['vfield_rel']:.4f} (need < {TOL_VFIELD})")
    print(f"\nlearned A (reactants) best rule:\n{np.round(A[:, ev['best']], 3)}")
    print(f"target  A best rule:\n{A_star[:, 0]}")
    print(f"\nlearned B (products) best rule:\n{np.round(B[:, ev['best']], 3)}")
    print(f"target  B best rule:\n{B_star[:, 0]}")
    # failure-mode diagnosis
    if verdict == "KILL":
        if ev["ident_ratio"] <= TOL_IDENTITY:
            print("DIAGNOSIS: collapse to trivial identity rule (learned ~no reaction).")
        elif max(ev["cosines"]) <= TOL_COSINE and np.std(k) < 0.2 * (np.mean(k) + 1e-9):
            print("DIAGNOSIS: collapse to uniform mush (no rule differentiated).")
        elif max(ev["cosines"]) <= TOL_COSINE:
            print("DIAGNOSIS: structure not recovered (wrong/diffuse stoichiometry).")
        elif not ev["spurious_ok"]:
            print("DIAGNOSIS: target found but spurious rules not suppressed by L1.")
        elif ev["rrmse"] >= TOL_TEST_RRMSE or ev["vfield_rel"] >= TOL_VFIELD:
            print("DIAGNOSIS: structure close but functional fit insufficient.")
    return verdict


if __name__ == "__main__":
    main()
