"""Simulator with a known nonlinear conditional mean (paper eq. 1: R_{t+1} = f(X_t) + eps).

Stylised facts we impose (and nothing more):
  * signals X_t iid N(0, I_d), already "standardised" like the paper's signals
  * f is a smooth nonlinear function (random sinusoid teacher) with NO linear component
    worth speaking of -> a linear model cannot capture it
  * eps ~ N(0,1); f is scaled so Var(f) = snr  (tiny, like real return predictability)
  * returns have unit conditional vol, matching the paper's vol-standardised returns

What this substitutes for: real data. It lets us test the *theory* (double ascent at
c = 1, ridge fixing it, ||beta|| falling for c > 1) in an environment where the
assumptions hold by construction. It says nothing about whether markets look like this.
"""
from __future__ import annotations

import numpy as np


def simulate_nonlinear(
    N: int,
    d: int = 5,
    snr: float = 0.05,
    n_teacher: int = 20,
    seed: int = 0,
    mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N, d), y (N,) with y[t] = mu + f(X[t]) + eps[t], and f(X) (N,)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))
    W = rng.standard_normal((n_teacher, d)) / np.sqrt(d)
    b = rng.uniform(0, 2 * np.pi, n_teacher)
    c = rng.standard_normal(n_teacher)
    f = np.sin(X @ W.T + b) @ c
    f = (f - f.mean()) / f.std() * np.sqrt(snr)
    y = mu + f + rng.standard_normal(N)  # mu: unconditional premium (per-period Sharpe)
    return X, y, f
