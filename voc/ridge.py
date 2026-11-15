"""Rolling ridge on nested random features, in the dual (T x T) form.

Paper eq. (9):  beta(z) = (zI + Psi)^-1 T^-1 sum_t S_t R_{t+1},  Psi = T^-1 sum S_t S_t'.
With S_tr the (T x P) training feature matrix and y the T training targets this equals

    beta = S_tr' (z T I + S_tr S_tr')^-1 y            (push-through identity)

so only a T x T system is solved, however large P is. For a test row f:

    pi = f' beta = k' alpha,   k = S_tr f,  alpha = (zT I + G)^-1 y,  G = S_tr S_tr'
    ||beta||^2 = alpha' G alpha

Nested models (first L columns) are obtained from one pass: G_L and k_L are
cumulative sums over blocks of columns, read off at each requested L.
Eigendecomposing G_L once makes every z on the grid cheap.

Scaling (our inference, not stated explicitly in the paper): the paper's S = P^{-1/2} h
uses the width P of the model. For a model with the first L columns we therefore use
S = F[:, :L] / sqrt(L)  (scale="L"), so G = G_raw / L and k = k_raw / L. With a fixed
1/sqrt(P_total) instead, "z = 1e-3" would not be near-ridgeless at small L and the
double-ascent dip in the paper's Figs 1-3 could not appear. scale="none" disables this
(used by the unit tests that compare against eq. 9 on arbitrary matrices).

Timing convention (no look-ahead):
    X[t]  = signal known at time t
    y[t]  = return earned from t to t+1  (R_{t+1} in the paper)
    prediction at t uses training pairs (X[tau], y[tau]) for tau = t-T, ..., t-1;
    y[t-1] is realised at t, so everything used is known at t.
    strategy return at t = pi_t * y[t].
"""
from __future__ import annotations

import numpy as np


def rolling_ridge_voc(
    F: np.ndarray,
    y: np.ndarray,
    T: int,
    L_grid: list[int] | np.ndarray,
    z_grid: list[float] | np.ndarray,
    t_start: int | None = None,
    scale: str = "L",
) -> dict[str, np.ndarray]:
    """Out-of-sample positions for every (L, z) with a rolling window of T periods.

    Returns dict with
        t     : (n,) integer time indices of the predictions (t_start .. N-1)
        pi    : (n, nL, nz) positions pi_t
        beta2 : (n, nL, nz) ||beta_t||^2
    """
    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=float)
    L_grid = np.asarray(L_grid, dtype=int)
    z_grid = np.asarray(z_grid, dtype=float)
    if np.any(np.diff(L_grid) <= 0):
        raise ValueError("L_grid must be strictly increasing")
    if L_grid[-1] > F.shape[1]:
        raise ValueError("L_grid exceeds number of features")
    if scale not in ("L", "none"):
        raise ValueError("scale must be 'L' or 'none'")
    N = F.shape[0]
    t_start = T if t_start is None else t_start
    if t_start < T:
        raise ValueError("t_start must be >= T (need a full training window)")
    ts = np.arange(t_start, N)
    nL, nz = len(L_grid), len(z_grid)
    edges = np.concatenate([[0], L_grid])
    pi = np.empty((len(ts), nL, nz))
    beta2 = np.empty((len(ts), nL, nz))
    zT = z_grid * T

    for i, t in enumerate(ts):
        Ftr = F[t - T : t]
        ytr = y[t - T : t]
        f = F[t]
        G = np.zeros((T, T))
        k = np.zeros(T)
        G_all = np.empty((nL, T, T))
        k_all = np.empty((nL, T))
        for j in range(nL):
            a, b = edges[j], edges[j + 1]
            blk = Ftr[:, a:b]
            G += blk @ blk.T
            k += blk @ f[a:b]
            s = 1.0 / L_grid[j] if scale == "L" else 1.0
            G_all[j] = G * s
            k_all[j] = k * s
        w, V = np.linalg.eigh(G_all)  # (nL, T), (nL, T, T)
        w = np.clip(w, 0.0, None)
        Vty = np.einsum("ltk,t->lk", V, ytr)  # V' y
        coef = Vty[:, :, None] / (w[:, :, None] + zT[None, None, :])  # (nL, T, nz)
        alpha = np.einsum("ltk,lkz->ltz", V, coef)
        pi[i] = np.einsum("lt,ltz->lz", k_all, alpha)
        beta2[i] = np.einsum("lk,lkz->lz", w, coef**2)
    return {"t": ts, "pi": pi, "beta2": beta2}
