"""Exact kernel ridge: the c -> infinity limit of the random-feature model (no sampling noise).

With raw random Fourier features F (2 columns per weight) and the 1/L scaling of ridge.py,
    G = F F' / L  ->  K / 2,    K(x, y) = (1/7) sum_i exp(-||x - y||^2 / (2 gamma_i))
because each (sin, cos) pair adds cos(w'(x - y)) and L = 2 * (number of weights).
So the exact counterpart of ridge.py with scale="L" uses K/2 in place of G and k/2 for k.
"""
from __future__ import annotations

import numpy as np

from .features import GAMMAS


def mixture_kernel(A: np.ndarray, B: np.ndarray, gammas: tuple[float, ...] = GAMMAS) -> np.ndarray:
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.mean([np.exp(-d2 / (2.0 * g)) for g in gammas], axis=0)


def rolling_kernel_ridge(X: np.ndarray, y: np.ndarray, T: int, z_grid, t_start: int | None = None) -> dict[str, np.ndarray]:
    """Same timing convention as rolling_ridge_voc. Returns t (n,), pi (n, nz), beta2 (n, nz)."""
    X, y, z_grid = np.asarray(X, float), np.asarray(y, float), np.asarray(z_grid, float)
    t_start = T if t_start is None else t_start
    ts = np.arange(t_start, len(y))
    pi = np.empty((len(ts), len(z_grid)))
    beta2 = np.empty_like(pi)
    for i, t in enumerate(ts):
        Xtr, ytr = X[t - T : t], y[t - T : t]
        G = 0.5 * mixture_kernel(Xtr, Xtr)
        k = 0.5 * mixture_kernel(X[t : t + 1], Xtr)[0]
        w, V = np.linalg.eigh(G)
        w = np.clip(w, 0.0, None)
        Vty = V.T @ ytr
        coef = Vty[:, None] / (w[:, None] + z_grid[None, :] * T)  # (T, nz)
        pi[i] = (V.T @ k) @ coef
        beta2[i] = (w[:, None] * coef**2).sum(0)
    return {"t": ts, "pi": pi, "beta2": beta2}
