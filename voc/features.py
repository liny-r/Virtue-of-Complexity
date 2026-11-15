"""Random Fourier features (paper eqs. 36-37).

Paper setup: a mixture of 7 Gaussian kernels with bandwidth grid gamma_i,
approximated by random Fourier features (Rahimi & Recht 2007).

    K(x1, x2) = (1/7) sum_i exp(-||x1 - x2||^2 / (2 gamma_i))              (36)

    S_{i,t} = P^{-1/2} [sin(gamma_[i/p] w_i' X_t), cos(gamma_[i/p] w_i' X_t)],
    w_i ~ N(0, I_d)                                                         (37)

NOTE (paper ambiguity, to verify): E[cos(a w'(x-y))] = exp(-a^2 ||x-y||^2 / 2) for
w ~ N(0, I). So eq. (37) taken literally (a = gamma) gives a kernel with variance
parameter 1/gamma^2, not gamma as in eq. (36). We therefore support two conventions:

    "kernel": frequencies w / sqrt(gamma)  -> matches eq. (36)  (default)
    "paper" : frequencies gamma * w        -> literal eq. (37)

Also: each of the P = 7p weights yields a (sin, cos) PAIR, so there are 2P columns.
Here L counts columns (features). This function returns RAW (unscaled) columns; the
1/sqrt(L) scaling of the paper's S is applied in ridge.py using the width L of the model
being fitted (see the note there). The first L columns of one fixed (seeded, permuted)
matrix give nested models.
"""
from __future__ import annotations

import numpy as np

GAMMAS = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def make_rff(
    X: np.ndarray,
    p: int,
    seed: int,
    gammas: tuple[float, ...] = GAMMAS,
    convention: str = "kernel",
) -> np.ndarray:
    """Return the (N, 2P) random-feature matrix, P = len(gammas) * p.

    Columns are ordered as adjacent (sin, cos) pairs, and the pairs are randomly
    permuted so every prefix of columns mixes all bandwidths (paper: "random
    permutation ... to get rid of an artificial ordering").
    """
    if convention not in ("kernel", "paper"):
        raise ValueError("convention must be 'kernel' or 'paper'")
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    n_w = len(gammas) * p
    W = rng.standard_normal((n_w, d))
    g = np.repeat(np.asarray(gammas, dtype=float), p)  # bandwidth of each weight
    scale = 1.0 / np.sqrt(g) if convention == "kernel" else g
    Z = X @ (W * scale[:, None]).T  # (N, P)
    F = np.empty((X.shape[0], 2 * n_w))
    F[:, 0::2] = np.sin(Z)
    F[:, 1::2] = np.cos(Z)
    perm = rng.permutation(n_w)
    cols = np.stack([2 * perm, 2 * perm + 1], axis=1).ravel()
    return F[:, cols]
