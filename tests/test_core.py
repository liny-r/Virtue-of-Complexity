import numpy as np

from voc import make_rff, rolling_ridge_voc, summarize
from voc.features import GAMMAS


def _primal(Ftr, ytr, f, z):
    """Paper eq. (9) literally: beta = (zI + Psi)^-1 T^-1 sum S_t y_t."""
    T, L = Ftr.shape
    Psi = Ftr.T @ Ftr / T
    beta = np.linalg.solve(z * np.eye(L) + Psi, Ftr.T @ ytr / T)
    return f @ beta, beta @ beta


import pytest


@pytest.mark.parametrize("scale", ["none", "L"])
def test_dual_matches_primal_eq9(scale):
    rng = np.random.default_rng(0)
    N, T = 40, 12
    F = rng.standard_normal((N, 60))
    y = rng.standard_normal(N)
    L_grid, z_grid = [4, 12, 30, 60], [1e-2, 1.0, 100.0]
    out = rolling_ridge_voc(F, y, T, L_grid, z_grid, scale=scale)
    for i in (0, 7, 20):
        t = out["t"][i]
        for jl, L in enumerate(L_grid):
            c = 1.0 / np.sqrt(L) if scale == "L" else 1.0  # S = F / sqrt(L)
            for jz, z in enumerate(z_grid):
                p, b2 = _primal(c * F[t - T : t, :L], y[t - T : t], c * F[t, :L], z)
                assert np.isclose(out["pi"][i, jl, jz], p, rtol=1e-6, atol=1e-9)
                assert np.isclose(out["beta2"][i, jl, jz], b2, rtol=1e-6, atol=1e-9)


def test_no_lookahead():
    """pi_t must not change if y[t:] or X[t+1:] are altered."""
    rng = np.random.default_rng(1)
    N, T, t0 = 60, 12, 35
    X = rng.standard_normal((N, 3))
    y = rng.standard_normal(N)
    F = make_rff(X, p=5, seed=0)
    base = rolling_ridge_voc(F, y, T, [10, 40], [0.1])["pi"]
    y2, X2 = y.copy(), X.copy()
    y2[t0:] = rng.standard_normal(N - t0)  # y[t0] is the realisation after the t0 forecast
    X2[t0 + 1 :] = rng.standard_normal((N - t0 - 1, 3))
    alt = rolling_ridge_voc(make_rff(X2, p=5, seed=0), y2, T, [10, 40], [0.1])["pi"]
    i = t0 - T  # row of t0 in the output (t_start = T)
    assert np.allclose(base[: i + 1], alt[: i + 1])


def test_rff_approximates_mixture_kernel():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((6, 3))
    F = make_rff(X, p=3000, seed=3, convention="kernel")
    n_w = 7 * 3000  # raw features: F F' / P -> (1/7) sum_i K_i  (sin*sin + cos*cos = cos)
    K_hat = F @ F.T / n_w
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    K = np.mean([np.exp(-d2 / (2 * g)) for g in GAMMAS], axis=0)
    assert np.abs(K_hat - K).max() < 0.05


def test_ridgeless_interpolates_when_L_gt_T():
    rng = np.random.default_rng(4)
    T = 10
    F = rng.standard_normal((T + 1, 50))
    y = rng.standard_normal(T + 1)
    out = rolling_ridge_voc(F, y, T, [50], [1e-10], scale="none")
    t = out["t"][0]
    Ftr, ytr = F[t - T : t], y[t - T : t]
    alpha = np.linalg.solve(Ftr @ Ftr.T + 1e-10 * T * np.eye(T), ytr)
    beta = Ftr.T @ alpha
    assert np.allclose(Ftr @ beta, ytr, atol=1e-6)  # exact in-sample fit
    # and it is the min-norm interpolator: any other interpolator has larger norm
    null = rng.standard_normal(50)
    null -= Ftr.T @ np.linalg.solve(Ftr @ Ftr.T, Ftr @ null)
    assert np.linalg.norm(beta + null) > np.linalg.norm(beta)


def test_summarize_shapes_and_sr():
    pi = np.ones((5, 2, 3))
    y = np.array([1.0, -1.0, 2.0, 0.5, 0.0])
    s = summarize(pi, y)
    assert s["SR"].shape == (2, 3)
    assert np.isclose(s["SR"][0, 0], y.mean() / y.std())


def test_exact_kernel_matches_many_random_features():
    from voc.kernel import rolling_kernel_ridge

    rng = np.random.default_rng(5)
    N, T = 30, 12
    X = rng.standard_normal((N, 2))
    y = rng.standard_normal(N)
    z = [0.1, 10.0]
    exact = rolling_kernel_ridge(X, y, T, z)["pi"]
    F = make_rff(X, p=4000, seed=7)
    approx = rolling_ridge_voc(F, y, T, [F.shape[1]], z)["pi"][:, 0, :]
    assert np.abs(exact - approx).max() < 0.05 * np.abs(exact).max() + 1e-3
