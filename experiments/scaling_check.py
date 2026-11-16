"""Sensitivity to the feature normalisation, judged against the paper's Figure 3 (Panel B shape).

Paper eq. (37) divides the sin/cos features by sqrt(P), but P is ambiguous (features in the model,
P = 7p weights, ...). Our main runs use S = F / sqrt(L) ("kappa = L": the model's own width), which
makes the effective ridge independent of L. Here the normalisation is a CONSTANT instead,
S = F / sqrt(kappa), for kappa in {1 (raw), T, P = 7p}:

    kappa = 1    raw sin/cos features, no normalisation
    kappa = T    12
    kappa = 7p   literal eq. (37) with P = 7p weights (3500 at p = 500)

What to look for in the paper's Fig. 3B (momentum signals, 159 factors, T = 12, c up to 1000):
  * z = 1e3 flat near 0.30; z = 1e2 and z = 1e1 do NOT coincide with it at high c (0.25, 0.20);
  * curves for z = 1 .. 1e2 peak around c = 3..10 and then FALL with c;
  * z <= 0.1 sit near 0.15-0.17 at high c; z = 1e-3 dips to ~0.07 near c = 1.
With a constant kappa the effective ridge shrinks as L grows, which produces the falling curves;
with kappa = L it does not, so all z >= 10 coincide.

Usage:  python -m experiments.scaling_check [--seeds 10] [--workers 8] [--fast]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.curves import load_all
from experiments.common import ANN, C_GRID, ROOT, Z_GRID, ctag, make_executor
from voc import make_rff, rolling_ridge_voc, summarize
from voc.preprocess import prepare_factor_dataset


def _one(args):
    """One factor: SR and R2 (nL, nz) per kappa, averaged over seeds."""
    r, T, p, seeds, L_grid, kappas, convention = args
    X, y = prepare_factor_dataset(r.dropna())
    Xv, yv = X.values, y.values
    sr = {k: np.zeros((len(L_grid), len(Z_GRID))) for k in kappas}
    r2 = {k: np.zeros((len(L_grid), len(Z_GRID))) for k in kappas}
    b2 = {k: np.zeros((len(L_grid), len(Z_GRID))) for k in kappas}
    for s in range(seeds):
        F = make_rff(Xv, p=p, seed=1000 + s, convention=convention)
        for k, kap in kappas.items():
            out = rolling_ridge_voc(F / np.sqrt(kap), yv, T, L_grid, Z_GRID, scale="none")
            m = summarize(out["pi"], yv[out["t"]], out["beta2"])
            sr[k] += m["SR"] * ANN / seeds
            r2[k] += m["R2"] / seeds
            b2[k] += m["beta2"] / seeds
    return sr, r2, b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--p", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--convention", default="paper", choices=["paper", "kernel"])
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.p, a.seeds, a.limit = 60, 1, 8
    T = a.T
    kappas = {"1 (raw)": 1.0, f"T = {T}": float(T), f"7p = {7 * a.p}": float(7 * a.p)}
    L_grid = sorted({int(round(c * T / 2)) * 2 for c in C_GRID if c * T <= 14 * a.p})
    c = np.array(L_grid) / T
    series = load_all()
    names = list(series)[: a.limit]
    SR = {k: np.zeros((len(names), len(L_grid), len(Z_GRID))) for k in kappas}
    R2 = {k: np.zeros_like(SR[k]) for k in kappas}
    B2 = {k: np.zeros_like(SR[k]) for k in kappas}
    t0 = time.time()
    ex = make_executor(a.workers)
    args = [(series[f], T, a.p, a.seeds, L_grid, kappas, a.convention) for f in names]
    for i, (sr, r2, b2) in enumerate(ex.map(_one, args, chunksize=2) if ex else map(_one, args)):
        for k in kappas:
            SR[k][i], R2[k][i], B2[k][i] = sr[k], r2[k], b2[k]
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(names)} factors, {time.time() - t0:.0f}s", flush=True)
    if ex:
        ex.shutdown()
    tag = ctag(a.convention) + ("_fast" if a.fast else "")
    np.savez(ROOT / "figures" / f"scaling_check{tag}.npz", seeds=a.seeds, c=c, z=Z_GRID, names=names,
             **{f"SR_{j}": SR[k] for j, k in enumerate(kappas)}, **{f"R2_{j}": R2[k] for j, k in enumerate(kappas)},
             **{f"B2_{j}": B2[k] for j, k in enumerate(kappas)}, kappas=list(kappas))

    fig, ax = plt.subplots(2, len(kappas), figsize=(4.6 * len(kappas), 7), sharex=True)
    for j, k in enumerate(kappas):
        sr, r2 = SR[k].mean(0), R2[k].mean(0)
        for q, z in enumerate(Z_GRID):
            col = plt.cm.viridis(q / (len(Z_GRID) - 1))
            ax[0, j].plot(c, sr[:, q], color=col, label=f"log10 z = {np.log10(z):g}")
            ax[1, j].plot(c, r2[:, q], color=col)
        ax[0, j].set_title(f"kappa = {k}: Sharpe, mean of {len(names)} factors")
        ax[1, j].set_title("OOS $R^2$"); ax[1, j].set_ylim(-1, 0.1)
        ax[1, j].set_xlabel("c = L / T")
        for r in (0, 1):
            ax[r, j].set_xscale("log"); ax[r, j].axvline(1, color="k", ls=":", lw=1)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"scaling_check{tag}.png", dpi=130)

    pick = [int(np.argmin(abs(c - v))) for v in (1, 16, 64, c.max())]
    print("||beta|| (sqrt of mean ||beta||^2) at z=1e-3 for c =", np.round(c[[0, 3, 5, 7, -1]], 2))
    for k in kappas:
        print(f"  kappa = {k}:", np.round(np.sqrt(B2[k].mean(0))[[0, 3, 5, 7, -1], 0], 2))
    print(f"done in {time.time() - t0:.0f}s; mean Sharpe over {len(names)} factors; columns c =", np.round(c[pick], 1))
    for k in kappas:
        print(f"\nkappa = {k}")
        for q, z in enumerate(Z_GRID):
            print(f"  log10z={np.log10(z):+.0f}:", np.round(SR[k].mean(0)[pick, q], 3))


if __name__ == "__main__":
    main()
