"""Simulation check of the three theory predictions (paper section 1 / Kelly et al. 2022).

  (1) OOS R^2 is generally negative while OOS Sharpe is positive.
  (2) With (almost) no ridge, Sharpe dips to ~0 at c = P/T = 1 ("double ascent").
  (3) With enough ridge, Sharpe is monotone increasing in complexity c.
  Lemma 1: ||beta|| decreases in L for L > T in the ridgeless limit.

Usage:  python -m experiments.sim_theory [--seeds 20] [--T 12] [--fast]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voc import make_rff, rolling_ridge_voc, summarize
from voc.simulate import simulate_nonlinear

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
C_GRID = [1 / 6, 1 / 3, 2 / 3, 1, 2, 4, 8, 16, 32, 64, 128, 256]
Z_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]


def run(N: int, T: int, d: int, snr: float, p: int, seeds: int, data_seeds: int, mu: float = 0.0, convention: str = "paper"):
    """Returns c grid and, per metric, an array (data_seeds, nL, nz).

    Within a dataset, metrics are averaged over `seeds` random-feature draws
    (paper eqs. 47-51). Across datasets we keep the draws separate so we can
    report Monte-Carlo standard errors.
    """
    P_cols = 2 * 7 * p
    L_grid = sorted({int(round(c * T / 2)) * 2 for c in C_GRID if c * T <= P_cols})
    per_data: dict[str, list[np.ndarray]] = {}
    for ds in range(data_seeds):
        X, y, _ = simulate_nonlinear(N, d=d, snr=snr, seed=ds, mu=mu)
        acc: dict[str, list[np.ndarray]] = {}
        for s in range(seeds):
            F = make_rff(X, p=p, seed=1000 + s, convention=convention)
            out = rolling_ridge_voc(F, y, T, L_grid, Z_GRID)
            m = summarize(out["pi"], y[out["t"]], out["beta2"])
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
        for k, v in acc.items():
            per_data.setdefault(k, []).append(np.mean(v, axis=0))
    return np.array(L_grid) / T, {k: np.array(v) for k, v in per_data.items()}


def plot(c, raw, T, path: Path, mu: float = 0.0):
    m = {k: v.mean(0) for k, v in raw.items()}
    se = {k: v.std(0, ddof=1) / np.sqrt(len(v)) for k, v in raw.items()}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cmap = plt.cm.viridis
    for j, z in enumerate(Z_GRID):
        col = cmap(j / (len(Z_GRID) - 1))
        lab = f"log10 z = {np.log10(z):g}"
        ax[0].plot(c, m["SR"][:, j], color=col, label=lab)
        ax[0].fill_between(c, m["SR"][:, j] - 2 * se["SR"][:, j], m["SR"][:, j] + 2 * se["SR"][:, j], color=col, alpha=0.12, lw=0)
        ax[1].plot(c, m["R2"][:, j], color=col)
        ax[2].plot(c, np.sqrt(m["beta2"][:, j]), color=col)
    for a, t in zip(ax, ["OOS Sharpe (per period)", "OOS R$^2$", r"$\|\hat\beta\|$"]):
        a.set_xscale("log")
        a.axvline(1, color="k", ls=":", lw=1)
        a.set_xlabel("complexity c = L / T")
        a.set_title(t)
    ax[1].set_ylim(-0.5, 0.1)
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Simulated nonlinear DGP, T={T}, drift={mu:g}: mean over {len(raw['SR'])} datasets (Sharpe band = ±2 s.e.)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1200)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--mu", type=float, default=0.0, help="unconditional premium (per-period Sharpe of buy-and-hold)")
    ap.add_argument("--snr", type=float, default=0.05)
    ap.add_argument("--p", type=int, default=300, help="weights per bandwidth; columns = 14p")
    ap.add_argument("--seeds", type=int, default=5, help="random-feature draws per dataset")
    ap.add_argument("--data-seeds", type=int, default=50, help="independent simulated datasets")
    ap.add_argument("--log10z", default="-3,-2,-1,0,1,2,3", help="comma-separated log10 ridge grid")
    ap.add_argument("--convention", default="paper", choices=["paper", "kernel"])
    ap.add_argument("--fast", action="store_true", help="quick smoke run")
    a = ap.parse_args()
    global Z_GRID
    Z_GRID = [10.0 ** float(v) for v in a.log10z.split(",")]
    if a.fast:
        a.N, a.p, a.seeds, a.data_seeds = 300, 60, 2, 3
    t0 = time.time()
    c, raw = run(a.N, a.T, a.d, a.snr, a.p, a.seeds, a.data_seeds, a.mu, a.convention)
    m = {k: v.mean(0) for k, v in raw.items()}
    FIG_DIR.mkdir(exist_ok=True)
    tag = (f"_mu{a.mu:g}" if a.mu else "") + ("" if a.log10z == "-3,-2,-1,0,1,2,3" else "_z" + a.log10z.replace(",", "_")) + ("_eq36" if a.convention == "kernel" else "")
    out = FIG_DIR / (f"sim_theory{tag}_fast.png" if a.fast else f"sim_theory{tag}.png")
    plot(c, raw, a.T, out, a.mu)
    np.savez(out.with_suffix(".npz"), c=c, z=Z_GRID, **raw)
    print(f"done in {time.time() - t0:.1f}s -> {out}")
    print("c        :", np.round(c, 2))
    for j, z in enumerate(Z_GRID):
        print(f"SR log10z={np.log10(z):+g}:", np.round(m["SR"][:, j], 3))
    # paired double-ascent test: SR(c=1) - SR(c=2/3), per dataset, for each z
    i1, i0 = int(np.argmin(abs(c - 1))), int(np.argmin(abs(c - 2 / 3)))
    d = raw["SR"][:, i1, :] - raw["SR"][:, i0, :]
    print("dip SR(c=1)-SR(c=2/3), mean (t-stat):")
    for j, z in enumerate(Z_GRID):
        t = d[:, j].mean() / (d[:, j].std(ddof=1) / np.sqrt(len(d)))
        print(f"  log10z={np.log10(z):+g}: {d[:, j].mean():+.4f} (t={t:+.1f})")
    print("R2 z=1e-3:", np.round(m["R2"][:, 0], 2))
    print("R2 z=1e+3:", np.round(m["R2"][:, -1], 4))


if __name__ == "__main__":
    main()
