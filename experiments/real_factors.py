"""Virtue-of-complexity curves on real equity factors (paper sec. 4.1.1 + 5, Panel B analogue).

Factors : Mkt-RF, SMB, HML, RMW, CMA, Mom (Fama-French library), sample cut at 2021-12 to
          match the paper. This is 6 of the paper's 159 factors.
Signals : X_t = (r_t, mean(r_{t-11..t-1})) -- the paper's "momentum" specification.
Model   : random Fourier features (mixture of 7 Gaussian kernels) + rolling ridge, window T.
Baselines the paper does NOT report:
    long     : pi_t = 1                       (unconditional exposure to the vol-scaled factor)
    trail    : pi_t = mean(y_{t-T..t-1})      (plain trailing-mean "momentum")
    linear   : ridge on the 2 raw signals     (L = d = 2, no random features)

Sharpe ratios are ANNUALISED (x sqrt(12)); the paper does not say whether its Fig. 3 is.

Usage:  python -m experiments.real_factors [--T 12] [--seeds 20] [--p 500] [--fast]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from voc import make_rff, rolling_ridge_voc, summarize
from voc.preprocess import prepare_factor_dataset

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
C_GRID = [1 / 6, 1 / 3, 2 / 3, 1, 2, 4, 8, 16, 32, 64, 128, 256, 500]
Z_GRID = [10.0**k for k in range(-3, 4)]
ANN = np.sqrt(12.0)


def load_factors(end: str = "2021-12") -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "ff_factors_monthly.csv", index_col=0, parse_dates=True)[:end]


def _sr(x: np.ndarray) -> float:
    return float(x.mean() / x.std())


def run_factor(
    r: pd.Series, T: int, p: int, seeds: int, L_grid: list[int], shuffle_seed: int | None = None
) -> dict[str, np.ndarray]:
    """shuffle_seed: if given, permute the signal rows in time (placebo): same marginal
    distribution of X, but no link between X_t and the return that follows."""
    X, y = prepare_factor_dataset(r.dropna())
    Xv, yv = X.values, y.values
    if shuffle_seed is not None:
        Xv = Xv[np.random.default_rng(shuffle_seed).permutation(len(Xv))]
    SR, R2 = [], []
    for s in range(seeds):
        F = make_rff(Xv, p=p, seed=1000 + s)
        out = rolling_ridge_voc(F, yv, T, L_grid, Z_GRID)
        m = summarize(out["pi"], yv[out["t"]])
        SR.append(m["SR"])
        R2.append(m["R2"])
    ts = out["t"]
    yo = yv[ts]
    trail = np.array([yv[t - T : t].mean() for t in ts])
    lin = rolling_ridge_voc(Xv, yv, T, [Xv.shape[1]], Z_GRID, scale="none")
    lin_sr = summarize(lin["pi"], yo)["SR"][0]  # (nz,)
    return {
        "SR": np.mean(SR, axis=0) * ANN,  # (nL, nz), paper eq. (51)
        "R2": np.mean(R2, axis=0),
        "long": _sr(yo) * ANN,
        "trail": _sr(trail * yo) * ANN,
        "linear": lin_sr * ANN,
        "n_oos": len(ts),
        "start": str(X.index[ts[0]].date()),
    }


def plot(c, res, T, path: Path):
    sr = np.mean([res[f]["SR"] for f in FACTORS], axis=0)
    r2 = np.mean([res[f]["R2"] for f in FACTORS], axis=0)
    long_ = np.mean([res[f]["long"] for f in FACTORS])
    trail = np.mean([res[f]["trail"] for f in FACTORS])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for j, z in enumerate(Z_GRID):
        col = plt.cm.viridis(j / (len(Z_GRID) - 1))
        ax[0].plot(c, sr[:, j], color=col, label=f"log10 z = {np.log10(z):g}")
        ax[1].plot(c, r2[:, j], color=col)
    ax[0].axhline(long_, color="r", ls="--", lw=1, label="always long (no timing)")
    ax[0].axhline(trail, color="k", ls="--", lw=1, label=f"trailing {T}m mean")
    ax[0].set_title("Annualised OOS Sharpe, mean of 6 factors")
    ax[1].set_title("OOS R$^2$, mean of 6 factors")
    ax[1].set_ylim(-1.0, 0.1)
    for a in ax:
        a.set_xscale("log")
        a.axvline(1, color="k", ls=":", lw=1)
        a.set_xlabel("complexity c = L / T")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Real factors (FF5+Mom, momentum signals), T={T}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--p", type=int, default=500, help="weights per bandwidth; columns = 14p")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--end", default="2021-12")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.p, a.seeds = 60, 2
    P_cols = 14 * a.p
    L_grid = sorted({int(round(c * a.T / 2)) * 2 for c in C_GRID if c * a.T <= P_cols})
    c = np.array(L_grid) / a.T
    df = load_factors(a.end)
    t0 = time.time()
    res = {f: run_factor(df[f], a.T, a.p, a.seeds, L_grid) for f in FACTORS}
    tag = f"_T{a.T}" + ("_fast" if a.fast else "")
    out_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    plot(c, res, a.T, out_dir / f"real_factors{tag}.png")
    np.savez(out_dir / f"real_factors{tag}.npz", c=c, z=Z_GRID, **{f"{f}_{k}": v for f in FACTORS for k, v in res[f].items() if k in ("SR", "R2", "long", "trail", "linear")})
    print(f"done in {time.time() - t0:.0f}s (T={a.T}, seeds={a.seeds}, columns={P_cols})")
    print("OOS start / n:", {f: (res[f]["start"], res[f]["n_oos"]) for f in FACTORS})
    sr = np.mean([res[f]["SR"] for f in FACTORS], axis=0)
    pick = [int(np.argmin(abs(c - v))) for v in (1 / 6, 1, 16, c.max())]
    print("\nMean annualised Sharpe over 6 factors (rows: z; cols: c =", np.round(c[pick], 2), ")")
    for j, z in enumerate(Z_GRID):
        print(f"  log10z={np.log10(z):+.0f}:", np.round(sr[pick, j], 3))
    print("\nBaselines, mean annualised Sharpe over 6 factors:")
    print("  always long :", round(np.mean([res[f]["long"] for f in FACTORS]), 3))
    print(f"  trailing-{a.T}m :", round(np.mean([res[f]["trail"] for f in FACTORS]), 3))
    print("  linear ridge, log10z=-3..3:", np.round(np.mean([res[f]["linear"] for f in FACTORS], axis=0), 3))
    print("\nPer factor (c max, z=1e3 | long | trail):")
    for f in FACTORS:
        print(f"  {f:7}", round(res[f]["SR"][-1, -1], 3), "|", round(res[f]["long"], 3), "|", round(res[f]["trail"], 3))
    r2 = np.mean([res[f]["R2"] for f in FACTORS], axis=0)
    print("\nMean R2 at c=1 (z=1e-3, 1e3):", np.round(r2[pick[1], [0, -1]], 3), " at c max:", np.round(r2[-1, [0, -1]], 4))


if __name__ == "__main__":
    main()
