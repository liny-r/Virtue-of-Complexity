"""Paper Figures 1-4 (and 5 for T = 60, 120) on the paper's equity-factor universe.

Universe: the paper's 159 factors = 6 French-library factors (Mkt-RF, SMB, HML, RMW, CMA, Mom)
+ 153 US JKP characteristic factors (capped value-weighted, jkpfactors.com), cut at 2021-12.
Signals: the 2 momentum signals (paper Panel "Equities with Momentums"). Model: random Fourier
features of a mixture of 7 Gaussian kernels, rolling ridge with window T, z in 1e-3..1e3.
Reported curve = equal average across factors (paper sec. 5.2).

Two JKP series (ivol_hxz4_21d, iskew_hxz4_21d) have interior gaps in 1967-69; we use the
contiguous stretch after the last gap.

Outputs (T = 12 shown):
    figures/curves_T12.npz   per-factor arrays (SR, R2, ER, Vol, B2, baselines), so any subset can be re-averaged
    figures/fig1_T12.png     R^2 and ||beta||            (paper Fig. 1)
    figures/fig2_T12.png     expected return and vol     (paper Fig. 2)
    figures/fig3_T12.png     Sharpe + baselines          (paper Fig. 3; Fig. 5 for T = 60, 120)
    figures/fig4_T12.png     per-factor normalised Sharpe at z = 1e3 (paper Fig. 4)

Features follow the literal eq. (37) (frequencies gamma*w) unless --convention kernel (eq. 36; files end in _eq36).

Usage:  python -m experiments.curves [--signals momentum|wg] [--T 12] [--seeds 20] [--p 500] [--workers 8] [--convention paper|kernel] [--limit N] [--fast]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import (C_GRID, FACTORS, ROOT, Z_GRID, ctag, load_factors, load_wg_signals, make_executor, run_many)


def load_all(end: str = "2021-12") -> dict[str, pd.Series]:
    out = {f: s.dropna() for f, s in load_factors(end).items() if f in FACTORS}
    jkp = pd.read_csv(ROOT / "data" / "jkp_factors_usa_monthly.csv", index_col=0, parse_dates=True)[:end]
    for name, s in jkp.items():
        s = s.loc[s.first_valid_index():]
        if s.isna().any():  # keep the contiguous stretch after the last gap
            s = s.loc[s[s.isna()].index[-1]:].iloc[1:]
        out[name] = s
    return out


# Line colours of the paper's figures, one per log10 z = -3 ... 3 (sampled from its legend).
PAPER_COLORS = ["#952b33", "#c95c2d", "#e4b345", "#82aa45", "#6dbbe9", "#3071b7", "#743489"]


def _lines(ax, c, y, ylab=None):
    for j, z in enumerate(Z_GRID):
        ax.plot(c, y[:, j], color=PAPER_COLORS[j], label=f"log10 z = {np.log10(z):g}")
    ax.set_xscale("log"); ax.axvline(1, color="k", ls=":", lw=1); ax.set_xlabel("complexity c = L / T")
    if ylab:
        ax.set_title(ylab)


def make_figures(d: dict, tag: str, T: int):
    c, n = d["c"], len(d["names"])
    out = ROOT / "figures"
    mean = lambda k: d[k].mean(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    _lines(ax[0], c, mean("R2"), f"OOS $R^2$ (eq. 47), mean of {n} factors"); ax[0].set_ylim(-1, 0.1)
    _lines(ax[1], c, np.sqrt(mean("B2")), "$\\|\\hat\\beta\\|$ (sqrt of eq. 48), mean of factors"); ax[1].set_ylim(0, 10)
    ax[0].legend(fontsize=7); fig.suptitle(f"Fig. 1 analogue, {'Welch-Goyal' if '_wg' in tag else 'momentum'} signals, T={T}"); fig.tight_layout()
    fig.savefig(out / f"fig1{tag}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    _lines(ax[0], c, mean("ER"), "Expected return per month (eq. 49)")
    _lines(ax[1], c, mean("Vol"), "Volatility per month, std (eq. 50)"); ax[1].set_ylim(0, 3)
    ax[0].legend(fontsize=7); fig.suptitle(f"Fig. 2 analogue, T={T}"); fig.tight_layout()
    fig.savefig(out / f"fig2{tag}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    _lines(ax, c, mean("SR"), f"Annualised OOS Sharpe (eq. 51), mean of {n} factors")
    ax.axhline(d["long"].mean(), color="r", ls="--", lw=1, label="always long (not in paper)")
    ax.axhline(d["trail"].mean(), color="k", ls="--", lw=1, label=f"trailing {T}m mean (not in paper)")
    ax.legend(fontsize=7); fig.suptitle(f"Fig. {3 if T == 12 else 5} analogue, T={T}"); fig.tight_layout()
    fig.savefig(out / f"fig3{tag}.png", dpi=130); plt.close(fig)

    sr = d["SR"][:, :, Z_GRID.index(1e3)]  # (n, nL) at z = 1e3, as in the paper's Fig. 4
    lo, hi = sr.min(1, keepdims=True), sr.max(1, keepdims=True)
    nrm = (sr - lo) / np.where(hi - lo > 0, hi - lo, 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(c, nrm.T, color="tab:blue", alpha=0.07, lw=0.8)
    ax.plot(c, nrm.mean(0), color="k", lw=2, label="mean over factors")
    ax.fill_between(c, np.percentile(nrm, 25, 0), np.percentile(nrm, 75, 0), color="tab:orange", alpha=0.35, label="25-75% of factors")
    ax.set_xscale("log"); ax.set_xlabel("complexity c = L / T"); ax.legend(fontsize=7)
    ax.set_title(f"Fig. 4 analogue: per-factor Sharpe, min-max normalised (z=1e3, T={T})", fontsize=9); fig.tight_layout()
    fig.savefig(out / f"fig4{tag}.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--p", type=int, default=500, help="weights per bandwidth; columns = 14p")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N factors (timing / debugging)")
    ap.add_argument("--signals", default="momentum", choices=["momentum", "wg"],
                    help="'momentum': 2 momentum signals (paper's second study); 'wg': 15 Welch-Goyal predictors (first study; files end in _wg)")
    ap.add_argument("--convention", default="paper", choices=["paper", "kernel"],
                    help="'paper': frequencies gamma*w, eq. (37) literally (main); 'kernel': w/sqrt(gamma), eq. (36) (outputs tagged _eq36)")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.p, a.seeds, a.limit = 60, 1, 12
    P_cols = 14 * a.p
    L_grid = sorted({int(round(c * a.T / 2)) * 2 for c in C_GRID if c * a.T <= P_cols})
    c = np.array(L_grid) / a.T
    series = load_all()
    names = list(series)[: a.limit]
    t0 = time.time()
    ex = make_executor(a.workers)
    res = run_many(series, names, a.T, a.p, a.seeds, L_grid, ex, log_every=20, convention=a.convention,
                   signals=load_wg_signals() if a.signals == "wg" else None)
    if ex:
        ex.shutdown()
    d = {k: np.stack([res[f][k] for f in names]) for k in ("SR", "R2", "ER", "Vol", "B2", "linear")}
    d["long"] = np.array([res[f]["long"] for f in names]); d["trail"] = np.array([res[f]["trail"] for f in names])
    d.update(names=np.array(names), c=c, z=np.array(Z_GRID))
    tag = f"_T{a.T}" + ("_wg" if a.signals == "wg" else "") + ctag(a.convention) + ("_fast" if a.fast else "")
    np.savez(ROOT / "figures" / f"curves{tag}.npz", seeds=a.seeds, T=a.T, p=a.p, **d)
    make_figures(d, tag, a.T)

    sr, r2 = d["SR"].mean(0), d["R2"].mean(0)
    print(f"done in {time.time() - t0:.0f}s: {len(names)} factors, T={a.T}, seeds={a.seeds}, columns={P_cols}")
    pick = [int(np.argmin(abs(c - v))) for v in (1 / 6, 1, 16, c.max())]
    print("\nMean annualised Sharpe (rows: z; cols: c =", np.round(c[pick], 2), ")")
    for j, z in enumerate(Z_GRID):
        print(f"  log10z={np.log10(z):+.0f}:", np.round(sr[pick, j], 3))
    print(f"\nalways long {d['long'].mean():.3f}; trailing {d['trail'].mean():.3f}; linear ridge", np.round(d["linear"].mean(0), 3))
    print("mean R2 at c=1 (z=1e-3,1e3):", np.round(r2[pick[1], [0, -1]], 3))


if __name__ == "__main__":
    main()
