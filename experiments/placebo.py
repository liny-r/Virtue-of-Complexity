"""Placebo test: how much of the Sharpe comes from the signals, and how much from elsewhere?

Real run    : signals X_t in their true time order.
Placebo run : rows of X permuted in time (same distribution, no link to y_{t}), repeated
              over many permutations. Targets y keep their true order, so anything that
              only uses past returns of the factor itself (unconditional premium, trailing
              mean) is untouched; only the signal -> return mapping is destroyed.

If the real curve sits on top of the placebo band, the signals add nothing beyond that.

Usage:  python -m experiments.placebo [--shuffles 20] [--seeds 5] [--fast]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.real_factors import C_GRID, FACTORS, ROOT, Z_GRID, load_factors, run_factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--p", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5, help="feature draws per placebo shuffle")
    ap.add_argument("--real-seeds", type=int, default=20)
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.p, a.seeds, a.real_seeds, a.shuffles = 60, 1, 2, 4
    L_grid = sorted({int(round(c * a.T / 2)) * 2 for c in C_GRID if c * a.T <= 14 * a.p})
    c = np.array(L_grid) / a.T
    df = load_factors()
    t0 = time.time()
    real = np.mean([run_factor(df[f], a.T, a.p, a.real_seeds, L_grid)["SR"] for f in FACTORS], axis=0)
    plc = np.array(
        [
            np.mean([run_factor(df[f], a.T, a.p, a.seeds, L_grid, shuffle_seed=k)["SR"] for f in FACTORS], axis=0)
            for k in range(a.shuffles)
        ]
    )  # (shuffles, nL, nz) -- mean over 6 factors, one row per permutation
    base = [run_factor(df[f], a.T, a.p, 1, L_grid[:1]) for f in FACTORS]
    long_, trail = np.mean([b["long"] for b in base]), np.mean([b["trail"] for b in base])
    pm, ps = plc.mean(0), plc.std(0, ddof=1)
    tag = "_fast" if a.fast else ""
    np.savez(ROOT / "figures" / f"placebo{tag}.npz", c=c, z=Z_GRID, real=real, placebo=plc, long=long_, trail=trail)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for a_, j in zip(ax, [0, 2, 6]):
        a_.fill_between(c, pm[:, j] - 2 * ps[:, j], pm[:, j] + 2 * ps[:, j], color="tab:orange", alpha=0.3, label="placebo mean ±2 sd")
        a_.plot(c, pm[:, j], color="tab:orange")
        a_.plot(c, real[:, j], color="tab:blue", label="real signals")
        a_.axhline(long_, color="r", ls="--", lw=1, label="always long")
        a_.axhline(trail, color="k", ls="--", lw=1, label="trailing mean")
        a_.set_xscale("log")
        a_.set_title(f"log10 z = {np.log10(Z_GRID[j]):g}")
        a_.set_xlabel("complexity c = L / T")
    ax[0].set_ylabel("annualised OOS Sharpe (mean of 6 factors)")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Placebo: real vs time-shuffled signals ({a.shuffles} shuffles)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"placebo{tag}.png", dpi=140)

    print(f"done in {time.time() - t0:.0f}s")
    pick = [int(np.argmin(abs(c - v))) for v in (1 / 6, 1, 16, c.max())]
    print("c =", np.round(c[pick], 2), "\n   real | placebo mean (sd) | real - placebo in sd units")
    for j in (0, 2, 4, 6):
        print(f"log10z={np.log10(Z_GRID[j]):+.0f}")
        for i in pick:
            print(f"   c={c[i]:6.2f}: {real[i, j]:.3f} | {pm[i, j]:.3f} ({ps[i, j]:.3f}) | {(real[i, j] - pm[i, j]) / ps[i, j]:+.1f}")
    gain_r = real[-1, 6] - real[0, 6]
    gain_p = plc[:, -1, 6] - plc[:, 0, 6]
    print(f"\nSharpe gain c_min -> c_max at z=1e3: real {gain_r:+.3f}; placebo {gain_p.mean():+.3f} (sd {gain_p.std(ddof=1):.3f})")
    print(f"always long {long_:.3f}; trailing mean {trail:.3f}")


if __name__ == "__main__":
    main()
