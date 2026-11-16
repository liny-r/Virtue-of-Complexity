"""Placebo test (beyond the paper): how much of the Sharpe comes from the signals?

Real run    : signals X_t in their true time order (read from figures/curves_T{T}.npz).
Placebo run : rows of X permuted in time (same distribution, no link to y_t), repeated over many
              permutations; targets keep their true order, so anything that uses only the factor's
              own past returns (unconditional premium, trailing mean) is untouched and only the
              signal -> return mapping is destroyed. `n_real // shuffles` feature draws per shuffle (1 when the
              counts match), cycling through the SAME seed pool as the real run (the smallest models depend strongly on the draw).

Run experiments.curves (same --T) first. Outputs figures/placebo.* for T = 12, figures/placebo_T{T}.* otherwise.
Usage:  python -m experiments.placebo [--T 12] [--shuffles 20] [--workers 8] [--convention paper|kernel] [--fast]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.curves import load_all
from experiments.common import C_GRID, ROOT, Z_GRID, ctag, make_executor, run_many


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--p", type=int, default=500)
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--convention", default="paper", choices=["paper", "kernel"])
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    ref = f"curves_T{a.T}" + ctag(a.convention) + ("_fast" if a.fast else "")
    if a.fast:
        a.p, a.shuffles = 60, 3
    R = np.load(ROOT / "figures" / f"{ref}.npz")
    names = [str(n) for n in R["names"]]
    n_real = int(R["seeds"])
    draws = max(1, n_real // a.shuffles)  # feature draws per shuffle, so the shuffles together cover the real run's seed pool
    L_grid = sorted({int(round(c * a.T / 2)) * 2 for c in C_GRID if c * a.T <= 14 * a.p})
    c = np.array(L_grid) / a.T
    assert np.allclose(c, R["c"]), "grid differs from the reference run"
    series = load_all()
    real = R["SR"].mean(0)  # (nL, nz)
    t0 = time.time()
    ex = make_executor(a.workers)
    plc = []
    for k in range(a.shuffles):
        res = run_many(series, names, a.T, a.p, draws, L_grid, ex, shuffle_seed=k, seed_offset=(k * draws) % n_real, convention=a.convention)
        plc.append(np.mean([res[f]["SR"] for f in names], axis=0))
        print(f"  shuffle {k + 1}/{a.shuffles}, {time.time() - t0:.0f}s", flush=True)
    if ex:
        ex.shutdown()
    plc = np.array(plc)
    long_, trail = float(R["long"].mean()), float(R["trail"].mean())
    pm, ps = plc.mean(0), plc.std(0, ddof=1)
    stem = ("placebo" if a.T == 12 else f"placebo_T{a.T}") + ctag(a.convention) + ("_fast" if a.fast else "")
    np.savez(ROOT / "figures" / f"{stem}.npz", c=c, z=Z_GRID, real=real, placebo=plc, long=long_, trail=trail)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for a_, j in zip(ax, [0, 2, 6]):
        a_.fill_between(c, pm[:, j] - 2 * ps[:, j], pm[:, j] + 2 * ps[:, j], color="tab:orange", alpha=0.3, label="placebo mean ±2 sd")
        a_.plot(c, pm[:, j], color="tab:orange")
        a_.plot(c, real[:, j], color="tab:blue", label="real signals")
        a_.axhline(long_, color="r", ls="--", lw=1, label="always long")
        a_.axhline(trail, color="k", ls="--", lw=1, label="trailing mean")
        a_.set_xscale("log"); a_.set_title(f"log10 z = {np.log10(Z_GRID[j]):g}"); a_.set_xlabel("complexity c = L / T")
    ax[0].set_ylabel(f"annualised OOS Sharpe (mean of {len(names)})"); ax[0].legend(fontsize=7)
    fig.suptitle(f"Placebo: real vs time-shuffled signals (T={a.T}, {a.shuffles} shuffles)"); fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{stem}.png", dpi=140)

    pick = [int(np.argmin(abs(c - v))) for v in (1 / 6, 1, 16, c.max())]
    print(f"done in {time.time() - t0:.0f}s\nc =", np.round(c[pick], 2), "\n   real | placebo mean (sd) | real - placebo in sd units")
    for j in (0, 2, 4, 6):
        print(f"log10z={np.log10(Z_GRID[j]):+.0f}")
        for i in pick:
            print(f"   c={c[i]:6.2f}: {real[i, j]:.3f} | {pm[i, j]:.3f} ({ps[i, j]:.3f}) | {(real[i, j] - pm[i, j]) / ps[i, j]:+.1f}")
    g = plc[:, -1, 6] - plc[:, 0, 6]
    print(f"\nSharpe gain c_min -> c_max at z=1e3: real {real[-1, 6] - real[0, 6]:+.3f}; placebo {g.mean():+.3f} (sd {g.std(ddof=1):.3f})")
    print(f"always long {long_:.3f}; trailing mean {trail:.3f}")


if __name__ == "__main__":
    main()
