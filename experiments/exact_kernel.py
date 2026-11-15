"""Mechanism check: is the placebo's rise with c just random-feature approximation error?

Exact mixture-kernel ridge = the c -> infinity limit with no feature sampling. Run it on
real and time-shuffled signals and compare with the random-feature results at c_min / c_max.

If the exact-kernel placebo equals the c_max random-feature placebo (~0.375 at z=1e3),
the placebo's rise from c_min to c_max is the random-feature kernel converging to the exact
kernel, and the placebo level itself is what that kernel does with uninformative signals.

Usage:  python -m experiments.exact_kernel [--shuffles 20]
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.real_factors import ANN, FACTORS, ROOT, Z_GRID, load_factors
from voc.kernel import rolling_kernel_ridge
from voc.metrics import summarize
from voc.preprocess import prepare_factor_dataset


def exact_sr(r, shuffle_seed=None, T=12):
    X, y = prepare_factor_dataset(r.dropna())
    Xv, yv = X.values, y.values
    if shuffle_seed is not None:
        Xv = Xv[np.random.default_rng(shuffle_seed).permutation(len(Xv))]
    out = rolling_kernel_ridge(Xv, yv, T, Z_GRID)
    return summarize(out["pi"][:, None, :], yv[out["t"]])["SR"][0] * ANN  # (nz,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--ref", default="placebo", help="stem of figures/<ref>.npz holding the random-feature results")
    a = ap.parse_args()
    df = load_factors()
    real = np.mean([exact_sr(df[f], None, a.T) for f in FACTORS], axis=0)
    plc = np.array([np.mean([exact_sr(df[f], k, a.T) for f in FACTORS], axis=0) for k in range(a.shuffles)])
    ref = np.load(ROOT / "figures" / f"{a.ref}.npz")
    suffix = "_fast" if a.ref.endswith("_fast") else ""  # never overwrite the full-run result
    np.savez(ROOT / "figures" / f"exact_kernel{suffix}.npz", z=Z_GRID, real=real, placebo=plc)
    print("mean annualised Sharpe over 6 factors; z columns:", [f"1e{int(np.log10(z)):+d}" for z in Z_GRID])
    print("exact kernel, real     :", np.round(real, 3))
    print("exact kernel, placebo  :", np.round(plc.mean(0), 3), "(sd", np.round(plc.std(0, ddof=1), 3)[[0, 3, 6]], "at z=1e-3,1,1e3)")
    print("RFF c=%.0f, real       :" % ref["c"][-1], np.round(ref["real"][-1], 3))
    print("RFF c=%.0f, placebo    :" % ref["c"][-1], np.round(ref["placebo"].mean(0)[-1], 3))
    print("RFF c=%.2f, placebo    :" % ref["c"][0], np.round(ref["placebo"].mean(0)[0], 3))
    print("RFF c=%.2f, real       :" % ref["c"][0], np.round(ref["real"][0], 3))
    print(f"always long {float(ref['long']):.3f}; trailing mean {float(ref['trail']):.3f}")


if __name__ == "__main__":
    main()
