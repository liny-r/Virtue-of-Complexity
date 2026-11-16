"""Mechanism check (beyond the paper): is the placebo's rise with c just random-feature error?

Exact mixture-kernel ridge = the c -> infinity limit with no feature sampling. Run it on the 159
factors with real and time-shuffled signals and compare with the random-feature results at c_min / c_max.
If the exact-kernel placebo equals the c_max random-feature placebo, the placebo's rise is the sampled
kernel converging to the exact kernel.

Run experiments.curves and experiments.placebo first. Usage:  python -m experiments.exact_kernel [--shuffles 20] [--workers 8]
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.curves import load_all
from experiments.common import ANN, ROOT, Z_GRID, ctag, make_executor
from voc.kernel import rolling_kernel_ridge
from voc.metrics import summarize
from voc.preprocess import prepare_factor_dataset


def exact_sr(args):
    r, shuffle_seed, T, convention = args
    X, y = prepare_factor_dataset(r.dropna())
    Xv, yv = X.values, y.values
    if shuffle_seed is not None:
        Xv = Xv[np.random.default_rng(shuffle_seed).permutation(len(Xv))]
    out = rolling_kernel_ridge(Xv, yv, T, Z_GRID, convention=convention)
    return summarize(out["pi"][:, None, :], yv[out["t"]])["SR"][0] * ANN  # (nz,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--convention", default="paper", choices=["paper", "kernel"])
    ap.add_argument("--fast", action="store_true", help="3 shuffles, first 12 factors; reads the _fast reference runs")
    a = ap.parse_args()
    series = load_all()
    names = list(series)[: 12 if a.fast else None]
    if a.fast:
        a.shuffles = 3
    ex = make_executor(a.workers)
    mp = ex.map if ex else map
    real = np.mean(list(mp(exact_sr, [(series[f], None, a.T, a.convention) for f in names])), axis=0)
    plc = np.array([np.mean(list(mp(exact_sr, [(series[f], k, a.T, a.convention) for f in names])), axis=0) for k in range(a.shuffles)])
    if ex:
        ex.shutdown()
    suffix = ctag(a.convention) + ("_fast" if a.fast else "")
    ref = np.load(ROOT / "figures" / f"placebo{suffix}.npz")
    np.savez(ROOT / "figures" / f"exact_kernel{suffix}.npz", z=Z_GRID, real=real, placebo=plc)
    print(f"mean annualised Sharpe over {len(names)} factors; z columns:", [f"1e{int(np.log10(z)):+d}" for z in Z_GRID])
    print("exact kernel, real     :", np.round(real, 3))
    print("exact kernel, placebo  :", np.round(plc.mean(0), 3), "(sd", np.round(plc.std(0, ddof=1), 3)[[0, 3, 6]], "at z=1e-3,1,1e3)")
    print("RFF c=%.0f, real       :" % ref["c"][-1], np.round(ref["real"][-1], 3))
    print("RFF c=%.0f, placebo    :" % ref["c"][-1], np.round(ref["placebo"].mean(0)[-1], 3))
    print(f"always long {float(ref['long']):.3f}; trailing mean {float(ref['trail']):.3f}")


if __name__ == "__main__":
    main()
