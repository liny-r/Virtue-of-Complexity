"""Shared helpers for the equity-factor experiments (paper sec. 4.1.1 + 5, Panel B analogue).

Universe : 159 factors, see experiments/curves.py (6 French factors + 153 JKP US factors).
Signals  : X_t = (r_t, mean(r_{t-11..t-1})) -- the paper's "momentum" specification.
Model    : random Fourier features (mixture of 7 Gaussian kernels) + rolling ridge, window T.
Reported per factor and (L, z): OOS R^2, ||beta||^2, ER, Vol, Sharpe (paper eqs. 41-46), averaged
over random-feature seeds (eqs. 47-51). Sharpe ratios are ANNUALISED (x sqrt(12)); ER and Vol are
per month. Baselines the paper does NOT report (used only in the "beyond the paper" checks):
    long     : pi_t = 1                       (unconditional exposure to the vol-scaled factor)
    trail    : pi_t = mean(y_{t-T..t-1})      (plain trailing-mean "momentum")
    linear   : ridge on the 2 raw signals     (L = d = 2, no random features)
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from voc import make_rff, rolling_ridge_voc, summarize
from voc.preprocess import prepare_factor_dataset

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]  # the French-library part of the universe
C_GRID = [1 / 6, 1 / 3, 2 / 3, 1, 2, 4, 8, 16, 32, 64, 128, 256, 500]
Z_GRID = [10.0**k for k in range(-3, 4)]
ANN = np.sqrt(12.0)


def load_wg_signals() -> pd.DataFrame:
    """The 15 raw Welch-Goyal predictors (see voc.preprocess.wg_raw_signals)."""
    from voc.preprocess import wg_raw_signals

    return wg_raw_signals(pd.read_csv(ROOT / "data" / "welch_goyal_monthly.csv", index_col=0, parse_dates=True))


def ctag(convention: str) -> str:
    """File-name tag: the main results use the literal eq. (37); eq. (36) outputs end in _eq36."""
    return "_eq36" if convention == "kernel" else ""


def load_factors(end: str = "2021-12") -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "ff_factors_monthly.csv", index_col=0, parse_dates=True)[:end]


def _sr(x: np.ndarray) -> float:
    return float(x.mean() / x.std())


def run_factor(
    r: pd.Series, T: int, p: int, seeds: int, L_grid: list[int], shuffle_seed: int | None = None, seed_offset: int = 0,
    convention: str = "paper",
    signals: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """shuffle_seed: if given, permute the signal rows in time (placebo): same marginal
    distribution of X, but no link between X_t and the return that follows.
    Feature draws use seeds 1000 + seed_offset + s. The smallest models (L = 2, 4) depend heavily
    on which frequency is drawn, so real and placebo runs must average over the SAME pool of seeds."""
    X, y = prepare_factor_dataset(r.dropna(), signals)
    Xv, yv = X.values, y.values
    if shuffle_seed is not None:
        Xv = Xv[np.random.default_rng(shuffle_seed).permutation(len(Xv))]
    acc = {k: [] for k in ("SR", "R2", "ER", "Vol", "B2")}
    for s in range(seeds):
        F = make_rff(Xv, p=p, seed=1000 + seed_offset + s, convention=convention)
        out = rolling_ridge_voc(F, yv, T, L_grid, Z_GRID)
        m = summarize(out["pi"], yv[out["t"]], out["beta2"])
        for k, key in (("SR", "SR"), ("R2", "R2"), ("ER", "ER"), ("Vol", "Vol"), ("B2", "beta2")):
            acc[k].append(m[key])
    ts = out["t"]
    yo = yv[ts]
    trail = np.array([yv[t - T : t].mean() for t in ts])
    lin = rolling_ridge_voc(Xv, yv, T, [Xv.shape[1]], Z_GRID, scale="none")
    lin_sr = summarize(lin["pi"], yo)["SR"][0]  # (nz,)
    return {
        "SR": np.mean(acc["SR"], axis=0) * ANN,  # (nL, nz), paper eq. (51), annualised
        "R2": np.mean(acc["R2"], axis=0),
        "ER": np.mean(acc["ER"], axis=0),  # per month, eq. (49)
        "Vol": np.mean(acc["Vol"], axis=0),  # per month, std of the strategy return, eq. (50)
        "B2": np.mean(acc["B2"], axis=0),  # time-average of ||beta||^2, eq. (48)
        "long": _sr(yo) * ANN,
        "trail": _sr(trail * yo) * ANN,
        "linear": lin_sr * ANN,
        "n_oos": len(ts),
        "start": str(X.index[ts[0]].date()),
    }


# ---- optional process pool (results are identical to the serial run: seeds are deterministic) ----
def _task(args):
    return run_factor(*args)


def make_executor(workers: int):
    if workers <= 1:
        return None
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"  # one BLAS thread per worker; set before the workers import numpy
    return ProcessPoolExecutor(max_workers=workers)


def run_many(series, names, T, p, seeds, L_grid, ex=None, shuffle_seed=None, seed_offset=0, log_every=0, convention="paper", signals=None):
    """run_factor over many factors, optionally on a process pool; returns {name: result}."""
    args = [(series[f], T, p, seeds, L_grid, shuffle_seed, seed_offset, convention, signals) for f in names]
    out = {}
    it = ex.map(_task, args, chunksize=2) if ex is not None else map(_task, args)
    for i, (f, res) in enumerate(zip(names, it)):
        out[f] = res
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(names)} factors", flush=True)
    return out
