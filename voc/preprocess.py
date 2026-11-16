"""Signal / return standardisation for the equity-factor study (paper section 4.1.1).

Conventions (all strictly past-only, so no look-ahead):
  * y[t] = r[t+1] / sigma_t, sigma_t = sqrt(mean(r[t-11..t]^2))           (13)-(14)
  * signals: X1_t = r[t],  X2_t = mean(r[t-11..t-1])                       (sec 4.1.1)
  * each signal divided by an expanding-window RMS using data up to t-1,
    requiring >= 36 observations, then clipped at +-5                      (15)-(16)

NOTE (paper ambiguity): eq. (15) sums X[t-tau]^2 for tau = 1..t-37 and divides by t-37, which read
literally is the mean over observations 37..t-1: it keeps the most recent data but DROPS THE FIRST 36
observations. We read the intent as "expanding window over all data before t, at least 36 months".
For the Momentum factor the two readings differ by <= 15% (median 1%) once 120 observations exist,
and by more only in the first few months of the sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_returns(r: pd.Series, window: int = 12) -> pd.Series:
    """Return y indexed by information date t: y[t] = r[t+1] / sigma_t."""
    sigma = np.sqrt((r**2).rolling(window, min_periods=window).mean())
    return (r.shift(-1) / sigma).rename("y")


def momentum_signals(r: pd.Series) -> pd.DataFrame:
    """Raw 2-signal set: lagged return and trailing 11-month mean (excluding t)."""
    x1 = r.rename("ret_1m")
    x2 = r.shift(1).rolling(11, min_periods=11).mean().rename("ret_11m")
    return pd.concat([x1, x2], axis=1)


def standardize_signals(X: pd.DataFrame, min_obs: int = 36, clip: float = 5.0) -> pd.DataFrame:
    rms = np.sqrt((X**2).expanding(min_periods=min_obs).mean().shift(1))
    return (X / rms).clip(-clip, clip)


WG_NAMES = ["dfy", "infl", "svar", "de", "lty", "tms", "tbl", "dfr", "dp", "dy", "ltr", "ep", "b/m", "ntis", "mkt"]


def wg_raw_signals(wg: pd.DataFrame, end: str = "2020-12") -> pd.DataFrame:
    """The paper's 15 Welch-Goyal predictors (footnote 10), built from data/welch_goyal_monthly.csv.

    dfy = BAA - AAA, tms = lty - tbl, dfr = corpr - ltr, dp = log D12 - log Index, dy = log D12 - log Index(-1),
    ep = log E12 - log Index, de = log D12 - log E12; infl, svar, lty, tbl, ltr, b/m, ntis as given; and the
    aggregate market return (CRSP value-weighted) of month t. Inflation is lagged one month (CPI is published
    with a delay, Goyal's convention), so every column is known at the end of month t. The paper uses the series
    through 2020 (its sample); rows with any missing value are dropped (ntis starts in 1926-12).
    """
    idx = wg["Index"]
    out = pd.DataFrame(
        {
            "dfy": wg["BAA"] - wg["AAA"],
            "infl": wg["infl"].shift(1),
            "svar": wg["svar"],
            "de": np.log(wg["D12"]) - np.log(wg["E12"]),
            "lty": wg["lty"],
            "tms": wg["lty"] - wg["tbl"],
            "tbl": wg["tbl"],
            "dfr": wg["corpr"] - wg["ltr"],
            "dp": np.log(wg["D12"]) - np.log(idx),
            "dy": np.log(wg["D12"]) - np.log(idx.shift(1)),
            "ltr": wg["ltr"],
            "ep": np.log(wg["E12"]) - np.log(idx),
            "b/m": wg["b/m"],
            "ntis": wg["ntis"],
            "mkt": wg["CRSP_SPvw"],
        }
    )
    return out.loc[:end].dropna()[WG_NAMES]


def prepare_factor_dataset(r: pd.Series, raw_signals: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Aligned (X, y) for one factor, with NaN rows (warm-up, last obs) dropped.

    raw_signals=None: the 2 momentum signals built from the factor itself (paper's second study).
    raw_signals=DataFrame (e.g. wg_raw_signals): common predictors, standardised by their own expanding-window
    RMS (eq. 15-16) over their whole history, then aligned to the factor's dates (paper's first study)."""
    if raw_signals is None:
        X = standardize_signals(momentum_signals(r))
    else:
        X = standardize_signals(raw_signals).reindex(r.index)
    y = standardize_returns(r)
    df = pd.concat([X, y], axis=1).dropna()
    return df[X.columns], df["y"]
