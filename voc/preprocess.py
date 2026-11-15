"""Signal / return standardisation for the equity-factor study (paper section 4.1.1).

Conventions (all strictly past-only, so no look-ahead):
  * y[t] = r[t+1] / sigma_t, sigma_t = sqrt(mean(r[t-11..t]^2))           (13)-(14)
  * signals: X1_t = r[t],  X2_t = mean(r[t-11..t-1])                       (sec 4.1.1)
  * each signal divided by an expanding-window RMS using data up to t-1,
    requiring >= 36 observations, then clipped at +-5                      (15)-(16)

NOTE (paper ambiguity): eq. (15) sums X[t-tau]^2 for tau = 1..t-37, which read literally
drops the 36 most recent AND the first 36 observations. We read the intent as
"expanding window, at least 36 months of history, using only data before t".
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


def prepare_factor_dataset(r: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Aligned (X, y) for one factor, with NaN rows (warm-up, last obs) dropped."""
    X = standardize_signals(momentum_signals(r))
    y = standardize_returns(r)
    df = pd.concat([X, y], axis=1).dropna()
    return df[X.columns], df["y"]
