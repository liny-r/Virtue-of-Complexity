"""Out-of-sample metrics (paper eqs. 41-46), computed per (L, z) cell."""
from __future__ import annotations

import numpy as np


def summarize(pi: np.ndarray, y_oos: np.ndarray, beta2: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """pi: (n, nL, nz) positions; y_oos: (n,) realised next-period returns.

    R2  = 1 - sum (y - pi)^2 / sum (y - ybar)^2                     (41)
    ER  = mean(pi * y)                                              (44)
    Vol = std(pi * y)  (population)                                 (45)
    SR  = ER / Vol, per period (not annualised)                     (46)
    ||beta||^2 = time-average of ||beta_t||^2                       (42)
    """
    y = np.asarray(y_oos, dtype=float)[:, None, None]
    strat = pi * y
    ER = strat.mean(axis=0)
    vol = strat.std(axis=0)
    out = {
        "R2": 1.0 - ((y - pi) ** 2).sum(axis=0) / ((y - y.mean()) ** 2).sum(),
        "ER": ER,
        "Vol": vol,
        "SR": np.where(vol > 0, ER / np.where(vol > 0, vol, 1.0), np.nan),
    }
    if beta2 is not None:
        out["beta2"] = beta2.mean(axis=0)
    return out
