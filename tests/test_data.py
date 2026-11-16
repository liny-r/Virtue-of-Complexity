"""Tests for the data layer and for seed handling in the experiments."""
import numpy as np
import pandas as pd

from experiments.curves import load_all
from experiments.common import run_factor


def test_universe_is_159_clean_series():
    series = load_all()
    assert len(series) == 159  # 6 French factors + 153 JKP
    assert {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"} <= set(series)
    for name, s in series.items():
        assert s.index.max() <= pd.Timestamp("2021-12-31"), name
        assert not s.isna().any(), name  # leading NaNs trimmed, interior gaps cut
        assert len(s) >= 400, name


def test_data_matches_paper_table1():
    paper = pd.read_csv("data/paper_table1.csv")
    assert len(paper) == 159
    series = load_all()
    ours, theirs = [], []
    for _, row in paper.iterrows():
        if row.factor in series:  # "intrinsic_value" has no counterpart in the public JKP file
            s = series[row.factor]
            ours.append(s.mean() / s.std() * np.sqrt(12))
            theirs.append(row.sharpe)
    assert len(ours) == 158
    assert np.corrcoef(ours, theirs)[0, 1] > 0.99
    assert np.max(np.abs(np.array(ours) - np.array(theirs))) < 0.15


def test_seed_offset_selects_the_same_draws():
    """run_factor averages over seeds 1000+offset+s, so three single-seed runs at offsets 0,1,2
    average to one three-seed run (the placebo relies on this to reuse the real run's seed pool)."""
    r = pd.Series(np.random.default_rng(0).normal(0.0, 0.04, 160), index=pd.date_range("2000-01-31", periods=160, freq="ME"))
    kw = dict(T=12, p=10, L_grid=[2, 4, 12])
    three = run_factor(r, seeds=3, **kw)["SR"]
    singles = [run_factor(r, seeds=1, seed_offset=k, **kw)["SR"] for k in range(3)]
    np.testing.assert_allclose(three, np.mean(singles, axis=0), atol=1e-12)


def test_welch_goyal_predictors():
    from experiments.common import load_wg_signals
    from voc.preprocess import WG_NAMES, prepare_factor_dataset

    W = load_wg_signals()
    assert list(W.columns) == WG_NAMES and len(WG_NAMES) == 15
    assert not W.isna().any().any()
    assert W.index.max() == pd.Timestamp("2020-12-31")
    assert W.index.min() <= pd.Timestamp("1927-01-31")
    # identities that define the derived predictors
    raw = pd.read_csv("data/welch_goyal_monthly.csv", index_col=0, parse_dates=True)
    np.testing.assert_allclose(W["dfy"], (raw["BAA"] - raw["AAA"]).loc[W.index])
    np.testing.assert_allclose(W["tms"], (raw["lty"] - raw["tbl"]).loc[W.index])
    np.testing.assert_allclose(W["infl"], raw["infl"].shift(1).loc[W.index])  # lagged: known at the end of month t
    # standardised signals for one factor: past-only scaling, clipped, aligned with the factor's dates
    r = load_all()["Mkt-RF"]
    X, y = prepare_factor_dataset(r, W)
    assert X.shape[1] == 15 and np.abs(X.values).max() <= 5.0 and X.index.max() <= pd.Timestamp("2020-12-31")
    # changing signal values after t must not change the standardised value at t
    W2 = W.copy()
    W2.iloc[-24:] *= 10.0
    X2, _ = prepare_factor_dataset(r, W2)
    t = X.index[-60]
    np.testing.assert_allclose(X.loc[t].values, X2.loc[t].values)
