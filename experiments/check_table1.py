"""Consistency check of our equity-factor data against the paper's Table 1 (159 factors).

data/paper_table1.csv holds the paper's Table 1 (start month, annualised mean %, vol %, Sharpe),
transcribed from the PDF. We recompute the same statistics from our data, truncated at 2021-12:
    French library : Mkt-RF, SMB, HML, RMW, CMA, Mom   (data/ff_factors_monthly.csv)
    JKP US factors : 153 capped value-weighted series  (data/jkp_factors_usa_monthly.csv)
Annualised mean = 12 x mean; vol = sqrt(12) x std; SR = mean/std x sqrt(12).

The paper's row "intrinsic value" has no series of that name in the public JKP file; the file has
"ival_me" instead. They are different factors (paper: mean -1.43%, SR -0.15; ival_me: +3.99%, 0.40),
so that row is reported as unmatched and excluded from the statistics.

Usage:  python -m experiments.check_table1
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import ROOT


def main():
    paper = pd.read_csv(ROOT / "data" / "paper_table1.csv")
    ours = pd.concat(
        [
            pd.read_csv(ROOT / "data" / "ff_factors_monthly.csv", index_col=0, parse_dates=True)[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]],
            pd.read_csv(ROOT / "data" / "jkp_factors_usa_monthly.csv", index_col=0, parse_dates=True),
        ],
        axis=1,
        sort=True,
    )[:"2021-12"]
    rows = []
    for f in paper.factor:
        if f not in ours:
            continue
        s = ours[f].dropna()
        rows.append((f, s.index[0].strftime("%b-%y"), s.mean() * 1200, s.std() * 12**0.5 * 100, s.mean() / s.std() * 12**0.5))
    o = pd.DataFrame(rows, columns=["factor", "start_o", "mean_o", "vol_o", "sr_o"]).merge(paper, on="factor")
    unmatched = sorted(set(paper.factor) - set(o.factor))
    print(f"paper rows {len(paper)}; matched by name {len(o)}; unmatched in paper: {unmatched}")
    print(f"start month equal: {(o.start_o == o.start).sum()}/{len(o)}; differ:", o[o.start_o != o.start][["factor", "start_o", "start"]].values.tolist())
    for lab, a, b in [("mean %", "mean_o", "mean_pct"), ("vol %", "vol_o", "vol_pct"), ("Sharpe", "sr_o", "sharpe")]:
        d = o[a] - o[b]
        print(f"{lab:7}: corr {np.corrcoef(o[a], o[b])[0, 1]:.4f}; ours-paper mean {d.mean():+.3f}, sd {d.std():.3f}, max |diff| {d.abs().max():.3f}")
    print(f"mean Sharpe over matched factors: ours {o.sr_o.mean():.3f}, paper {o.sharpe.mean():.3f}")
    print("largest Sharpe gaps:")
    print(o.reindex((o.sr_o - o.sharpe).abs().sort_values(ascending=False).index).head(5)[["factor", "mean_o", "mean_pct", "vol_o", "vol_pct", "sr_o", "sharpe"]].round(2).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (x, y, t) in zip(ax, [("mean_pct", "mean_o", "annualised mean, %"), ("vol_pct", "vol_o", "annualised vol, %"), ("sharpe", "sr_o", "annualised Sharpe")]):
        a.scatter(o[x], o[y], s=10)
        lim = [min(o[x].min(), o[y].min()), max(o[x].max(), o[y].max())]
        a.plot(lim, lim, "k--", lw=1)
        a.set_xlabel("paper Table 1"); a.set_ylabel("this repo's data"); a.set_title(t)
    fig.suptitle(f"Table 1 consistency, {len(o)} factors")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "table1_check.png", dpi=130)


if __name__ == "__main__":
    main()
