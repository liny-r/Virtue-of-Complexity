"""Slide figure: mean Sharpe at T = 12 (momentum signals) under two feature normalisations,
drawn from saved results (no re-run): the main 1/sqrt(L) and kappa = 1 (raw features).

    python -m experiments.plot_scaling_slide  ->  figures/scaling_slide.png
Line colours follow the paper's figures (experiments.curves.PAPER_COLORS).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import ROOT
from experiments.curves import PAPER_COLORS
from experiments.plot_paper_axes import broken_panel

FIG = ROOT / "figures"


def main():
    main_d = np.load(FIG / "curves_T12.npz", allow_pickle=True)
    sc = np.load(FIG / "scaling_check.npz", allow_pickle=True)
    panels = [("main: features scaled by $1/\\sqrt{L}$", main_d["c"], main_d["SR"].mean(0)),
              ("$\\kappa=1$: raw features (no scaling)", sc["c"], sc["SR_0"].mean(0))]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(11, 3.6)); gs = GridSpec(1, 2, figure=fig, wspace=0.25, bottom=0.2, top=0.86, left=0.07, right=0.98)
    for n, (title, c, sr) in enumerate(panels):
        broken_panel(fig, gs[n], c, sr, title, ylabel="mean OOS Sharpe, 159 factors" if n == 0 else None)
    fig.savefig(FIG / "scaling_slide.png", dpi=150)


if __name__ == "__main__":
    main()
