"""Slide figures on the paper's axes: linear c up to 50, an axis break, and a short panel at the far right
for the largest model (c = 500 here, 1,000 in the paper). Drawn from saved results (no re-run).

    python -m experiments.plot_paper_axes  ->  figures/fig3_T12_paperaxes.png, figures/fig1_T12_paperaxes.png
Line colours follow the paper (experiments.curves.PAPER_COLORS). Panel A = Welch-Goyal, panel B = momentum.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from experiments.common import ROOT
from experiments.curves import PAPER_COLORS

FIG = ROOT / "figures"


def broken_panel(fig, spec, c, y, title, ylim=None, ylabel=None, c_end=500, fs=10, hlines=()):
    inner = spec.subgridspec(1, 2, width_ratios=[8, 1], wspace=0.08)
    a, b = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
    for j in range(y.shape[1]):
        a.plot(c, y[:, j], color=PAPER_COLORS[j], lw=1.6)
        b.hlines(y[-1, j], c_end - 10, c_end, color=PAPER_COLORS[j], lw=1.6)
    for v, col in hlines:
        a.axhline(v, color=col, ls="--", lw=1); b.axhline(v, color=col, ls="--", lw=1)
    a.axvline(1, color="gray", ls="--", lw=1)
    a.set_xlim(0, 50); b.set_xlim(c_end - 10, c_end); b.set_xticks([c_end])
    if ylim:
        a.set_ylim(*ylim)
    b.set_ylim(a.get_ylim()); b.set_yticks([])
    a.spines["right"].set_visible(False); b.spines["left"].set_visible(False)
    a.spines["top"].set_visible(False); b.spines["top"].set_visible(False)
    a.set_title(title, fontsize=fs + 1); a.set_xlabel("c = L/T", fontsize=fs)
    a.tick_params(labelsize=fs - 1); b.tick_params(labelsize=fs - 1)
    if ylabel:
        a.set_ylabel(ylabel, fontsize=fs)
    d = 0.02
    a.plot([1 - d, 1 + d], [-d, d], transform=a.transAxes, color="k", clip_on=False, lw=1)
    b.plot([-d * 6, d * 6], [-d, d], transform=b.transAxes, color="k", clip_on=False, lw=1)
    return a, b


def paper_layout(d_by_study, metrics, fname, col_titles, row_titles, figsize=(10, 6.2)):
    """Paper's Fig. 1 / Fig. 2 layout: one row per study (A, B), one column per metric.
    metrics: list of (key, transform, ylim)."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.95, bottom=0.08, top=0.8, left=0.07, right=0.985)
    for r, d in enumerate(d_by_study):
        for k, (key, tf, ylim) in enumerate(metrics):
            a, _ = broken_panel(fig, gs[r, k], d["c"], tf(d[key].mean(0)), "", ylim=ylim, fs=9)
            if k == 0:
                fig.text(0.5, gs[r, 0].get_position(fig).y1 + 0.04, row_titles[r], fontsize=11, ha="center")
    for k, t in enumerate(col_titles):
        fig.text(0.07 + 0.47 * k + 0.2, 0.93, t, fontsize=13, ha="center")
    fig.savefig(FIG / fname, dpi=150); plt.close(fig)


def placebo_panel(fig, spec, c, real, pm, ps, title, hlines=(), ylabel=None, c_end=500, fs=10, show="both", ylim=None):
    """Real signals (blue) vs the time-shuffled placebo (orange, mean +/- 2 sd), on the paper's axes."""
    inner = spec.subgridspec(1, 2, width_ratios=[8, 1], wspace=0.08)
    a, b = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
    xe = np.array([c_end - 10, c_end])
    if show in ("both", "placebo"):
        a.fill_between(c, pm - 2 * ps, pm + 2 * ps, color="tab:orange", alpha=0.3)
        a.plot(c, pm, color="tab:orange", lw=1.6, label="placebo mean ±2 sd")
        b.fill_between(xe, [pm[-1] - 2 * ps[-1]] * 2, [pm[-1] + 2 * ps[-1]] * 2, color="tab:orange", alpha=0.3)
        b.hlines(pm[-1], *xe, color="tab:orange", lw=1.6)
    if show in ("both", "real"):
        a.plot(c, real, color="tab:blue", lw=1.6, label="real signals")
        b.hlines(real[-1], *xe, color="tab:blue", lw=1.6)
    for v, col, lab in hlines:
        a.axhline(v, color=col, ls="--", lw=1, label=lab); b.axhline(v, color=col, ls="--", lw=1)
    a.axvline(1, color="gray", ls=":", lw=1)
    a.set_xlim(0, 50); b.set_xlim(*xe); b.set_xticks([c_end])
    if ylim:
        a.set_ylim(*ylim)
    b.set_ylim(a.get_ylim()); b.set_yticks([])
    for ax_, side in ((a, "right"), (b, "left")):
        ax_.spines[side].set_visible(False); ax_.spines["top"].set_visible(False); ax_.tick_params(labelsize=fs - 1)
    a.set_title(title, fontsize=fs + 1); a.set_xlabel("c = L/T", fontsize=fs)
    if ylabel:
        a.set_ylabel(ylabel, fontsize=fs)
    d = 0.02
    a.plot([1 - d, 1 + d], [-d, d], transform=a.transAxes, color="k", clip_on=False, lw=1)
    b.plot([-d * 6, d * 6], [-d, d], transform=b.transAxes, color="k", clip_on=False, lw=1)
    return a


def main():
    dA, dB = np.load(FIG / "curves_T12_wg.npz"), np.load(FIG / "curves_T12.npz")
    studies = [("Panel A: Equities with WG factors", dA), ("Panel B: Equities with Momentums", dB)]
    # Fig. 3 (Sharpe): same layout as the paper's panels A and B
    fig = plt.figure(figsize=(12, 3.5)); gs = GridSpec(1, 2, figure=fig, wspace=0.25, bottom=0.2, top=0.85, left=0.06, right=0.98)
    for k, (t, d) in enumerate(studies):
        broken_panel(fig, gs[k], d["c"], d["SR"].mean(0), t, ylabel="OOS Sharpe" if k == 0 else None)
    fig.savefig(FIG / "fig3_T12_paperaxes.png", dpi=150); plt.close(fig)
    # Fig. 1 (R^2 on top, coefficient norm below)
    fig = plt.figure(figsize=(12, 6.4)); gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.5, bottom=0.09, top=0.93, left=0.07, right=0.98)
    for k, (t, d) in enumerate(studies):
        broken_panel(fig, gs[0, k], d["c"], d["R2"].mean(0), t, ylim=(-1, 0.1), ylabel="OOS $R^2$" if k == 0 else None)
        broken_panel(fig, gs[1, k], d["c"], np.sqrt(d["B2"].mean(0)), t, ylim=(0, 10), ylabel=r"$\|\hat\beta\|$" if k == 0 else None)
    fig.savefig(FIG / "fig1_T12_paperaxes.png", dpi=150); plt.close(fig)
    # Paper's Fig. 1 and Fig. 2 layout (rows: study A, B; columns: the two metrics)
    rows = ["Panel A: Equity with WG factors", "Panel B: Equity with Factor Momentums"]
    paper_layout([dA, dB], [("R2", lambda y: y, (-0.5, 0.1)), ("B2", np.sqrt, (0, 10))], "fig1_T12_paperlayout.png",
                 ["$R^2$", r"$\|\hat\beta\|$"], rows)
    paper_layout([dA, dB], [("ER", lambda y: y, (-0.002, 0.047)), ("Vol", lambda y: y, (0, 2.5))], "fig2_T12_paperlayout.png",
                 ["Expected return (per month)", "Volatility (per month)"], rows)
    # Placebo (momentum signals, T = 12): all seven ridge levels in one graph per case
    p = np.load(FIG / "placebo.npz")
    c, real, plc = p["c"], p["real"], p["placebo"]
    pm = plc.mean(0)
    hl = [(float(p["long"]), "r"), (float(p["trail"]), "k")]
    ylim = (0.02, 0.37)
    fig = plt.figure(figsize=(12, 3.9)); gs = GridSpec(1, 2, figure=fig, wspace=0.25, bottom=0.2, top=0.84, left=0.065, right=0.99)
    broken_panel(fig, gs[0], c, real, "Real signals", ylim=ylim, ylabel="OOS Sharpe (mean of 159)", hlines=hl, fs=11)
    broken_panel(fig, gs[1], c, pm, "Time-shuffled placebo (mean of 20 shuffles)", ylim=ylim, hlines=hl, fs=11)
    h = [plt.Line2D([], [], color=PAPER_COLORS[j], lw=1.8) for j in range(7)] + [plt.Line2D([], [], color="r", ls="--"), plt.Line2D([], [], color="k", ls="--")]
    l = [f"log10(z) = {j - 3}" for j in range(7)] + ["always long", "trailing mean"]
    fig.legend(h, l, loc="upper center", ncol=9, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / "placebo_paperaxes.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
