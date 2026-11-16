"""Print every number quoted in the README from the saved figures/*.npz files (no recomputation).

Usage:  python -m experiments.report_numbers [--tag ""|_eq36] > figures/report_numbers.log
tag "" = main results (eq. 37 literal); "_eq36" = the eq. 36 sensitivity set.
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.common import ROOT

FIG = ROOT / "figures"
ZS = [0, 1, 2, 3, 4, 5, 6]


def load(name):
    p = FIG / f"{name}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def cidx(c, target):
    return int(np.argmin(np.abs(np.log(c) - np.log(target))))


def curves_block(label, d):
    c, z = d["c"], d["z"]
    sr, long_ = d["SR"], d["long"]
    m = sr.mean(0)
    print(f"\n## {label}: T={int(d['T'])}, seeds={int(d['seeds'])}, factors={sr.shape[0]}")
    print("c grid:", np.round(c, 2))
    for j in ZS:
        print(f"  Sharpe z=1e{int(round(np.log10(z[j]))):+d}:", np.round(m[:, j], 3))
    lo, hi = 0, len(c) - 1
    for j in (0, 3, 6):
        diff = sr[:, hi, j] - sr[:, lo, j]
        print(f"  z=1e{int(round(np.log10(z[j]))):+d}: c_max - c_min = {diff.mean():+.3f} (s.e. {diff.std(ddof=1) / np.sqrt(len(diff)):.3f}); "
              f"{(diff > 0).mean():.0%} of factors positive; ratio {m[hi, j] / m[lo, j]:.2f}x")
    j0 = int(np.argmin(m[:, 0]))
    print(f"  z=1e-3 minimum {m[j0, 0]:.3f} at c={c[j0]:.2f}; c=1: {m[cidx(c, 1), 0]:.3f}; c_max: {m[hi, 0]:.3f}")
    gap = sr[:, hi, 6] - d["long"]
    print(f"  always long {d['long'].mean():.3f}; trailing {d['trail'].mean():.3f}; linear ridge {np.round(d['linear'].mean(0), 3)}")
    print(f"  model(z=1e3,c_max) - always long = {gap.mean():+.3f} (s.e. {gap.std(ddof=1) / np.sqrt(len(gap)):.3f}); "
          f"- trailing = {(sr[:, hi, 6] - d['trail']).mean():+.3f} (s.e. {(sr[:, hi, 6] - d['trail']).std(ddof=1) / np.sqrt(len(gap)):.3f})")
    names = [str(n) for n in d["names"]]
    fr = [i for i, n in enumerate(names) if n in ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")]
    jk = [i for i in range(len(names)) if i not in fr]
    if len(fr) == 6:
        print(f"  6 French: model {sr[fr, hi, 6].mean():.3f}, always long {d['long'][fr].mean():.3f}; "
              f"153 JKP: model {sr[jk, hi, 6].mean():.3f}, always long {d['long'][jk].mean():.3f}")
    # fig 4 statistic (z=1e3, min-max normalised)
    s = sr[:, :, 6]
    nrm = (s - s.min(1, keepdims=True)) / np.where(s.max(1, keepdims=True) - s.min(1, keepdims=True) > 0, s.max(1, keepdims=True) - s.min(1, keepdims=True), 1)
    print(f"  Fig.4 normalised Sharpe mean: c=1 {nrm[:, cidx(c, 1)].mean():.2f}, c_max {nrm[:, hi].mean():.2f}; "
          f"share >0.9 at c_max {(nrm[:, hi] > 0.9).mean():.0%}, <0.5 {(nrm[:, hi] < 0.5).mean():.0%}; "
          f"end above c_min value {(s[:, hi] > s[:, lo]).mean():.0%}")
    if int(d["T"]) == 12:
        print("  Fig.1/2 quantities at c_max, mean of factors: z, R2, ||b||, ER, Vol")
        for j in ZS:
            print(f"    z=1e{int(round(np.log10(z[j]))):+d}: R2 {d['R2'][:, hi, j].mean():+.3f}  ||b|| {np.sqrt(d['B2'][:, hi, j]).mean():.3f}  "
                  f"ER {d['ER'][:, hi, j].mean():.4f}  Vol {d['Vol'][:, hi, j].mean():.3f}")
        print("  at z=1e-3: R2 by c", np.round(d["R2"][:, :, 0].mean(0), 2))
        print("  at z=1e-3: ||b|| by c", np.round(np.sqrt(d["B2"][:, :, 0]).mean(0), 2))


def placebo_block(name):
    p = load(name)
    if p is None:
        return
    c, z = p["c"], p["z"]
    print(f"\n## {name}: shuffles={p['placebo'].shape[0]}")
    for j in (0, 3, 6):
        pm, ps = p["placebo"].mean(0)[:, j], p["placebo"].std(0, ddof=1)[:, j]
        sel = [0, cidx(c, 1), cidx(c, 16), len(c) - 1]
        print(f"  z=1e{int(round(np.log10(z[j]))):+d}  c      :", np.round(c[sel], 2))
        print("      real        :", np.round(p["real"][sel, j], 3))
        print("      placebo     :", np.round(pm[sel], 3), "sd", np.round(ps[sel], 3))
        print("      real-placebo:", np.round(p["real"][sel, j] - pm[sel], 3), " (sd units", np.round((p["real"][sel, j] - pm[sel]) / ps[sel], 1), ")")
        print(f"      rise c_min->c_max: real {p['real'][-1, j] - p['real'][0, j]:+.3f}, placebo {pm[-1] - pm[0]:+.3f}")


def sim_block(name):
    d = load(name)
    if d is None:
        return
    c, z = d["c"], d["z"]
    sr, r2 = d["SR"], d["R2"]
    print(f"\n## {name}: {sr.shape[0]} datasets; c = {np.round(c, 2)}")
    i1, i23 = cidx(c, 1), cidx(c, 2 / 3)
    for j in range(len(z)):
        dip = sr[:, i1, j] - sr[:, i23, j]
        t = dip.mean() / (dip.std(ddof=1) / np.sqrt(len(dip)))
        print(f"  z=1e{int(round(np.log10(z[j]))):+d}: Sharpe c=2/3 {sr[:, i23, j].mean():.4f}, c=1 {sr[:, i1, j].mean():.4f}, c=2 {sr[:, cidx(c, 2), j].mean():.4f}, "
              f"c_max {sr[:, -1, j].mean():.4f}; dip c=1 vs 2/3 {dip.mean():+.4f} (t={t:+.1f}); R2(c=1) {r2[:, i1, j].mean():+.2f}")
    print("  ||b|| at z=1e-3 by c:", np.round(np.sqrt(d["beta2"][:, :, 0]).mean(0), 2))


def scaling_block(tag):
    d = load(f"scaling_check{tag}")
    if d is None:
        return
    c, z, ks = d["c"], d["z"], d["kappas"]
    print(f"\n## scaling_check{tag}: kappas {ks}, seeds {int(d['seeds'])}")
    for k in range(3):
        print(f" kappa={ks[k]}")
        for j in ZS:
            print(f"   z=1e{int(round(np.log10(z[j]))):+d}: SR c_max {d[f'SR_{k}'][:, -1, j].mean():.3f}   R2 {d[f'R2_{k}'][:, -1, j].mean():+.2f}")
        b = np.sqrt(d[f"B2_{k}"][:, :, 0]).mean(0)
        print(f"   ||b|| at z=1e-3: peak {b.max():.2f} at c={c[int(np.argmax(b))]:.2f}, at c_max {b[-1]:.2f}")


def exact_block(tag):
    e = load(f"exact_kernel{tag}")
    if e is None:
        return
    print(f"\n## exact_kernel{tag}: real {np.round(e['real'], 3)}; placebo {np.round(e['placebo'].mean(0), 3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help='"" (eq. 37, main) or "_eq36"')
    a = ap.parse_args()
    t = a.tag
    for T in (12, 60, 120):
        d = load(f"curves_T{T}{t}")
        if d is not None:
            curves_block(f"curves T={T}{t}", d)
    placebo_block(f"placebo{t}")
    placebo_block(f"placebo_T60{t}")
    placebo_block(f"placebo_T120{t}")
    exact_block(t)
    scaling_block(t)
    for n in ("sim_theory", "sim_theory_mu0.08", "sim_theory_z-12_-8_-4_-3"):
        sim_block(n + t)
    if t == "":
        for T in (12, 60, 120):
            d = load(f"curves_T{T}_wg")
            if d is not None:
                curves_block(f"Welch-Goyal curves T={T}", d)


if __name__ == "__main__":
    main()
