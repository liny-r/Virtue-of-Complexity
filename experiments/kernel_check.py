"""Kernel checks (beyond the paper). Prints three things; the full-run output is figures/kernel_check.log.

1. Random features vs the exact mixture kernel, for both conventions (eq. 36: frequencies w/sqrt(gamma);
   eq. 37 literal: frequencies gamma*w). The error should be of order 1/sqrt(number of weights).
2. How the kernel looks on our standardised signals: mean off-diagonal K per bandwidth. Wide bandwidths
   are nearly constant under eq. 36; under the literal eq. 37 it is the LARGE gammas that are nearly
   delta functions and the small gammas that are nearly constant (gamma -> 0 = linear model, as the paper says).
3. Why the placebo rises like the real curve: at infinite ridge the exact-kernel position is
   sum_s w_s y_s with w_s = K(x_t, x_s) / sum K. We measure how uniform w is (effective number of past
   returns used out of T) and the correlation of the position with the trailing mean, for real and
   time-shuffled signals (20 factors), under both kernel conventions.

Usage:  python -m experiments.kernel_check
"""
from __future__ import annotations

import numpy as np

from experiments.curves import load_all
from voc import make_rff
from voc.features import GAMMAS
from voc.kernel import mixture_kernel
from voc.preprocess import prepare_factor_dataset


def main():
    S = load_all()
    pool = ["Mom", "Mkt-RF", "SMB", "HML", "age", "at_be", "be_me", "cop_at"]
    Xs = np.vstack([prepare_factor_dataset(S[f].dropna())[0].values for f in pool])
    X = Xs[np.random.default_rng(0).choice(len(Xs), 400, replace=False)]
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    iu = np.triu_indices(len(X), 1)
    dist = np.sqrt(d2[iu])
    print(f"signals: std per column {Xs.std(0).round(2)}; pairwise distance quantiles 10/50/90% {np.quantile(dist, [.1, .5, .9]).round(2)}")

    def K(g, conv):
        return np.exp(-d2 / (2 * g)) if conv == "kernel" else np.exp(-g**2 * d2 / 2)

    print("\n[2] mean off-diagonal kernel value per bandwidth")
    print("    gamma        " + " ".join(f"{g:>6}" for g in GAMMAS))
    for conv, lab in (("kernel", "eq. 36 (w/sqrt g)"), ("paper", "eq. 37 (g*w)    ")):
        print(f"    {lab}  " + " ".join(f"{K(g, conv)[iu].mean():6.3f}" for g in GAMMAS))

    print("\n[1] random features (p = 4000 weights per bandwidth block of 7) vs exact mixture kernel, max |error| on 120 points")
    for conv in ("kernel", "paper"):
        F = make_rff(X[:120], p=4000, seed=1, convention=conv)
        Kf = F @ F.T / (len(GAMMAS) * 4000)
        Ke = np.mean([K(g, conv)[:120, :120] for g in GAMMAS], axis=0)
        print(f"    convention {conv:6}: {np.abs(Kf - Ke).max():.4f}   (1/sqrt(4000) = {1 / np.sqrt(4000):.4f})")

    print("\n[3] infinite-ridge exact-kernel position vs trailing mean (T = 12, every 8th factor, first 20)")
    T = 12
    for conv, clab in (("paper", "eq. 37 literal (main)"), ("kernel", "eq. 36")):
        for lab, shuf in (("real signals", None), ("shuffled signals", 0)):
            neff, maxw, cor = [], [], []
            for f in list(S)[::8][:20]:
                Xf, yf = prepare_factor_dataset(S[f].dropna())
                Xv, yv = Xf.values, yf.values
                if shuf is not None:
                    Xv = Xv[np.random.default_rng(shuf).permutation(len(Xv))]
                pos, tm = [], []
                for t in range(T, len(yv)):
                    k = mixture_kernel(Xv[t : t + 1], Xv[t - T : t], convention=conv)[0]
                    w = k / k.sum()
                    neff.append(1 / (w**2).sum()); maxw.append(w.max()); pos.append(w @ yv[t - T : t]); tm.append(yv[t - T : t].mean())
                cor.append(np.corrcoef(pos, tm)[0, 1])
            print(f"    {clab:22} {lab:17}: effective number of past returns used {np.mean(neff):.1f} of {T}; largest weight {np.mean(maxw):.2f} (uniform = {1 / T:.2f}); corr(position, trailing mean) {np.mean(cor):.3f}")


if __name__ == "__main__":
    main()
