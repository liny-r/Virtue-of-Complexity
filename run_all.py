"""Single entry point: regenerates every figure and number reported in the README.

    python run_all.py              # main results (literal eq. 37 features), about 4 h on an 8-core laptop
    python run_all.py --eq36       # also the eq. (36) sensitivity set (files end in _eq36), about 2.5 h more
    python run_all.py --quick      # reduced run that exercises the whole pipeline (< 3 min)
    python run_all.py --workers 4  # processes for the per-factor loops (default: all cores, max 8)

Headline result (paper Figs 1-4 analogues, 159 factors, T = 12): `python -m experiments.curves`
alone regenerates figures/fig1_T12.png ... fig4_T12.png and figures/curves_T12.npz.
Quick mode writes *_fast outputs and is NOT the reported result.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sh(*args: str) -> None:
    print("\n$", sys.executable, *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eq36", action="store_true", help="also run the eq. (36) kernel-convention sensitivity set")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args()
    q, w = a.quick, str(a.workers)
    fast = ["--fast"] if q else []

    sh("-m", "pytest", "-q")
    if not (ROOT / "data" / "ff_factors_monthly.csv").exists():
        sh("data/fetch_data.py")
    elif not (ROOT / "data" / "jkp_factors_usa_monthly.csv").exists():
        sh("data/fetch_data.py", "--jkp")

    # A. the paper's own results: Figs 1-4 on the 159 equity factors, T = 12 (features: literal eq. 37)
    sh("-m", "experiments.curves", "--workers", w, *(fast if q else ["--seeds", "100"]))  # 100 draws, as in the paper
    sh("-m", "experiments.check_table1")  # our data vs the paper's Table 1

    # A2. the paper's first study: the same figures with the 15 Welch-Goyal predictors (files end in _wg)
    sh("-m", "experiments.curves", "--signals", "wg", "--workers", w, *(fast if q else ["--seeds", "100"]))

    # B. paper Fig. 5: longer training windows (10 feature draws)
    if not q:
        for T in (60, 120):
            sh("-m", "experiments.curves", "--T", str(T), "--seeds", "10", "--workers", w)
            sh("-m", "experiments.curves", "--signals", "wg", "--T", str(T), "--seeds", "10", "--workers", w)

    # C. beyond the paper: kernel look, simulator, placebo, exact kernel, feature-normalisation check
    sh("-m", "experiments.kernel_check")
    sh("-m", "experiments.sim_theory", *fast)
    if not q:
        sh("-m", "experiments.sim_theory", "--mu", "0.08")
        sh("-m", "experiments.sim_theory", "--log10z=-12,-8,-4,-3")
    sh("-m", "experiments.placebo", "--workers", w, *(fast if q else ["--shuffles", "20"]))
    if not q:
        for T in (60, 120):
            sh("-m", "experiments.placebo", "--T", str(T), "--shuffles", "10", "--workers", w)
    sh("-m", "experiments.exact_kernel", "--workers", w, *(fast if q else ["--shuffles", "20"]))
    sh("-m", "experiments.scaling_check", "--workers", w, *(fast if q else ["--seeds", "10"]))

    # D. sensitivity: features of eq. (36) (frequencies w/sqrt(gamma)) instead of the literal eq. (37)
    if a.eq36:
        e = ["--convention", "kernel"]
        sh("-m", "experiments.curves", *e, "--workers", w, *(fast if q else ["--seeds", "20"]))
        sh("-m", "experiments.placebo", *e, "--workers", w, *(fast if q else ["--shuffles", "20"]))
        sh("-m", "experiments.exact_kernel", *e, "--workers", w, *(fast if q else ["--shuffles", "20"]))
        sh("-m", "experiments.scaling_check", *e, "--workers", w, *(fast if q else ["--seeds", "10"]))
        sh("-m", "experiments.sim_theory", *e, *fast)
        if not q:
            sh("-m", "experiments.sim_theory", *e, "--mu", "0.08")
            sh("-m", "experiments.sim_theory", *e, "--log10z=-12,-8,-4,-3")
            for T in (60, 120):
                sh("-m", "experiments.curves", *e, "--T", str(T), "--seeds", "10", "--workers", w)
                sh("-m", "experiments.placebo", *e, "--T", str(T), "--shuffles", "10", "--workers", w)

    # E. print every number quoted in the README from the saved results (no recomputation)
    if not q:
        for tag, f in (("", "report_numbers.log"), ("_eq36", "report_numbers_eq36.log")):
            if tag == "" or a.eq36:
                out = subprocess.run([sys.executable, "-m", "experiments.report_numbers", "--tag", tag], cwd=ROOT, check=True, capture_output=True, text=True).stdout
                (ROOT / "figures" / f).write_text(out)


if __name__ == "__main__":
    main()
