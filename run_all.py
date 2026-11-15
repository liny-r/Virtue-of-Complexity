"""Single entry point: regenerates every figure and number reported in the README.

    python run_all.py            # full run (~45 min on a laptop, mostly the T=120 robustness run)
    python run_all.py --quick    # reduced run that exercises the whole pipeline

Headline result: figures/real_factors_T12.png (Sharpe vs complexity on real factors, with
baselines) and figures/placebo.png (same, with the time-shuffled-signal placebo).
Quick mode writes *_fast outputs and is NOT the reported result.
"""
from __future__ import annotations

import argparse
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
    q = ap.parse_args().quick

    sh("-m", "pytest", "-q")
    if not (ROOT / "data" / "ff_factors_monthly.csv").exists():
        sh("data/fetch_data.py")

    # 1. simulator: do the theory's predictions hold when its assumptions hold by construction?
    sh("-m", "experiments.sim_theory", *(["--fast"] if q else []))
    if not q:
        sh("-m", "experiments.sim_theory", "--mu", "0.08")
        sh("-m", "experiments.sim_theory", "--log10z=-12,-8,-4,-3")

    # 2. real factors (headline): VoC curves + baselines the paper omits
    sh("-m", "experiments.real_factors", *(["--fast"] if q else []))

    # 3. placebo: time-shuffled signals
    sh("-m", "experiments.placebo", *(["--fast"] if q else []))

    # 4. mechanism check: exact kernel ridge (c -> infinity)
    sh("-m", "experiments.exact_kernel", *(["--shuffles", "3", "--ref", "placebo_fast"] if q else []))

    # 5. robustness to the training window (paper Fig. 5 analogue)
    if not q:
        for T in (60, 120):
            sh("-m", "experiments.real_factors", "--T", str(T), "--seeds", "5")


if __name__ == "__main__":
    main()
