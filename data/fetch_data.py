"""Download monthly Fama-French factors + Momentum from Kenneth French's data library.

Output: data/ff_factors_monthly.csv, columns Mkt-RF, SMB, HML, RMW, CMA, Mom, RF as
DECIMAL monthly returns, index = month-end timestamps.

Sources, matched to the start dates in the paper's Table 1:
    Mkt-RF, SMB, HML : 3-factor file (from 1926-07)
    RMW, CMA         : 5-factor 2x3 file (from 1963-07)
    Mom              : momentum file (from 1927-01)

Usage:  python data/fetch_data.py
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "three": "F-F_Research_Data_Factors_CSV.zip",
    "five": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}
OUT = Path(__file__).resolve().parent / "ff_factors_monthly.csv"


def _download(name: str) -> str:
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0 (course project)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    return z.read(z.namelist()[0]).decode("latin-1")


def _parse_monthly(text: str) -> pd.DataFrame:
    """The files hold a monthly block followed by an annual block; keep only the monthly rows."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if re.match(r"^\s*\d{6}\s*,", ln))
    header = [h.strip() for h in lines[start - 1].split(",")][1:]
    rows = []
    for ln in lines[start:]:
        if not re.match(r"^\s*\d{6}\s*,", ln):
            break
        parts = [p.strip() for p in ln.split(",")]
        rows.append([parts[0]] + [float(x) for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["date"] + header)
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.replace([-99.99, -999], float("nan"))
    return df / 100.0


def main() -> None:
    three = _parse_monthly(_download(FILES["three"]))[["Mkt-RF", "SMB", "HML", "RF"]]
    five = _parse_monthly(_download(FILES["five"]))[["RMW", "CMA"]]
    mom = _parse_monthly(_download(FILES["mom"]))
    mom.columns = ["Mom"]
    df = three.join(five, how="outer").join(mom, how="outer")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]]
    df.to_csv(OUT, float_format="%.6f")
    print(f"wrote {OUT} {df.shape} {df.index.min():%Y-%m} .. {df.index.max():%Y-%m}")


if __name__ == "__main__":
    main()
