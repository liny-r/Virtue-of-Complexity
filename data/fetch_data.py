"""Download monthly Fama-French factors + Momentum from Kenneth French's data library.

Output: data/ff_factors_monthly.csv, columns Mkt-RF, SMB, HML, RMW, CMA, Mom, RF as
DECIMAL monthly returns, index = month-end timestamps.

Sources, matched to the start dates in the paper's Table 1:
    Mkt-RF, SMB, HML : 3-factor file (from 1926-07)
    RMW, CMA         : 5-factor 2x3 file (from 1963-07)
    Mom              : momentum file (from 1927-01)

Also (--jkp, or by default when absent): the 153 JKP characteristic-based US factors from
jkpfactors.com (Jensen-Kelly-Pedersen 2023), capped value-weighted, monthly. The paper uses
these (plus FF5 + Mom = 159 factors) for its equity-factor results.
Output: data/jkp_factors_usa_monthly.csv, one column per factor, DECIMAL monthly returns.

Also: the Welch-Goyal (2008) monthly predictor data, updated by Amit Goyal (sites.google.com/view/agoyal145),
used for the paper's first equity-factor study (15 predictors). Output: data/welch_goyal_monthly.csv with the
raw columns needed to build the 15 predictors (see voc/preprocess.py), 1926-01 .. 2021-12. Needs openpyxl.

Usage:  python data/fetch_data.py          # FF factors + JKP factors + Welch-Goyal
        python data/fetch_data.py --jkp    # JKP only
        python data/fetch_data.py --wg     # Welch-Goyal only
"""
from __future__ import annotations

import argparse
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
JKP_URL = "https://jkpfactors-data.s3.amazonaws.com/public/%5Busa%5D_%5Ball_factors%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip"
JKP_OUT = Path(__file__).resolve().parent / "jkp_factors_usa_monthly.csv"
WG_URL = "https://docs.google.com/spreadsheets/d/1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ/export?format=xlsx"
WG_OUT = Path(__file__).resolve().parent / "welch_goyal_monthly.csv"
WG_COLS = ["Index", "D12", "E12", "b/m", "tbl", "AAA", "BAA", "lty", "ntis", "infl", "ltr", "corpr", "svar", "CRSP_SPvw"]


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


def fetch_jkp() -> None:
    req = urllib.request.Request(JKP_URL, headers={"User-Agent": "Mozilla/5.0 (course project)"})
    with urllib.request.urlopen(req, timeout=300) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    long = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["name", "date", "ret"], parse_dates=["date"])
    wide = long.pivot(index="date", columns="name", values="ret").sort_index()
    wide.index.name = "date"
    wide.to_csv(JKP_OUT, float_format="%.6f")
    print(f"wrote {JKP_OUT} {wide.shape} {wide.index.min():%Y-%m} .. {wide.index.max():%Y-%m}")


def fetch_wg() -> None:
    req = urllib.request.Request(WG_URL, headers={"User-Agent": "Mozilla/5.0 (course project)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    m = pd.read_excel(io.BytesIO(raw), sheet_name="Monthly")
    m.index = pd.to_datetime(m.pop("yyyymm").astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    m.index.name = "date"
    m = m.loc["1926-01":"2021-12", WG_COLS]
    m.to_csv(WG_OUT, float_format="%.8g")
    print(f"wrote {WG_OUT} {m.shape} {m.index.min():%Y-%m} .. {m.index.max():%Y-%m}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jkp", action="store_true", help="fetch only the JKP factors")
    ap.add_argument("--wg", action="store_true", help="fetch only the Welch-Goyal predictors")
    args = ap.parse_args()
    if args.jkp:
        return fetch_jkp()
    if args.wg:
        return fetch_wg()
    three = _parse_monthly(_download(FILES["three"]))[["Mkt-RF", "SMB", "HML", "RF"]]
    five = _parse_monthly(_download(FILES["five"]))[["RMW", "CMA"]]
    mom = _parse_monthly(_download(FILES["mom"]))
    mom.columns = ["Mom"]
    df = three.join(five, how="outer").join(mom, how="outer")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]]
    df.to_csv(OUT, float_format="%.6f")
    print(f"wrote {OUT} {df.shape} {df.index.min():%Y-%m} .. {df.index.max():%Y-%m}")
    fetch_jkp()
    fetch_wg()


if __name__ == "__main__":
    main()
