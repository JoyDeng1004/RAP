"""Token-by-token comparison of two run_pdm_score_v1 CSVs (shared tokens only).

T1 step 6 uses it to check that RIG_PRESET=raw reproduces the existing official
CSV exactly; later it gives the paired difference of a preset against raw.

    python tools/compare_csv_tokens.py <csv or dir A> <csv or dir B>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(arg: str) -> pd.DataFrame:
    p = Path(arg)
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if len(csvs) != 1:
            raise SystemExit(f"{p}: expected one csv, found {[c.name for c in csvs]}")
        p = csvs[0]
    df = pd.read_csv(p, index_col=0)
    df = df[df["token"] != "average"].set_index("token")
    print(f"{p}: {len(df)} tokens, valid={int(df['valid'].astype(bool).sum())}")
    return df


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = a.index.intersection(b.index)
    cols = [c for c in a.columns if c in b.columns and c != "valid" and pd.api.types.is_numeric_dtype(a[c])]
    print(f"shared tokens: {len(common)}")
    identical = True
    for c in cols:
        x, y = a.loc[common, c].to_numpy(float), b.loc[common, c].to_numpy(float)
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        d = np.abs(x - y)
        identical &= bool(same.all())
        print(f"  {c:32s} equal {int(same.sum()):6d}/{len(common)}  max|diff| {np.nanmax(d) if len(d) else 0:.3g}  "
              f"mean A {np.nanmean(x):.4f}  mean B {np.nanmean(y):.4f}  B-A {np.nanmean(y) - np.nanmean(x):+.4f}")
    print("IDENTICAL" if identical else "DIFFERENT")


if __name__ == "__main__":
    main()
