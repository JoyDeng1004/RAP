#!/usr/bin/env python3
"""Post-run check of a run_pdm_score*.py result CSV against the metric cache it scored.

A PASS means the scorer consumed every cached token and produced v2 (EPDMS) metrics:

  rows        token rows (rows whose token is in the cache index) == cache size;
              no duplicates. Other rows (e.g. "average") are listed as aggregates.
  valid       every token row has valid == True. Invalid rows are tokens where the
              agent or scorer raised; they are excluded from the average silently.
  columns     prints every column; with --expect-v2, requires a traffic-light and
              a lane-keeping column (EPDMS terms that v1 PDMS does not have).
  summary     mean of every numeric column over valid token rows.

Only the csv and the cache's metadata csv are read; nothing is imported from navsim.

  python scripts/data_audit/verify_pdm_score_csv.py --result-dir <output_dir> \\
      --cache $NAVSIM_EXP_ROOT/metric_cache_v2_navtest_q --expect-v2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

V2_MARKERS = ("traffic_light", "lane_keeping")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=Path, required=True, help="output_dir of the scoring run")
    parser.add_argument("--csv", type=Path, help="result csv; default: newest *.csv in --result-dir")
    parser.add_argument("--cache", type=Path, required=True, help="metric cache that was scored")
    parser.add_argument("--expect-v2", action="store_true", help="require EPDMS-only columns")
    args = parser.parse_args()

    failures: list[str] = []
    result_csv = args.csv or max(args.result_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, default=None)
    if result_csv is None:
        print(f"FAIL\n  no *.csv in {args.result_dir}")
        return 1
    print(f"result: {result_csv}\ncache: {args.cache}\n")

    metadata_csv = next(p for p in sorted((args.cache / "metadata").iterdir()) if ".csv" in p.name)
    with open(metadata_csv) as f:
        cache_tokens = {r[0].split("/")[-2] for r in list(csv.reader(f))[1:]}

    df = pd.read_csv(result_csv, index_col=0)
    if "token" not in df.columns:
        print(f"FAIL\n  no token column; columns: {list(df.columns)}")
        return 1
    df["token"] = df["token"].astype(str)
    is_token = df["token"].isin(cache_tokens)
    tokens = df[is_token]
    aggregates = df.loc[~is_token, "token"].tolist()

    # rows
    dup = int(tokens["token"].duplicated().sum())
    missing = cache_tokens - set(tokens["token"])
    print(f"[rows] total: {len(df)} | token rows: {len(tokens)} | cache tokens: {len(cache_tokens)} | "
          f"not scored: {len(missing)} | duplicate: {dup}")
    print(f"[rows] aggregate rows: {aggregates[:10]}{' ...' if len(aggregates) > 10 else ''}")
    if missing:
        failures.append(f"rows: {len(missing)} cached tokens have no result row, e.g. {sorted(missing)[:3]}")
    if dup:
        failures.append(f"rows: {dup} duplicate token rows")

    # valid
    if "valid" in tokens.columns:
        valid = tokens["valid"].astype(str).str.lower().eq("true")
        print(f"[valid] valid: {int(valid.sum())} | invalid: {int((~valid).sum())}")
        if (~valid).any():
            failures.append(f"valid: {int((~valid).sum())} invalid rows, e.g. {tokens.loc[~valid, 'token'].head(3).tolist()}")
        tokens = tokens[valid]
    else:
        print("[valid] no valid column")

    # columns
    columns = [c for c in df.columns if c not in ("token", "valid")]
    print(f"[columns] ({len(columns)}): {columns}")
    if args.expect_v2:
        absent = [m for m in V2_MARKERS if not any(m in c for c in columns)]
        if absent:
            failures.append(f"columns: no column matching {absent}; this looks like a v1 PDMS result")

    # summary
    numeric = tokens[columns].apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(skipna=True).dropna()
    print("[summary] mean over valid token rows (sanity only; the reportable score is the devkit's aggregate row):")
    for name, value in means.items():
        print(f"  {name:45s} {value:.4f}")
    all_nan = [c for c in columns if numeric[c].isna().all()]
    if all_nan:
        print(f"  note: non-numeric or all-NaN columns: {all_nan}")

    # aggregate rows as written by the devkit (these carry the official, possibly weighted, score)
    for _, row in df[~is_token].iterrows():
        values = pd.to_numeric(row[columns], errors="coerce").dropna()
        print(f"[aggregate] {row['token']}: " + ", ".join(f"{k}={v:.4f}" for k, v in values.items()))

    print("\nPASS" if not failures else "\nFAIL\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
