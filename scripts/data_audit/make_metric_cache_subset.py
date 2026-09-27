#!/usr/bin/env python3
"""Build a smoke-test view of a metric cache: a new metadata CSV listing a
log-stratified random sample of the source cache's pkls. Nothing is copied.

run_pdm_score*.py scores SceneLoader tokens ∩ MetricCacheLoader tokens, and
MetricCacheLoader only reads <cache>/metadata/*.csv, whose rows are absolute
pkl paths. Pointing metric_cache_path at the output dir therefore scores only
the sampled tokens while reading the original pkls. The source cache is only read.

  python scripts/data_audit/make_metric_cache_subset.py \\
      --cache $NAVSIM_EXP_ROOT/metric_cache_v2_navtest_q \\
      --out $NAVSIM_EXP_ROOT/metric_cache_v2_navtest_q_smoke --per-log 2
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="source metric cache")
    parser.add_argument("--out", type=Path, required=True, help="new dir; only metadata/ is written")
    parser.add_argument("--per-log", type=int, default=2, help="tokens sampled from each log")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    csvs = sorted(p for p in (args.cache / "metadata").iterdir() if ".csv" in p.name)
    if len(csvs) != 1:
        print(f"expected 1 csv in {args.cache / 'metadata'}, found {len(csvs)}", file=sys.stderr)
        return 1
    header, *rows = csvs[0].read_text().splitlines()

    by_log: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_log[Path(row).parts[-4]].append(row)  # <cache>/<log>/<scenario_type>/<token>/metric_cache.pkl

    rng = random.Random(args.seed)
    sample = [r for log in sorted(by_log) for r in rng.sample(by_log[log], min(args.per_log, len(by_log[log])))]

    metadata_dir = args.out / "metadata"
    if metadata_dir.exists() and any(metadata_dir.iterdir()):
        print(f"{metadata_dir} is not empty", file=sys.stderr)
        return 1
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metadata_dir / f"{args.out.name}_metadata_node_0.csv"
    out_csv.write_text("\n".join([header, *sample]) + "\n")

    print(f"source: {csvs[0]} ({len(rows)} tokens, {len(by_log)} logs)")
    print(f"sample: {len(sample)} tokens ({args.per_log}/log, seed {args.seed}) → {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
