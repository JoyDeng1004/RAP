#!/usr/bin/env python3
"""Post-run audit of an evaluation metric cache (run_metric_caching.py).

Checks what run_pdm_score.py will actually see, so a PASS here means the
evaluation will score every token of the split:

  devkit      navsim and MetricCache are imported from --devkit (RAP = v1, the
              upstream checkout = v2). v1 and v2 share the module path of
              MetricCache, so a v2 pkl unpickled under RAP's v1 navsim passes
              isinstance silently; the resolved files are printed and asserted.
  metadata    exactly one CSV in <cache>/metadata/. MetricCacheLoader reads only
              the first CSV that iterdir() returns, so a second one silently
              drops tokens.
  index       CSV rows are unique and every listed file exists and is non-empty;
              no metric_cache.pkl on disk is missing from the CSV (an h_rt kill
              leaves pkls without an index).
  coverage    tokens from SceneLoader(split), built as run_pdm_score.py builds
              it, minus cached tokens must be empty. Those are the tokens
              run_pdm_score.py would skip with only a warning.
  loadable    a sample (or --all) of pkls decompresses into the devkit's
              MetricCache; core fields must not be None, every other dataclass
              field gets a None count.
  compare     optional: token set against another cache (e.g. v1 vs v2 navtest).

Exit code 0 on PASS, 1 on FAIL.

  # v1
  python scripts/data_audit/verify_eval_metric_cache.py --cache metric_cache_v1_navtest
  # v2 (run from a dir without a navsim/ package)
  python scripts/data_audit/verify_eval_metric_cache.py --devkit $NAVSIM_DEVKIT_ROOT \\
      --cache $NAVSIM_EXP_ROOT/metric_cache_v2_navtest_q --all --workers 12
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import inspect
import lzma
import os
import pickle
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
FILTER_SUBDIR = "navsim/planning/script/config/common/train_test_split"
# Fields present in both v1 and v2 MetricCache that the scorer cannot do without.
CORE_FIELDS = ("trajectory", "ego_state", "observation", "centerline", "route_lane_ids", "drivable_area_map")

MetricCache = None  # bound in main() after the devkit is on sys.path; inherited by forked workers


def load_split(filter_dir: Path, split: str, scene_filter_cls):
    split_cfg = yaml.safe_load(open(filter_dir / f"{split}.yaml"))
    filter_name = split_cfg["defaults"][0]["scene_filter"]
    raw = yaml.safe_load(open(filter_dir / "scene_filter" / f"{filter_name}.yaml"))
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    return split_cfg["data_split"], scene_filter_cls(**raw)


def build_scene_loader(scene_loader_cls, sensor_config, data_root: Path, data_split: str, scene_filter):
    """Pass only the kwargs this devkit's SceneLoader accepts (v2 added the synthetic paths)."""
    synthetic = data_root / "navhard_two_stage"
    candidates = {
        "data_path": data_root / "navsim_logs" / data_split,
        "sensor_blobs_path": None,
        "original_sensor_path": None,
        "synthetic_sensor_path": synthetic / "sensor_blobs",
        "synthetic_scenes_path": synthetic / "synthetic_scene_pickles",
        "scene_filter": scene_filter,
        "sensor_config": sensor_config,
    }
    params = inspect.signature(scene_loader_cls.__init__).parameters
    return scene_loader_cls(**{k: v for k, v in candidates.items() if k in params})


def check_pkl(path: str):
    """Returns (path, error or None, tuple of None-valued field names)."""
    try:
        with lzma.open(path, "rb") as f:
            mc = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        return path, f"{type(e).__name__}: {e}", ()
    if not isinstance(mc, MetricCache):
        return path, f"type {type(mc).__module__}.{type(mc).__name__}", ()
    nones = tuple(f.name for f in dataclasses.fields(mc) if getattr(mc, f.name, None) is None)
    return path, None, nones


def read_index(cache: Path) -> list[str]:
    csvs = sorted(p for p in (cache / "metadata").iterdir() if ".csv" in p.name)
    with open(csvs[0]) as f:
        return [r[0] for r in csv.reader(f)][1:]


def main() -> int:
    global MetricCache
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devkit", type=Path, default=REPO, help="navsim checkout to import (RAP = v1)")
    parser.add_argument("--cache", type=Path, default=REPO / "metric_cache_v1_navtest")
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("OPENSCENE_DATA_ROOT", "")))
    parser.add_argument("--sample", type=int, default=300, help="pkls to load-check")
    parser.add_argument("--all", action="store_true", help="load-check every pkl")
    parser.add_argument("--workers", type=int, default=1, help="processes for the load check")
    parser.add_argument("--compare-cache", type=Path, help="another cache whose token set should match")
    args = parser.parse_args()

    devkit = args.devkit.resolve()
    sys.path.insert(0, str(devkit))
    import navsim  # noqa: E402
    from navsim.common.dataclasses import SceneFilter, SensorConfig  # noqa: E402
    from navsim.common.dataloader import SceneLoader  # noqa: E402
    from navsim.planning.metric_caching.metric_cache import MetricCache as _MetricCache  # noqa: E402

    MetricCache = _MetricCache
    failures: list[str] = []
    print(f"cache: {args.cache}\nsplit: {args.split}\ndata root: {args.data_root}\n")

    # devkit
    mc_file = Path(inspect.getfile(MetricCache)).resolve()
    field_names = [f.name for f in dataclasses.fields(MetricCache)]
    print(f"[devkit] navsim: {Path(navsim.__file__).resolve()}")
    print(f"[devkit] MetricCache: {mc_file}")
    print(f"[devkit] MetricCache fields ({len(field_names)}): {field_names}")
    if not mc_file.is_relative_to(devkit):
        print(f"\nFAIL\n  devkit: MetricCache resolved outside {devkit}; run from a dir without a navsim/ package")
        return 1

    # metadata
    if not args.cache.is_dir():
        print(f"FAIL\n  cache dir does not exist: {args.cache.resolve()}")
        return 1
    metadata_dir = args.cache / "metadata"
    if not metadata_dir.is_dir():
        n_pkl = sum(1 for _ in args.cache.rglob("metric_cache.pkl"))
        print(f"FAIL\n  no metadata/ dir; {n_pkl} metric_cache.pkl on disk (job likely killed before writing the csv)")
        return 1
    csvs = sorted(p for p in metadata_dir.iterdir() if ".csv" in p.name)
    print(f"[metadata] csv files: {[p.name for p in csvs]}")
    if len(csvs) != 1:
        failures.append(f"metadata: expected 1 csv, found {len(csvs)}")
    if not csvs:
        print("\nFAIL\n  " + "\n  ".join(failures))
        return 1

    # index
    rows = read_index(args.cache)
    csv_tokens = [r.split("/")[-2] for r in rows]
    missing_files = [r for r in rows if not os.path.isfile(r) or os.path.getsize(r) == 0]
    on_disk = {str(p) for p in args.cache.rglob("metric_cache.pkl")}
    unindexed = on_disk - set(rows)
    dup = len(csv_tokens) - len(set(csv_tokens))
    print(f"[index] csv rows: {len(rows)} | unique tokens: {len(set(csv_tokens))} | pkl on disk: {len(on_disk)}")
    print(f"[index] listed but missing/empty: {len(missing_files)} | on disk but not in csv: {len(unindexed)}")
    if dup:
        failures.append(f"index: {dup} duplicate tokens in csv")
    if missing_files:
        failures.append(f"index: {len(missing_files)} csv rows point to missing/empty files, e.g. {missing_files[:2]}")
    if unindexed:
        failures.append(f"index: {len(unindexed)} pkls not in csv, e.g. {sorted(unindexed)[:2]}")

    # coverage
    data_split, scene_filter = load_split(devkit / FILTER_SUBDIR, args.split, SceneFilter)
    scene_loader = build_scene_loader(
        SceneLoader, SensorConfig.build_no_sensors(), args.data_root, data_split, scene_filter
    )
    expected = set(scene_loader.tokens)
    cached = set(csv_tokens)
    uncached = expected - cached
    unused = cached - expected
    yaml_tokens = set(scene_filter.tokens or [])
    print(
        f"[coverage] split yaml tokens: {len(yaml_tokens)} | SceneLoader tokens: {len(expected)} | "
        f"to evaluate: {len(expected & cached)} | missing cache: {len(uncached)} | unused cache: {len(unused)}"
    )
    if yaml_tokens and not yaml_tokens <= expected:
        failures.append(f"coverage: {len(yaml_tokens - expected)} split yaml tokens not yielded by SceneLoader")
    if uncached:
        failures.append(f"coverage: {len(uncached)} tokens would be skipped by run_pdm_score, e.g. {sorted(uncached)[:3]}")
    if unused:
        print(f"  note: {len(unused)} cached tokens outside the split (ignored by evaluation)")

    # loadable
    to_check = rows if args.all else random.Random(0).sample(rows, min(args.sample, len(rows)))
    bad: list[str] = []
    none_counts: Counter = Counter()
    with Pool(args.workers) as pool:
        for i, (path, err, nones) in enumerate(pool.imap_unordered(check_pkl, to_check, chunksize=16), 1):
            if err:
                bad.append(f"{path} ({err})")
            none_counts.update(nones)
            if i % 1000 == 0:
                print(f"  loaded {i}/{len(to_check)}")
    print(f"[loadable] checked: {len(to_check)} | unreadable/wrong type: {len(bad)}")
    print(f"[loadable] None count per field: {dict(none_counts) or 'none'}")
    if bad:
        failures.append(f"loadable: {len(bad)} bad pkls, e.g. {bad[:2]}")
    core_none = {k: none_counts[k] for k in CORE_FIELDS if none_counts[k]}
    if core_none:
        failures.append(f"loadable: core fields None {core_none}")
    absent = [k for k in CORE_FIELDS if k not in field_names]
    if absent:
        print(f"  note: core fields not in this MetricCache: {absent}")

    # compare
    if args.compare_cache:
        other = {r.split("/")[-2] for r in read_index(args.compare_cache)}
        print(
            f"[compare] {args.compare_cache}: {len(other)} tokens | shared: {len(cached & other)} | "
            f"only here: {len(cached - other)} | only there: {len(other - cached)}"
        )
        if cached != other:
            failures.append(f"compare: token set differs from {args.compare_cache}")

    total = sum(os.path.getsize(p) for p in on_disk)
    print(f"[size] {total / 1e9:.2f} GB over {len(on_disk)} pkls ({total / max(len(on_disk), 1) / 1e6:.3f} MB each)")

    print("\nPASS" if not failures else "\nFAIL\n  " + "\n  ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
