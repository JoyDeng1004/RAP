"""Dump per-token start points (+ log / frame type / time) from the v2 navhard metric cache.

SceneAggregator weights a stage-2 scene by exp(-d^2 / (2 * 0.1)), d = distance between the
stage-1 trajectory endpoint and the stage-2 scene's `start_point` (= metric_cache.ego_state.rear_axle).
run_pdm_score.py drops both from the CSV; this script recovers the start points (model-independent).

Read-only over the cache; writes one CSV into a NEW directory (refuses to reuse an existing one).
Must run with the v2 devkit (PYTHONPATH=$NAVSIM_DEVKIT_ROOT) from a cwd without a navsim/ dir.

  cd /tmp && PYTHONPATH=$NAVSIM_DEVKIT_ROOT $PY $RAP/scripts/analysis/navhard_dump_points_v2.py \
      --cache $NAVSIM_EXP_ROOT/metric_cache_v2_navhard_two_stage --out-dir <new dir> --workers 8
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

LOADER = None


def _init(cache):
    global LOADER
    from navsim.common.dataloader import MetricCacheLoader
    LOADER = MetricCacheLoader(Path(cache))


def _one(tok):
    mc = LOADER.get_from_token(tok)
    ra = mc.ego_state.rear_axle
    return dict(token=tok, log_name=str(mc.log_name), frame_type=str(mc.scene_type),
                start_time=float(mc.timepoint.time_s), start_point_x=float(ra.x),
                start_point_y=float(ra.y), start_heading=float(ra.heading))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    import navsim
    dk = os.environ.get("NAVSIM_DEVKIT_ROOT", "")
    print(f"[env] navsim -> {navsim.__file__}")
    if not dk or not str(Path(navsim.__file__).resolve()).startswith(str(Path(dk).resolve())):
        raise SystemExit("[FAIL] navsim 不是 v2 devkit；请用 PYTHONPATH=$NAVSIM_DEVKIT_ROOT 并在不含 navsim/ 的目录下运行")
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"[FAIL] 输出目录已存在，不覆盖：{out}")

    _init(a.cache)
    tokens = sorted(LOADER.tokens)
    print(f"[load] cache tokens: {len(tokens)}")
    with Pool(a.workers, initializer=_init, initargs=(a.cache,)) as pool:
        rows = list(pool.imap(_one, tokens, chunksize=16))
    df = pd.DataFrame(rows)
    print(df["frame_type"].value_counts().to_string())
    print(f"logs: {df['log_name'].nunique()}")
    out.mkdir(parents=True, exist_ok=False)
    f = out / "start_points.csv"
    df.to_csv(f, index=False, float_format="%.17g")  # full precision: weights are exp(-d^2/0.2)
    print(f"[write] {f}  rows={len(df)}")


if __name__ == "__main__":
    sys.exit(main())
