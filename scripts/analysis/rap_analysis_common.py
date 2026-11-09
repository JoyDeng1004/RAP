"""Shared read-only helpers for scripts/analysis/*.

Only pandas/numpy/stdlib (rap env, Python 3.9). Nothing here writes files.
"""
import glob
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# aggregate rows written by the devkits' run_pdm_score*.py
V1_AGG_TOKENS = ("average",)
V2_AGG_TOKENS = (
    "average_all_frames",
    "extended_pdm_score_stage_one",
    "extended_pdm_score_stage_two",
    "extended_pdm_score_combined",
)


def find_one(path: str, pattern: str) -> Path:
    """`path` is a file, or a directory holding exactly one file matching `pattern`."""
    p = Path(path)
    if p.is_file():
        return p
    hits = sorted(glob.glob(str(p / pattern)))
    if len(hits) != 1:
        raise SystemExit(f"[FAIL] expected exactly one {pattern} under {p}, found {len(hits)}: {hits[:5]}")
    return Path(hits[0])


def read_eval_csv(path: str, agg_tokens: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Return (token rows, aggregate rows, resolved csv path). First CSV column is the index."""
    f = find_one(path, "*.csv")
    df = pd.read_csv(f, index_col=0)
    for col in ("token", "valid"):
        if col not in df.columns:
            raise SystemExit(f"[FAIL] {f}: missing column {col!r}; columns = {list(df.columns)}")
    df["token"] = df["token"].astype(str)
    is_agg = df["token"].isin(list(agg_tokens))
    rows, agg = df[~is_agg].copy(), df[is_agg].copy()
    rows["valid"] = rows["valid"].astype(str).str.lower().isin(["true", "1", "1.0"])
    if rows["token"].duplicated().any():
        raise SystemExit(f"[FAIL] {f}: duplicated tokens, e.g. {rows.loc[rows['token'].duplicated(), 'token'].iloc[0]}")
    return rows, agg, f


def v1_cache_index(cache_root: str) -> pd.DataFrame:
    """token -> log_name, scenario_type from `<cache>/metadata/*.csv` (path .../<log>/<type>/<token>/metric_cache.pkl)."""
    meta = sorted(glob.glob(os.path.join(cache_root, "metadata", "*.csv")))
    if not meta:
        raise SystemExit(f"[FAIL] no metadata csv under {cache_root}/metadata")
    recs = []
    for m in meta:
        with open(m) as fh:
            lines = fh.read().splitlines()[1:]  # same as MetricCacheLoader: first line is a header
        for line in lines:
            parts = Path(line.strip()).parts
            if len(parts) < 4:
                continue
            recs.append((parts[-2], parts[-4], parts[-3]))
    idx = pd.DataFrame(recs, columns=["token", "log_name", "scenario_type"]).drop_duplicates("token")
    return idx.set_index("token")


def _log_hypergeom(x: int, n_g: int, k: int, n: int) -> float:
    """log P(X=x), X ~ Hypergeom(population n, k successes, n_g draws)."""
    def lc(a, b):
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)
    return lc(k, x) + lc(n - k, n_g - x) - lc(n, n_g)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for [[a, b], [c, d]] (same convention as scipy: sum of P <= P_obs)."""
    n_g, k, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, n_g - (n - k)), min(n_g, k)
    lp_obs = _log_hypergeom(a, n_g, k, n)
    tot = 0.0
    for x in range(lo, hi + 1):
        lp = _log_hypergeom(x, n_g, k, n)
        if lp <= lp_obs + 1e-7:
            tot += math.exp(lp)
    return min(1.0, tot)


def bh_qvalues(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def quat_to_yaw(q: Sequence[float]) -> float:
    """[w, x, y, z] -> yaw."""
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_named(items: Optional[List[str]]) -> Dict[str, str]:
    """['E3=/a/b', 'official=/c'] -> {'E3': '/a/b', 'official': '/c'}."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"[FAIL] expected NAME=PATH, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out
