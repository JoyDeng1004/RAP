#!/usr/bin/env python3
"""Generate reproducible, nested, log-level few-shot training splits.

Write YAML files for the ``train_val_test_log_split`` config group:

    python navsim/planning/script/run_training.py ... \\
        train_val_test_log_split=navsim_kshot_p05_seed0

The sampler stratifies logs by city, turn fraction, mean speed, and acceleration
p95. Lower fractions for a seed are prefixes of the same stratified ordering.
``--guarantee city`` ensures every city appears in the smallest split.

    python tools/make_kshot_splits.py \\
        --logs-dir $OPENSCENE_DATA_ROOT/navsim_logs/trainval \\
        --scene-filter navsim/planning/script/config/common/train_test_split/scene_filter/navall.yaml \\
        --base-split navsim/planning/script/config/training/train_val_test_log_split/nuplan_trainval.yaml \\
        --name-prefix navsim_kshot \\
        --fractions 0.01 0.02 0.05 0.10 1.0 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Allow direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from log_universe import resolve_universe  # noqa: E402

CMD_LEFT, CMD_FORWARD, CMD_RIGHT, CMD_UNKNOWN = 0, 1, 2, 3


@dataclass
class LogFeature:
    """Scalar log features used for stratification."""

    name: str
    n_frames: int
    city: str
    turn_frac: float      # Fraction of left- or right-turn frames.
    speed_mean: float     # m/s
    accel_p95: float      # m/s^2; proxy for scene difficulty.


def extract_log_feature(args: Tuple[Path, str]) -> Optional[LogFeature]:
    """Extract stratification features from one log pickle."""
    pkl_path, name = args
    try:
        frames = pickle.loads(pkl_path.read_bytes())
    except Exception:
        return None

    cmds: List[int] = []
    speeds: List[float] = []
    accels: List[float] = []
    cities: List[str] = []

    for frame in frames:
        # Metadata augmentation may add invalid placeholder frames.
        if not frame.get("is_valid", True):
            continue
        cmd = np.asarray(frame.get("driving_command", [0, 0, 0, 1])).reshape(-1)
        cmds.append(int(np.argmax(cmd)) if cmd.any() else CMD_UNKNOWN)

        state = list(frame.get("ego_dynamic_state") or [])
        state = (state + [0.0, 0.0, 0.0, 0.0])[:4]
        vx, vy, ax, ay = (float(v) for v in state)
        speeds.append(math.hypot(vx, vy))
        accels.append(math.hypot(ax, ay))

        cities.append(str(frame.get("map_location", "unknown")))

    if not cmds:
        return None

    turning = sum(1 for c in cmds if c in (CMD_LEFT, CMD_RIGHT))
    return LogFeature(
        name=name,
        n_frames=len(cmds),
        city=Counter(cities).most_common(1)[0][0],
        turn_frac=turning / len(cmds),
        speed_mean=float(np.mean(speeds)),
        accel_p95=float(np.percentile(accels, 95)),
    )


def load_features(
    logs_dir: Path, names: Sequence[str], jobs: int, cache: Optional[Path]
) -> List[LogFeature]:
    """Load per-log features, optionally reusing a JSON cache."""
    if cache is not None and cache.is_file():
        rows = json.loads(cache.read_text())
        by_name = {r["name"]: LogFeature(**r) for r in rows}
        if set(by_name) >= set(names):
            print(f"[kshot] Reusing feature cache {cache} ({len(by_name)} logs)")
            return [by_name[n] for n in names]
        print(f"[kshot] Feature cache is incomplete; rescanning: {cache}")

    tasks = [(logs_dir / f"{n}.pkl", n) for n in names]
    print(f"[kshot] Scanning {len(tasks)} log pickles (jobs={jobs})...")
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(extract_log_feature, tasks, chunksize=8))
    else:
        results = [extract_log_feature(t) for t in tasks]

    bad = [n for (_, n), r in zip(tasks, results) if r is None]
    if bad:
        print(f"[kshot] WARN excluded {len(bad)} logs without readable valid frames, e.g. {bad[:3]}")
    feats = [r for r in results if r is not None]
    if not feats:
        raise SystemExit("FAIL no log yielded extractable features")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([asdict(f) for f in feats], ensure_ascii=False, indent=1))
        print(f"[kshot] Saved features to {cache}")
    return feats


def quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin values, reducing bins when edges collapse."""
    if n_bins <= 1:
        return np.zeros(len(values), dtype=int)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    if edges.size == 0:
        return np.zeros(len(values), dtype=int)
    return np.searchsorted(edges, values, side="right")


def build_strata(
    feats: Sequence[LogFeature], turn_bins: int, speed_bins: int, accel_bins: int
) -> List[Tuple[str, ...]]:
    """Build one stratification key per log."""
    turn = quantile_bin(np.array([f.turn_frac for f in feats]), turn_bins)
    speed = quantile_bin(np.array([f.speed_mean for f in feats]), speed_bins)
    accel = quantile_bin(np.array([f.accel_p95 for f in feats]), accel_bins)
    return [
        (f.city, f"turn{t}", f"spd{s}", f"acc{a}")
        for f, t, s, a in zip(feats, turn, speed, accel)
    ]


def stratified_order(
    feats: Sequence[LogFeature],
    strata: Sequence[Tuple[str, ...]],
    seed: int,
    guarantee_dims: Sequence[str],
) -> List[int]:
    """Return a balanced global ordering whose prefixes are stratified samples.

    After shuffling a stratum, its i-th log receives score
    ``(i + 0.5) / |stratum|``. Sorting globally by score yields proportional,
    nested prefixes.
    """
    rng = np.random.default_rng(seed)

    members: Dict[Tuple[str, ...], List[int]] = {}
    for i, key in enumerate(strata):
        members.setdefault(key, []).append(i)

    scores = np.empty(len(feats), dtype=float)
    for key in sorted(members):
        idxs = members[key]
        for rank, i in enumerate(rng.permutation(idxs)):
            scores[i] = (rank + 0.5) / len(idxs)

    # Promote one log per requested value to prevent absent strata in small splits.
    dim_of = {"city": 0, "turn": 1, "speed": 2, "accel": 3}
    for dim in guarantee_dims:
        if dim not in dim_of:
            raise SystemExit(f"FAIL unknown --guarantee dimension: {dim} (choices: {sorted(dim_of)})")
        pos = dim_of[dim]
        best: Dict[str, int] = {}
        for i, key in enumerate(strata):
            val = key[pos]
            if val not in best or scores[i] < scores[best[val]]:
                best[val] = i
        for i in best.values():
            scores[i] -= 1.0  # Promote while preserving their relative order.

    tiebreak = rng.random(len(feats))
    return sorted(range(len(feats)), key=lambda i: (scores[i], tiebreak[i]))


def tv_distance(sub: Sequence[str], full: Sequence[str]) -> float:
    """Total variation distance between categorical distributions; 0 is exact."""
    p, q = Counter(sub), Counter(full)
    n_p, n_q = sum(p.values()), sum(q.values())
    if n_p == 0 or n_q == 0:
        return float("nan")
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p[k] / n_p - q[k] / n_q) for k in keys)


def load_base_split(path: Path) -> Dict[str, List[str]]:
    """Load ``val_logs`` and ``test_logs`` from an existing split YAML."""
    data = yaml.safe_load(path.read_text()) or {}
    # Support both top-level and one-level nested Hydra configs.
    if "val_logs" not in data and len(data) == 1:
        data = next(iter(data.values())) or {}
    out = {k: [str(x) for x in (data.get(k) or [])] for k in ("val_logs", "test_logs")}
    if not out["val_logs"]:
        raise SystemExit(f"FAIL {path} has no val_logs to inherit")
    return out


def write_split_yaml(
    path: Path,
    train_logs: Sequence[str],
    base: Dict[str, List[str]],
    header: Sequence[str],
) -> None:
    lines = ["# @package _global_"]
    lines += [f"# {h}" for h in header]
    lines.append("")
    lines.append("restrict_train_logs: true")
    lines.append("")
    for key, values in (("train_logs", train_logs),
                        ("val_logs", base["val_logs"]),
                        ("test_logs", base["test_logs"])):
        lines.append(f"{key}:")
        lines.extend(f"  - {v}" for v in values)
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--logs-dir", type=Path, required=True,
                        help="$OPENSCENE_DATA_ROOT/navsim_logs/<data_split>")
    parser.add_argument("--scene-filter", type=Path, default=None,
                        help="scene_filter YAML; intersect its log_names with logs-dir")
    parser.add_argument("--base-split", type=Path, required=True,
                        help="train_val_test_log_split YAML providing val_logs and test_logs")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("navsim/planning/script/config/training/train_val_test_log_split"))
    parser.add_argument("--name-prefix", required=True, help="for example, navsim_kshot")
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.01, 0.02, 0.05, 0.10, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--turn-bins", type=int, default=3)
    parser.add_argument("--speed-bins", type=int, default=3)
    parser.add_argument("--accel-bins", type=int, default=2)
    parser.add_argument("--guarantee", nargs="*", default=["city"],
                        help="dimensions guaranteed in small splits (city/turn/speed/accel); pass '' to disable")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="JSON cache for per-log features")
    parser.add_argument("--dry-run", action="store_true", help="print the report without writing YAML files")
    args = parser.parse_args()

    universe, missing = resolve_universe(args.logs_dir, args.scene_filter)
    if missing:
        print(f"[kshot] WARN scene filter references {len(missing)} missing logs, e.g. {missing[:3]}")

    base = load_base_split(args.base_split)
    val_set = set(base["val_logs"]) | set(base["test_logs"])
    pool_names = [n for n in universe if n not in val_set]
    print(f"[kshot] Universe: {len(universe)} -> candidate pool after val/test exclusion: {len(pool_names)}")

    feats = load_features(args.logs_dir, pool_names, args.jobs, args.feature_cache)
    strata = build_strata(feats, args.turn_bins, args.speed_bins, args.accel_bins)
    print(f"[kshot] Strata: {len(set(strata))} (cities: {len({s[0] for s in strata})})")

    guarantee = [g for g in args.guarantee if g]
    fractions = sorted(set(args.fractions))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report: List[str] = [
        "# K-shot split report",
        "",
        f"- logs-dir: `{args.logs_dir}`",
        f"- Candidate pool: {len(feats)} logs / {sum(f.n_frames for f in feats)} frames",
        f"- Stratification: city x turn({args.turn_bins}) x speed({args.speed_bins}) x accel({args.accel_bins})"
        f" = {len(set(strata))} strata, guarantee={guarantee or 'none'}",
        "",
        "TV is the total variation distance from the candidate pool (0 is exact).",
        "",
        "| split | logs | frames | TV(city) | TV(turn) | TV(speed) | TV(accel) |",
        "|---|---|---|---|---|---|---|",
    ]

    for seed in args.seeds:
        order = stratified_order(feats, strata, seed, guarantee)
        prev_set: set = set()
        prev_name = ""
        for frac in fractions:
            k = min(len(feats), max(1, round(frac * len(feats))))
            picked = order[:k]
            names = sorted(feats[i].name for i in picked)

            # Fail rather than silently emitting non-nested splits.
            assert prev_set <= set(names), (
                f"nested split violation: {prev_name} is not a subset of {frac} (seed={seed})"
            )
            prev_set, prev_name = set(names), f"{frac:g}"

            tag = f"p{round(frac * 100):03d}"
            split_name = f"{args.name_prefix}_{tag}_seed{seed}"
            n_frames = sum(feats[i].n_frames for i in picked)

            tvs = [
                tv_distance([strata[i][d] for i in picked], [s[d] for s in strata])
                for d in range(4)
            ]
            report.append(
                f"| {split_name} | {k} | {n_frames} | "
                + " | ".join(f"{t:.3f}" for t in tvs) + " |"
            )

            if args.dry_run:
                continue
            write_split_yaml(
                args.out_dir / f"{split_name}.yaml",
                names,
                base,
                header=[
                    "Generated by tools/make_kshot_splits.py -- do not edit manually.",
                    f"Fraction: {frac:g}; seed: {seed}; train logs: {k}/{len(feats)}; frames: {n_frames}.",
                    "Nested: lower fractions for the same seed are subsets of this split.",
                ],
            )

    report_path = args.out_dir / f"_{args.name_prefix}_report.md"
    text = "\n".join(report) + "\n"
    print()
    print(text)
    if not args.dry_run:
        report_path.write_text(text)
        print(f"[kshot] Wrote YAML files to {args.out_dir}")
        print(f"[kshot] Wrote report to {report_path}")


if __name__ == "__main__":
    main()
