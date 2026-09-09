#!/usr/bin/env python3
"""Freeze Stage-A source inventory and emit the 54-log Hydra split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
DOCUMENTED_COUNTS = {
    "split_entries": {"train_logs": 13_180, "val_logs": 1_381, "test_logs": 1_349},
    "raster_logs": 64,
    "split_intersection": {"train_logs": 44, "val_logs": 10, "test_logs": 10},
    "camera_files": {"CAM_F0": 51_898, "CAM_L0": 51_898, "CAM_R0": 51_900, "CAM_B0": 48},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_files(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file()) if path.is_dir() else 0


def difference(actual: Any, documented: Any) -> Any:
    if isinstance(actual, dict):
        return {key: difference(actual[key], documented[key]) for key in actual}
    return actual - documented


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster-src-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emit-scene-filter", type=Path, required=True)
    args = parser.parse_args()

    split = yaml.safe_load(args.split_config.read_text())
    expected_keys = {"train_logs", "val_logs", "test_logs"}
    assert set(split) == expected_keys, sorted(split)
    split_sets = {key: set(split[key]) for key in expected_keys}
    for key, values in split.items():
        assert len(values) == len(set(values)), f"duplicate entries in {key}"
    assert not (split_sets["train_logs"] & split_sets["val_logs"])
    assert not (split_sets["train_logs"] & split_sets["test_logs"])
    assert not (split_sets["val_logs"] & split_sets["test_logs"])

    all_raster_dirs = sorted(path for path in args.raster_src_root.iterdir() if path.is_dir())
    excluded = []
    valid_dirs = []
    for path in all_raster_dirs:
        if path.name == "missing_camera":
            excluded.append({"name": path.name, "reason": "not_a_log_unconditional_exclusion"})
        else:
            valid_dirs.append(path)
    assert len(excluded) == 1, "missing_camera directory must be present and excluded exactly once"

    raster_logs = {path.name for path in valid_dirs}
    intersection = {key: sorted(raster_logs & values) for key, values in split_sets.items()}
    paired_logs = sorted(intersection["train_logs"] + intersection["val_logs"])

    # Structural gates stay hard even when documented absolute counts drift.
    assert len(paired_logs) == 54, f"paired log_names must contain exactly 54 logs, got {len(paired_logs)}"
    assert len(paired_logs) == len(set(paired_logs))
    assert not (set(paired_logs) & split_sets["test_logs"]), "test log leaked into paired log_names"
    assert set(paired_logs) == ((raster_logs & split_sets["train_logs"]) | (raster_logs & split_sets["val_logs"]))

    camera_counts = {
        camera: sum(count_files(log_dir / camera) for log_dir in valid_dirs) for camera in CAMERAS
    }
    assert camera_counts["CAM_F0"] == camera_counts["CAM_L0"], (
        camera_counts["CAM_F0"], camera_counts["CAM_L0"]
    )

    scene_filter_dir = args.emit_scene_filter / "scene_filter"
    scene_filter_dir.mkdir(parents=True, exist_ok=True)
    scene_filter_path = scene_filter_dir / "paired_54log.yaml"
    trainval_path = args.emit_scene_filter / "paired_trainval.yaml"
    scene_filter = {
        "_target_": "navsim.common.dataclasses.SceneFilter",
        "_convert_": "all",
        "num_history_frames": 4,
        "num_future_frames": 10,
        "frame_interval": 1,
        "has_route": True,
        "max_scenes": None,
        "tokens": None,
        "log_names": paired_logs,
    }
    trainval = {"defaults": [{"scene_filter": "paired_54log"}], "data_split": "trainval"}
    scene_filter_path.write_text(yaml.safe_dump(scene_filter, sort_keys=False), encoding="utf-8")
    trainval_path.write_text(yaml.safe_dump(trainval, sort_keys=False), encoding="utf-8")

    actual = {
        "split_entries": {key: len(split[key]) for key in sorted(expected_keys)},
        "raster_logs": len(valid_dirs),
        "split_intersection": {key: len(intersection[key]) for key in sorted(expected_keys)},
        "camera_files": camera_counts,
    }
    output = {
        "camera_file_counts": camera_counts,
        "documented_counts": DOCUMENTED_COUNTS,
        "documented_count_differences": difference(actual, DOCUMENTED_COUNTS),
        "excluded_raster_directories": excluded,
        "log_root": str(args.log_root.resolve()),
        "log_root_pkl_count": sum(1 for path in args.log_root.iterdir() if path.suffix == ".pkl"),
        "paired_scene_filter": {
            "path": str(scene_filter_path),
            "sha256": sha256_file(scene_filter_path),
            "log_names_count": len(paired_logs),
        },
        "paired_trainval": {"path": str(trainval_path), "sha256": sha256_file(trainval_path)},
        "raster_log_count": len(valid_dirs),
        "raster_log_names": sorted(raster_logs),
        "raster_source_root": str(args.raster_src_root.resolve()),
        "split_config": str(args.split_config.resolve()),
        "split_config_sha256": sha256_file(args.split_config),
        "split_entry_counts": actual["split_entries"],
        "split_intersection_counts": actual["split_intersection"],
        "split_intersection_log_names": intersection,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
