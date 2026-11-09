#!/usr/bin/env python3
"""Per-log sensor_blobs coverage audit.

Metadata (navsim_logs/<split>/*.pkl), not the blob tree, is the frame inventory:
the expected file list is derived from frame["cams"][CAMERA]["data_path"], which is
exactly the path navsim/common/dataclasses.py resolves against sensor_blobs_path.

The audit separates two causes that must never be conflated:

  declared_missing  data_path starts with "missing_camera/" -> the dataset itself
                    states no image exists. A dataset characteristic. Not fixable
                    by downloading.
  absent_on_disk    data_path is a normal path but the file is not present (or is
                    zero-byte). A local availability problem. Fixable by
                    re-downloading the owning log.

Per-log status:
  complete  every expected file present
  partial   some present, some absent
  absent    zero expected files present (the whole log directory is missing)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

DEFAULT_CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")
MISSING_CAMERA_PREFIX = "missing_camera"


def scan_log_dir(blob_root: Path, log_dir: str, cameras: Tuple[str, ...]) -> Dict[str, Dict[str, int]]:
    """One scandir per camera directory instead of one stat per expected file.

    Returns {camera: {filename: size}}. Missing directories yield an empty dict,
    which is what makes a fully absent log cheap to detect on a parallel FS.
    """
    present: Dict[str, Dict[str, int]] = {}
    for camera in cameras:
        camera_dir = blob_root / log_dir / camera
        entries: Dict[str, int] = {}
        try:
            with os.scandir(camera_dir) as iterator:
                for entry in iterator:
                    if entry.is_file(follow_symlinks=True):
                        try:
                            entries[entry.name] = entry.stat(follow_symlinks=True).st_size
                        except OSError:
                            entries[entry.name] = -1
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            pass
        present[camera] = entries
    return present


def audit_log(
    log_path: Path,
    blob_root: Path,
    cameras: Tuple[str, ...],
) -> Dict[str, Any]:
    with log_path.open("rb") as handle:
        frames = pickle.load(handle)
    assert isinstance(frames, list), f"metadata is not a list: {log_path}"

    log_name = log_path.stem
    expected = 0
    present = 0
    zero_byte = 0
    declared_missing = 0
    camera_exists_false = 0
    per_camera_absent: Counter = Counter()
    missing_paths: List[str] = []
    log_dirs: set[str] = set()
    dir_cache: Dict[str, Dict[str, Dict[str, int]]] = {}

    for frame in frames:
        if frame.get("camera_exists", True) is False:
            camera_exists_false += 1
        cams = frame.get("cams") or {}
        for camera in cameras:
            entry = cams.get(camera)
            if entry is None or not entry.get("data_path"):
                declared_missing += 1
                continue
            rel = PurePosixPath(str(entry["data_path"]))
            if rel.parts and rel.parts[0] == MISSING_CAMERA_PREFIX:
                declared_missing += 1
                continue
            assert len(rel.parts) >= 3, f"unexpected data_path layout: {rel}"
            log_dir = rel.parts[0]
            log_dirs.add(log_dir)
            if log_dir not in dir_cache:
                dir_cache[log_dir] = scan_log_dir(blob_root, log_dir, cameras)
            expected += 1
            size = dir_cache[log_dir].get(rel.parts[1], {}).get(rel.parts[-1])
            if size is None:
                per_camera_absent[camera] += 1
                missing_paths.append(rel.as_posix())
            elif size == 0:
                zero_byte += 1
                per_camera_absent[camera] += 1
                missing_paths.append(rel.as_posix())
            else:
                present += 1

    absent = expected - present
    if expected == 0:
        status = "no_expected_files"
    elif absent == 0:
        status = "complete"
    elif present == 0:
        status = "absent"
    else:
        status = "partial"

    return {
        "log_name": log_name,
        "status": status,
        "n_frames": len(frames),
        "blob_dirs": sorted(log_dirs),
        "expected_files": expected,
        "present_files": present,
        "absent_files": absent,
        "zero_byte_files": zero_byte,
        "declared_missing_paths": declared_missing,
        "frames_camera_exists_false": camera_exists_false,
        "absent_by_camera": dict(per_camera_absent),
        "missing_paths": missing_paths,
    }


def decode_check(blob_root: Path, reports: List[Dict[str, Any]], sample: int, seed: int) -> Dict[str, Any]:
    """Open a random sample of present files. Catches truncated transfers that
    a size>0 check cannot see."""
    try:
        from PIL import Image
    except ImportError:
        return {"skipped": "Pillow not available"}
    candidates: List[str] = []
    for report in reports:
        for log_dir in report["blob_dirs"]:
            camera_dir = blob_root / log_dir
            try:
                for camera in sorted(os.listdir(camera_dir)):
                    files = sorted(os.listdir(camera_dir / camera))[:2]
                    candidates.extend(f"{log_dir}/{camera}/{name}" for name in files)
            except OSError:
                continue
    if not candidates:
        return {"checked": 0, "corrupt": []}
    random.Random(seed).shuffle(candidates)
    corrupt: List[str] = []
    for rel in candidates[:sample]:
        try:
            with Image.open(blob_root / rel) as image:
                image.verify()
        except Exception as error:  # noqa: BLE001 - any decode failure is a finding
            corrupt.append(f"{rel}\t{type(error).__name__}: {error}")
    return {"checked": min(sample, len(candidates)), "corrupt": corrupt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-split", default="trainval", help="trainval | test | mini")
    parser.add_argument("--openscene-data-root", type=Path, default=os.environ.get("OPENSCENE_DATA_ROOT"))
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--blob-root", type=Path, default=None)
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dump-missing-files", action="store_true", help="write every absent relative path")
    parser.add_argument("--decode-sample", type=int, default=0, help="open N present images to detect truncation")
    parser.add_argument("--decode-seed", type=int, default=20260826)
    args = parser.parse_args()

    if args.log_root is None or args.blob_root is None:
        assert args.openscene_data_root is not None, "set OPENSCENE_DATA_ROOT or pass --log-root/--blob-root"
    log_root = args.log_root or Path(args.openscene_data_root) / "navsim_logs" / args.data_split
    blob_root = args.blob_root or Path(args.openscene_data_root) / "sensor_blobs" / args.data_split
    cameras = tuple(name.strip() for name in args.cameras.split(",") if name.strip())
    assert log_root.is_dir(), f"log root not found: {log_root}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_paths = sorted(path for path in log_root.iterdir() if path.suffix == ".pkl")
    assert log_paths, f"no .pkl logs under {log_root}"

    reports: List[Dict[str, Any]] = []
    for index, log_path in enumerate(log_paths, start=1):
        reports.append(audit_log(log_path, blob_root, cameras))
        if index % 50 == 0 or index == len(log_paths):
            print(f"[{index}/{len(log_paths)}] {log_path.stem}", flush=True)

    by_status: Counter = Counter(report["status"] for report in reports)
    expected_total = sum(report["expected_files"] for report in reports)
    present_total = sum(report["present_files"] for report in reports)
    declared_total = sum(report["declared_missing_paths"] for report in reports)
    zero_byte_total = sum(report["zero_byte_files"] for report in reports)

    by_date: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for report in reports:
        key = report["log_name"][:10]
        by_date[key][0] += report["expected_files"]
        by_date[key][1] += report["present_files"]

    absent_logs = [r["log_name"] for r in reports if r["status"] == "absent"]
    partial_logs = [r for r in reports if r["status"] == "partial"]

    summary = {
        "data_split": args.data_split,
        "log_root": str(log_root),
        "blob_root": str(blob_root),
        "cameras": list(cameras),
        "logs_expected": len(reports),
        "logs_with_any_file": len(reports) - len(absent_logs) - by_status.get("no_expected_files", 0),
        "logs_complete": by_status.get("complete", 0),
        "logs_partial": by_status.get("partial", 0),
        "logs_absent": by_status.get("absent", 0),
        "logs_no_expected_files": by_status.get("no_expected_files", 0),
        "files_expected": expected_total,
        "files_present": present_total,
        "files_absent": expected_total - present_total,
        "files_zero_byte": zero_byte_total,
        "coverage_pct": round(100.0 * present_total / expected_total, 4) if expected_total else 0.0,
        "declared_missing_paths": declared_total,
        "frames_camera_exists_false": sum(r["frames_camera_exists_false"] for r in reports),
        "coverage_by_date": {
            key: {
                "expected": value[0],
                "present": value[1],
                "coverage_pct": round(100.0 * value[1] / value[0], 2) if value[0] else 0.0,
            }
            for key, value in sorted(by_date.items())
        },
    }
    if args.decode_sample:
        summary["decode_check"] = decode_check(blob_root, reports, args.decode_sample, args.decode_seed)

    (args.out_dir / "logs_absent.txt").write_text(
        "".join(f"{name}\n" for name in absent_logs), encoding="utf-8"
    )
    (args.out_dir / "logs_partial.txt").write_text(
        "".join(
            f"{r['log_name']}\t{r['present_files']}/{r['expected_files']}\tmissing={r['absent_files']}\n"
            for r in sorted(partial_logs, key=lambda r: -r["absent_files"])
        ),
        encoding="utf-8",
    )
    (args.out_dir / "redownload_logs.txt").write_text(
        "".join(f"{name}\n" for name in sorted(absent_logs + [r["log_name"] for r in partial_logs])),
        encoding="utf-8",
    )
    if args.dump_missing_files:
        with (args.out_dir / "missing_files.txt").open("w", encoding="utf-8") as handle:
            for report in reports:
                for rel in report["missing_paths"]:
                    handle.write(rel + "\n")

    for report in reports:
        report.pop("missing_paths")
    with (args.out_dir / "sensor_blob_coverage.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_log": reports}, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    print(
        f"\n应有 log 数 {summary['logs_expected']}"
        f" | 实有 log 数 {summary['logs_with_any_file']}"
        f" | 完整 {summary['logs_complete']}"
        f" | 部分缺失 {summary['logs_partial']}"
        f" | 完全缺失 {summary['logs_absent']}"
        f"\n文件覆盖 {summary['files_present']}/{summary['files_expected']}"
        f" = {summary['coverage_pct']}%"
        f" | 数据集声明无图 {summary['declared_missing_paths']} 条（非下载问题）"
    )


if __name__ == "__main__":
    main()
