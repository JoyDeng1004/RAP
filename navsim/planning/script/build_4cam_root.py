#!/usr/bin/env python3
"""Build the frozen Stage-A three-camera hard-link tree.

The fourth camera (CAM_B0) is generated only after Stage-A token selection.
Metadata, rather than files found in the raster root, is the frame inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List

import yaml


CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")
AUTHORIZED_MISSING_CAMERA_TOKENS = {
    "52f288435bb35105",
    "5183fe8b541e5b82",
    "e1c06719b4275c35",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict), f"expected JSON object: {path}"
    return value


def destination_audit(root: Path) -> Dict[str, Any]:
    entries = sorted(root.iterdir()) if root.is_dir() else []
    files = [entry for entry in root.rglob("*") if entry.is_file()] if entries else []
    return {
        "root": str(root.resolve()),
        "top_level_entries": [entry.name for entry in entries],
        "file_count": len(files),
        "sample_files": [str(path.relative_to(root)) for path in files[:20]],
    }


def metadata_frames(log_root: Path, log_names: List[str]) -> Iterator[tuple[str, Dict[str, Any]]]:
    seen_tokens = set()
    for log_name in log_names:
        log_path = log_root / f"{log_name}.pkl"
        assert log_path.is_file(), f"missing metadata log: {log_path}"
        with log_path.open("rb") as handle:
            frames = pickle.load(handle)
        assert isinstance(frames, list), f"metadata is not a list: {log_path}"
        for frame in frames:
            assert frame["log_name"] == log_name, (frame["log_name"], log_name)
            token = frame["token"]
            assert token not in seen_tokens, f"duplicate metadata frame token: {token}"
            seen_tokens.add(token)
            yield log_name, frame


def preflight_frames(
    log_root: Path, log_names: List[str], raster_src_root: Path
) -> tuple[List[tuple[str, Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Validate the complete inventory before creating the first hard link."""
    valid: List[tuple[str, Dict[str, Any]]] = []
    excluded: List[Dict[str, Any]] = []
    retained_camera_exists_false: List[Dict[str, str]] = []
    for log_name, frame in metadata_frames(log_root, log_names):
        if frame["token"] in AUTHORIZED_MISSING_CAMERA_TOKENS:
            assert frame.get("camera_exists", True) is False, frame["token"]
            missing = []
            for camera in CAMERAS:
                rel_path = Path(frame["cams"][camera]["data_path"])
                if rel_path.parts[0] == "missing_camera":
                    missing.append(
                        {
                            "camera": camera,
                            "rel_path": rel_path.as_posix(),
                            "source_exists": (raster_src_root / rel_path).is_file(),
                        }
                    )
            excluded.append(
                {
                    "frame_token": frame["token"],
                    "log_name": log_name,
                    "missing_camera_paths": missing,
                    "reason": "authorized_camera_exists_false_exclusion",
                }
            )
            continue

        if frame.get("camera_exists", True) is False:
            retained_camera_exists_false.append(
                {
                    "frame_token": frame["token"],
                    "log_name": log_name,
                    "reason": "all_step_0_3_camera_paths_valid",
                }
            )

        camera_paths: Dict[str, str] = {}
        for camera in CAMERAS:
            rel_path = Path(frame["cams"][camera]["data_path"])
            assert len(rel_path.parts) >= 3, rel_path
            assert rel_path.parts[0] == log_name, (log_name, rel_path)
            assert rel_path.parts[1] == camera, (camera, rel_path)
            assert (raster_src_root / rel_path).is_file(), f"missing source raster: {rel_path}"
            camera_paths[camera] = rel_path.as_posix()
        # Keep only the inventory fields needed during linking. Holding all loaded
        # metadata frames would retain large annotation arrays unnecessarily.
        valid.append((log_name, {"frame_token": frame["token"], "camera_paths": camera_paths}))

    excluded_tokens = {item["frame_token"] for item in excluded}
    assert excluded_tokens == AUTHORIZED_MISSING_CAMERA_TOKENS, (
        "camera_exists=False inventory differs from the three explicitly authorized exclusions: "
        f"expected={sorted(AUTHORIZED_MISSING_CAMERA_TOKENS)}, actual={sorted(excluded_tokens)}"
    )
    return valid, excluded, retained_camera_exists_false


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster-src-root", type=Path, required=True)
    parser.add_argument("--dest-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--log-list", type=Path, required=True)
    parser.add_argument("--source-checksums", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    assert args.workers >= 1

    frozen = load_json(args.source_checksums)
    recorded = frozen["paired_scene_filter"]["sha256"]
    actual = sha256_file(args.log_list)
    assert actual == recorded, (
        f"paired log-list checksum drifted: recorded={recorded}, actual={actual}; HALT"
    )

    scene_filter = yaml.safe_load(args.log_list.read_text(encoding="utf-8"))
    log_names = scene_filter["log_names"]
    assert isinstance(log_names, list) and len(log_names) == 54
    assert len(log_names) == len(set(log_names)), "duplicate log in paired log list"
    assert log_names == sorted(log_names), "paired log list must be sorted"
    frozen_paired = set(
        frozen["split_intersection_log_names"]["train_logs"]
        + frozen["split_intersection_log_names"]["val_logs"]
    )
    frozen_test = set(frozen["split_intersection_log_names"]["test_logs"])
    assert set(log_names) == frozen_paired, "paired log list differs from frozen intersection"
    assert not set(log_names) & frozen_test, "test log in hard-link input"

    audit = destination_audit(args.dest_root)
    if audit["top_level_entries"]:
        print(json.dumps({"nonempty_destination_audit": audit}, indent=2, sort_keys=True))
        raise RuntimeError("destination is non-empty; audited and refusing to overwrite (HALT)")
    if args.manifest.exists() and args.manifest.stat().st_size:
        raise RuntimeError(f"manifest is non-empty; refusing to overwrite (HALT): {args.manifest}")

    valid_frames, excluded_frames, retained_camera_exists_false = preflight_frames(
        args.log_root, log_names, args.raster_src_root
    )
    metadata_frame_total = len(valid_frames) + len(excluded_frames)
    expected_frame_total = len(valid_frames)

    args.dest_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.manifest.with_name(f"{args.manifest.stem}_summary.json")
    if summary_path.exists() and summary_path.stat().st_size:
        raise RuntimeError(f"summary is non-empty; refusing to overwrite (HALT): {summary_path}")

    counts: Counter[str] = Counter()
    link_jobs = [
        (log_name, frame["frame_token"], camera, Path(frame["camera_paths"][camera]))
        for log_name, frame in valid_frames
        for camera in CAMERAS
    ]

    def link_one(job: tuple[str, str, str, Path]) -> Dict[str, Any]:
        log_name, frame_token, camera, rel_path = job
        assert rel_path.parts[0] == log_name, (log_name, rel_path)
        assert len(rel_path.parts) >= 3 and rel_path.parts[1] == camera, rel_path
        src = args.raster_src_root / rel_path
        dst = args.dest_root / rel_path
        assert src.is_file(), f"missing source raster: {src}"
        assert not src.is_symlink(), f"source raster must not be a symlink: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists(), f"destination collision: {dst}"

        src_sha256 = sha256_file(src)
        inode_src = src.stat().st_ino
        os.link(src, dst)
        inode_dst = dst.stat().st_ino
        linked_sha256 = sha256_file(dst)
        assert inode_src == inode_dst, f"not a hard link: {src} -> {dst}"
        assert src_sha256 == linked_sha256, f"hard-link checksum mismatch: {dst}"
        return {
            "camera": camera,
            "frame_token": frame_token,
            "inode_dst": inode_dst,
            "inode_src": inode_src,
            "linked_sha256": linked_sha256,
            "log_name": log_name,
            "rel_path": rel_path.as_posix(),
            "src_sha256": src_sha256,
        }

    with args.manifest.open("x", encoding="utf-8") as manifest:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # executor.map preserves input order, so the manifest remains deterministic.
            for record in executor.map(link_one, link_jobs):
                manifest.write(
                    json.dumps(record, sort_keys=True) + "\n"
                )
                counts[record["camera"]] += 1

    dest_logs = sorted(path.name for path in args.dest_root.iterdir() if path.is_dir())
    assert dest_logs == log_names, "destination log set differs from paired log list"
    for camera in CAMERAS:
        assert counts[camera] == expected_frame_total, (camera, counts[camera], expected_frame_total)
    assert not set(dest_logs) & frozen_test, "test log leaked into destination"

    summary = {
        "camera_counts": dict(sorted(counts.items())),
        "dest_root": str(args.dest_root.resolve()),
        "excluded_camera_missing_frame_count": len(excluded_frames),
        "excluded_camera_missing_frames": excluded_frames,
        "expected_frame_total": expected_frame_total,
        "log_count": len(dest_logs),
        "log_list": str(args.log_list.resolve()),
        "log_list_sha256": actual,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "metadata_frame_total": metadata_frame_total,
        "retained_camera_exists_false_frames": retained_camera_exists_false,
        "test_log_intersection": [],
        "workers": args.workers,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
