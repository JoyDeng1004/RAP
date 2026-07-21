#!/usr/bin/env python3
"""Audit Step 0-A for one explicit metadata/raster dataset pair.

The audit deliberately accepts an explicit raster root.  RAP's norm, aug and
perturbed datasets use three different raster directory names, so deriving the
path with a single ``sensor_blobs -> rendered_sensor_blobs`` replacement would
silently audit the wrong files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import platform
import random
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

PROCESS_DATA_DIR = Path(__file__).resolve().parents[1] / "process_data"
if str(PROCESS_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_DATA_DIR))
os.environ.setdefault("NUPLAN_DB_PATH", "/nonexistent/step0_audit_db")
os.environ.setdefault("NUPLAN_SENSOR_PATH", "/nonexistent/step0_audit_sensor")

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.common.dataclasses import Scene
from process_data.create_openscene_metadata import extract_map_features
from process_data.helpers.renderer import ScenarioRenderer


ALL_CAMERAS = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0")


def parse_camera_list(value: str) -> tuple[str, ...]:
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(cameras) - set(ALL_CAMERAS)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown cameras: {sorted(unknown)}")
    return cameras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--raster-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("norm", "aug", "perturbed"), required=True)
    parser.add_argument(
        "--required-cameras",
        type=parse_camera_list,
        required=True,
        help="Comma-separated cameras expected to have stored raster files.",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--content-samples-per-camera", type=int, default=256)
    parser.add_argument("--rerender-samples", type=int, default=100)
    parser.add_argument("--evidence-samples", type=int, default=8)
    parser.add_argument("--max-metadata-files", type=int)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fields})


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reservoir_add(
    reservoir: list[dict[str, Any]],
    item: dict[str, Any],
    seen: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    index = rng.randrange(seen)
    if index < capacity:
        reservoir[index] = item


def iter_frames(
    paths: Sequence[Path], load_errors: list[dict[str, Any]]
) -> Iterable[tuple[Path, int, Mapping[str, Any]]]:
    for path in paths:
        try:
            with path.open("rb") as handle:
                frames = pickle.load(handle)
        except Exception as error:
            load_errors.append(
                {
                    "metadata_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
            continue
        if not isinstance(frames, list):
            continue
        for frame_index, frame in enumerate(frames):
            if isinstance(frame, Mapping):
                yield path, frame_index, frame


def usable_frame(frame: Mapping[str, Any]) -> bool:
    return frame.get("is_valid", True) is not False and isinstance(frame.get("cams"), Mapping)


def scan_paths(
    metadata_paths: Sequence[Path],
    raster_root: Path,
    manifest_path: Path,
    variant: str,
    required_cameras: Sequence[str],
    content_capacity: int,
    rerender_capacity: int,
    rng: random.Random,
) -> tuple[
    dict[str, Counter],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    Counter,
    list[dict[str, Any]],
]:
    counters = {camera: Counter() for camera in ALL_CAMERAS}
    content_samples = {camera: [] for camera in ALL_CAMERAS}
    content_seen = Counter()
    rerender_samples: list[dict[str, Any]] = []
    metadata_errors: list[dict[str, Any]] = []
    totals = Counter()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(manifest_path, "wt", encoding="utf-8", compresslevel=1) as manifest:
        for metadata_path, frame_index, frame in iter_frames(metadata_paths, metadata_errors):
            totals["frames_total"] += 1
            # The perturbed generator writes a 14-frame training window but
            # renders only batch[3], the current frame consumed by the loader.
            # Neighbouring frames are context/targets and are not raster inputs.
            if variant == "perturbed" and frame_index != 3:
                totals["frames_skipped_noncurrent_perturbed"] += 1
                continue
            if frame.get("is_valid", True) is False:
                totals["frames_skipped_is_valid_false"] += 1
                continue
            if not isinstance(frame.get("cams"), Mapping):
                totals["frames_skipped_no_cams"] += 1
                continue
            totals["frames_audited"] += 1
            cams = frame["cams"]
            if all(camera in cams for camera in required_cameras) and isinstance(frame.get("anns"), Mapping):
                totals["rerender_eligible"] += 1
                reservoir_add(
                    rerender_samples,
                    {
                        "metadata_path": str(metadata_path),
                        "frame_index": frame_index,
                        "token": frame.get("token", ""),
                    },
                    totals["rerender_eligible"],
                    rerender_capacity,
                    rng,
                )

            for camera in ALL_CAMERAS:
                counter = counters[camera]
                counter["records"] += 1
                camera_info = cams.get(camera)
                data_path = camera_info.get("data_path") if isinstance(camera_info, Mapping) else None
                if not data_path:
                    counter["metadata_camera_missing"] += 1
                    record = {
                        "metadata_path": str(metadata_path),
                        "frame_index": frame_index,
                        "frame_token": frame.get("token", ""),
                        "log_name": frame.get("log_name", ""),
                        "camera": camera,
                        "required": camera in required_cameras,
                        "data_path": "",
                        "raster_path": "",
                        "raster_exists": False,
                    }
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                relative_path = Path(str(data_path))
                raster_path = raster_root / relative_path
                exists = raster_path.is_file()
                counter["raster_exists" if exists else "raster_missing"] += 1
                if exists:
                    content_seen[camera] += 1
                    reservoir_add(
                        content_samples[camera],
                        {
                            "metadata_path": str(metadata_path),
                            "frame_index": frame_index,
                            "frame_token": frame.get("token", ""),
                            "log_name": frame.get("log_name", ""),
                            "camera": camera,
                            "raster_path": str(raster_path),
                        },
                        content_seen[camera],
                        content_capacity,
                        rng,
                    )
                record = {
                    "metadata_path": str(metadata_path),
                    "frame_index": frame_index,
                    "frame_token": frame.get("token", ""),
                    "log_name": frame.get("log_name", ""),
                    "camera": camera,
                    "required": camera in required_cameras,
                    "data_path": str(relative_path),
                    "raster_path": str(raster_path),
                    "raster_exists": exists,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    totals["metadata_read_errors"] = len(metadata_errors)
    totals["metadata_files_readable"] = len(metadata_paths) - len(metadata_errors)
    return counters, content_samples, rerender_samples, totals, metadata_errors


def audit_content(samples: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Counter]]:
    rows: list[dict[str, Any]] = []
    counters = {camera: Counter() for camera in ALL_CAMERAS}
    for camera, camera_samples in samples.items():
        for sample in camera_samples:
            path = Path(str(sample["raster_path"]))
            row = dict(sample)
            counters[camera]["samples"] += 1
            try:
                image = np.asarray(Image.open(path).convert("RGB"))
                foreground = np.any(image != 0, axis=-1)
                all_zero = not bool(foreground.any())
                expected_raw_shape = tuple(image.shape) == (1120, 1920, 3)
                cropped = image[20:-20]
                expected_crop_shape = tuple(cropped.shape) == (1080, 1920, 3)
                counters[camera]["all_zero" if all_zero else "nonzero"] += 1
                counters[camera]["raw_shape_ok" if expected_raw_shape else "raw_shape_bad"] += 1
                counters[camera]["crop_shape_ok" if expected_crop_shape else "crop_shape_bad"] += 1
                row.update(
                    {
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "raw_shape": "x".join(map(str, image.shape)),
                        "dtype": str(image.dtype),
                        "all_zero": all_zero,
                        "nonzero_fraction": float(foreground.mean()),
                        "raw_shape_ok": expected_raw_shape,
                        "crop_shape": "x".join(map(str, cropped.shape)),
                        "crop_shape_ok": expected_crop_shape,
                        "error": "",
                    }
                )
            except Exception as error:
                counters[camera]["errors"] += 1
                row["error"] = repr(error)
            rows.append(row)
    return rows, counters


def reconstruct_scenario(frame: Mapping[str, Any], map_cache: MutableMapping[str, Any]) -> dict[str, Any]:
    map_name = str(frame["map_location"])
    if map_name not in map_cache:
        map_cache[map_name] = Scene._build_map_api(map_name)
    map_api = map_cache[map_name]
    ego_position = np.asarray(frame["ego2global_translation"], dtype=np.float64)[:2]
    ego_heading = float(Quaternion(frame["ego2global_rotation"]).yaw_pitch_roll[0])
    traffic_lights = []
    for light in frame.get("traffic_lights", []) or []:
        if len(light) >= 3:
            traffic_lights.append(light)
            continue
        lane_id, is_red = light
        lane = map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
        if lane is None:
            continue
        path = lane.baseline_path.discrete_path
        accumulated = 0.0
        selected = path[0]
        for index, point in enumerate(path[1:], start=1):
            previous = path[index - 1]
            accumulated += float(np.hypot(point.x - previous.x, point.y - previous.y))
            selected = point
            if accumulated > 8.0:
                break
        traffic_lights.append((lane_id, is_red, [selected.x - ego_position[0], selected.y - ego_position[1]]))
    anns = frame["anns"]
    if anns.get("gt_boxes_world") is not None:
        boxes_world = np.asarray(anns["gt_boxes_world"])
    else:
        # Older norm metadata stores only ego-frame gt_boxes.  The production
        # generator formed gt_boxes_world by rotating centers into the world
        # orientation (without adding the ego translation, so coordinates stay
        # relative to ego) and adding ego yaw to box yaw.
        boxes_ego = np.asarray(anns.get("gt_boxes", np.zeros((0, 7))))
        boxes_world = boxes_ego.copy().reshape(-1, 7)
        if len(boxes_world):
            ego_rotation = Quaternion(frame["ego2global_rotation"])
            boxes_world[:, :3] = (ego_rotation.rotation_matrix @ boxes_ego[:, :3].T).T
            boxes_world[:, 6] = boxes_ego[:, 6] + ego_heading
    return {
        "map_features": extract_map_features(map_api, ego_position.tolist(), radius=200),
        "ego_pos": ego_position.tolist(),
        "ego_heading": ego_heading,
        "traffic_lights": traffic_lights,
        "anns": {
            "gt_boxes_world": boxes_world,
            "gt_names": np.asarray(anns.get("gt_names", [])),
        },
    }


def jpeg_roundtrip_rgb(image: np.ndarray) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", image[:, :, ::-1])
    if not success:
        raise RuntimeError("cv2.imencode('.jpg') failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded[:, :, ::-1]


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": "x".join(map(str, reference.shape)),
            "candidate_shape": "x".join(map(str, candidate.shape)),
        }
    delta = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    different = np.any(delta != 0, axis=-1)
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    ref_fg = np.any(reference != 0, axis=-1)
    cand_fg = np.any(candidate != 0, axis=-1)
    union = int(np.logical_or(ref_fg, cand_fg).sum())
    intersection = int(np.logical_and(ref_fg, cand_fg).sum())
    return {
        "shape_match": True,
        "exact_equal": not bool(different.any()),
        "different_pixel_count": int(different.sum()),
        "different_pixel_fraction": float(different.mean()),
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": int(delta.max()),
        "psnr": float("inf") if mse == 0 else float(20 * math.log10(255.0) - 10 * math.log10(mse)),
        "foreground_iou": 1.0 if union == 0 else intersection / union,
        "reference_nonzero_fraction": float(ref_fg.mean()),
        "candidate_nonzero_fraction": float(cand_fg.mean()),
    }


def save_evidence(
    root: Path,
    sample_index: int,
    token: str,
    camera: str,
    stored: np.ndarray,
    rerendered: np.ndarray,
) -> None:
    target = root / f"{sample_index:03d}_{token}" / camera
    target.mkdir(parents=True, exist_ok=True)
    difference = np.abs(stored.astype(np.int16) - rerendered.astype(np.int16)).astype(np.uint8)
    Image.fromarray(stored).save(target / "stored_loader_crop.png")
    Image.fromarray(rerendered).save(target / "rerender_loader_crop.png")
    Image.fromarray(difference).save(target / "absolute_difference.png")
    Image.fromarray(np.concatenate([stored, rerendered, difference], axis=1)).save(target / "panel.png")


def audit_rerender(
    specs: Sequence[Mapping[str, Any]],
    raster_root: Path,
    required_cameras: Sequence[str],
    evidence_root: Path,
    evidence_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    renderer = ScenarioRenderer(camera_channel_list=list(required_cameras))
    map_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for sample_index, spec in enumerate(specs):
        try:
            metadata_path = Path(str(spec["metadata_path"]))
            with metadata_path.open("rb") as handle:
                frames = pickle.load(handle)
            frame = frames[int(spec["frame_index"])]
            rendered = renderer.observe(reconstruct_scenario(frame, map_cache))
            for camera in required_cameras:
                raster_path = raster_root / Path(frame["cams"][camera]["data_path"])
                stored_bgr = cv2.imread(str(raster_path), cv2.IMREAD_COLOR)
                if stored_bgr is None:
                    raise FileNotFoundError(raster_path)
                stored = stored_bgr[:, :, ::-1][20:-20]
                current = jpeg_roundtrip_rgb(np.asarray(rendered[camera], dtype=np.uint8))[20:-20]
                row = {
                    "sample_index": sample_index,
                    "metadata_path": str(metadata_path),
                    "frame_index": int(spec["frame_index"]),
                    "frame_token": frame.get("token", ""),
                    "log_name": frame.get("log_name", ""),
                    "camera": camera,
                    "raster_path": str(raster_path),
                    "stored_sha256": sha256_file(raster_path),
                    **image_metrics(stored, current),
                }
                rows.append(row)
                if sample_index < evidence_limit and stored.shape == current.shape:
                    save_evidence(evidence_root, sample_index, str(frame.get("token", "unknown")), camera, stored, current)
        except Exception as error:
            errors.append({**dict(spec), "error": repr(error), "traceback": traceback.format_exc()})
    return rows, errors


def summarize(
    args: argparse.Namespace,
    metadata_paths: Sequence[Path],
    path_counters: Mapping[str, Counter],
    content_counters: Mapping[str, Counter],
    totals: Counter,
    rerender_rows: Sequence[Mapping[str, Any]],
    rerender_errors: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    per_camera: dict[str, dict[str, Any]] = {}
    for camera in ALL_CAMERAS:
        path = path_counters[camera]
        content = content_counters[camera]
        records = path["records"]
        per_camera[camera] = {
            "required": camera in args.required_cameras,
            "metadata_records": records,
            "metadata_camera_missing": path["metadata_camera_missing"],
            "raster_exists": path["raster_exists"],
            "raster_missing": path["raster_missing"],
            "raster_missing_fraction": path["raster_missing"] / records if records else float("nan"),
            "content_samples": content["samples"],
            "content_all_zero": content["all_zero"],
            "content_decode_errors": content["errors"],
            "content_raw_shape_bad": content["raw_shape_bad"],
            "content_crop_shape_bad": content["crop_shape_bad"],
        }
    path_pass = totals["metadata_read_errors"] == 0 and all(
        per_camera[camera]["metadata_camera_missing"] == 0 and per_camera[camera]["raster_missing"] == 0
        for camera in args.required_cameras
    )
    content_pass = all(
        per_camera[camera]["content_samples"] > 0
        and per_camera[camera]["content_decode_errors"] == 0
        and per_camera[camera]["content_raw_shape_bad"] == 0
        and per_camera[camera]["content_crop_shape_bad"] == 0
        for camera in args.required_cameras
    )
    all_zero_samples = sum(per_camera[camera]["content_all_zero"] for camera in args.required_cameras)
    rerender_pass = bool(rerender_rows) and not rerender_errors and all(row.get("exact_equal") for row in rerender_rows)
    if path_pass and content_pass and rerender_pass:
        overall = "PASS_WITH_WARNINGS" if all_zero_samples else "PASS"
    else:
        overall = "FAIL"
    content_status = "FAIL" if not content_pass else ("PASS_WITH_WARNINGS" if all_zero_samples else "PASS")
    summary = {
        "variant": args.variant,
        "overall_status": overall,
        "path_integrity_status": "PASS" if path_pass else "FAIL",
        "content_sample_status": content_status,
        "zero_perturbation_reproduction_status": "PASS" if rerender_pass else "FAIL",
        "metadata_dir": str(args.metadata_dir),
        "raster_root": str(args.raster_root),
        "raster_root_exists": args.raster_root.is_dir(),
        "required_cameras": list(args.required_cameras),
        "metadata_files": len(metadata_paths),
        **dict(totals),
        "rerender_samples_requested": args.rerender_samples,
        "rerender_camera_rows": len(rerender_rows),
        "rerender_exact_rows": sum(bool(row.get("exact_equal")) for row in rerender_rows),
        "rerender_errors": len(rerender_errors),
        "per_camera": per_camera,
    }
    lines = [
        f"# Step 0-A — {args.variant}",
        "",
        f"- 总体：**{overall}**",
        f"- 路径完整性：**{summary['path_integrity_status']}**",
        f"- 图像内容抽样：**{summary['content_sample_status']}**",
        f"- 零扰动重渲染：**{summary['zero_perturbation_reproduction_status']}**",
        f"- metadata：`{args.metadata_dir}`",
        f"- raster：`{args.raster_root}`",
        f"- metadata 文件：{len(metadata_paths)}，可读 {totals['metadata_files_readable']}，读取错误 {totals['metadata_read_errors']}",
        f"- 有效帧：{totals['frames_audited']} / 总帧 {totals['frames_total']}",
        "",
        "| Camera | required | records | metadata missing | raster missing | content sampled | all zero | bad shape | decode errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for camera in ALL_CAMERAS:
        row = per_camera[camera]
        lines.append(
            f"| {camera} | {row['required']} | {row['metadata_records']} | {row['metadata_camera_missing']} | "
            f"{row['raster_missing']} | {row['content_samples']} | {row['content_all_zero']} | "
            f"{row['content_raw_shape_bad'] + row['content_crop_shape_bad']} | {row['content_decode_errors']} |"
        )
    lines.extend(
        [
            "",
            f"零扰动比较：{len(rerender_rows)} 个 camera rows，逐像素完全相等 "
            f"{summary['rerender_exact_rows']} 个，错误 {len(rerender_errors)} 个。",
            "",
            "说明：路径存在性覆盖全部有效帧；图像解码、全零、shape 与 SHA256 使用按相机蓄水池抽样。",
        ]
    )
    return summary, "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    args.metadata_dir = args.metadata_dir.resolve()
    args.raster_root = args.raster_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.raster_root.is_dir():
        raise FileNotFoundError(f"Raster root does not exist: {args.raster_root}")
    metadata_paths = sorted(args.metadata_dir.glob("*.pkl"))
    if args.max_metadata_files is not None:
        metadata_paths = metadata_paths[: args.max_metadata_files]
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata pkl under {args.metadata_dir}")
    rng = random.Random(args.seed)

    path_counters, content_specs, rerender_specs, totals, metadata_errors = scan_paths(
        metadata_paths,
        args.raster_root,
        args.output_dir / "manifest.jsonl.gz",
        args.variant,
        args.required_cameras,
        args.content_samples_per_camera,
        args.rerender_samples,
        rng,
    )
    write_csv(args.output_dir / "metadata_errors.csv", metadata_errors)
    content_rows, content_counters = audit_content(content_specs)
    write_csv(args.output_dir / "content_sample.csv", content_rows)
    rerender_rows, rerender_errors = audit_rerender(
        rerender_specs,
        args.raster_root,
        args.required_cameras,
        args.output_dir / "evidence",
        args.evidence_samples,
    )
    write_csv(args.output_dir / "pixel_delta.csv", rerender_rows)
    write_csv(args.output_dir / "rerender_errors.csv", rerender_errors)
    integrity_rows = []
    for camera in ALL_CAMERAS:
        integrity_rows.append({"camera": camera, **path_counters[camera], **{f"content_{k}": v for k, v in content_counters[camera].items()}})
    write_csv(args.output_dir / "integrity.csv", integrity_rows)
    summary, markdown = summarize(
        args, metadata_paths, path_counters, content_counters, totals, rerender_rows, rerender_errors
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=jsonable) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=jsonable) + "\n", encoding="utf-8")
    environment = {
        "python": sys.version.replace(os.linesep, " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": getattr(Image, "__version__", "unknown"),
        "seed": args.seed,
    }
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=jsonable))
    return 0 if summary["overall_status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
