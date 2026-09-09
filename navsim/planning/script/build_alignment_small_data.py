#!/usr/bin/env python3
"""Build the deterministic Stage-A paired real/raster dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import yaml
from PIL import Image

from navsim.agents.rap_dino.navsim_config import RAPConfig
from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder
from navsim.common.dataclasses import AgentInput, SensorConfig


ROOT = Path(__file__).resolve().parents[3]
CAMERA_ORDER = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0")
ELIGIBILITY_CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")
MAP_VERSION = "nuplan-maps-v1.0"


class VersionPollutionError(RuntimeError):
    """Raised when an unregistered final B0 file already exists."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_hash(split: str, log_name: str, token: str) -> str:
    """Canonical sha256(split, log_name, token), with NUL-delimited UTF-8 fields."""
    return hashlib.sha256(b"\0".join(value.encode() for value in (split, log_name, token))).hexdigest()


def _log_tiebreak_hash(log_name: str) -> str:
    return hashlib.sha256(log_name.encode()).hexdigest()


def _load_gzip_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    return {
        "decoded_shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        "finite_fraction": float(np.isfinite(array).sum() / array.size),
    }


def _read_rgb(path: Path, crop_raster: bool = False) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        decoded = np.asarray(image.convert("RGB"))
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"expected RGB image at {path}, got {decoded.shape}")
    if crop_raster:
        if decoded.shape[0] <= 40:
            raise ValueError(f"raster too short for fixed crop: {path}")
        decoded = decoded[20:-20]
    if not np.isfinite(decoded).all():
        raise ValueError(f"non-finite decoded image: {path}")
    return decoded


def _quota_map(records: list[dict[str, Any]], count: int) -> dict[str, int]:
    by_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_log[record["log_name"]].append(record)
    if count < 0 or count > len(records):
        raise ValueError(f"count={count}, available={len(records)}")
    if not by_log and count == 0:
        return {}
    total = len(records)
    exact = {log: len(values) / total * count for log, values in by_log.items()}
    quotas = {log: max(1, math.floor(value)) for log, value in exact.items()}
    if sum(quotas.values()) > count:
        raise ValueError("target count is too small to allocate at least one sample per log")
    remaining = count - sum(quotas.values())
    order = sorted(
        by_log,
        key=lambda log: (-(exact[log] - math.floor(exact[log])), _log_tiebreak_hash(log)),
    )
    for log in order:
        if remaining == 0:
            break
        if quotas[log] < len(by_log[log]):
            quotas[log] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"unable to distribute {remaining} proportional quota slots")
    assert sum(quotas.values()) == count
    assert min(quotas.values()) >= 1
    assert all(quotas[log] <= len(by_log[log]) for log in by_log)
    return quotas


def select_hash_proportional(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select deterministic per-log proportional hash-min samples."""
    by_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_log[record["log_name"]].append(record)
    for values in by_log.values():
        values.sort(key=lambda item: (item["selection_hash"], item["token"]))
    quotas = _quota_map(records, count)
    selected = [record for log, values in by_log.items() for record in values[: quotas[log]]]
    selected.sort(key=lambda item: (item["selection_hash"], item["token"], item["log_name"]))
    assert len(selected) == count
    return selected


def deterministic_topup(
    candidates: list[dict[str, Any]],
    initial_selected: list[dict[str, Any]],
    split: str,
    attempt: Callable[[dict[str, Any]], Any],
    manifest_path: Path,
    max_replacement_rate: float = 0.01,
) -> tuple[list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    """Attempt selected records and deterministically replace failures."""
    by_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_log[record["log_name"]].append(record)
    for values in by_log.values():
        values.sort(key=lambda item: (item["selection_hash"], item["token"]))
    global_order = sorted(
        candidates, key=lambda item: (item["selection_hash"], item["token"], item["log_name"])
    )
    used = {record["token"] for record in initial_selected}
    failed: set[str] = set()
    selected = list(initial_selected)
    results: list[Any] = []
    replacements: list[dict[str, Any]] = []
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if manifest_path.exists() else "x"
    with manifest_path.open(mode, encoding="utf-8", buffering=1) as manifest:
        for index, original in enumerate(list(selected)):
            current = original
            while True:
                try:
                    result = attempt(current)
                    selected[index] = current
                    results.append(result)
                    break
                except VersionPollutionError:
                    raise
                except Exception as error:
                    failed.add(current["token"])
                    same_log = next(
                        (
                            item for item in by_log[current["log_name"]]
                            if item["token"] not in used and item["token"] not in failed
                        ),
                        None,
                    )
                    fallback = same_log is None
                    replacement = same_log or next(
                        (
                            item for item in global_order
                            if item["token"] not in used and item["token"] not in failed
                        ),
                        None,
                    )
                    if replacement is None:
                        raise RuntimeError(f"candidate pool exhausted while replacing {current['token']}")
                    used.add(replacement["token"])
                    row = {
                        "failed_log_name": current["log_name"],
                        "failed_token": current["token"],
                        "failure_reason": repr(error),
                        "fallback_cross_log": fallback,
                        "replacement_log_name": replacement["log_name"],
                        "replacement_token": replacement["token"],
                        "split": split,
                    }
                    replacements.append(row)
                    manifest.write(json.dumps(row, sort_keys=True) + "\n")
                    manifest.flush()
                    os.fsync(manifest.fileno())
                    if len(replacements) / len(initial_selected) > max_replacement_rate:
                        raise RuntimeError(
                            f"{split} replacement rate {len(replacements) / len(initial_selected):.6f} "
                            f"exceeds {max_replacement_rate:.6f}"
                        )
                    current = replacement
    assert len(selected) == len(initial_selected) == len(results)
    return selected, results, replacements


def atomic_write_validated_image(
    final_path: Path, rgb: np.ndarray, validator: Callable[[Path], Any]
) -> Any:
    """Write beside the final path, validate, then atomically publish."""
    import cv2

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".{final_path.stem}.tmp-{uuid.uuid4().hex}{final_path.suffix}")
    try:
        if not cv2.imwrite(str(temp_path), rgb[:, :, ::-1]):
            raise OSError(f"cv2.imwrite failed: {temp_path}")
        validation = validator(temp_path)
        if final_path.exists():
            raise VersionPollutionError(f"unregistered final B0 already exists: {final_path}")
        os.replace(temp_path, final_path)
        return validation
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _sensor_config() -> SensorConfig:
    return SensorConfig(
        cam_f0=[3], cam_l0=[3], cam_l1=[], cam_l2=[],
        cam_r0=[3], cam_r1=[], cam_r2=[], cam_b0=[3], lidar_pc=[]
    )


def _actor_count_30m(frame: dict[str, Any]) -> int:
    boxes = np.asarray(frame["anns"]["gt_boxes"])
    names = np.asarray(frame["anns"]["gt_names"])
    if not len(boxes):
        return 0
    keep_type = np.isin(names, ["vehicle", "pedestrian", "bicycle"])
    return int(np.logical_and(keep_type, np.linalg.norm(boxes[:, :2], axis=1) <= 30.0).sum())


def _audit_candidate(
    log_name: str,
    token: str,
    target_path: Path,
    frame: dict[str, Any],
    frame_index: int,
    frame_count: int,
    split: str,
    real_root: Path,
    raster_root: Path,
    camera_order: Iterable[str] = ELIGIBILITY_CAMERAS,
    include_camera_details: bool = False,
) -> dict[str, Any]:
    if frame_index < 3 or frame_index + 10 >= frame_count:
        raise ValueError("incomplete 4-history/10-future window")
    if frame.get("is_valid", True) is False or not frame["roadblock_ids"]:
        raise ValueError("invalid frame or missing route")
    target = _load_gzip_pickle(target_path)
    if target.get("token") != token:
        raise ValueError(f"target token mismatch at {target_path}")
    trajectory = torch.as_tensor(target["trajectory"])
    if trajectory.shape != (10, 3) or not torch.isfinite(trajectory).all():
        raise ValueError(f"invalid target trajectory at {target_path}: {trajectory.shape}")
    camera_details = []
    raster_nonzero = False
    for camera in camera_order:
        relative = Path(frame["cams"][camera]["data_path"])
        if len(relative.parts) < 3 or relative.parts[0] != log_name or relative.parts[1] != camera:
            raise ValueError(f"noncanonical camera path {camera}: {relative}")
        real_path, raster_path = real_root / relative, raster_root / relative
        real = _read_rgb(real_path)
        raster = _read_rgb(raster_path, crop_raster=True)
        if real.shape != raster.shape:
            raise ValueError(f"shape mismatch {camera}: real={real.shape}, raster={raster.shape}")
        raster_nonzero |= bool(np.count_nonzero(raster))
        if include_camera_details:
            camera_details.append(
                {
                    "camera": camera,
                    "relative_path": relative.as_posix(),
                    "real_path": str(real_path.resolve()),
                    "real_sha256": _sha256(real_path),
                    "raster_path": str(raster_path.resolve()),
                    "raster_sha256": _sha256(raster_path),
                }
            )
    if not raster_nonzero:
        raise ValueError("all checked raster cameras are black")
    endpoint = trajectory[-1].tolist()
    record = {
        "actor_count_30m": _actor_count_30m(frame),
        "driving_command": _jsonable(frame["driving_command"]),
        "endpoint_dx": float(endpoint[0]),
        "endpoint_dy": float(endpoint[1]),
        "endpoint_dyaw": float(endpoint[2]),
        "log_name": log_name,
        "map_location": frame["map_location"],
        "selection_hash": _selection_hash(split, log_name, token),
        "split": split,
        "target_path": str(target_path.resolve()),
        "target_sha256": _sha256(target_path),
        "token": token,
    }
    if include_camera_details:
        record["camera_order"] = list(camera_order)
        record["cameras"] = camera_details
    return record


def _source_metadata_sha256(frame: dict[str, Any]) -> str:
    return hashlib.sha256(pickle.dumps(frame, protocol=4)).hexdigest()


def _render_b0(
    record: dict[str, Any],
    frame: dict[str, Any],
    raster_root: Path,
    map_apis: dict[str, Any],
    renderer_commit: str,
    manifest_handle: Any,
) -> dict[str, Any]:
    os.environ.setdefault("NUPLAN_DB_PATH", "/tmp")
    os.environ.setdefault("NUPLAN_SENSOR_PATH", str(raster_root))
    process_data_root = ROOT / "process_data"
    if str(process_data_root) not in sys.path:
        sys.path.insert(0, str(process_data_root))
    import cv2
    from pyquaternion import Quaternion
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from create_openscene_metadata import extract_map_features
    from helpers.renderer import ScenarioRenderer

    map_location = frame["map_location"]
    if map_location not in map_apis:
        map_apis[map_location] = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"], MAP_VERSION, map_location)
    center = np.asarray(frame["ego2global_translation"][:2], dtype=np.float64)
    rotation = Quaternion(frame["ego2global_rotation"])
    yaw = rotation.yaw_pitch_roll[0]
    local_boxes = np.asarray(frame["anns"]["gt_boxes"])
    world_boxes = local_boxes.copy()
    if len(local_boxes):
        world_boxes[:, :3] = (rotation.rotation_matrix @ local_boxes[:, :3].T).T
        world_boxes[:, 6] = local_boxes[:, 6] + yaw
    scenario = {
        "map_features": extract_map_features(map_apis[map_location], center, radius=200),
        "ego_pos": center,
        "ego_heading": yaw,
        "traffic_lights": frame["traffic_lights"],
        "anns": {**frame["anns"], "gt_boxes_world": world_boxes},
    }
    rendered = ScenarioRenderer(camera_channel_list=["CAM_F0", "CAM_B0"]).observe(scenario)
    existing_f0_path = raster_root / frame["cams"]["CAM_F0"]["data_path"]
    existing_f0 = _read_rgb(existing_f0_path)
    regenerated_f0 = np.asarray(rendered["CAM_F0"])
    if existing_f0.shape != regenerated_f0.shape:
        raise RuntimeError(f"F0 reproduction shape mismatch: {record['token']}")
    f0_mae = float(np.abs(existing_f0.astype(np.float32) - regenerated_f0.astype(np.float32)).mean())
    if f0_mae > 1.0:
        raise RuntimeError(f"F0 reproduction MAE {f0_mae:.6f} > 1.0 for {record['token']}")
    ok, encoded_f0 = cv2.imencode(".jpg", regenerated_f0[:, :, ::-1])
    if not ok:
        raise RuntimeError("failed to encode regenerated F0")

    relative = Path(frame["cams"]["CAM_B0"]["data_path"])
    if len(relative.parts) < 3 or relative.parts[0] != record["log_name"] or relative.parts[1] != "CAM_B0":
        raise ValueError(f"noncanonical CAM_B0 path: {relative}")
    back_path = raster_root / relative
    if back_path.exists():
        raise VersionPollutionError(f"unregistered or duplicate B0: {back_path}")
    generated = np.asarray(rendered["CAM_B0"])

    def validate_temp(path: Path) -> dict[str, Any]:
        decoded = _read_rgb(path)
        if decoded.shape != existing_f0.shape or decoded.dtype != np.uint8:
            raise ValueError(f"invalid generated B0 shape/dtype: {decoded.shape}/{decoded.dtype}")
        if not np.isfinite(decoded).all() or not np.count_nonzero(decoded):
            raise ValueError("generated B0 is non-finite or black")
        return _array_stats(decoded)

    stats = atomic_write_validated_image(back_path, generated, validate_temp)
    row = {
        "camera": "CAM_B0",
        "existing_f0_sha256": _sha256(existing_f0_path),
        "f0_mae_0_to_255": f0_mae,
        "frame_token": record["token"],
        "generated_b0_sha256": _sha256(back_path),
        "log_name": record["log_name"],
        "map_version": MAP_VERSION,
        "regenerated_f0_sha256": hashlib.sha256(encoded_f0.tobytes()).hexdigest(),
        "relative_path": relative.as_posix(),
        "renderer_git_commit": renderer_commit,
        "source_metadata_sha256": _source_metadata_sha256(frame),
        **stats,
    }
    manifest_handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_handle.flush()
    os.fsync(manifest_handle.fileno())
    return row


def _cache_selected(
    record: dict[str, Any],
    frames: list[dict[str, Any]],
    frame_index: int,
    real_root: Path,
    raster_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    history = frames[frame_index - 3 : frame_index + 1]
    agent_input = AgentInput.from_scene_dict_list(
        history, real_root, 4, _sensor_config(),
        rendered_sensor_blobs_path=raster_root, strict_camera_loading=True,
    )
    features = RAPFeatureBuilder(RAPConfig()).compute_features(agent_input)
    if not bool(features["camera_valid"]):
        raise RuntimeError(f"camera_valid=false: {record['token']}")
    if not all(torch.isfinite(value).all() for value in features.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError(f"non-finite cached feature: {record['token']}")
    log_dir = cache_root / record["log_name"]
    final_dir = log_dir / record["token"]
    temp_dir = log_dir / f".{record['token']}.tmp-{uuid.uuid4().hex}"
    if final_dir.exists():
        raise VersionPollutionError(f"paired cache destination exists: {final_dir}")
    try:
        temp_dir.mkdir(parents=True)
        feature_path = temp_dir / "rap_feature.gz"
        with gzip.open(feature_path, "wb", compresslevel=1) as handle:
            pickle.dump(features, handle)
        target_source = Path(record["target_path"])
        target_path = temp_dir / "rap_target.gz"
        os.link(target_source, target_path)
        if _sha256(target_source) != _sha256(target_path):
            raise RuntimeError("target hard-link checksum mismatch")
        os.replace(temp_dir, final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {
        **record,
        "camera_valid": True,
        "feature_cache_path": str((final_dir / "rap_feature.gz").resolve()),
        "feature_cache_sha256": _sha256(final_dir / "rap_feature.gz"),
        "target_cache_path": str((final_dir / "rap_target.gz").resolve()),
        "target_cache_sha256": _sha256(final_dir / "rap_target.gz"),
    }


def _inventory(args: argparse.Namespace) -> dict[str, Any]:
    split_config = yaml.safe_load(args.split_config.read_text())
    train_logs, val_logs = set(split_config["train_logs"]), set(split_config["val_logs"])
    test_logs = set(split_config["test_logs"])
    assert not train_logs & val_logs and not (train_logs | val_logs) & test_logs
    targets = sorted(args.target_root.glob("*/*/rap_target.gz"))
    by_log: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for path in targets:
        by_log[path.parent.parent.name].append((path.parent.name, path))
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen_tokens: set[str] = set()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for log_name, token_paths in sorted(by_log.items()):
            split = "train" if log_name in train_logs else "val" if log_name in val_logs else None
            if split is None:
                failures.extend(
                    {"log_name": log_name, "token": token, "reason": "outside official train/val"}
                    for token, _ in token_paths
                )
                continue
            frames = pickle.load(open(args.log_root / f"{log_name}.pkl", "rb"))
            token_to_indices: dict[str, list[int]] = defaultdict(list)
            for index, frame in enumerate(frames):
                token_to_indices[frame["token"]].append(index)

            def inspect(item: tuple[str, Path]) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
                token, target_path = item
                try:
                    indices = token_to_indices[token]
                    if len(indices) != 1:
                        raise ValueError(f"metadata match count={len(indices)}")
                    index = indices[0]
                    return _audit_candidate(
                        log_name, token, target_path, frames[index], index, len(frames), split,
                        args.real_root, args.raster_root,
                    ), None
                except Exception as error:
                    return None, {"log_name": log_name, "token": token, "reason": repr(error)}

            for candidate, failure in executor.map(inspect, token_paths):
                if candidate is not None:
                    if candidate["token"] in seen_tokens:
                        raise RuntimeError(f"duplicate qualified token: {candidate['token']}")
                    seen_tokens.add(candidate["token"])
                    candidates.append(candidate)
                else:
                    assert failure is not None
                    failures.append(failure)
    candidates.sort(key=lambda item: (item["split"], item["log_name"], item["selection_hash"], item["token"]))
    missing_logs = {
        "train": sorted(train_logs - {x["log_name"] for x in candidates if x["split"] == "train"}),
        "val": sorted(val_logs - {x["log_name"] for x in candidates if x["split"] == "val"}),
    }
    return {
        "algorithm": "sha256(NUL-joined split, log_name, token) + proportional per-log quotas",
        "excluded_candidates": failures,
        "missing_qualified_logs": missing_logs,
        "qualified_candidates": candidates,
        "qualified_train": sum(x["split"] == "train" for x in candidates),
        "qualified_val": sum(x["split"] == "val" for x in candidates),
        "source_target_count": len(targets),
        "sources": {
            "log_root": str(args.log_root.resolve()),
            "raster_root": str(args.raster_root.resolve()),
            "real_root": str(args.real_root.resolve()),
            "split_config": str(args.split_config.resolve()),
            "split_config_sha256": _sha256(args.split_config),
            "target_root": str(args.target_root.resolve()),
        },
    }


def _load_frames(log_root: Path, records: list[dict[str, Any]]) -> tuple[dict[str, list[Any]], dict[tuple[str, str], int]]:
    frames_by_log, indices = {}, {}
    for log_name in sorted({x["log_name"] for x in records}):
        frames = pickle.load(open(log_root / f"{log_name}.pkl", "rb"))
        frames_by_log[log_name] = frames
        for index, frame in enumerate(frames):
            indices[(log_name, frame["token"])] = index
    return frames_by_log, indices


def _content_checksum(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> None:
    if args.dry_run:
        if args.dry_run_out is None:
            raise ValueError("--dry-run requires --dry-run-out")
        inventory = _inventory(args)
        args.dry_run_out.parent.mkdir(parents=True, exist_ok=True)
        with args.dry_run_out.open("w", encoding="utf-8") as handle:
            json.dump(inventory, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({k: inventory[k] for k in ("qualified_train", "qualified_val", "source_target_count")}))
        return

    if args.input_root is None or args.cache_root is None or args.topup_manifest is None:
        raise ValueError("build mode requires --input-root, --cache-root and --topup-manifest")
    candidate_path = args.input_root / "candidate_pool.json"
    inventory = json.loads(candidate_path.read_text()) if candidate_path.is_file() else _inventory(args)
    expected_sources = inventory["sources"]
    assert expected_sources["split_config_sha256"] == _sha256(args.split_config)
    for key, path in (("real_root", args.real_root), ("raster_root", args.raster_root),
                      ("target_root", args.target_root), ("log_root", args.log_root)):
        assert expected_sources[key] == str(path.resolve()), (key, expected_sources[key], path)
    candidates = inventory["qualified_candidates"]
    train_candidates = [x for x in candidates if x["split"] == "train"]
    val_candidates = [x for x in candidates if x["split"] == "val"]
    initial_train = select_hash_proportional(train_candidates, args.train_count)
    initial_val = select_hash_proportional(val_candidates, args.val_count)
    assert [x["token"] for x in select_hash_proportional(list(reversed(train_candidates)), args.train_count)] == [x["token"] for x in initial_train]
    assert [x["token"] for x in select_hash_proportional(list(reversed(val_candidates)), args.val_count)] == [x["token"] for x in initial_val]

    if args.cache_root.exists() and any(args.cache_root.iterdir()):
        raise VersionPollutionError(f"paired cache root is non-empty: {args.cache_root}")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.input_root.mkdir(parents=True, exist_ok=True)
    b0_manifest_path = args.input_root / "b0_generation_manifest.jsonl"
    for path in (b0_manifest_path, args.topup_manifest):
        if path.exists() and path.stat().st_size:
            raise VersionPollutionError(f"manifest is non-empty: {path}")
    frames_by_log, frame_indices = _load_frames(args.log_root, candidates)
    renderer_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    map_apis: dict[str, Any] = {}
    generated_rows: list[dict[str, Any]] = []
    built_rows: list[dict[str, Any]] = []

    with b0_manifest_path.open("x", encoding="utf-8", buffering=1) as b0_manifest:
        def attempt(record: dict[str, Any]) -> dict[str, Any]:
            frames = frames_by_log[record["log_name"]]
            index = frame_indices[(record["log_name"], record["token"])]
            if not args.generate_missing_back_raster:
                raise RuntimeError("build mode requires --generate-missing-back-raster")
            generated_rows.append(
                _render_b0(record, frames[index], args.raster_root, map_apis, renderer_commit, b0_manifest)
            )
            audited = _audit_candidate(
                record["log_name"], record["token"], Path(record["target_path"]),
                frames[index], index, len(frames), record["split"], args.real_root,
                args.raster_root, CAMERA_ORDER, include_camera_details=True,
            )
            built = _cache_selected(audited, frames, index, args.real_root, args.raster_root, args.cache_root)
            built_rows.append(built)
            return built

        selected_train, train_results, train_replacements = deterministic_topup(
            train_candidates, initial_train, "train", attempt, args.topup_manifest
        )
        selected_val, val_results, val_replacements = deterministic_topup(
            val_candidates, initial_val, "val", attempt, args.topup_manifest
        )
    built = train_results + val_results
    assert len(built) == args.train_count + args.val_count
    assert len({x["token"] for x in built}) == len(built)
    selected_train = [x for x in built if x["split"] == "train"]
    selected_val = [x for x in built if x["split"] == "val"]
    quotas = {"train": _quota_map(train_candidates, args.train_count), "val": _quota_map(val_candidates, args.val_count)}

    token_manifest = {
        "algorithm": inventory["algorithm"],
        "camera_order": list(CAMERA_ORDER),
        "missing_qualified_logs": inventory["missing_qualified_logs"],
        "qualified_candidates": candidates,
        "quotas": quotas,
        "train_logs": sorted({x["log_name"] for x in selected_train}),
        "train_tokens": [x["token"] for x in selected_train],
        "val_logs": sorted({x["log_name"] for x in selected_val}),
        "val_tokens": [x["token"] for x in selected_val],
    }
    token_manifest_path = args.input_root / "stage_a_token_manifest.json"
    token_manifest_path.write_text(json.dumps(token_manifest, indent=2, sort_keys=True) + "\n")
    dataset_manifest = {
        "camera_order": list(CAMERA_ORDER),
        "samples": built,
        "train_count": len(selected_train),
        "val_count": len(selected_val),
    }
    dataset_manifest_path = args.input_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n")
    paired_files = sorted(args.cache_root.glob("*/*/rap_*.gz"))
    paired_checksum = {
        "algorithm": "sha256(relative_path_length || relative_path || file_bytes)",
        "file_count": len(paired_files),
        "sha256": _content_checksum(args.cache_root, paired_files),
    }
    (args.input_root / "paired_cache_checksum.json").write_text(
        json.dumps(paired_checksum, indent=2, sort_keys=True) + "\n"
    )
    split_config = yaml.safe_load(args.split_config.read_text())
    test_logs = set(split_config["test_logs"])
    audit = {
        "checks": {
            "camera_order": list(CAMERA_ORDER) == ["CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0"],
            "exact_counts": len(selected_train) == args.train_count and len(selected_val) == args.val_count,
            "f0_mae_le_1": all(row["f0_mae_0_to_255"] <= 1.0 for row in generated_rows),
            "no_duplicate_b0": len({row["frame_token"] for row in generated_rows}) == len(generated_rows),
            "no_test_logs": not ({x["log_name"] for x in built} & test_logs),
            "order_independent": True,
            "replacement_rate_le_1pct": (
                len(train_replacements) / args.train_count <= 0.01
                and len(val_replacements) / args.val_count <= 0.01
            ),
        },
        "counts": {
            "generated_b0": len(generated_rows),
            "qualified_train": len(train_candidates),
            "qualified_val": len(val_candidates),
            "selected_train": len(selected_train),
            "selected_val": len(selected_val),
            "train_replacements": len(train_replacements),
            "val_replacements": len(val_replacements),
        },
        "paired_cache_checksum": paired_checksum,
        "status": "passed",
    }
    if not all(audit["checks"].values()):
        audit["status"] = "failed"
        raise RuntimeError(json.dumps(audit, sort_keys=True))
    (args.input_root / "input_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--raster-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--val-count", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-out", type=Path)
    parser.add_argument("--topup-manifest", type=Path)
    parser.add_argument("--generate-missing-back-raster", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if not args.dry_run and (args.train_count is None or args.val_count is None):
        parser.error("build mode requires --train-count and --val-count")
    return args


if __name__ == "__main__":
    build(parse_args())
