"""Build and audit the paired raw-image cache for the alignment smoke test.

The script deliberately uses three independent, explicit sources:

* real camera files from ``sensor_blobs/trainval``;
* raster camera files from ``dataset_norm/rendered_sensor_blobs``;
* token/target records from ``cache/rap_ego``.

It never derives one root from another and never substitutes a zero image.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import shutil
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from navsim.agents.rap_dino.navsim_config import RAPConfig
from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder
from navsim.common.dataclasses import AgentInput, SensorConfig


ROOT = Path(__file__).resolve().parents[3]
CAMERA_ORDER = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0")
SAMPLING_NAMESPACE = "alignment-small-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gzip_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


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
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "nonzero_fraction": float(np.count_nonzero(array) / array.size),
        "finite_fraction": float(np.isfinite(array).sum() / array.size),
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.float().mean()),
        "std": float(tensor.float().std()),
        "finite_fraction": float(torch.isfinite(tensor).float().mean()),
    }


def _read_rgb(path: Path, crop_raster: bool) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        decoded = np.asarray(image.convert("RGB"))
    if decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"Expected an RGB image at {path}, got {decoded.shape}")
    if crop_raster:
        if decoded.shape[0] <= 40:
            raise ValueError(f"Raster image is too short to crop 20 pixels per side: {path}")
        decoded = decoded[20:-20]
    return decoded


def _hash_key(split: str, log_name: str, token: str) -> str:
    text = f"{SAMPLING_NAMESPACE}:{split}:{log_name}:{token}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_hash_round_robin(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select hash-min tokens in rounds, with at most one token per log per round."""
    by_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_log[record["log_name"]].append(record)
    for values in by_log.values():
        values.sort(key=lambda item: (item["selection_hash"], item["token"]))

    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        round_values = [
            values[round_index]
            for values in by_log.values()
            if round_index < len(values)
        ]
        if not round_values:
            break
        round_values.sort(
            key=lambda item: (item["selection_hash"], item["log_name"], item["token"])
        )
        selected.extend(round_values[: count - len(selected)])
        round_index += 1
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} samples, but only selected {len(selected)}")
    return selected


def _sensor_config() -> SensorConfig:
    return SensorConfig(
        cam_f0=[3], cam_l0=[3], cam_l1=[], cam_l2=[],
        cam_r0=[3], cam_r1=[], cam_r2=[], cam_b0=[3], lidar_pc=[]
    )


def _audit_candidate(
    log_name: str,
    token: str,
    target_path: Path,
    frame: dict[str, Any],
    split: str,
    real_root: Path,
    raster_root: Path,
    camera_order: tuple[str, ...] = CAMERA_ORDER,
) -> dict[str, Any]:
    target = _load_gzip_pickle(target_path)
    if target.get("token") != token:
        raise ValueError(f"Target token mismatch at {target_path}")
    cameras = []
    all_raster_black = True
    for camera_name in camera_order:
        relative_path = Path(frame["cams"][camera_name]["data_path"])
        real_path = real_root / relative_path
        raster_path = raster_root / relative_path
        real = _read_rgb(real_path, crop_raster=False)
        raster = _read_rgb(raster_path, crop_raster=True)
        if real.shape != raster.shape:
            raise ValueError(
                f"Decoded shape mismatch for {log_name}/{token}/{camera_name}: "
                f"real={real.shape}, raster={raster.shape}"
            )
        all_raster_black &= bool(raster.max() == 0)
        cameras.append(
            {
                "camera": camera_name,
                "relative_path": str(relative_path),
                "real": {"path": str(real_path.resolve()), "bytes": real_path.stat().st_size,
                         "sha256": _sha256(real_path), "decoded": _array_stats(real)},
                "raster": {"path": str(raster_path.resolve()), "bytes": raster_path.stat().st_size,
                           "sha256": _sha256(raster_path), "decoded_after_crop": _array_stats(raster)},
            }
        )
    if all_raster_black:
        raise ValueError(f"All four raster cameras are pure black for {log_name}/{token}")
    return {
        "split": split,
        "log_name": log_name,
        "token": token,
        "selection_hash": _hash_key(split, log_name, token),
        "target_path": str(target_path.resolve()),
        "target_sha256": _sha256(target_path),
        "target_shape": list(target["trajectory"].shape),
        "camera_order": list(camera_order),
        "cameras": cameras,
        "qualified": True,
    }


def _render_missing_back_camera(
    record: dict[str, Any],
    frame: dict[str, Any],
    raster_root: Path,
    map_apis: dict[str, Any],
) -> dict[str, Any]:
    """Render CAM_B0 with the repository's original rasterization code."""
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
        map_apis[map_location] = get_maps_api(
            os.environ["NUPLAN_MAPS_ROOT"], "nuplan-maps-v1.0", map_location
        )
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
    front_path = raster_root / frame["cams"]["CAM_F0"]["data_path"]
    front_source = _read_rgb(front_path, crop_raster=False)
    front_regenerated = rendered["CAM_F0"]
    front_mae = float(
        np.abs(front_source.astype(np.float32) - front_regenerated.astype(np.float32)).mean()
    )
    if front_mae > 1.0:
        raise RuntimeError(
            f"Renderer reproduction MAE {front_mae:.4f} exceeds 1/255 for {record['token']}"
        )

    back_path = raster_root / frame["cams"]["CAM_B0"]["data_path"]
    generated = False
    if not back_path.exists():
        back_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(back_path), rendered["CAM_B0"][:, :, ::-1]):
            raise OSError(f"cv2.imwrite failed for {back_path}")
        generated = True
    decoded_back = _read_rgb(back_path, crop_raster=False)
    return {
        "log_name": record["log_name"],
        "token": record["token"],
        "path": str(back_path.resolve()),
        "generated_now": generated,
        "sha256": _sha256(back_path),
        "renderer": "process_data.helpers.renderer.ScenarioRenderer",
        "renderer_camera_channels": ["CAM_F0", "CAM_B0"],
        "front_reproduction_mae_0_to_255": front_mae,
        "decoded_back": _array_stats(decoded_back),
    }


def _build_selected(
    record: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    frame_index: int,
    real_root: Path,
    raster_root: Path,
    cache_root: Path,
    input_root: Path,
) -> dict[str, Any]:
    if frame_index < 3:
        raise ValueError(f"Token {record['token']} lacks three history frames")
    history = frame_rows[frame_index - 3 : frame_index + 1]
    if history[-1]["token"] != record["token"]:
        raise AssertionError("Current-frame token changed during cache construction")

    agent_input = AgentInput.from_scene_dict_list(
        scene_dict_list=history,
        sensor_blobs_path=real_root,
        rendered_sensor_blobs_path=raster_root,
        num_history_frames=4,
        sensor_config=_sensor_config(),
        strict_camera_loading=True,
    )
    features = RAPFeatureBuilder(RAPConfig()).compute_features(agent_input)
    if not bool(features["camera_valid"]):
        raise RuntimeError(f"camera_valid is false for selected token {record['token']}")
    if not all(torch.isfinite(value).all() for value in features.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError(f"Non-finite feature tensor for selected token {record['token']}")

    sample_cache = cache_root / record["log_name"] / record["token"]
    sample_cache.mkdir(parents=True, exist_ok=True)
    feature_path = sample_cache / "rap_feature.gz"
    with gzip.open(feature_path, "wb", compresslevel=1) as file:
        pickle.dump(features, file)
    target_path = Path(record["target_path"])
    cached_target_path = sample_cache / "rap_target.gz"
    shutil.copy2(target_path, cached_target_path)

    reloaded_features = _load_gzip_pickle(feature_path)
    for key, value in features.items():
        if isinstance(value, torch.Tensor) and not torch.equal(value, reloaded_features[key]):
            raise RuntimeError(f"Feature cache round-trip mismatch: {record['token']} {key}")
    if _sha256(target_path) != _sha256(cached_target_path):
        raise RuntimeError(f"Target byte mismatch for {record['token']}")

    sample_input = input_root / record["split"] / record["token"]
    for modality in ("real", "raster"):
        (sample_input / modality).mkdir(parents=True, exist_ok=True)
    for camera in record["cameras"]:
        camera_name = camera["camera"].lower()
        shutil.copy2(camera["real"]["path"], sample_input / "real" / f"{camera_name}.jpg")
        shutil.copy2(camera["raster"]["path"], sample_input / "raster" / f"{camera_name}.jpg")

    target = _load_gzip_pickle(target_path)
    with (sample_input / "target.json").open("w", encoding="utf-8") as file:
        json.dump(_jsonable(target), file, ensure_ascii=False, indent=2)
    selected_audit = {
        **record,
        "camera_valid": bool(features["camera_valid"]),
        "feature_cache_path": str(feature_path.resolve()),
        "feature_cache_sha256": _sha256(feature_path),
        "cached_target_path": str(cached_target_path.resolve()),
        "cached_target_sha256": _sha256(cached_target_path),
        "target_byte_identical": True,
        "tensor_stats": {
            key: _tensor_stats(value)
            for key, value in features.items()
            if isinstance(value, torch.Tensor)
        },
        "ego_status": _jsonable(features["ego_status"]),
        "lidar2img": _jsonable(features["lidar2img"]),
    }
    with (sample_input / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(selected_audit, file, ensure_ascii=False, indent=2)
    return selected_audit


def build(args: argparse.Namespace) -> None:
    split_config = yaml.safe_load(args.split_config.read_text(encoding="utf-8"))
    train_logs, val_logs = set(split_config["train_logs"]), set(split_config["val_logs"])
    if train_logs & val_logs:
        raise RuntimeError("Official train/val log lists overlap")

    targets = sorted(args.target_root.glob("*/*/rap_target.gz"))
    log_to_targets: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for path in targets:
        log_to_targets[path.parent.parent.name].append((path.parent.name, path))

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    frames_by_log: dict[str, list[dict[str, Any]]] = {}
    frame_indices: dict[tuple[str, str], int] = {}
    for log_name, token_paths in sorted(log_to_targets.items()):
        split = "train" if log_name in train_logs else "val" if log_name in val_logs else None
        if split is None:
            failures.extend(
                {"log_name": log_name, "token": token, "reason": "log_not_in_official_train_or_val"}
                for token, _ in token_paths
            )
            continue
        log_path = args.log_root / f"{log_name}.pkl"
        with log_path.open("rb") as file:
            frames = pickle.load(file)
        frames_by_log[log_name] = frames
        token_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, frame in enumerate(frames):
            token_to_indices[frame["token"]].append(index)
        for token, target_path in token_paths:
            try:
                indices = token_to_indices[token]
                if len(indices) != 1:
                    raise ValueError(f"metadata match count is {len(indices)}, expected 1")
                frame_index = indices[0]
                frame_indices[(log_name, token)] = frame_index
                # The original renderer output only F0/L0/R0. Eligibility is
                # established on those existing files before the missing B0 is
                # generated for the selected smoke subset.
                candidates.append(_audit_candidate(
                    log_name, token, target_path, frames[frame_index], split,
                    args.real_root, args.raster_root,
                    camera_order=("CAM_F0", "CAM_L0", "CAM_R0"),
                ))
            except Exception as error:
                failures.append({"log_name": log_name, "token": token, "reason": repr(error)})

    qualified_train = [item for item in candidates if item["split"] == "train"]
    qualified_val = [item for item in candidates if item["split"] == "val"]
    selected_train = select_hash_round_robin(qualified_train, args.train_count)
    selected_val = select_hash_round_robin(qualified_val, args.val_count)
    preselected = selected_train + selected_val

    args.input_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    generated_back = []
    if args.generate_missing_back_raster:
        map_apis: dict[str, Any] = {}
        for record in preselected:
            frame = frames_by_log[record["log_name"]][
                frame_indices[(record["log_name"], record["token"])]
            ]
            generated_back.append(
                _render_missing_back_camera(record, frame, args.raster_root, map_apis)
            )
    selected = []
    for record in preselected:
        frame = frames_by_log[record["log_name"]][
            frame_indices[(record["log_name"], record["token"])]
        ]
        selected.append(_audit_candidate(
            record["log_name"], record["token"], Path(record["target_path"]), frame,
            record["split"], args.real_root, args.raster_root,
        ))
    selected_train = [item for item in selected if item["split"] == "train"]
    selected_val = [item for item in selected if item["split"] == "val"]
    built = [
        _build_selected(
            record, frames_by_log[record["log_name"]],
            frame_indices[(record["log_name"], record["token"])],
            args.real_root, args.raster_root, args.cache_root, args.input_root,
        )
        for record in selected
    ]

    token_manifest = {
        "algorithm": "SHA256 namespace + per-log hash-min round-robin",
        "namespace": SAMPLING_NAMESPACE,
        "selection_eligibility_camera_order": ["CAM_F0", "CAM_L0", "CAM_R0"],
        "selected_cache_camera_order": list(CAMERA_ORDER),
        "back_camera_generation": generated_back,
        "train_tokens": [item["token"] for item in selected_train],
        "val_tokens": [item["token"] for item in selected_val],
        "train_logs": [item["log_name"] for item in selected_train],
        "val_logs": [item["log_name"] for item in selected_val],
        "qualified_candidates": candidates,
        "excluded_candidates": failures,
    }
    manifest_path = args.input_root / "token_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(token_manifest, file, ensure_ascii=False, indent=2)

    rows = [
        {
            "split": item["split"], "log_name": item["log_name"], "token": item["token"],
            "selection_hash": item["selection_hash"],
            "feature_cache_path": item["feature_cache_path"],
            "target_cache_path": item["cached_target_path"],
        }
        for item in built
    ]
    with (args.input_root / "samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected_train_logs = {item["log_name"] for item in selected_train}
    selected_val_logs = {item["log_name"] for item in selected_val}
    audit = {
        "status": "passed",
        "scope": "32/16 alignment smoke subset",
        "sources": {
            "real_root": str(args.real_root.resolve()),
            "raster_root": str(args.raster_root.resolve()),
            "target_root": str(args.target_root.resolve()),
            "log_root": str(args.log_root.resolve()),
            "split_config": str(args.split_config.resolve()),
        },
        "counts": {
            "source_targets": len(targets), "qualified_train": len(qualified_train),
            "qualified_val": len(qualified_val), "excluded": len(failures),
            "selected_train": len(selected_train), "selected_val": len(selected_val),
        },
        "checks": {
            "independent_real_and_raster_roots": args.real_root.resolve() != args.raster_root.resolve(),
            "selected_train_val_token_overlap": bool(
                set(token_manifest["train_tokens"]) & set(token_manifest["val_tokens"])
            ),
            "selected_train_val_log_overlap": bool(selected_train_logs & selected_val_logs),
            "all_selected_camera_valid": all(item["camera_valid"] for item in built),
            "all_selected_targets_byte_identical": all(item["target_byte_identical"] for item in built),
            "all_selected_tensors_finite": all(
                stats["finite_fraction"] == 1.0
                for item in built for stats in item["tensor_stats"].values()
            ),
            "all_generated_front_reproduction_mae_le_1": all(
                item["front_reproduction_mae_0_to_255"] <= 1.0
                for item in generated_back
            ),
        },
        "token_manifest": str(manifest_path.resolve()),
        "paired_cache": str(args.cache_root.resolve()),
    }
    if audit["checks"]["selected_train_val_token_overlap"]:
        raise RuntimeError("Selected train/val tokens overlap")
    if audit["checks"]["selected_train_val_log_overlap"]:
        raise RuntimeError("Selected train/val logs overlap")
    with (args.input_root / "input_audit.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-root", type=Path, default=Path(
        "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/trainval"))
    parser.add_argument("--raster-root", type=Path, default=ROOT / "dataset_norm/rendered_sensor_blobs")
    parser.add_argument("--target-root", type=Path, default=ROOT / "cache/rap_ego")
    parser.add_argument("--log-root", type=Path, default=Path(
        "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/navsim_logs/trainval"))
    parser.add_argument("--split-config", type=Path, default=ROOT /
        "navsim/planning/script/config/training/default_train_val_test_log_split.yaml")
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs/alignment_small/input_data")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs/alignment_small/paired_cache")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument(
        "--generate-missing-back-raster",
        action="store_true",
        help="Generate selected CAM_B0 rasters with the repository renderer after validating F0 reproduction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
