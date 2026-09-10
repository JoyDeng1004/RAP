#!/usr/bin/env python3
"""Render one extracted NAVSIM CAM_F0 frame with RAP's rasterizer.

This lightweight adapter consumes the 14-frame subset produced by
``extract_one_scene.py``.  It uses the selected NAVSIM frame's calibrated
CAM_F0 intrinsics/extrinsics.  When a nuPlan ``map.gpkg`` is supplied, nearby
lanes, crosswalks, boundaries, and traffic-light lane connectors are read via
GDAL's ``ogr2ogr`` command without requiring the full nuPlan Python stack.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers.renderer import ScenarioRenderer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one NAVSIM front-camera raster with RAP."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "navsim_one_scene" / "dataset",
        help="Root containing scene_manifest.json, navsim_logs, and sensor_blobs.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Metadata/sensor split. Defaults to scene_manifest.json.",
    )
    parser.add_argument(
        "--log-name",
        default=None,
        help="Log basename. Defaults to scene_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "poster_pairs" / "navsim",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame in the extracted log; defaults to the current frame (history - 1).",
    )
    parser.add_argument(
        "--map-gpkg",
        type=Path,
        default=None,
        help="Optional nuPlan map.gpkg used to include HD-map and traffic lights.",
    )
    parser.add_argument("--map-radius", type=float, default=120.0)
    return parser.parse_args()


def _ogr_geojson(
    map_path: Path,
    *,
    layer: str | None = None,
    center: np.ndarray | None = None,
    radius: float | None = None,
    sql: str | None = None,
    simplify: float = 0.25,
) -> dict[str, Any]:
    command = ["ogr2ogr", "-f", "GeoJSON", "/vsistdout/", str(map_path)]
    if layer is not None:
        command.append(layer)
    command.extend(["-t_srs", "EPSG:32611"])
    if center is not None and radius is not None:
        x, y = map(float, center[:2])
        command.extend(
            [
                "-spat",
                str(x - radius),
                str(y - radius),
                str(x + radius),
                str(y + radius),
                "-spat_srs",
                "EPSG:32611",
            ]
        )
    if sql is not None:
        command.extend(["-dialect", "SQLITE", "-sql", sql])
    command.extend(["-simplify", str(simplify)])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _polygon_rings(geometry: dict[str, Any]) -> list[np.ndarray]:
    if geometry["type"] == "Polygon":
        return [np.asarray(geometry["coordinates"][0], dtype=np.float32)]
    if geometry["type"] == "MultiPolygon":
        return [
            np.asarray(polygon[0], dtype=np.float32)
            for polygon in geometry["coordinates"]
        ]
    return []


def _polylines(geometry: dict[str, Any]) -> list[np.ndarray]:
    if geometry["type"] == "LineString":
        return [np.asarray(geometry["coordinates"], dtype=np.float32)]
    if geometry["type"] == "MultiLineString":
        return [np.asarray(line, dtype=np.float32) for line in geometry["coordinates"]]
    return []


def load_map_features(
    map_path: Path, center: np.ndarray, radius: float
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    specifications = (
        ("lanes_polygons", "LANE", _polygon_rings, "polygon"),
        ("crosswalks", "CROSSWALK", _polygon_rings, "polygon"),
        ("boundaries", "BOUNDARY", _polylines, "polyline"),
    )
    for layer, feature_type, converter, coordinate_key in specifications:
        collection = _ogr_geojson(
            map_path, layer=layer, center=center, radius=radius
        )
        for feature_index, feature in enumerate(collection["features"]):
            geometry = feature.get("geometry")
            if geometry is None:
                continue
            for part_index, coordinates in enumerate(converter(geometry)):
                if len(coordinates) < 2:
                    continue
                local = coordinates.copy()
                local[:, :2] -= center[None, :2]
                features[f"{layer}_{feature_index}_{part_index}"] = {
                    "type": feature_type,
                    coordinate_key: local,
                }
    return features


def _point_along_line(points: np.ndarray, distance: float) -> np.ndarray:
    if len(points) == 1:
        return points[0]
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    traversed = 0.0
    for index, segment_length in enumerate(lengths):
        if segment_length > 0 and traversed + segment_length >= distance:
            ratio = (distance - traversed) / segment_length
            return points[index] + ratio * (points[index + 1] - points[index])
        traversed += float(segment_length)
    return points[-1]


def load_traffic_lights(
    map_path: Path,
    center: np.ndarray,
    statuses: list[tuple[int, bool]],
) -> list[tuple[int, bool, list[float]]]:
    if not statuses:
        return []
    identifiers = sorted({int(status[0]) for status in statuses})
    identifier_sql = ",".join(map(str, identifiers))
    collection = _ogr_geojson(
        map_path,
        sql=(
            "SELECT CAST(fid AS TEXT) AS source_id, geom "
            f"FROM lane_connectors WHERE fid IN ({identifier_sql})"
        ),
        simplify=0.05,
    )
    connector_paths: dict[int, np.ndarray] = {}
    for feature in collection["features"]:
        geometry = feature.get("geometry")
        lines = _polylines(geometry) if geometry is not None else []
        if lines:
            connector_paths[int(feature["properties"]["source_id"])] = lines[0]

    lights = []
    for lane_connector_id, is_red in statuses:
        path = connector_paths.get(int(lane_connector_id))
        if path is None:
            continue
        position = _point_along_line(path, 8.0)[:2] - center[:2]
        lights.append((int(lane_connector_id), bool(is_red), position.tolist()))
    return lights


def boxes_in_map_aligned_frame(boxes: np.ndarray, ego_yaw: float) -> np.ndarray:
    result = boxes.copy()
    cosine, sine = math.cos(ego_yaw), math.sin(ego_yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    result[:, :2] = result[:, :2] @ rotation.T
    result[:, 6] += ego_yaw
    return result


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    manifest_path = dataset_root / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    split = args.split or manifest["split"]
    log_name = args.log_name or manifest["log_name"]
    log_path = dataset_root / "navsim_logs" / split / f"{log_name}.pkl"
    with log_path.open("rb") as stream:
        frames = pickle.load(stream)

    frame_index = (
        args.frame_index
        if args.frame_index is not None
        else int(manifest["num_history_frames"]) - 1
    )
    if not 0 <= frame_index < len(frames):
        raise IndexError(f"frame-index {frame_index} is outside [0, {len(frames) - 1}]")

    frame = frames[frame_index]
    cam = frame["cams"]["CAM_F0"]
    real_path = dataset_root / "sensor_blobs" / split / cam["data_path"]
    if not real_path.is_file():
        raise FileNotFoundError(real_path)

    boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float32).copy()
    names = np.asarray(frame["anns"]["gt_names"])
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"Expected gt_boxes with shape (N, >=7), got {boxes.shape}")

    map_features: dict[str, dict[str, Any]] = {}
    traffic_lights: list[tuple[int, bool, list[float]]] = []
    ego_yaw = 0.0
    render_boxes = boxes
    if args.map_gpkg is not None:
        map_path = args.map_gpkg.resolve()
        if not map_path.is_file():
            raise FileNotFoundError(map_path)
        center = np.asarray(frame["ego2global_translation"], dtype=np.float64)
        ego_matrix = np.asarray(frame["ego2global"], dtype=np.float64)
        ego_yaw = math.atan2(float(ego_matrix[1, 0]), float(ego_matrix[0, 0]))
        map_features = load_map_features(map_path, center, args.map_radius)
        traffic_lights = load_traffic_lights(
            map_path, center, list(frame.get("traffic_lights", []))
        )
        # NAVSIM gt_boxes is ego-local, whereas RAP's map renderer uses axes
        # parallel to the global UTM frame and centered on the ego position.
        render_boxes = boxes_in_map_aligned_frame(boxes, ego_yaw)

    scenario = {
        "ego_heading": ego_yaw,
        "traffic_lights": traffic_lights,
        "map_features": map_features,
        "anns": {
            "gt_boxes_world": render_boxes,
            "gt_names": names,
        },
    }

    renderer = ScenarioRenderer(camera_channel_list=["CAM_F0"])
    # Use this NAVSIM frame's calibration rather than the renderer module's
    # approximate static rig. ScenarioRenderer.observe applies RAP's original
    # virtual shift (x -= 2 m, z += 0.8 m), so pre-compensate it here to make
    # the effective extrinsics equal to the NAVSIM calibration.
    camera_model = {
        "distortion": np.asarray(cam["distortion"], dtype=np.float64).copy(),
        "intrinsics": np.asarray(cam["cam_intrinsic"], dtype=np.float64).copy(),
        "sensor2lidar_rotation": np.asarray(
            cam["sensor2lidar_rotation"], dtype=np.float64
        ).copy(),
        "sensor2lidar_translation": np.asarray(
            cam["sensor2lidar_translation"], dtype=np.float64
        ).copy(),
    }
    prepared_camera_model = {
        key: np.asarray(value).copy() for key, value in camera_model.items()
    }
    prepared_camera_model["sensor2lidar_translation"] += np.array(
        [2.0, 0.0, -0.8], dtype=np.float64
    )
    renderer.camera_models["CAM_F0"] = prepared_camera_model

    raster_rgb = renderer.observe(scenario)["CAM_F0"]
    real_bgr = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise RuntimeError(f"OpenCV could not read {real_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    token = frame["token"]
    real_output = output_dir / f"navsim_{token}_real.jpg"
    raster_output = output_dir / f"navsim_{token}_raster_navsim_camera.png"
    preview_output = output_dir / f"navsim_{token}_pair_preview.jpg"
    calibration_output = output_dir / f"navsim_{token}_camera_params.npz"
    record_output = output_dir / f"navsim_{token}_render_record.json"

    shutil.copy2(real_path, real_output)
    if not cv2.imwrite(str(raster_output), raster_rgb[:, :, ::-1]):
        raise RuntimeError(f"Failed to write {raster_output}")

    # Preview only: resize the raster to the real image height.  The original
    # scientific outputs above remain untouched.
    raster_preview_bgr = cv2.resize(
        raster_rgb[:, :, ::-1],
        (real_bgr.shape[1], real_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    preview = np.concatenate([real_bgr, raster_preview_bgr], axis=1)
    if not cv2.imwrite(str(preview_output), preview):
        raise RuntimeError(f"Failed to write {preview_output}")

    np.savez(
        calibration_output,
        intrinsics=camera_model["intrinsics"],
        sensor2lidar_rotation=camera_model["sensor2lidar_rotation"],
        sensor2lidar_translation=camera_model["sensor2lidar_translation"],
        distortion=camera_model["distortion"],
    )

    record = {
        "dataset": "NAVSIM/OpenScene mini",
        "log_name": log_name,
        "token": token,
        "scene_token": frame.get("scene_token"),
        "frame_index": frame_index,
        "map_location": frame.get("map_location"),
        "camera": "CAM_F0",
        "camera_parameters": "NAVSIM frame calibration",
        "rap_camera_translation_adjustment_m": {"x": -2.0, "y": 0.0, "z": 0.8},
        "rap_camera_translation_adjustment_compensated": True,
        "hd_map_features_included": bool(map_features),
        "map_feature_count": len(map_features),
        "traffic_light_status_count": len(frame.get("traffic_lights", [])),
        "traffic_lights_rendered": len(traffic_lights),
        "traffic_light_encoding": "red when is_red=True; green otherwise",
        "annotation_boxes": int(len(boxes)),
        "real_shape_hwc": list(real_bgr.shape),
        "raster_shape_hwc": list(raster_rgb.shape),
        "raster_nonzero_fraction": float(np.count_nonzero(raster_rgb) / raster_rgb.size),
        "real_image": str(real_output),
        "raster_image": str(raster_output),
        "pair_preview": str(preview_output),
        "camera_params_npz": str(calibration_output),
    }
    record_output.write_text(json.dumps(record, indent=2))

    print(json.dumps(record, indent=2))
    print(f"render_record: {record_output}")


if __name__ == "__main__":
    main()
