#!/usr/bin/env python3
"""Render one nuScenes CAM_FRONT sample with NAVSIM and native cameras.

The scene geometry and annotations are held fixed.  Render A injects the
NAVSIM CAM_F0 calibration saved by ``render_navsim_scene.py``; Render B uses
the nuScenes CAM_FRONT calibrated_sensor record.  Both paths call RAP's
original ScenarioRenderer.  Its hard-coded camera translation adjustment is
pre-compensated so the effective calibration is exactly the requested one.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers.renderer import ScenarioRenderer  # noqa: E402


DEFAULT_SAMPLE_TOKEN = "c235638ed66145988d17f9d0601923f2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-render one nuScenes sample with NAVSIM/native cameras."
    )
    parser.add_argument(
        "--nuscenes-root",
        type=Path,
        default=REPO_ROOT / "nuscenes-mini" / "nuscenes",
    )
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--sample-token", default=DEFAULT_SAMPLE_TOKEN)
    parser.add_argument(
        "--navsim-camera",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs/poster_pairs/navsim"
            / "navsim_c82e3ccb51965d6a_camera_params.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "poster_pairs" / "nuscenes_cross_camera",
    )
    parser.add_argument("--map-radius", type=float, default=80.0)
    return parser.parse_args()


def load_table(version_root: Path, name: str) -> list[dict[str, Any]]:
    return json.loads((version_root / f"{name}.json").read_text())


def by_token(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["token"]: record for record in records}


def quaternion_rotation(q: list[float]) -> np.ndarray:
    """nuScenes quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = map(float, q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_yaw(rotation: np.ndarray) -> float:
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def localize_points(
    points_xy: np.ndarray, ego_translation: np.ndarray, ego_rotation: np.ndarray
) -> np.ndarray:
    points_xyz = np.column_stack(
        [points_xy, np.full(len(points_xy), ego_translation[2], dtype=np.float64)]
    )
    return ((ego_rotation.T @ (points_xyz - ego_translation).T).T)[:, :2]


def extract_map_features(
    map_data: dict[str, Any],
    ego_translation: np.ndarray,
    ego_rotation: np.ndarray,
    radius: float,
) -> dict[str, dict[str, Any]]:
    nodes = {record["token"]: record for record in map_data["node"]}
    lines = {record["token"]: record for record in map_data["line"]}
    polygons = {record["token"]: record for record in map_data["polygon"]}
    features: dict[str, dict[str, Any]] = {}

    def node_xy(tokens: list[str]) -> np.ndarray:
        return np.asarray([[nodes[t]["x"], nodes[t]["y"]] for t in tokens], dtype=np.float64)

    def keep(points_local: np.ndarray) -> bool:
        return len(points_local) >= 2 and bool(
            np.any(np.linalg.norm(points_local, axis=1) <= radius)
        )

    for layer_name in ("lane", "lane_connector"):
        for record in map_data[layer_name]:
            polygon = polygons.get(record.get("polygon_token"))
            if polygon is None:
                continue
            local = localize_points(
                node_xy(polygon["exterior_node_tokens"]), ego_translation, ego_rotation
            )
            if keep(local):
                features[f"{layer_name}:{record['token']}"] = {
                    "type": "LANE",
                    "polygon": local.astype(np.float32),
                }

    for record in map_data["ped_crossing"]:
        polygon = polygons.get(record.get("polygon_token"))
        if polygon is None:
            continue
        local = localize_points(
            node_xy(polygon["exterior_node_tokens"]), ego_translation, ego_rotation
        )
        if keep(local):
            features[f"ped_crossing:{record['token']}"] = {
                "type": "CROSSWALK",
                "polygon": local.astype(np.float32),
            }

    for layer_name in ("lane_divider", "road_divider"):
        for record in map_data[layer_name]:
            line = lines.get(record.get("line_token"))
            if line is None:
                continue
            local = localize_points(
                node_xy(line["node_tokens"]), ego_translation, ego_rotation
            )
            if keep(local):
                features[f"{layer_name}:{record['token']}"] = {
                    "type": "BOUNDARY",
                    "polyline": local.astype(np.float32),
                }

    return features


def traffic_light_positions(map_data: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = {record["token"]: record for record in map_data["node"]}
    lines = {record["token"]: record for record in map_data["line"]}
    positions = []
    for light in map_data["traffic_light"]:
        pose = light.get("pose", {})
        x = float(pose.get("tx") or 0.0)
        y = float(pose.get("ty") or 0.0)
        z = float(pose.get("tz") or 5.0)
        if x == 0.0 and y == 0.0:
            line = lines.get(light.get("line_token"))
            line_nodes = (
                [nodes[token] for token in line["node_tokens"] if token in nodes]
                if line
                else []
            )
            if line_nodes:
                x = float(np.mean([node["x"] for node in line_nodes]))
                y = float(np.mean([node["y"] for node in line_nodes]))
        if x != 0.0 or y != 0.0:
            positions.append({"token": light["token"], "global_xyz": np.array([x, y, z])})
    return positions


def project_visible_traffic_lights(
    lights: list[dict[str, Any]],
    ego_translation: np.ndarray,
    ego_rotation: np.ndarray,
    camera_translation: np.ndarray,
    camera_rotation: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    visible = []
    for light in lights:
        ego_xyz = ego_rotation.T @ (light["global_xyz"] - ego_translation)
        camera_xyz = camera_rotation.T @ (ego_xyz - camera_translation)
        if camera_xyz[2] <= 0.1 or camera_xyz[2] >= 120.0:
            continue
        projected = intrinsics @ camera_xyz
        u = float(projected[0] / projected[2])
        v = float(projected[1] / projected[2])
        if 0 <= u < width and 0 <= v < height:
            visible.append(
                {
                    "token": light["token"],
                    "depth_m": float(camera_xyz[2]),
                    "pixel_uv_native": [u, v],
                    "ego_xyz": ego_xyz,
                }
            )
    return sorted(visible, key=lambda item: item["depth_m"])


def render(
    scenario: dict[str, Any], camera: dict[str, np.ndarray], width: int, height: int
) -> np.ndarray:
    renderer = ScenarioRenderer(
        camera_channel_list=["CAM_F0"], width=width, height=height
    )
    prepared_camera = {
        key: np.asarray(value).copy() for key, value in camera.items()
    }
    # ScenarioRenderer.observe() subsequently applies x -= 2 and z += 0.8.
    # Pre-compensate that virtual-camera shift so the effective extrinsics are
    # exactly the NAVSIM/native calibration being compared.
    prepared_camera["sensor2lidar_translation"] += np.array(
        [2.0, 0.0, -0.8], dtype=np.float64
    )
    renderer.camera_models["CAM_F0"] = prepared_camera
    return renderer.observe(scenario)["CAM_F0"]


def add_preview_label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (20, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    args = parse_args()
    root = args.nuscenes_root.resolve()
    version_root = root / args.version
    output_dir = args.output_dir.resolve()

    samples = by_token(load_table(version_root, "sample"))
    scenes = by_token(load_table(version_root, "scene"))
    logs = by_token(load_table(version_root, "log"))
    sample_data = load_table(version_root, "sample_data")
    calibrations = by_token(load_table(version_root, "calibrated_sensor"))
    sensors = by_token(load_table(version_root, "sensor"))
    ego_poses = by_token(load_table(version_root, "ego_pose"))
    annotation_records = load_table(version_root, "sample_annotation")
    annotations = by_token(annotation_records)
    instances = by_token(load_table(version_root, "instance"))
    categories = by_token(load_table(version_root, "category"))

    sample = samples[args.sample_token]
    scene = scenes[sample["scene_token"]]
    log = logs[scene["log_token"]]
    native_sd = next(
        record
        for record in sample_data
        if record["sample_token"] == sample["token"]
        and record["is_key_frame"]
        and sensors[
            calibrations[record["calibrated_sensor_token"]]["sensor_token"]
        ]["channel"]
        == "CAM_FRONT"
    )
    native_calibration = calibrations[native_sd["calibrated_sensor_token"]]
    ego_pose = ego_poses[native_sd["ego_pose_token"]]

    ego_translation = np.asarray(ego_pose["translation"], dtype=np.float64)
    ego_rotation = quaternion_rotation(ego_pose["rotation"])
    native_camera_rotation = quaternion_rotation(native_calibration["rotation"])
    native_camera_translation = np.asarray(
        native_calibration["translation"], dtype=np.float64
    )
    native_intrinsics = np.asarray(
        native_calibration["camera_intrinsic"], dtype=np.float64
    )

    box_rows = []
    names = []
    sample_annotation_tokens = [
        record["token"]
        for record in annotation_records
        if record["sample_token"] == sample["token"]
    ]
    for annotation_token in sample_annotation_tokens:
        annotation = annotations[annotation_token]
        global_center = np.asarray(annotation["translation"], dtype=np.float64)
        local_center = ego_rotation.T @ (global_center - ego_translation)
        global_box_rotation = quaternion_rotation(annotation["rotation"])
        local_yaw = rotation_yaw(ego_rotation.T @ global_box_rotation)
        width, length, height = map(float, annotation["size"])
        box_rows.append(
            [*local_center.tolist(), length, width, height, local_yaw]
        )
        instance = instances[annotation["instance_token"]]
        names.append(categories[instance["category_token"]]["name"])
    boxes = np.asarray(box_rows, dtype=np.float32).reshape(-1, 7)

    map_path = root / "maps" / "expansion" / f"{log['location']}.json"
    map_data = json.loads(map_path.read_text())
    map_features = extract_map_features(
        map_data, ego_translation, ego_rotation, args.map_radius
    )
    visible_lights = project_visible_traffic_lights(
        traffic_light_positions(map_data),
        ego_translation,
        ego_rotation,
        native_camera_translation,
        native_camera_rotation,
        native_intrinsics,
        int(native_sd["width"]),
        int(native_sd["height"]),
    )

    # The chosen frame was visually inspected and contains green signals.
    # nuScenes does not provide the active light state. Keep the closest map
    # heads as selection evidence, but do not invent a state in the raster.
    selected_lights = visible_lights[:2]

    scenario = {
        "ego_heading": 0.0,
        "traffic_lights": [],
        "map_features": map_features,
        "anns": {"gt_boxes_world": boxes, "gt_names": np.asarray(names)},
    }

    with np.load(args.navsim_camera) as navsim_file:
        navsim_camera = {
            "intrinsics": navsim_file["intrinsics"],
            "sensor2lidar_rotation": navsim_file["sensor2lidar_rotation"],
            "sensor2lidar_translation": navsim_file["sensor2lidar_translation"],
            "distortion": navsim_file["distortion"],
        }
    native_camera = {
        "intrinsics": native_intrinsics,
        "sensor2lidar_rotation": native_camera_rotation,
        "sensor2lidar_translation": native_camera_translation,
        "distortion": np.zeros(5, dtype=np.float64),
    }

    raster_a_rgb = render(scenario, navsim_camera, width=1920, height=1120)
    raster_b_rgb = render(
        scenario,
        native_camera,
        width=int(native_sd["width"]),
        height=int(native_sd["height"]),
    )

    real_source = root / native_sd["filename"]
    real_bgr = cv2.imread(str(real_source), cv2.IMREAD_COLOR)
    if real_bgr is None:
        raise RuntimeError(f"OpenCV could not read {real_source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    token = sample["token"]
    real_output = output_dir / f"nuscenes_{token}_real.jpg"
    raster_a_output = output_dir / f"nuscenes_{token}_raster_A_navsim_camera.png"
    raster_b_output = output_dir / f"nuscenes_{token}_raster_B_native_camera.png"
    preview_output = output_dir / f"nuscenes_{token}_real_A_B_preview.jpg"
    navsim_overlay_output = (
        output_dir / f"nuscenes_{token}_navsim_camera_overlay_diagnostic.jpg"
    )
    overlay_output = output_dir / f"nuscenes_{token}_native_overlay_diagnostic.jpg"
    camera_output = output_dir / f"nuscenes_{token}_camera_comparison.npz"
    record_output = output_dir / f"nuscenes_{token}_render_record.json"

    shutil.copy2(real_source, real_output)
    cv2.imwrite(str(raster_a_output), raster_a_rgb[:, :, ::-1])
    cv2.imwrite(str(raster_b_output), raster_b_rgb[:, :, ::-1])

    target_size = (real_bgr.shape[1], real_bgr.shape[0])
    raster_a_preview = cv2.resize(
        raster_a_rgb[:, :, ::-1], target_size, interpolation=cv2.INTER_NEAREST
    )
    raster_b_bgr = raster_b_rgb[:, :, ::-1]
    preview = cv2.hconcat(
        [
            add_preview_label(real_bgr, "nuScenes Real / CAM_FRONT"),
            add_preview_label(raster_a_preview, "Raster A / NAVSIM camera"),
            add_preview_label(raster_b_bgr, "Raster B / nuScenes camera"),
        ]
    )
    cv2.imwrite(str(preview_output), preview)

    navsim_overlay = real_bgr.copy()
    navsim_mask = np.any(raster_a_preview != 0, axis=2)
    navsim_overlay[navsim_mask] = cv2.addWeighted(
        real_bgr[navsim_mask], 0.35, raster_a_preview[navsim_mask], 0.65, 0
    )
    cv2.imwrite(str(navsim_overlay_output), navsim_overlay)

    overlay = real_bgr.copy()
    mask = np.any(raster_b_bgr != 0, axis=2)
    overlay[mask] = cv2.addWeighted(real_bgr[mask], 0.35, raster_b_bgr[mask], 0.65, 0)
    cv2.imwrite(str(overlay_output), overlay)

    np.savez(
        camera_output,
        navsim_intrinsics=navsim_camera["intrinsics"],
        navsim_sensor2ego_rotation=navsim_camera["sensor2lidar_rotation"],
        navsim_sensor2ego_translation=navsim_camera["sensor2lidar_translation"],
        nuscenes_intrinsics=native_camera["intrinsics"],
        nuscenes_sensor2ego_rotation=native_camera["sensor2lidar_rotation"],
        nuscenes_sensor2ego_translation=native_camera["sensor2lidar_translation"],
    )

    record = {
        "dataset": "nuScenes v1.0-mini",
        "scene": scene["name"],
        "sample_token": token,
        "location": log["location"],
        "real_image_has_visible_green_traffic_lights": True,
        "traffic_light_selection": {
            "nearby_visible_map_lights": len(visible_lights),
            "closest_visual_evidence_lights": len(selected_lights),
            "rendered_lights": 0,
            "state_source": "visible in the real image; nuScenes has no active-state annotation",
            "lights": [
                {
                    "token": item["token"],
                    "depth_m": item["depth_m"],
                    "pixel_uv_native": item["pixel_uv_native"],
                }
                for item in selected_lights
            ],
        },
        "annotation_boxes": len(boxes),
        "map_features": len(map_features),
        "rap_internal_translation_adjustment_compensated": True,
        "effective_camera_extrinsics": "exact requested NAVSIM/native calibration",
        "render_A": {
            "camera": "NAVSIM CAM_F0 from step 5",
            "shape_hwc": list(raster_a_rgb.shape),
            "nonzero_fraction": float(
                np.count_nonzero(raster_a_rgb) / raster_a_rgb.size
            ),
            "path": str(raster_a_output),
        },
        "render_B": {
            "camera": "nuScenes native CAM_FRONT",
            "shape_hwc": list(raster_b_rgb.shape),
            "nonzero_fraction": float(
                np.count_nonzero(raster_b_rgb) / raster_b_rgb.size
            ),
            "path": str(raster_b_output),
        },
        "real_image": str(real_output),
        "preview": str(preview_output),
        "navsim_camera_overlay_diagnostic": {
            "path": str(navsim_overlay_output),
            "real_weight": 0.35,
            "raster_weight": 0.65,
            "display_normalization": "NAVSIM raster resized 1920x1120 -> 1600x900",
            "purpose": "cross-camera misalignment diagnostic",
        },
        "native_overlay_diagnostic": str(overlay_output),
        "camera_comparison_npz": str(camera_output),
    }
    record_output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"render_record: {record_output}")


if __name__ == "__main__":
    main()
