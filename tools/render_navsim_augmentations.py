#!/usr/bin/env python3
"""Render RAP's two scene-level augmentations for one extracted NAVSIM frame.

The implementation mirrors the geometry in:

* process_data/create_openscene_metadata_purturbed.py
* process_data/create_openscene_metadata_aug.py

Unlike a 2-D image warp, both augmentations rebuild the scene in a changed ego
frame and invoke ScenarioRenderer again.  The frame's calibrated NAVSIM CAM_F0
model is kept fixed across panels so the comparison isolates the augmentation.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers.renderer import ScenarioRenderer  # noqa: E402
from tools.render_navsim_scene import (  # noqa: E402
    load_map_features,
    load_traffic_lights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            REPO_ROOT
            / "navsim_one_scene/dataset/navsim_logs/poster"
            / "2021.06.09.14.58.55_veh-35_01894_02311.pkl"
        ),
    )
    parser.add_argument(
        "--map-gpkg",
        type=Path,
        default=(
            REPO_ROOT
            / "navsim_one_scene/dataset/maps/us-nv-las-vegas-strip"
            / "9.15.1915/map.gpkg"
        ),
    )
    parser.add_argument(
        "--original-raster",
        type=Path,
        default=None,
        help="Existing original Raster. Defaults to the poster_pairs NAVSIM output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/poster_pairs/navsim/augmentations",
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--map-radius", type=float, default=120.0)
    return parser.parse_args()


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def ego_pose(frame: dict[str, Any]) -> tuple[np.ndarray, float]:
    center = np.asarray(frame["ego2global_translation"], dtype=np.float64).copy()
    matrix = np.asarray(frame["ego2global"], dtype=np.float64)
    yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    return center, yaw


def boxes_in_absolute_world(
    frame: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centers/yaws in world coordinates plus names and track tokens."""
    boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float64)
    names = np.asarray(frame["anns"]["gt_names"])
    tracks = np.asarray(frame["anns"]["track_tokens"])
    center, ego_yaw = ego_pose(frame)
    cosine, sine = math.cos(ego_yaw), math.sin(ego_yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)

    world_boxes = boxes.copy()
    world_boxes[:, :2] = center[:2] + boxes[:, :2] @ rotation.T
    world_boxes[:, 2] = center[2] + boxes[:, 2]
    world_boxes[:, 6] = np.array(
        [normalize_angle(ego_yaw + float(yaw)) for yaw in boxes[:, 6]]
    )
    return world_boxes, names, tracks, boxes


def track_world_xy(frame: dict[str, Any], track_token: str) -> np.ndarray | None:
    tracks = np.asarray(frame["anns"]["track_tokens"]).astype(str)
    matches = np.flatnonzero(tracks == track_token)
    if not len(matches):
        return None
    boxes, _, _, _ = boxes_in_absolute_world(frame)
    return boxes[int(matches[0]), :2]


def select_cross_agent(
    frames: list[dict[str, Any]], frame_index: int
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    """Select the nearest vehicle satisfying the augmentation script's filters."""
    frame = frames[frame_index]
    world_boxes, names, tracks, local_boxes = boxes_in_absolute_world(frame)
    metrics: list[dict[str, Any]] = []

    for index in np.flatnonzero(names.astype(str) == "vehicle"):
        index = int(index)
        track = str(tracks[index])
        distance = float(np.linalg.norm(local_boxes[index, :2]))
        window = range(frame_index - 3, frame_index + 11)
        window_valid = all(
            0 <= j < len(frames)
            and track in np.asarray(frames[j]["anns"]["track_tokens"]).astype(str)
            for j in window
        )
        mean_cv_error = None
        if window_valid:
            current = track_world_xy(frames[frame_index], track)
            following = track_world_xy(frames[frame_index + 1], track)
            dt = (
                float(frames[frame_index + 1]["timestamp"])
                - float(frames[frame_index]["timestamp"])
            ) / 1e6
            if current is not None and following is not None and dt > 0:
                velocity = (following - current) / dt
                errors = []
                for step in range(1, 11):
                    truth = track_world_xy(frames[frame_index + step], track)
                    if truth is None:
                        errors = []
                        break
                    prediction = current + velocity * (step * 0.5)
                    errors.append(float(np.linalg.norm(prediction - truth)))
                if errors:
                    mean_cv_error = float(np.mean(errors))

        eligible = bool(
            window_valid and mean_cv_error is not None and mean_cv_error > 0.5
        )
        metrics.append(
            {
                "box_index": index,
                "track_token": track,
                "distance_from_original_ego_m": distance,
                "present_in_14_frame_window": window_valid,
                "mean_constant_velocity_error_m": mean_cv_error,
                "passes_source_filters": eligible,
            }
        )

    eligible_metrics = [metric for metric in metrics if metric["passes_source_filters"]]
    pool = eligible_metrics or metrics
    if not pool:
        raise RuntimeError("No vehicle annotation is available for cross-agent synthesis")
    selected = min(pool, key=lambda metric: metric["distance_from_original_ego_m"])
    return int(selected["box_index"]), selected, metrics


def navsim_camera_model(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    cam = frame["cams"]["CAM_F0"]
    model = {
        "distortion": np.asarray(cam["distortion"], dtype=np.float64).copy(),
        "intrinsics": np.asarray(cam["cam_intrinsic"], dtype=np.float64).copy(),
        "sensor2lidar_rotation": np.asarray(
            cam["sensor2lidar_rotation"], dtype=np.float64
        ).copy(),
        "sensor2lidar_translation": np.asarray(
            cam["sensor2lidar_translation"], dtype=np.float64
        ).copy(),
    }
    # Undo ScenarioRenderer.observe's hard-coded x -= 2, z += 0.8 shift.
    model["sensor2lidar_translation"] += np.array([2.0, 0.0, -0.8])
    return model


def render_view(
    *,
    frame: dict[str, Any],
    map_path: Path,
    camera_model: dict[str, np.ndarray],
    viewpoint_center: np.ndarray,
    viewpoint_yaw: float,
    world_boxes: np.ndarray,
    names: np.ndarray,
    map_radius: float,
) -> tuple[np.ndarray, dict[str, int]]:
    local_world_aligned_boxes = world_boxes.copy()
    local_world_aligned_boxes[:, :3] -= viewpoint_center[None, :3]
    map_features = load_map_features(map_path, viewpoint_center, map_radius)
    traffic_lights = load_traffic_lights(
        map_path,
        viewpoint_center,
        list(frame.get("traffic_lights", [])),
    )
    scenario = {
        "ego_heading": viewpoint_yaw,
        "traffic_lights": traffic_lights,
        "map_features": map_features,
        "anns": {
            "gt_boxes_world": local_world_aligned_boxes.astype(np.float32),
            "gt_names": names,
        },
    }
    renderer = ScenarioRenderer(camera_channel_list=["CAM_F0"])
    renderer.camera_models["CAM_F0"] = {
        key: value.copy() for key, value in camera_model.items()
    }
    image = renderer.observe(scenario)["CAM_F0"]
    return image, {
        "agents": int(len(world_boxes)),
        "map_features": int(len(map_features)),
        "traffic_lights": int(len(traffic_lights)),
    }


def labelled_panel(image_bgr: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    header_height = 92
    panel = np.zeros(
        (image_bgr.shape[0] + header_height, image_bgr.shape[1], 3), dtype=np.uint8
    )
    panel[:header_height] = (28, 28, 28)
    panel[header_height:] = image_bgr
    cv2.putText(
        panel,
        title,
        (30, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        subtitle,
        (30, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata.resolve()
    map_path = args.map_gpkg.resolve()
    output_dir = (args.output_dir / args.token).resolve()
    original_path = (
        args.original_raster.resolve()
        if args.original_raster is not None
        else (
            REPO_ROOT
            / "outputs/poster_pairs/navsim"
            / f"navsim_{args.token}_raster_navsim_camera.png"
        ).resolve()
    )
    for path in (metadata_path, map_path, original_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with metadata_path.open("rb") as stream:
        frames = pickle.load(stream)
    matches = [index for index, frame in enumerate(frames) if frame["token"] == args.token]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frame for {args.token}, found {len(matches)}")
    frame_index = matches[0]
    frame = frames[frame_index]
    original_bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise RuntimeError(f"OpenCV could not read {original_path}")

    world_boxes, names, tracks, _ = boxes_in_absolute_world(frame)
    original_center, original_yaw = ego_pose(frame)
    camera_model = navsim_camera_model(frame)

    # Recovery-oriented perturbation: match get_ego_params()' first three draws.
    rng = random.Random(args.seed)
    delta_x = rng.uniform(-0.5, 0.5)
    delta_y = rng.uniform(-0.5, 0.5)
    delta_yaw_deg = rng.uniform(-15.0, 15.0)
    recovery_center = original_center.copy()
    recovery_center[:2] += [delta_x, delta_y]
    recovery_yaw = normalize_angle(original_yaw + math.radians(delta_yaw_deg))
    recovery_rgb, recovery_counts = render_view(
        frame=frame,
        map_path=map_path,
        camera_model=camera_model,
        viewpoint_center=recovery_center,
        viewpoint_yaw=recovery_yaw,
        world_boxes=world_boxes,
        names=names,
        map_radius=args.map_radius,
    )

    # Cross-agent synthesis: use the nearest neighbor passing the source's
    # 14-frame persistence and non-constant-motion filters.
    selected_index, selected_metric, all_metrics = select_cross_agent(
        frames, frame_index
    )
    cross_center = world_boxes[selected_index, :3].copy()
    cross_yaw = float(world_boxes[selected_index, 6])
    keep = np.arange(len(world_boxes)) != selected_index
    cross_boxes = world_boxes[keep].copy()
    cross_names = names[keep].copy()

    # The selected vehicle becomes the virtual ego and is removed.  Match the
    # source script by adding the original ego back as an Agent.
    original_ego_box = np.array(
        [
            original_center[0],
            original_center[1],
            original_center[2],
            4.6,
            1.8,
            5.0,
            original_yaw,
        ],
        dtype=np.float64,
    )
    cross_boxes = np.vstack([cross_boxes, original_ego_box])
    cross_names = np.append(cross_names, "vehicle")
    cross_rgb, cross_counts = render_view(
        frame=frame,
        map_path=map_path,
        camera_model=camera_model,
        viewpoint_center=cross_center,
        viewpoint_yaw=cross_yaw,
        world_boxes=cross_boxes,
        names=cross_names,
        map_radius=args.map_radius,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    original_output = output_dir / f"navsim_{args.token}_original_raster.png"
    recovery_output = output_dir / f"navsim_{args.token}_recovery_oriented.png"
    cross_output = output_dir / f"navsim_{args.token}_cross_agent.png"
    comparison_output = output_dir / f"navsim_{args.token}_augmentation_triptych.jpg"
    record_output = output_dir / f"navsim_{args.token}_augmentation_record.json"

    if not cv2.imwrite(str(original_output), original_bgr):
        raise RuntimeError(f"Failed to write {original_output}")
    if not cv2.imwrite(str(recovery_output), recovery_rgb[:, :, ::-1]):
        raise RuntimeError(f"Failed to write {recovery_output}")
    if not cv2.imwrite(str(cross_output), cross_rgb[:, :, ::-1]):
        raise RuntimeError(f"Failed to write {cross_output}")

    recovery_bgr = recovery_rgb[:, :, ::-1]
    cross_bgr = cross_rgb[:, :, ::-1]
    if original_bgr.shape != recovery_bgr.shape:
        original_bgr = cv2.resize(
            original_bgr,
            (recovery_bgr.shape[1], recovery_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    panels = [
        labelled_panel(original_bgr, "Original Raster", f"token: {args.token}"),
        labelled_panel(
            recovery_bgr,
            "Recovery-oriented perturbation",
            f"dx={delta_x:+.3f}m  dy={delta_y:+.3f}m  dyaw={delta_yaw_deg:+.2f}deg",
        ),
        labelled_panel(
            cross_bgr,
            "Cross-agent view synthesis",
            "virtual ego: " + str(tracks[selected_index]),
        ),
    ]
    triptych = np.concatenate(panels, axis=1)
    if not cv2.imwrite(
        str(comparison_output), triptych, [cv2.IMWRITE_JPEG_QUALITY, 95]
    ):
        raise RuntimeError(f"Failed to write {comparison_output}")

    record = {
        "token": args.token,
        "frame_index": frame_index,
        "source_scripts": [
            "process_data/create_openscene_metadata_purturbed.py",
            "process_data/create_openscene_metadata_aug.py",
        ],
        "camera_policy": (
            "NAVSIM CAM_F0 frame calibration held constant across panels; "
            "RAP renderer translation offset pre-compensated"
        ),
        "recovery_oriented": {
            "seed": args.seed,
            "delta_x_m": delta_x,
            "delta_y_m": delta_y,
            "delta_yaw_deg": delta_yaw_deg,
            "source_ranges": {"x_y_m": [-0.5, 0.5], "yaw_deg": [-15.0, 15.0]},
            "render_counts": recovery_counts,
        },
        "cross_agent": {
            "selection_policy": (
                "nearest vehicle passing source 14-frame persistence and "
                "mean CV-error > 0.5 m filters; nearest vehicle fallback"
            ),
            "selected": selected_metric,
            "selected_world_position": cross_center.tolist(),
            "selected_world_yaw_rad": cross_yaw,
            "candidate_metrics": all_metrics,
            "render_counts": cross_counts,
        },
        "outputs": {
            "original": str(original_output),
            "recovery_oriented": str(recovery_output),
            "cross_agent": str(cross_output),
            "comparison": str(comparison_output),
        },
    }
    record_output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"record: {record_output}")


if __name__ == "__main__":
    main()
