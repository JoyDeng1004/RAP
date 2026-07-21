#!/usr/bin/env python3
"""Step 0-C raster component/ownership audit for dataset_norm.

This program deliberately reconstructs the old norm metadata through the same
map API path used by ``audit_step0_0a.py``.  The stored pickle files do not
contain ``map_features`` and some do not contain ``gt_boxes_world``.

The ``scan`` command is shardable.  ``aggregate`` concatenates shard outputs,
checks ownership invariants, computes log-bootstrap confidence intervals and
writes the canonical Step 0-C tables/summary.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import random
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("NUPLAN_DB_PATH", "/nonexistent/step0_audit_db")
os.environ.setdefault("NUPLAN_SENSOR_PATH", "/nonexistent/step0_audit_sensor")

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from process_data.helpers.renderer import (  # noqa: E402
    COLOR_TABLE,
    ScenarioRenderer,
    _sutherland_hodgman,
    camera_params,
    draw_cuboid_at,
    draw_cuboids_with_occlusion,
    draw_polygon_depth,
    draw_polyline_depth,
    project_points_cam,
    vehicle_corners_local,
    world_to_camera_T,
    yaw_to_rot,
)
from tools.audit_step0_0a import image_metrics, jpeg_roundtrip_rgb, reconstruct_scenario  # noqa: E402

CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")
H, W = 1120, 1920
MODEL_H, MODEL_W = 448, 768
FACE_IDXS = ([0, 1, 5, 4], [2, 3, 7, 6], [3, 0, 4, 7], [1, 2, 6, 5], [3, 2, 1, 0], [4, 5, 6, 7])
LIGHT_FACE_IDXS = ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])
SEED = 20260717


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--manifest", type=Path, required=True)
    scan.add_argument("--raster-root", type=Path, required=True)
    scan.add_argument("--real-root", type=Path, required=True)
    scan.add_argument("--output-dir", type=Path, required=True)
    scan.add_argument("--max-frames", type=int, default=32)
    scan.add_argument("--num-shards", type=int, default=1)
    scan.add_argument("--shard-index", type=int, default=0)
    scan.add_argument("--seed", type=int, default=SEED)
    scan.add_argument("--rgb-check-frames", type=int, default=100)
    scan.add_argument("--panel-frames", type=int, default=2)
    scan.add_argument("--save-variants-frames", type=int, default=32)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--parts-root", type=Path, required=True)
    agg.add_argument("--output-dir", type=Path, required=True)
    agg.add_argument("--seed", type=int, default=SEED)
    agg.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key, "")) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_frame_specs(manifest: Path, max_frames: int, seed: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    with gzip.open(manifest, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("camera") != "CAM_F0" or not row.get("required"):
                continue
            specs.append({k: row[k] for k in ("metadata_path", "frame_index", "frame_token", "log_name")})
    rng = random.Random(seed)
    rng.shuffle(specs)
    return specs[:max_frames] if max_frames > 0 else specs


def camera_geometry(scenario: Mapping[str, Any], camera: str, shifted: bool = True) -> tuple[np.ndarray, np.ndarray]:
    model = camera_params[camera]
    cam_t = model["sensor2lidar_translation"].copy()
    if shifted:
        cam_t[2] += 0.8
        cam_t[0] -= 2.0
    return world_to_camera_T(np.zeros(3), scenario["ego_heading"], cam_t, model["sensor2lidar_rotation"]), model["intrinsics"]


def model_mask(mask: np.ndarray) -> np.ndarray:
    cropped = mask[20:-20]
    resized = cv2.resize(cropped, (768, 432), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(resized, 0, 16, 0, 0, cv2.BORDER_CONSTANT, value=0)


def model_rgb(rgb: np.ndarray) -> np.ndarray:
    cropped = rgb[20:-20]
    resized = cv2.resize(cropped, (768, 432), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(resized, 0, 16, 0, 0, cv2.BORDER_CONSTANT, value=0)


def count_id(mask: np.ndarray, ident: int) -> int:
    return int(np.count_nonzero(mask == ident))


def light_corners(light: Sequence[Any]) -> np.ndarray:
    xy = light[2]
    center = np.asarray([xy[0], xy[1], 5.0], dtype=np.float32)
    local = np.asarray([
        [0.25, 0.25, 0], [0.25, -0.25, 0], [-0.25, -0.25, 0], [-0.25, 0.25, 0],
        [0.25, 0.25, 1], [0.25, -0.25, 1], [-0.25, -0.25, 1], [-0.25, 0.25, 1],
    ], dtype=np.float32)
    return local + center


def draw_cuboid_id(canvas: np.ndarray, corners: np.ndarray, T: np.ndarray, K: np.ndarray, ident: int, faces=LIGHT_FACE_IDXS) -> int:
    pts_cam = (T[:3, :3] @ corners.T + T[:3, 3:4]).T
    uv, valid = project_points_cam(pts_cam, K, canvas.shape)
    for idxs in faces:
        idx = list(idxs)
        if not np.any(valid[idx]) or np.all(pts_cam[idx, 2] <= 0):
            continue
        cv2.fillConvexPoly(canvas, uv[idx].astype(np.int32), int(ident), cv2.LINE_8)
    return int(valid.sum())


def agent_corners(box: np.ndarray) -> np.ndarray:
    local = vehicle_corners_local(float(box[3]), float(box[4]), float(box[5]))
    return (yaw_to_rot(float(box[6])) @ local.T).T + box[:3]


def draw_agents_id(canvas: np.ndarray, boxes: np.ndarray, T: np.ndarray, K: np.ndarray, id_offset: int) -> list[int]:
    faces: list[tuple[float, np.ndarray, int]] = []
    valid_counts: list[int] = []
    for agent_index, box in enumerate(boxes):
        corners = agent_corners(box)
        pts_cam = (T[:3, :3] @ corners.T + T[:3, 3:4]).T
        uv, valid = project_points_cam(pts_cam, K, canvas.shape)
        valid_counts.append(int(valid.sum()))
        if valid.sum() < 4:
            continue
        for idxs in FACE_IDXS:
            idx = list(idxs)
            face = pts_cam[idx]
            if np.all(face[:, 2] <= 0) or not np.any(valid[idx]):
                continue
            depth = float(np.mean(face[:, 2].clip(0, 120)))
            faces.append((depth, uv[idx].astype(np.int32), id_offset + agent_index))
    faces.sort(key=lambda item: item[0], reverse=True)
    for _, polygon, ident in faces:
        cv2.fillConvexPoly(canvas, polygon, int(ident), cv2.LINE_8)
    return valid_counts


def clip_polygon_near(points: np.ndarray, near: float = 1e-3) -> np.ndarray:
    output: list[np.ndarray] = []
    previous = points[-1]
    previous_inside = previous[2] >= near
    for current in points:
        inside = current[2] >= near
        if inside != previous_inside:
            ratio = (near - previous[2]) / (current[2] - previous[2])
            output.append(previous + ratio * (current - previous))
        if inside:
            output.append(current)
        previous, previous_inside = current, inside
    return np.asarray(output, dtype=np.float32)


def draw_agent_strict_clip(canvas: np.ndarray, box: np.ndarray, T: np.ndarray, K: np.ndarray) -> None:
    corners = agent_corners(box)
    pts_cam = (T[:3, :3] @ corners.T + T[:3, 3:4]).T
    for idxs in FACE_IDXS:
        clipped3d = clip_polygon_near(pts_cam[list(idxs)])
        if len(clipped3d) < 3:
            continue
        uv_h = (K @ clipped3d.T).T
        uv = uv_h[:, :2] / uv_h[:, 2:3]
        clipped2d = _sutherland_hodgman(uv, canvas.shape[1], canvas.shape[0])
        if len(clipped2d) >= 3:
            cv2.fillConvexPoly(canvas, clipped2d.astype(np.int32), 1, cv2.LINE_8)


def map_category(ftype: str) -> str:
    if "LANE" in ftype:
        return "lane"
    if "CROSSWALK" in ftype or "SPEED_BUMP" in ftype:
        return "crosswalk_or_speed_bump"
    if "BOUNDARY" in ftype or "SOLID" in ftype:
        return "boundary_or_solid"
    return "unsupported_by_renderer"


def map_geometry(feat: Mapping[str, Any], category: str) -> tuple[np.ndarray | None, str]:
    key = "polyline" if category == "boundary_or_solid" else "polygon"
    value = feat.get(key)
    if value is None:
        return None, key
    return np.asarray(value, dtype=np.float32), key


def draw_polyline_id(canvas: np.ndarray, points: np.ndarray, T: np.ndarray, K: np.ndarray, ident: int, radius: int, depth_max: float = 120) -> None:
    pts3d = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])
    pts_cam = (T[:3, :3] @ pts3d.T + T[:3, 3:4]).T
    for index in range(len(pts_cam) - 1):
        p1, p2 = pts_cam[index].copy(), pts_cam[index + 1].copy()
        if min(p1[2], p2[2]) < 1e-3:
            if p1[2] < 1e-3 and p2[2] < 1e-3:
                continue
            if p1[2] < 1e-3:
                p1 = p1 + ((1e-3 - p1[2]) / (p2[2] - p1[2])) * (p2 - p1)
            else:
                p2 = p2 + ((1e-3 - p2[2]) / (p1[2] - p2[2])) * (p1 - p2)
        uv = []
        for point in (p1, p2):
            q = K @ point
            uv.append((int(round(q[0] / q[2])), int(round(q[1] / q[2]))))
        ok, a, b = cv2.clipLine((0, 0, canvas.shape[1] - 1, canvas.shape[0] - 1), uv[0], uv[1])
        if ok:
            cv2.line(canvas, a, b, int(ident), radius, cv2.LINE_8)


def draw_polygon_id(canvas: np.ndarray, points: np.ndarray, T: np.ndarray, K: np.ndarray, ident: int) -> None:
    pts3d = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])
    pts_cam = (T[:3, :3] @ pts3d.T + T[:3, 3:4]).T
    front = pts_cam[:, 2] > 1e-6
    if not np.any(front):
        return
    q = (K @ pts_cam[front].T).T
    uv = q[:, :2] / q[:, 2:3]
    clipped = _sutherland_hodgman(uv, canvas.shape[1], canvas.shape[0])
    if len(clipped) >= 3:
        cv2.fillConvexPoly(canvas, clipped.astype(np.int32), int(ident), cv2.LINE_8)


def draw_map_id(canvas: np.ndarray, feat: Mapping[str, Any], category: str, T: np.ndarray, K: np.ndarray, ident: int) -> tuple[bool, str]:
    points, kind = map_geometry(feat, category)
    if points is None or not len(points):
        return False, kind
    if category == "lane":
        if float(np.min(np.linalg.norm(points, axis=1))) > 120:
            return False, kind
        draw_polyline_id(canvas, points, T, K, ident, 2)
    elif category == "crosswalk_or_speed_bump":
        draw_polygon_id(canvas, points, T, K, ident)
        draw_polyline_id(canvas, points, T, K, ident, 8)
    elif category == "boundary_or_solid":
        draw_polyline_id(canvas, points, T, K, ident, 10)
    else:
        return False, kind
    return True, kind


def render_variant(scenario: Mapping[str, Any], camera: str, components: set[str]) -> np.ndarray:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    T, K = camera_geometry(scenario, camera)
    if "traffic_light" in components:
        for light in scenario["traffic_lights"]:
            color = COLOR_TABLE["traffic_light_red" if light[1] else "traffic_light_green"].tolist()
            draw_cuboid_at(canvas, [light[2][0], light[2][1], 5], (0.5, 0.5, 1), T, K, color_rgb=color, thickness=-1)
    if "map" in components:
        for feat in scenario["map_features"].values():
            category = map_category(str(feat["type"]))
            points, _ = map_geometry(feat, category)
            if points is None or not len(points):
                continue
            pts3d = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])
            if category == "lane":
                if float(np.min(np.linalg.norm(points, axis=1))) <= 120:
                    draw_polyline_depth(canvas, pts3d, T, K, COLOR_TABLE["lanelines"], radius=2, depth_max=120)
            elif category == "crosswalk_or_speed_bump":
                draw_polygon_depth(canvas, pts3d, T, K, COLOR_TABLE["crosswalks"], 120)
                draw_polyline_depth(canvas, pts3d, T, K, COLOR_TABLE["lanelines"], depth_max=120)
            elif category == "boundary_or_solid":
                draw_polyline_depth(canvas, pts3d, T, K, COLOR_TABLE["road_boundaries"], radius=10, depth_max=120)
    if "agent" in components:
        draw_cuboids_with_occlusion(canvas, np.asarray(scenario["anns"]["gt_boxes_world"]), T, K)
    return canvas


def ownership_pass(scenario: Mapping[str, Any], camera: str) -> dict[str, Any]:
    T, K = camera_geometry(scenario, camera)
    lights = list(scenario["traffic_lights"])
    maps = list(scenario["map_features"].items())
    boxes = np.asarray(scenario["anns"]["gt_boxes_world"])
    light_base, map_base, agent_base = 1, 1 + len(lights), 1 + len(lights) + len(maps)
    ownership = np.zeros((H, W), dtype=np.int32)
    light_valid: list[int] = []
    for index, light in enumerate(lights):
        light_valid.append(draw_cuboid_id(ownership, light_corners(light), T, K, light_base + index))
    after_lights = ownership.copy()
    map_meta: list[tuple[str, str, str]] = []
    for index, (feature_id, feat) in enumerate(maps):
        category = map_category(str(feat["type"]))
        _, kind = draw_map_id(ownership, feat, category, T, K, map_base + index)
        map_meta.append((str(feature_id), str(feat["type"]), kind))
    after_map = ownership.copy()
    valid_counts = draw_agents_id(ownership, boxes, T, K, agent_base)
    return {
        "T": T, "K": K, "lights": lights, "maps": maps, "boxes": boxes,
        "light_base": light_base, "map_base": map_base, "agent_base": agent_base,
        "light_valid": light_valid, "valid_counts": valid_counts, "map_meta": map_meta,
        "after_lights": after_lights, "after_map": after_map, "final": ownership,
    }


def primary_reason(flags: Mapping[str, bool], order: Sequence[tuple[str, str]]) -> str:
    for flag, label in order:
        if flags.get(flag):
            return label
    return "not_dropped"


def audit_camera(base: Mapping[str, Any], scenario: Mapping[str, Any], camera: str, variants_root: Path | None, compute_variants: bool) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[str, np.ndarray]]:
    state = ownership_pass(scenario, camera)
    T, K = state["T"], state["K"]
    after_lights, after_map, final = state["after_lights"], state["after_map"], state["final"]
    model_after_lights, model_after_map, model_final = map(model_mask, (after_lights, after_map, final))
    common = {**base, "camera": camera}
    light_rows: list[dict] = []
    for index, light in enumerate(state["lights"]):
        ident = state["light_base"] + index
        isolated = np.zeros((H, W), np.int32)
        valid_count = draw_cuboid_id(isolated, light_corners(light), T, K, ident)
        isolated_m = model_mask(isolated)
        areas = [count_id(mask, ident) for mask in (isolated, after_lights, after_map, final)]
        areas_m = [count_id(mask, ident) for mask in (isolated_m, model_after_lights, model_after_map, model_final)]
        distance = float(np.linalg.norm(np.asarray(light[2], dtype=float)))
        flags = {
            "geometry_zero": areas[0] == 0,
            "light_overwrite": areas[0] > 0 and areas[1] == 0,
            "map_overwrite": areas[1] > 0 and areas[2] == 0,
            "agent_overwrite": areas[2] > 0 and areas[3] == 0,
        }
        reason = primary_reason(flags, (("geometry_zero", "zero_area_after_rasterization"), ("light_overwrite", "fully_overwritten_by_light"), ("map_overwrite", "fully_overwritten_by_map"), ("agent_overwrite", "fully_overwritten_by_agent")))
        light_rows.append({**common, "traffic_light_index": index, "lane_connector_id": light[0], "is_red": bool(light[1]), "distance": distance,
            "valid_corner_count": valid_count, "isolated_area_render": areas[0], "visible_after_lights_render": areas[1], "visible_after_map_render": areas[2], "visible_final_render": areas[3],
            "isolated_area_model": areas_m[0], "visible_after_lights_model": areas_m[1], "visible_after_map_model": areas_m[2], "visible_final_model": areas_m[3],
            "light_overwrite_fraction": 1 - areas[1] / max(areas[0], 1), "map_overwrite_fraction": (areas[1] - areas[2]) / max(areas[1], 1),
            "agent_overwrite_fraction": (areas[2] - areas[3]) / max(areas[2], 1), **flags, "drop_reason": reason})
    map_rows: list[dict] = []
    for index, (feature_id, feat) in enumerate(state["maps"]):
        ident = state["map_base"] + index
        category = map_category(str(feat["type"]))
        isolated = np.zeros((H, W), np.int32)
        supported, kind = draw_map_id(isolated, feat, category, T, K, ident)
        isolated_m = model_mask(isolated)
        areas = [count_id(mask, ident) for mask in (isolated, after_map, final)]
        areas_m = [count_id(mask, ident) for mask in (isolated_m, model_after_map, model_final)]
        points, _ = map_geometry(feat, category)
        distance = float(np.min(np.linalg.norm(points, axis=1))) if points is not None and len(points) else float("nan")
        pts3d = np.column_stack([points, np.zeros(len(points))]) if points is not None and len(points) else np.empty((0, 3))
        depths = (T[:3, :3] @ pts3d.T + T[:3, 3:4]).T[:, 2] if len(pts3d) else np.empty(0)
        flags = {
            "unsupported": category == "unsupported_by_renderer",
            "beyond_depth": category == "lane" and math.isfinite(distance) and distance > 120,
            "behind": len(depths) > 0 and bool(np.all(depths <= 0)),
            "geometry_zero": supported and areas[0] == 0,
            "map_overwrite": areas[0] > 0 and areas[1] == 0,
            "agent_overwrite": areas[1] > 0 and areas[2] == 0,
        }
        reason = primary_reason(flags, (("unsupported", "unsupported_by_renderer"), ("beyond_depth", "beyond_depth_max"), ("behind", "behind_camera"), ("geometry_zero", "outside_fov_after_clipping"), ("map_overwrite", "fully_overwritten_by_map"), ("agent_overwrite", "fully_overwritten_by_agent")))
        map_rows.append({**common, "map_feature_id": feature_id, "raw_ftype": feat["type"], "map_category": category, "renderer_supported": category != "unsupported_by_renderer",
            "distance_min": distance, "geometry_kind": kind, "isolated_area_render": areas[0], "visible_after_map_render": areas[1], "visible_final_render": areas[2],
            "isolated_area_model": areas_m[0], "visible_after_map_model": areas_m[1], "visible_final_model": areas_m[2],
            "map_overwrite_fraction": 1 - areas[1] / max(areas[0], 1), "agent_overwrite_fraction": (areas[1] - areas[2]) / max(areas[1], 1), **flags, "drop_reason": reason})
    agent_rows: list[dict] = []
    anns = scenario["anns"]
    names = np.asarray(anns.get("gt_names", []))
    for index, box in enumerate(state["boxes"]):
        ident = state["agent_base"] + index
        isolated = np.zeros((H, W), np.int32)
        valid_count = draw_agents_id(isolated, box.reshape(1, -1), T, K, ident)[0]
        clipped = np.zeros((H, W), np.uint8)
        draw_agent_strict_clip(clipped, box, T, K)
        isolated_m, clipped_m = model_mask(isolated), model_mask(clipped)
        areas = [count_id(isolated, ident), count_id(final, ident), int(np.count_nonzero(clipped))]
        areas_m = [count_id(isolated_m, ident), count_id(model_final, ident), int(np.count_nonzero(clipped_m))]
        corners = agent_corners(box)
        depths = (T[:3, :3] @ corners.T + T[:3, 3:4]).T[:, 2]
        flags = {"behind": bool(np.all(depths <= 0)), "valid_lt4": valid_count < 4, "zero": valid_count >= 4 and areas[0] == 0, "occluded": areas[0] > 0 and areas[1] == 0}
        reason = primary_reason(flags, (("behind", "behind_camera"), ("valid_lt4", "valid_corner_lt4"), ("zero", "zero_area_after_rasterization"), ("occluded", "fully_occluded_by_agent")))
        instance_tokens = anns.get("instance_tokens", [])
        track_tokens = anns.get("track_tokens", [])
        visibility = areas[1] / max(areas[0], 1)
        agent_rows.append({**common, "agent_index": index, "instance_token": instance_tokens[index] if index < len(instance_tokens) else "", "track_token": track_tokens[index] if index < len(track_tokens) else "",
            "class": names[index] if index < len(names) else "", "center_x": box[0], "center_y": box[1], "center_z": box[2], "distance": float(np.linalg.norm(box[:2])),
            "valid_corner_count": valid_count, "drop_reason": reason, "isolated_area_prod_render": areas[0], "visible_area_prod_render": areas[1], "isolated_area_clip_render": areas[2],
            "isolated_area_prod_model": areas_m[0], "visible_area_prod_model": areas_m[1], "isolated_area_clip_model": areas_m[2], "raster_visibility": visibility,
            "agent_occlusion_fraction": 1 - visibility, "fov_drop_threshold_4": areas[0] == 0 and areas_m[2] >= 4,
            "fov_drop_threshold_16": areas[0] == 0 and areas_m[2] >= 16, "fov_drop_threshold_64": areas[0] == 0 and areas_m[2] >= 64, **flags})
    component_ids = {
        "traffic_light": range(state["light_base"], state["map_base"]),
        "map": range(state["map_base"], state["agent_base"]),
        "agent": range(state["agent_base"], state["agent_base"] + len(state["boxes"])),
    }
    coverage_rows: list[dict] = []
    for scale, masks in (("render", (after_lights, after_map, final)), ("model", (model_after_lights, model_after_map, model_final))):
        lights_mask, maps_mask, final_mask = masks
        component_only = {
            "traffic_light": int(np.isin(lights_mask, list(component_ids["traffic_light"])).sum()),
            "map": int(np.isin(maps_mask, list(component_ids["map"])).sum()),
            "agent": int(np.isin(final_mask, list(component_ids["agent"])).sum()),
        }
        final_counts = {key: int(np.isin(final_mask, list(ids)).sum()) for key, ids in component_ids.items()}
        total = int(final_mask.size)
        blank = int(np.count_nonzero(final_mask == 0))
        row = {**common, "scale": scale, "total_pixels": total,
            "traffic_light_component_only_pixels": component_only["traffic_light"], "map_component_only_pixels": component_only["map"], "agent_component_only_pixels": component_only["agent"],
            "final_traffic_light_owned_pixels": final_counts["traffic_light"], "final_map_owned_pixels": final_counts["map"], "final_agent_owned_pixels": final_counts["agent"], "final_blank_pixels": blank,
            "traffic_light_coverage": final_counts["traffic_light"] / total, "map_coverage": final_counts["map"] / total, "agent_coverage": final_counts["agent"] / total,
            "blank_fraction": blank / total, "ownership_sum_pixels": blank + sum(final_counts.values()), "ownership_complete": blank + sum(final_counts.values()) == total}
        if scale == "model":
            token_mask = cv2.resize(final_mask, (48, 28), interpolation=cv2.INTER_NEAREST)
            token_counts = {key: int(np.isin(token_mask, list(ids)).sum()) for key, ids in component_ids.items()}
            token_blank = int(np.count_nonzero(token_mask == 0))
            row.update({"total_tokens": int(token_mask.size), "final_traffic_light_owned_tokens": token_counts["traffic_light"],
                "final_map_owned_tokens": token_counts["map"], "final_agent_owned_tokens": token_counts["agent"], "final_blank_tokens": token_blank,
                "traffic_light_token_coverage": token_counts["traffic_light"] / token_mask.size, "map_token_coverage": token_counts["map"] / token_mask.size,
                "agent_token_coverage": token_counts["agent"] / token_mask.size, "blank_token_fraction": token_blank / token_mask.size,
                "ownership_sum_tokens": token_blank + sum(token_counts.values()), "token_ownership_complete": token_blank + sum(token_counts.values()) == token_mask.size})
        coverage_rows.append(row)
    variants: dict[str, np.ndarray] = {}
    if compute_variants:
        variants = {
            "full": render_variant(scenario, camera, {"traffic_light", "map", "agent"}),
            "blank": render_variant(scenario, camera, set()),
            "agent_only": render_variant(scenario, camera, {"agent"}),
            "map_only": render_variant(scenario, camera, {"map"}),
            "traffic_light_only": render_variant(scenario, camera, {"traffic_light"}),
            "full_without_agents": render_variant(scenario, camera, {"traffic_light", "map"}),
            "full_without_map": render_variant(scenario, camera, {"traffic_light", "agent"}),
            "full_without_traffic_lights": render_variant(scenario, camera, {"map", "agent"}),
        }
    if variants_root is not None:
        target = variants_root / str(base["frame_token"]) / camera
        target.mkdir(parents=True, exist_ok=True)
        for name, image in variants.items():
            Image.fromarray(image).save(target / f"{name}.png")
    return agent_rows, map_rows, light_rows, coverage_rows, {**variants, "after_lights": after_lights, "after_map": after_map, "final_ownership": final}


def ownership_rgb(mask: np.ndarray, light_end: int, map_end: int) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), np.uint8)
    out[(mask > 0) & (mask < light_end)] = (220, 45, 45)
    out[(mask >= light_end) & (mask < map_end)] = (45, 100, 220)
    out[mask >= map_end] = (235, 135, 35)
    return out


def save_panel(path: Path, real: np.ndarray, variants: Mapping[str, np.ndarray], title: str, light_end: int, map_end: int) -> None:
    tiles = [real, variants["full"], ownership_rgb(variants["after_lights"], light_end, map_end), ownership_rgb(variants["after_map"], light_end, map_end), ownership_rgb(variants["final_ownership"], light_end, map_end), variants["agent_only"], variants["map_only"], variants["traffic_light_only"], variants["full_without_agents"], variants["full_without_map"], variants["full_without_traffic_lights"]]
    thumbs = [np.asarray(Image.fromarray(tile).resize((384, 224))) for tile in tiles]
    blank = np.zeros_like(thumbs[0])
    thumbs.append(blank)
    canvas = np.concatenate([np.concatenate(thumbs[i:i + 4], axis=1) for i in range(0, 12, 4)], axis=0)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), title + " | real/full/ownership stages/component ablations", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def scan(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_specs = stable_frame_specs(args.manifest, args.max_frames, args.seed)
    specs = [spec for index, spec in enumerate(all_specs) if index % args.num_shards == args.shard_index]
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        grouped[Path(spec["metadata_path"])].append(spec)
    agents: list[dict] = []
    maps: list[dict] = []
    lights: list[dict] = []
    coverage: list[dict] = []
    reproduction: list[dict] = []
    visualization: list[dict] = []
    variant_index: list[dict] = []
    errors: list[dict] = []
    map_cache: dict[str, Any] = {}
    production = ScenarioRenderer(list(CAMERAS))
    completed = 0
    for metadata_path, group in grouped.items():
        with metadata_path.open("rb") as handle:
            frames = pickle.load(handle)
        for spec in sorted(group, key=lambda item: int(item["frame_index"])):
            try:
                frame = frames[int(spec["frame_index"])]
                scenario = reconstruct_scenario(frame, map_cache)
                scenario["anns"]["instance_tokens"] = frame["anns"].get("instance_tokens", [])
                scenario["anns"]["track_tokens"] = frame["anns"].get("track_tokens", [])
                rendered = production.observe(scenario) if completed < args.rgb_check_frames else None
                base = {"split": "mini", "log": frame.get("log_name", spec["log_name"]), "frame_token": frame.get("token", spec["frame_token"])}
                for camera in CAMERAS:
                    save_variants = completed < args.save_variants_frames
                    need_variants = save_variants or completed < args.panel_frames
                    a, m, l, c, variants = audit_camera(base, scenario, camera, args.output_dir / "variants" if save_variants else None, need_variants)
                    agents.extend(a); maps.extend(m); lights.extend(l); coverage.extend(c)
                    raster_path = args.raster_root / frame["cams"][camera]["data_path"]
                    if rendered is not None:
                        stored_bgr = cv2.imread(str(raster_path), cv2.IMREAD_COLOR)
                        stored = stored_bgr[:, :, ::-1][20:-20]
                        candidate = jpeg_roundtrip_rgb(rendered[camera])[20:-20]
                        reproduction.append({**base, "camera": camera, "raster_path": str(raster_path), **image_metrics(stored, candidate)})
                    if save_variants:
                        for name in ("full", "blank", "agent_only", "map_only", "traffic_light_only", "full_without_agents", "full_without_map", "full_without_traffic_lights"):
                            variant_index.append({**base, "camera": camera, "variant": name, "path": str(args.output_dir / "variants" / str(base["frame_token"]) / camera / f"{name}.png")})
                    if completed < args.panel_frames:
                        real_path = args.real_root / frame["cams"][camera]["data_path"]
                        real = np.asarray(Image.open(real_path).convert("RGB").resize((W, H)))
                        panel_path = args.output_dir / "panels" / "overview" / f"{base['frame_token']}_{camera}.png"
                        light_end = 1 + len(scenario["traffic_lights"])
                        map_end = light_end + len(scenario["map_features"])
                        save_panel(panel_path, real, variants, f"{base['frame_token']} {camera}", light_end, map_end)
                        visualization.append({**base, "camera": camera, "panel_type": "overview", "element_type": "all", "element_id": "", "selection_rule": "stratified_random_control", "rank": completed,
                            "real_path": str(real_path), "raster_path": str(raster_path), "panel_path": str(panel_path), "heatmap_npy_path": ""})
                completed += 1
            except Exception as error:
                errors.append({**spec, "error": repr(error), "traceback": traceback.format_exc()})
    for name, rows in (("agent_raster_visibility.csv", agents), ("map_element_visibility.csv", maps), ("traffic_light_visibility.csv", lights), ("component_coverage.csv", coverage), ("rgb_reproduction.csv", reproduction), ("visualization_index.csv", visualization), ("variant_index.csv", variant_index), ("errors.csv", errors)):
        write_csv(args.output_dir / name, rows)
    config = {**vars(args), "frames_selected_global": len(all_specs), "frames_in_shard": len(specs), "frames_completed": completed, "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "opencv": cv2.__version__}}
    (args.output_dir / "config.json").write_text(json.dumps(config, default=jsonable, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(f"{len(errors)} frames failed; see {args.output_dir / 'errors.csv'}")


def numeric(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def log_bootstrap(rows: Sequence[Mapping[str, str]], key: str, replicates: int, seed: int) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = numeric(row, key)
        if math.isfinite(value):
            grouped[row["log"]].append(value)
    logs = sorted(grouped)
    values = [value for log in logs for value in grouped[log]]
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "logs": 0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        picked = [rng.choice(logs) for _ in logs]
        sample = [value for log in picked for value in grouped[log]]
        estimates.append(float(np.mean(sample)))
    return {"mean": float(np.mean(values)), "ci_low": float(np.percentile(estimates, 2.5)), "ci_high": float(np.percentile(estimates, 97.5)), "n": len(values), "logs": len(logs)}


def distance_bin(value: float) -> str:
    for low, high in ((0, 10), (10, 20), (20, 40), (40, 80), (80, 120)):
        if low <= value < high:
            return f"[{low},{high})"
    return "outside_bins"


def grouped_summary(rows: Sequence[Mapping[str, str]], group_keys: Sequence[str], kind: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        keys = []
        for key in group_keys:
            keys.append(distance_bin(numeric(row, "distance" if kind != "map" else "distance_min")) if key == "distance_bin" else row.get(key, ""))
        groups[tuple(keys)].append(row)
    output: list[dict[str, Any]] = []
    for key_values, members in sorted(groups.items()):
        base = dict(zip(group_keys, key_values)); base["metadata_count"] = len(members)
        if kind == "agent":
            base.update({"production_nonzero": sum(numeric(row, "isolated_area_prod_render") > 0 for row in members), "final_nonzero": sum(numeric(row, "visible_area_prod_render") > 0 for row in members),
                "fov_drop_4": sum(row.get("fov_drop_threshold_4") == "True" for row in members), "fov_drop_16": sum(row.get("fov_drop_threshold_16") == "True" for row in members),
                "fov_drop_64": sum(row.get("fov_drop_threshold_64") == "True" for row in members), "mean_raster_visibility": float(np.nanmean([numeric(row, "raster_visibility") for row in members]))})
        elif kind == "map":
            base.update({"renderer_supported": sum(row.get("renderer_supported") == "True" for row in members), "isolated_nonzero": sum(numeric(row, "isolated_area_render") > 0 for row in members),
                "final_nonzero": sum(numeric(row, "visible_final_render") > 0 for row in members), "fully_overwritten_by_map": sum(row.get("drop_reason") == "fully_overwritten_by_map" for row in members),
                "fully_overwritten_by_agent": sum(row.get("drop_reason") == "fully_overwritten_by_agent" for row in members)})
        else:
            base.update({"isolated_nonzero": sum(numeric(row, "isolated_area_render") > 0 for row in members), "final_nonzero": sum(numeric(row, "visible_final_render") > 0 for row in members),
                "fully_overwritten_by_map": sum(row.get("drop_reason") == "fully_overwritten_by_map" for row in members), "fully_overwritten_by_agent": sum(row.get("drop_reason") == "fully_overwritten_by_agent" for row in members),
                "median_final_area_model": float(np.nanmedian([numeric(row, "visible_final_model") for row in members]))})
        output.append(base)
    return output


def concatenate_csv(parts: Sequence[Path], name: str, output: Path) -> None:
    """Concatenate identically shaped shard CSVs without materializing rows."""
    wrote_header = False
    with output.open("wb") as target:
        for part in parts:
            source_path = part / name
            if not source_path.exists() or not source_path.stat().st_size:
                continue
            with source_path.open("rb") as source:
                header = source.readline()
                if not wrote_header:
                    target.write(header); wrote_header = True
                shutil.copyfileobj(source, target, length=8 << 20)


def stream_grouped_summary(path: Path, group_keys: Sequence[str], kind: str) -> tuple[list[dict[str, Any]], Counter]:
    groups: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    final_areas: dict[tuple[str, ...], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys = tuple(distance_bin(numeric(row, "distance" if kind != "map" else "distance_min")) if key == "distance_bin" else row.get(key, "") for key in group_keys)
            state = groups[keys]; state["metadata_count"] += 1; totals["rows"] += 1
            if kind == "agent":
                state["production_nonzero"] += numeric(row, "isolated_area_prod_render") > 0
                state["final_nonzero"] += numeric(row, "visible_area_prod_render") > 0
                for threshold in (4, 16, 64):
                    hit = row.get(f"fov_drop_threshold_{threshold}") == "True"
                    state[f"fov_drop_{threshold}"] += hit; totals[f"fov_drop_{threshold}"] += hit
                value = numeric(row, "raster_visibility")
                if math.isfinite(value): state["visibility_sum"] += value; state["visibility_n"] += 1
            elif kind == "map":
                supported = row.get("renderer_supported") == "True"; iso = numeric(row, "isolated_area_render") > 0; final = numeric(row, "visible_final_render") > 0
                state["renderer_supported"] += supported; state["isolated_nonzero"] += iso; state["final_nonzero"] += final
                state["fully_overwritten_by_map"] += row.get("drop_reason") == "fully_overwritten_by_map"; state["fully_overwritten_by_agent"] += row.get("drop_reason") == "fully_overwritten_by_agent"
                totals["renderer_supported"] += supported; totals["isolated_nonzero"] += iso; totals["final_nonzero"] += final
            else:
                iso = numeric(row, "isolated_area_render") > 0; final = numeric(row, "visible_final_render") > 0
                state["isolated_nonzero"] += iso; state["final_nonzero"] += final
                state["fully_overwritten_by_map"] += row.get("drop_reason") == "fully_overwritten_by_map"; state["fully_overwritten_by_agent"] += row.get("drop_reason") == "fully_overwritten_by_agent"
                totals["isolated_nonzero"] += iso; totals["final_nonzero"] += final
                area = numeric(row, "visible_final_model")
                if math.isfinite(area): final_areas[keys].append(area)
    output = []
    for keys, state in sorted(groups.items()):
        row: dict[str, Any] = dict(zip(group_keys, keys)); row.update({key: value for key, value in state.items() if key not in ("visibility_sum", "visibility_n")})
        if kind == "agent": row["mean_raster_visibility"] = state["visibility_sum"] / max(state["visibility_n"], 1)
        if kind == "traffic_light": row["median_final_area_model"] = float(np.median(final_areas[keys])) if final_areas[keys] else float("nan")
        output.append(row)
    return output, totals


def aggregate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ("agent_raster_visibility.csv", "map_element_visibility.csv", "traffic_light_visibility.csv", "component_coverage.csv", "rgb_reproduction.csv", "visualization_index.csv", "variant_index.csv", "errors.csv")
    part_dirs = sorted(path.parent for path in args.parts_root.glob("shard_*/config.json"))
    if not part_dirs:
        raise RuntimeError(f"no completed shards under {args.parts_root}")
    for name in names:
        concatenate_csv(part_dirs, name, args.output_dir / name)
    coverage = read_csv(args.output_dir / "component_coverage.csv")
    reproduction = read_csv(args.output_dir / "rgb_reproduction.csv")
    visual_rows = read_csv(args.output_dir / "visualization_index.csv")
    error_rows = read_csv(args.output_dir / "errors.csv")
    html_rows = []
    for row in visual_rows:
        panel = row.get("panel_path", "")
        html_rows.append(
            "<figure><img loading='lazy' src='{}'><figcaption>{} {} · {} · {}</figcaption></figure>".format(
                panel, row.get("frame_token", ""), row.get("camera", ""), row.get("panel_type", ""), row.get("selection_rule", "")
            )
        )
    html = """<!doctype html><meta charset='utf-8'><title>Step 0-C panels</title>
<style>body{{background:#111;color:#eee;font-family:sans-serif}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px}}figure{{margin:0}}img{{width:100%}}figcaption{{padding:4px}}</style>
<h1>Step 0-C visualization index</h1><main>{}</main>""".format("\n".join(html_rows))
    (args.output_dir / "panel_index.html").write_text(html, encoding="utf-8")
    ownership_bad = sum(row.get("ownership_complete") != "True" for row in coverage)
    exact = sum(row.get("exact_equal") == "True" for row in reproduction)
    unique_frames = len({row["frame_token"] for row in coverage})
    metrics: dict[str, Any] = {}
    for scale in ("render", "model"):
        subset = [row for row in coverage if row["scale"] == scale]
        for key in ("traffic_light_coverage", "map_coverage", "agent_coverage", "blank_fraction"):
            metrics[f"{scale}.{key}"] = log_bootstrap(subset, key, args.bootstrap_replicates, args.seed)
    model_subset = [row for row in coverage if row["scale"] == "model"]
    for key in ("traffic_light_token_coverage", "map_token_coverage", "agent_token_coverage", "blank_token_fraction"):
        metrics[f"token.{key}"] = log_bootstrap(model_subset, key, args.bootstrap_replicates, args.seed)
    agent_summary, agent_totals = stream_grouped_summary(args.output_dir / "agent_raster_visibility.csv", ("camera", "class", "distance_bin"), "agent")
    map_summary, map_totals = stream_grouped_summary(args.output_dir / "map_element_visibility.csv", ("camera", "map_category"), "map")
    light_summary, light_totals = stream_grouped_summary(args.output_dir / "traffic_light_visibility.csv", ("camera", "is_red", "distance_bin"), "traffic_light")
    fov = {str(threshold): {"numerator": int(agent_totals[f"fov_drop_{threshold}"]), "denominator_all_agents": int(agent_totals["rows"])} for threshold in (4, 16, 64)}
    summary = {
        "status": "PASS" if not error_rows and ownership_bad == 0 and reproduction and exact == len(reproduction) else "FAIL",
        "shards": len(part_dirs), "frames": unique_frames, "camera_rows": len(coverage) // 2,
        "rgb_reproduction": {"rows": len(reproduction), "exact_equal": exact}, "ownership_invariant_failures": ownership_bad,
        "agents": {"rows": int(agent_totals["rows"]), "fov_drop_model_thresholds": fov},
        "maps": {"rows": int(map_totals["rows"]), "renderer_supported": int(map_totals["renderer_supported"]), "isolated_nonzero": int(map_totals["isolated_nonzero"]), "final_nonzero": int(map_totals["final_nonzero"])},
        "traffic_lights": {"rows": int(light_totals["rows"]), "isolated_nonzero": int(light_totals["isolated_nonzero"]), "final_nonzero": int(light_totals["final_nonzero"])},
        "coverage_log_bootstrap_95ci": metrics, "errors": len(error_rows),
    }
    (args.output_dir / "geometry_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "agent_summary.csv", agent_summary)
    write_csv(args.output_dir / "map_summary.csv", map_summary)
    write_csv(args.output_dir / "traffic_light_summary.csv", light_summary)
    coverage_summary = []
    for scale in ("render", "model"):
        subset = [row for row in coverage if row["scale"] == scale]
        for key in ("traffic_light_coverage", "map_coverage", "agent_coverage", "blank_fraction"):
            values = np.asarray([numeric(row, key) for row in subset], dtype=float)
            boot = metrics[f"{scale}.{key}"]
            coverage_summary.append({"scale": scale, "metric": key, **boot, "median": float(np.nanmedian(values)), "p10": float(np.nanpercentile(values, 10)),
                "p25": float(np.nanpercentile(values, 25)), "p75": float(np.nanpercentile(values, 75)), "p90": float(np.nanpercentile(values, 90))})
    write_csv(args.output_dir / "component_coverage_summary.csv", coverage_summary)
    lines = ["# Step 0-C raster component audit", "", f"- 几何/ownership 状态：**{summary['status']}**", f"- 扫描：{unique_frames} 帧 × F0/L0/R0（{len(part_dirs)} shards）", f"- production RGB 重现：{exact}/{len(reproduction)} camera rows exact equal", f"- ownership 互斥完备性失败：{ownership_bad}", f"- agent rows：{agent_totals['rows']}；模型尺度 FOV drop（阈值 4/16/64 px）：{fov['4']['numerator']}/{fov['16']['numerator']}/{fov['64']['numerator']}（分母均为全部 agent-camera {agent_totals['rows']}）", f"- map rows：{map_totals['rows']}；renderer 支持 {map_totals['renderer_supported']}；isolated 非零 {map_totals['isolated_nonzero']}；final 非零 {map_totals['final_nonzero']}", f"- traffic-light rows：{light_totals['rows']}；isolated 非零 {light_totals['isolated_nonzero']}；final 非零 {light_totals['final_nonzero']}", "", "## Final ownership coverage（log-level bootstrap 95% CI）", "", "| scale | component | mean | 95% CI | rows/logs |", "|---|---|---:|---:|---:|"]
    for label, value in metrics.items():
        scale, key = label.split(".")
        lines.append(f"| {scale} | {key} | {value['mean']:.6f} | [{value['ci_low']:.6f}, {value['ci_high']:.6f}] | {value['n']}/{value['logs']} |")
    lines += ["", "说明：coverage 是最终互斥 ownership，不把 component-only 重叠面积相加。这里仅描述 raster target 内容，不推断 planner 性能因果。", ""]
    (args.output_dir / "geometry_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.command == "scan":
        scan(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
