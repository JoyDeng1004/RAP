#!/usr/bin/env python3
"""Generate top-k dual-projection evidence panels from completed 0-C tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("NUPLAN_DB_PATH", "/nonexistent/step0_audit_db")
os.environ.setdefault("NUPLAN_SENSOR_PATH", "/nonexistent/step0_audit_sensor")

import cv2
import numpy as np
from PIL import Image, ImageDraw

from process_data.helpers.renderer import camera_params, world_to_camera_T
from tools.audit_step0_0a import reconstruct_scenario
from tools.audit_step0_0c_geometry import agent_corners, light_corners, map_category, map_geometry

EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
COLORS = {"final_visible": (60, 220, 80), "partial": (255, 210, 40), "missing": (245, 60, 60), "unsupported": (190, 70, 230), "outside": (145, 145, 145)}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def truth(row: Mapping[str, str], key: str) -> bool:
    return row.get(key) == "True"


def number(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def keep(heap: list[tuple[float, int, dict[str, str]]], score: float, serial: int, row: Mapping[str, str], k: int) -> None:
    if not math.isfinite(score):
        return
    item = (score, serial, dict(row))
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def candidates(root: Path, k: int) -> list[dict[str, str]]:
    heaps: dict[str, list[tuple[float, int, dict[str, str]]]] = defaultdict(list)
    serial = 0
    for row in read_rows(root / "agent_raster_visibility.csv"):
        serial += 1
        if truth(row, "fov_drop_threshold_64"):
            keep(heaps[f"agent_fov_{row['camera']}"], number(row, "isolated_area_clip_model"), serial, row, k)
        if number(row, "isolated_area_prod_model") >= 16:
            keep(heaps[f"agent_occlusion_{row['camera']}"], number(row, "agent_occlusion_fraction"), serial, row, k)
    for row in read_rows(root / "map_element_visibility.csv"):
        serial += 1
        if not truth(row, "renderer_supported"):
            keep(heaps[f"map_unsupported_{row['camera']}"], -number(row, "distance_min"), serial, row, k)
        overwrite = max(number(row, "map_overwrite_fraction"), number(row, "agent_overwrite_fraction"))
        if number(row, "isolated_area_model") >= 16:
            keep(heaps[f"map_overwrite_{row['camera']}"], overwrite, serial, row, k)
    for row in read_rows(root / "traffic_light_visibility.csv"):
        serial += 1
        overwrite = max(number(row, "light_overwrite_fraction"), number(row, "map_overwrite_fraction"), number(row, "agent_overwrite_fraction"))
        if number(row, "isolated_area_model") >= 4:
            keep(heaps[f"light_overwrite_{row['camera']}"], overwrite, serial, row, k)
    output = []
    for rule, heap in heaps.items():
        for rank, (_, _, row) in enumerate(sorted(heap, reverse=True), 1):
            row["selection_rule"] = rule; row["rank"] = str(rank)
            output.append(row)
    return output


def frame_specs(manifest: Path, tokens: set[str]) -> dict[str, dict[str, Any]]:
    output = {}
    with gzip.open(manifest, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("camera") == "CAM_F0" and row.get("frame_token") in tokens:
                output[row["frame_token"]] = row
    return output


def camera_T(frame: Mapping[str, Any], scenario: Mapping[str, Any], camera: str, shifted: bool) -> tuple[np.ndarray, np.ndarray]:
    if shifted:
        model = camera_params[camera]
        translation = model["sensor2lidar_translation"].copy(); translation[2] += 0.8; translation[0] -= 2
        rotation, K = model["sensor2lidar_rotation"], model["intrinsics"]
    else:
        model = frame["cams"][camera]
        translation, rotation, K = model["sensor2lidar_translation"], model["sensor2lidar_rotation"], model["cam_intrinsic"]
    return world_to_camera_T(np.zeros(3), scenario["ego_heading"], translation, rotation), K


def project(points: np.ndarray, T: np.ndarray, K: np.ndarray, shifted: bool) -> tuple[np.ndarray, np.ndarray]:
    cam = (T[:3, :3] @ points.T + T[:3, 3:4]).T
    q = (K @ cam.T).T
    uv = q[:, :2] / np.maximum(q[:, 2:3], 1e-6)
    if shifted:
        uv[:, 1] -= 20
    return uv, cam[:, 2]


def dashed_line(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thickness: int = 3) -> None:
    a, b = np.asarray(p1, float), np.asarray(p2, float); length = float(np.linalg.norm(b - a))
    if length == 0: return
    for start in np.arange(0, length, 14):
        x = a + (b - a) * (start / length); y = a + (b - a) * (min(start + 8, length) / length)
        cv2.line(image, tuple(np.round(x).astype(int)), tuple(np.round(y).astype(int)), color, thickness, cv2.LINE_AA)


def draw_segments(image: np.ndarray, uv: np.ndarray, depth: np.ndarray, segments: Sequence[tuple[int, int]], color: tuple[int, int, int], dashed: bool) -> None:
    h, w = image.shape[:2]
    for i, j in segments:
        if depth[i] <= 1e-3 or depth[j] <= 1e-3: continue
        p1, p2 = tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[j]).astype(int))
        ok, a, b = cv2.clipLine((0, 0, w - 1, h - 1), p1, p2)
        if ok:
            dashed_line(image, a, b, color) if dashed else cv2.line(image, a, b, color, 3, cv2.LINE_AA)


def element_geometry(row: Mapping[str, str], scenario: Mapping[str, Any]) -> tuple[np.ndarray, list[tuple[int, int]], str]:
    if "agent_index" in row:
        index = int(row["agent_index"]); points = agent_corners(np.asarray(scenario["anns"]["gt_boxes_world"])[index]); return points, list(EDGES), f"agent {index} {row.get('class','')}"
    if "map_feature_id" in row:
        feature_id = row["map_feature_id"]
        feature = next(value for key, value in scenario["map_features"].items() if str(key) == feature_id)
        points2d, _ = map_geometry(feature, map_category(str(feature["type"])))
        if points2d is None:
            fallback = feature.get("polyline", feature.get("polygon"))
            if fallback is None:
                raise ValueError(f"map feature {feature_id} has no drawable geometry")
            points2d = np.asarray(fallback, dtype=np.float32)
        points = np.column_stack([points2d, np.zeros(len(points2d))]); segments = [(i, i + 1) for i in range(len(points) - 1)]
        if "CROSSWALK" in str(feature["type"]) or "SPEED_BUMP" in str(feature["type"]): segments.append((len(points) - 1, 0))
        return points, segments, f"map {row['map_feature_id']} {row.get('raw_ftype','')}"
    index = int(row["traffic_light_index"]); points = light_corners(scenario["traffic_lights"][index]); return points, list(EDGES), f"light {index} lane={row.get('lane_connector_id','')} red={row.get('is_red','')}"


def status(row: Mapping[str, str]) -> str:
    if row.get("drop_reason") == "unsupported_by_renderer": return "unsupported"
    if row.get("drop_reason") in ("behind_camera", "outside_fov_after_clipping", "valid_corner_lt4"): return "outside"
    final_key = "visible_area_prod_render" if "agent_index" in row else "visible_final_render"
    isolated_key = "isolated_area_prod_render" if "agent_index" in row else "isolated_area_render"
    final, isolated = number(row, final_key), number(row, isolated_key)
    if final <= 0: return "missing"
    if final < isolated: return "partial"
    return "final_visible"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step0_alignment_audit/0c_raster_components"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/step0_alignment_audit/0a_norm/manifest.jsonl.gz"))
    parser.add_argument("--real-root", type=Path, default=Path("/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/mini"))
    parser.add_argument("--raster-root", type=Path, default=Path("dataset_norm/rendered_sensor_blobs"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(); rows = candidates(args.output_dir, args.top_k); specs = frame_specs(args.manifest, {row["frame_token"] for row in rows})
    map_cache: dict[str, Any] = {}; frame_cache: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}; index_rows = []
    for row in rows:
        token, camera = row["frame_token"], row["camera"]
        if token not in frame_cache:
            spec = specs[token]
            with Path(spec["metadata_path"]).open("rb") as handle: frame = pickle.load(handle)[int(spec["frame_index"])]
            frame_cache[token] = (frame, reconstruct_scenario(frame, map_cache))
        frame, scenario = frame_cache[token]
        points, segments, label = element_geometry(row, scenario); state = status(row); color = COLORS[state]
        rel = Path(frame["cams"][camera]["data_path"]); real_path, raster_path = args.real_root / rel, args.raster_root / rel
        real = np.asarray(Image.open(real_path).convert("RGB")); overlay = real.copy()
        for shifted in (False, True):
            T, K = camera_T(frame, scenario, camera, shifted); uv, depth = project(points, T, K, shifted)
            draw_segments(overlay, uv, depth, segments, color, shifted)
        raster = np.asarray(Image.open(raster_path).convert("RGB"))[20:-20]
        tiles = [np.asarray(Image.fromarray(image).resize((640, 360))) for image in (real, overlay, raster)]
        panel = Image.fromarray(np.concatenate(tiles, axis=1)); draw = ImageDraw.Draw(panel); draw.rectangle((0, 0, panel.width, 32), fill=(0, 0, 0))
        draw.text((8, 7), f"{token} {camera} | solid=real-calibrated dashed=raster-shifted | {label} | {state} | {row['selection_rule']} rank={row['rank']}", fill=color)
        panel_path = args.output_dir / "panels" / "dual_projection" / f"{row['selection_rule']}_{row['rank']}_{token}_{camera}.png"; panel_path.parent.mkdir(parents=True, exist_ok=True); panel.save(panel_path)
        element_type = "agent" if "agent_index" in row else ("map" if "map_feature_id" in row else "traffic_light")
        element_id = row.get("agent_index", row.get("map_feature_id", row.get("traffic_light_index", "")))
        index_rows.append({"split": "mini", "log": row["log"], "frame_token": token, "camera": camera, "panel_type": "dual_projection", "element_type": element_type, "element_id": element_id,
            "selection_rule": row["selection_rule"], "rank": row["rank"], "real_path": str(real_path), "raster_path": str(raster_path), "panel_path": str(panel_path), "heatmap_npy_path": ""})
    write_csv(args.output_dir / "overlay_visualization_index.csv", index_rows)
    print(f"wrote {len(index_rows)} dual-projection panels", flush=True)


if __name__ == "__main__": main()
