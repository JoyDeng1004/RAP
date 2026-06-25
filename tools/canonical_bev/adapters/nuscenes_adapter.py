"""nuScenes (nuscenes-devkit + map expansion) -> ego-frame SceneMeta.

Pure metadata -> canonical BEV; no RAP code involved.  nuScenes ego frame is
already x-forward / y-left, matching our convention, so the global->ego step is a
plain inverse of the LIDAR_TOP-timestamp ego pose.

Prereqs (NOT installed on this dev machine; runs where data lives):
    pip install nuscenes-devkit
    nuScenes data root with the map-expansion under <root>/maps/.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import CanonBEVConfig
from ..scene_meta import SceneMeta

# raw nuScenes category prefix -> canonical name
_CATEGORY_PREFIX = [
    ("vehicle.", "vehicle"),
    ("human.pedestrian.", "pedestrian"),
    ("cyclist", "bicycle"),
]

# nuScenes map polygon layers -> SceneMeta source keys
_POLYGON_LAYERS = {
    "drivable": ["drivable_area"],
    "walkway": ["walkway", "ped_crossing"],
}
# nuScenes map line layers -> SceneMeta source keys
_POLYLINE_LAYERS = {
    "centerline": ["lane_divider", "road_divider"],  # dividers first; true lane
                                                      # centerline added below if available
}


def _normalize_name(cat: str) -> str:
    for pre, norm in _CATEGORY_PREFIX:
        if cat.startswith(pre):
            return norm
    return cat


def _quat_yaw(q) -> float:
    """Yaw (rad) of a (w, x, y, z) quaternion about +z."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def sample_to_meta(nusc, nusc_map, sample_token: str,
                   cfg: Optional[CanonBEVConfig] = None) -> SceneMeta:
    """Convert one nuScenes ``sample`` into an ego-frame ``SceneMeta``."""
    cfg = cfg or CanonBEVConfig()
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    t = np.array(ego["translation"], dtype=np.float64)          # global
    eyaw = _quat_yaw(ego["rotation"])
    c, s = np.cos(eyaw), np.sin(eyaw)

    def to_ego(points_xy: np.ndarray) -> np.ndarray:
        d = np.asarray(points_xy, dtype=np.float64) - t[:2]
        x = c * d[:, 0] + s * d[:, 1]
        y = -s * d[:, 0] + c * d[:, 1]
        return np.stack([x, y], axis=1).astype(np.float32)

    # --- agents ---
    boxes, names = [], []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cx, cy = to_ego(np.array([ann["translation"][:2]]))[0]
        w, l, h = ann["size"]                      # nuScenes order = (w, l, h)
        yaw = _quat_yaw(ann["rotation"]) - eyaw
        boxes.append([cx, cy, ann["translation"][2], l, w, h, yaw])
        names.append(_normalize_name(ann["category_name"]))
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)

    # --- map ---
    radius = cfg.canon_bev_range * 1.5
    map_polygons = {k: [] for k in _POLYGON_LAYERS}
    map_polylines = {k: [] for k in _POLYLINE_LAYERS}
    records = nusc_map.get_records_in_radius(
        float(t[0]), float(t[1]), radius,
        sorted({l for v in list(_POLYGON_LAYERS.values()) + list(_POLYLINE_LAYERS.values()) for l in v}),
    )
    for key, layers in _POLYGON_LAYERS.items():
        for layer in layers:
            for tok in records.get(layer, []):
                for poly in _polygon_coords(nusc_map, layer, tok):
                    map_polygons[key].append(to_ego(poly))
    for key, layers in _POLYLINE_LAYERS.items():
        for layer in layers:
            for tok in records.get(layer, []):
                line = _line_coords(nusc_map, layer, tok)
                if line is not None:
                    map_polylines[key].append(to_ego(line))

    # best-effort true lane centerlines via arcline discretization
    try:
        _add_lane_centerlines(nusc_map, float(t[0]), float(t[1]), radius,
                              to_ego, map_polylines["centerline"])
    except Exception:
        pass

    return SceneMeta(
        boxes=boxes, names=names,
        map_polygons=map_polygons, map_polylines=map_polylines,
        ego_size=cfg.default_ego_size,
        meta={"dataset": "nuscenes", "sample_token": sample_token,
              "location": nusc_map.map_name},
    )


def _polygon_coords(nusc_map, layer, token):
    """Yield exterior-ring coord arrays for a polygon-ish record."""
    rec = nusc_map.get(layer, token)
    poly_tokens = rec.get("polygon_tokens") or ([rec["polygon_token"]] if "polygon_token" in rec else [])
    for pt in poly_tokens:
        poly = nusc_map.extract_polygon(pt)
        yield np.array(poly.exterior.coords)


def _line_coords(nusc_map, layer, token):
    rec = nusc_map.get(layer, token)
    if "line_token" not in rec:
        return None
    line = nusc_map.extract_line(rec["line_token"])
    if line.is_empty:
        return None
    return np.array(line.coords)


def _add_lane_centerlines(nusc_map, x, y, radius, to_ego, out_list):
    from nuscenes.map_expansion.arcline_path_utils import discretize_lane
    recs = nusc_map.get_records_in_radius(x, y, radius, ["lane", "lane_connector"])
    for layer in ["lane", "lane_connector"]:
        for tok in recs.get(layer, []):
            path = nusc_map.arcline_path_3.get(tok)
            if not path:
                continue
            pts = np.array(discretize_lane(path, resolution_meters=1.0))[:, :2]
            if len(pts) >= 2:
                out_list.append(to_ego(pts))


def load_sample_meta(dataroot: str, sample_token: str, version: str = "v1.0-trainval",
                     cfg: Optional[CanonBEVConfig] = None) -> SceneMeta:
    """Convenience loader: open NuScenes + the right NuScenesMap, then convert."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=log["location"])
    return sample_to_meta(nusc, nusc_map, sample_token, cfg=cfg)
