"""NAVSIM (OpenScene / nuPlan map) -> ego-frame SceneMeta.

Reads metadata only (boxes + map + ego). No RAP training code is touched.
The global->ego transform mirrors RAP's
``RAPTargetBuilder._geometry_local_coords`` (rap_features.py): translate to ego,
then rotate by -heading, so +x = forward, +y = left.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import CanonBEVConfig
from ..scene_meta import SceneMeta

# Map-layer groups -> SceneMeta source keys (filled lazily to avoid importing
# nuplan at module import time).
_POLYGON_GROUPS = {
    "drivable": ["LANE", "INTERSECTION"],
    "walkway": ["WALKWAYS", "CROSSWALK"],
}
_POLYLINE_GROUPS = {
    "centerline": ["LANE", "LANE_CONNECTOR"],
}

# raw navsim category -> canonical name
_NAME_MAP = {
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
}


def _to_ego(points_xy: np.ndarray, origin) -> np.ndarray:
    """(M,2) global xy -> ego xy.  origin: (ex, ey, eheading)."""
    ex, ey, eh = origin
    c, s = np.cos(eh), np.sin(eh)
    d = np.asarray(points_xy, dtype=np.float64) - np.array([ex, ey])
    x = c * d[:, 0] + s * d[:, 1]
    y = -s * d[:, 0] + c * d[:, 1]
    return np.stack([x, y], axis=1).astype(np.float32)


def scene_to_meta(scene, frame_idx: Optional[int] = None,
                  cfg: Optional[CanonBEVConfig] = None) -> SceneMeta:
    """Convert a navsim ``Scene`` (current frame) into a ``SceneMeta``."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    cfg = cfg or CanonBEVConfig()
    if frame_idx is None:
        # last history frame = present observation
        frame_idx = max(0, scene.scene_metadata.num_history_frames - 1)
    frame = scene.frames[frame_idx]

    ego_pose = frame.ego_status.ego_pose  # (x, y, heading) global
    origin = (float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2]))

    # --- agents: global -> ego ---
    boxes_g = np.asarray(frame.annotations.boxes, dtype=np.float32).reshape(-1, 7)
    names_raw = list(frame.annotations.names)
    boxes_e = boxes_g.copy()
    if len(boxes_g):
        centers = _to_ego(boxes_g[:, :2], origin)
        boxes_e[:, 0:2] = centers
        boxes_e[:, 6] = boxes_g[:, 6] - origin[2]
    names = [_NAME_MAP.get(n, n) for n in names_raw]

    # --- map: query proximal objects, transform exteriors / baselines ---
    radius = cfg.canon_bev_range * 1.5
    map_api = scene.map_api
    map_polygons = {k: [] for k in _POLYGON_GROUPS}
    map_polylines = {k: [] for k in _POLYLINE_GROUPS}

    def _layer(name):
        return getattr(SemanticMapLayer, name)

    all_poly_layers = sorted({l for ls in _POLYGON_GROUPS.values() for l in ls})
    all_line_layers = sorted({l for ls in _POLYLINE_GROUPS.values() for l in ls})
    from nuplan.common.actor_state.state_representation import Point2D
    point = Point2D(origin[0], origin[1])
    queried = map_api.get_proximal_map_objects(
        point=point, radius=radius,
        layers=[_layer(n) for n in set(all_poly_layers + all_line_layers)],
    )

    for key, layers in _POLYGON_GROUPS.items():
        for ln in layers:
            for obj in queried.get(_layer(ln), []):
                try:
                    ext = np.array(obj.polygon.exterior.coords)
                except Exception:
                    continue
                map_polygons[key].append(_to_ego(ext, origin))
    for key, layers in _POLYLINE_GROUPS.items():
        for ln in layers:
            for obj in queried.get(_layer(ln), []):
                try:
                    pts = np.array(obj.baseline_path.linestring.coords)
                except Exception:
                    continue
                map_polylines[key].append(_to_ego(pts, origin))

    return SceneMeta(
        boxes=boxes_e, names=names,
        map_polygons=map_polygons, map_polylines=map_polylines,
        ego_size=cfg.default_ego_size,
        meta={"dataset": "navsim", "token": getattr(frame, "token", None),
              "frame_idx": frame_idx},
    )


def load_scene_meta(token: str, data_path: str, sensor_blobs_path: str,
                    cfg: Optional[CanonBEVConfig] = None, **loader_kwargs) -> SceneMeta:
    """Convenience: build a SceneLoader, fetch one scene by token, convert."""
    from pathlib import Path

    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig

    scene_filter = loader_kwargs.pop("scene_filter", SceneFilter())
    loader = SceneLoader(
        data_path=Path(data_path), sensor_blobs_path=Path(sensor_blobs_path),
        scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors(),
        **loader_kwargs,
    )
    scene = loader.get_scene_from_token(token)
    return scene_to_meta(scene, cfg=cfg)
