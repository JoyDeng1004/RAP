"""Dataset-agnostic canonical BEV rasterizer.

Consumes an ego-frame :class:`SceneMeta` and paints a multi-channel raster
following the coordinate convention pinned by tests/canonical_bev/test_coord.py:

    +x = forward, +y = left, ego at center.
    col = size/2 + (y / bev_range) * (size/2)
    row = size/2 - (x / bev_range) * (size/2)
    => forward is "up"; grid_sample lockstep gx = y/bev_range, gy = -x/bev_range.

Channels are driven by ``cfg.channels`` (see config.py); this module knows only
how to paint boxes / polygons / polylines / ego, never which dataset they came
from.
"""

from __future__ import annotations

import cv2
import numpy as np

# Re-exported so the pinned unit test can reach them as canonical_bev.<NAME>.
from .config import (  # noqa: F401
    CanonBEVConfig,
    ChannelSpec,
    CANON_RASTER_CH,
    CH_VEHICLE,
    CH_PEDESTRIAN,
    CH_EGO,
    _VEHICLE_TYPES,
    _PEDESTRIAN_TYPES,
)
from .scene_meta import SceneMeta


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _xy_to_pixel(xy, size, bev_range):
    """Ego-frame (x,y) meters -> raster (col,row) pixels. See module docstring."""
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    half = size / 2.0
    col = half + (xy[:, 1] / bev_range) * half
    row = half - (xy[:, 0] / bev_range) * half
    return np.stack([col, row], axis=1)


def _box_corners(x, y, length, width, yaw):
    """Four corners (ego xy) of an oriented box centered at (x,y)."""
    dx, dy = length / 2.0, width / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _draw(channel, px_colrow, value=1.0, polyline=False, line_thickness=2):
    """Paint one shape into ``channel`` (float32 HxW), accumulating via max."""
    pts = np.round(px_colrow).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) == 0:
        return
    mask = np.zeros(channel.shape, dtype=np.uint8)
    if polyline:
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=line_thickness)
    else:
        cv2.fillPoly(mask, [pts], color=1)
    np.maximum(channel, mask.astype(channel.dtype) * value, out=channel)


# --------------------------------------------------------------------------- #
# per-kind painters  (signatures kept stable for test_coord.py)
# --------------------------------------------------------------------------- #
def _paint_boxes(raster, annotations, types, ch_idx, size, bev_range, value=1.0):
    """Paint footprints of boxes whose normalized name is in ``types``.

    ``annotations`` only needs ``.names`` (list[str]) and ``.boxes`` (N,7) in the
    ego frame: x, y, z, length, width, height, yaw.  Works for both a navsim
    ``Annotations`` and a ``SceneMeta``.
    """
    names = list(annotations.names)
    boxes = np.asarray(annotations.boxes, dtype=np.float32).reshape(-1, 7)
    types = set(types)
    channel = raster[ch_idx]
    for name, box in zip(names, boxes):
        if name not in types:
            continue
        corners = _box_corners(box[0], box[1], box[3], box[4], box[6])
        _draw(channel, _xy_to_pixel(corners, size, bev_range), value)


def _paint_polygons(raster, polys, ch_idx, size, bev_range, value=1.0):
    channel = raster[ch_idx]
    for poly in polys:
        if len(poly) < 3:
            continue
        _draw(channel, _xy_to_pixel(poly, size, bev_range), value)


def _paint_polylines(raster, lines, ch_idx, size, bev_range, thickness=2, value=1.0):
    channel = raster[ch_idx]
    for line in lines:
        if len(line) < 2:
            continue
        _draw(channel, _xy_to_pixel(line, size, bev_range), value,
              polyline=True, line_thickness=thickness)


def _paint_ego(raster, ego_size, ch_idx, size, bev_range, value=1.0):
    length, width = ego_size
    corners = _box_corners(0.0, 0.0, length, width, 0.0)
    _draw(raster[ch_idx], _xy_to_pixel(corners, size, bev_range), value)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_canonical_bev(scene_meta: SceneMeta, cfg: CanonBEVConfig = None) -> np.ndarray:
    """Ego-frame ``SceneMeta`` -> (C, H, W) float32 canonical BEV raster."""
    if cfg is None:
        cfg = CanonBEVConfig()
    size = cfg.canon_raster_size
    bev_range = cfg.canon_bev_range
    raster = np.zeros((cfg.num_channels, size, size), dtype=np.float32)

    for i, spec in enumerate(cfg.channels):
        if spec.kind == "box":
            _paint_boxes(raster, scene_meta, spec.types, i, size, bev_range)
        elif spec.kind == "map_polygon":
            _paint_polygons(raster, scene_meta.map_polygons.get(spec.source, []),
                            i, size, bev_range)
        elif spec.kind == "map_polyline":
            _paint_polylines(raster, scene_meta.map_polylines.get(spec.source, []),
                             i, size, bev_range, thickness=cfg.polyline_thickness)
        elif spec.kind == "ego":
            _paint_ego(raster, scene_meta.ego_size, i, size, bev_range)
        else:
            raise ValueError(f"unknown channel kind {spec.kind!r}")
    return raster
