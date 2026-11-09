"""Adapt Waymo Open Motion Dataset (WOMD) scenarios to RAP/OpenScene metadata.

WOMD v1.3.1 ships no camera images, so it is a raster-only source: every
anchor is rendered with the target (NAVSIM) rig by ``ScenarioRenderer``.
The module parses ``Scenario`` protos without tensorflow or the waymo devkit
(``protobuf>=4`` + numpy only) and rebuilds the ``scenario`` dictionary that
``ScenarioRenderer.observe`` consumes.

Frame contract (same as ``create_openscene_metadata.py``):
  * origin = ego rear axle projected to the ground, axes parallel to the
    global frame; rotation is applied by the renderer through ``ego_heading``.
  * ``gt_boxes_world`` z = box-center z minus ego ground z.
  * map features are drawn at z=0 by the renderer (flat-ground assumption).

Map mapping (see probe_womd.py results, 2026-10-28):
  * ``lane``      -> ``LANE`` polygon. WOMD stores only the centerline, so the
                     polygon is the centerline offset by per-side half widths
                     measured to the referenced boundary features. Bike lanes
                     are dropped (nuPlan lanes are vehicle lanes).
  * ``road_edge`` -> ``BOUNDARY`` polyline (edge of the drivable area).
  * ``crosswalk`` -> ``CROSSWALK`` polygon.
  * ``road_line``, ``speed_bump``, ``driveway``, ``stop_sign`` are dropped:
    the official nuPlan pipeline extracts none of them, and 60% of road lines
    are ``SOLID_*`` which the renderer's substring match would paint as road
    boundaries. Only the three exact type strings above are ever emitted.

Traffic lights follow ``is_red = (status == RED)`` from the official pipeline.
WOMD ``UNKNOWN`` states are dropped instead of being painted green.
WOMD has no route, so ``driving_command`` is inferred from the future ego
trajectory with the nuScenes thresholds (it leaks future information).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool

from process_data.helpers.nuscenes_adapter import driving_command_from_track, wrap_to_pi

# -----------------------------------------------------------------------------
# Scenario / Map proto schema (subset of waymo-open-dataset scenario.proto and
# map.proto, checked 2026-10-28). Undeclared fields become unknown fields.
# -----------------------------------------------------------------------------
_D, _F, _I64, _I32, _B, _S, _M = 1, 2, 3, 5, 8, 9, 11  # FieldDescriptorProto.Type
_OPT, _REP = 1, 3

_SCHEMA = {
    "MapPoint": [("x", 1, _D, _OPT), ("y", 2, _D, _OPT), ("z", 3, _D, _OPT)],
    "TrafficSignalLaneState": [
        ("lane", 1, _I64, _OPT), ("state", 2, _I32, _OPT),
        ("stop_point", 3, "MapPoint", _OPT),
    ],
    "DynamicMapState": [("lane_states", 1, "TrafficSignalLaneState", _REP)],
    "BoundarySegment": [
        ("lane_start_index", 1, _I32, _OPT), ("lane_end_index", 2, _I32, _OPT),
        ("boundary_feature_id", 3, _I64, _OPT), ("boundary_type", 4, _I32, _OPT),
    ],
    "LaneCenter": [
        ("speed_limit_mph", 1, _D, _OPT), ("type", 2, _I32, _OPT),
        ("interpolating", 3, _B, _OPT), ("polyline", 8, "MapPoint", _REP),
        ("entry_lanes", 9, _I64, _REP), ("exit_lanes", 10, _I64, _REP),
        ("left_boundaries", 13, "BoundarySegment", _REP),
        ("right_boundaries", 14, "BoundarySegment", _REP),
    ],
    "RoadEdge": [("type", 1, _I32, _OPT), ("polyline", 2, "MapPoint", _REP)],
    "RoadLine": [("type", 1, _I32, _OPT), ("polyline", 2, "MapPoint", _REP)],
    "Crosswalk": [("polygon", 1, "MapPoint", _REP)],
    "MapFeature": [
        ("id", 1, _I64, _OPT), ("lane", 3, "LaneCenter", _OPT),
        ("road_line", 4, "RoadLine", _OPT), ("road_edge", 5, "RoadEdge", _OPT),
        ("crosswalk", 8, "Crosswalk", _OPT),
    ],
    "ObjectState": [
        ("center_x", 2, _D, _OPT), ("center_y", 3, _D, _OPT), ("center_z", 4, _D, _OPT),
        ("length", 5, _F, _OPT), ("width", 6, _F, _OPT), ("height", 7, _F, _OPT),
        ("heading", 8, _F, _OPT), ("velocity_x", 9, _F, _OPT),
        ("velocity_y", 10, _F, _OPT), ("valid", 11, _B, _OPT),
    ],
    "Track": [
        ("id", 1, _I32, _OPT), ("object_type", 2, _I32, _OPT),
        ("states", 3, "ObjectState", _REP),
    ],
    "Scenario": [
        ("timestamps_seconds", 1, _D, _REP), ("tracks", 2, "Track", _REP),
        ("scenario_id", 5, _S, _OPT), ("sdc_track_index", 6, _I32, _OPT),
        ("dynamic_map_states", 7, "DynamicMapState", _REP),
        ("map_features", 8, "MapFeature", _REP),
        ("current_time_index", 10, _I32, _OPT),
    ],
}
_PACKAGE = "womd_adapter"


def _build_scenario_class():
    fdp = descriptor_pb2.FileDescriptorProto(
        name="womd_adapter.proto", package=_PACKAGE, syntax="proto2")
    for msg_name, fields in _SCHEMA.items():
        msg = fdp.message_type.add(name=msg_name)
        for name, number, ftype, label in fields:
            f = msg.field.add(name=name, number=number, label=label)
            if isinstance(ftype, str):
                f.type = _M
                f.type_name = f".{_PACKAGE}.{ftype}"
            else:
                f.type = ftype
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    desc = pool.FindMessageTypeByName(f"{_PACKAGE}.Scenario")
    from google.protobuf import message_factory
    try:
        return message_factory.GetMessageClass(desc)
    except AttributeError:  # protobuf < 4.21
        return message_factory.MessageFactory(pool).GetPrototype(desc)


ScenarioProto = _build_scenario_class()

# -----------------------------------------------------------------------------
# Enums and conventions
# -----------------------------------------------------------------------------
TRACK_TYPE_TO_NAME = {1: "vehicle", 2: "pedestrian", 3: "bicycle"}  # nuPlan vocabulary
LANE_TYPE_BIKE = 3
SIGNAL_RED = {1, 4, 7}            # ARROW_STOP, STOP, FLASHING_STOP
SIGNAL_NOT_RED = {2, 3, 5, 6, 8}  # ARROW_CAUTION, ARROW_GO, CAUTION, GO, FLASHING_CAUTION
SIGNAL_UNKNOWN = 0

# nuPlan Pacifica geometry (get_pacifica_parameters): ego poses refer to the
# rear axle, whereas WOMD stores the box center.
REAR_AXLE_TO_CENTER = 1.461
DEFAULT_LANE_HALF_WIDTH = 1.85
LANE_HALF_WIDTH_RANGE = (1.2, 2.5)
_WIDTH_SAMPLES = 16   # centerline samples per boundary segment
_WIDTH_CROP_M = 8.0   # boundary vertices kept around those samples
# Official ``set_light_position`` walks 8 m along the controlled connector.
TRAFFIC_LIGHT_OFFSET = 8.0
# WOMD runs at 10 Hz; nuScenes commands use 20 keyframes at 2 Hz = 10 s.
COMMAND_LOOKAHEAD_FRAMES = 100


def iter_tfrecord(path: Path) -> Iterator[bytes]:
    """Yield raw records of a TFRecord file without checking CRCs."""
    with open(path, "rb") as fh:
        while True:
            head = fh.read(12)
            if not head:
                return
            if len(head) < 12:
                raise IOError(f"{path}: truncated record header")
            (length,) = struct.unpack("<Q", head[:8])
            data = fh.read(length)
            if len(data) < length or len(fh.read(4)) < 4:
                raise IOError(f"{path}: truncated record body")
            yield data


def parse_scenario(raw: bytes):
    sc = ScenarioProto()
    sc.ParseFromString(raw)
    return sc


def _xyz(points) -> np.ndarray:
    return np.array([(p.x, p.y, p.z) for p in points], dtype=np.float64).reshape(-1, 3)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def point_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each 2D point to the nearest segment of a 2D polyline."""
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    return point_to_segments_distance(points, polyline[:-1], polyline[1:])


def point_to_segments_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each 2D point to the nearest of the segments ``a[i]-b[i]``."""
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(axis=2) / denom[None], 0.0, 1.0)
    nearest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - nearest, axis=2).min(axis=1)


def offset_polyline(centerline: np.ndarray, offset: float) -> np.ndarray:
    """Shift a 2D polyline sideways; positive offsets move to the left."""
    tangent = np.gradient(centerline, axis=0) if len(centerline) > 2 else np.repeat(
        centerline[-1:] - centerline[:1], len(centerline), axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norm, 1e-9)
    left_normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return centerline + offset * left_normal


def lane_polygon(centerline: np.ndarray, half_left: float, half_right: float) -> np.ndarray:
    """Closed lane outline: left edge forward, right edge backward."""
    left = offset_polyline(centerline, half_left)
    right = offset_polyline(centerline, -half_right)
    ring = np.concatenate([left, right[::-1]], axis=0)
    return np.concatenate([ring, ring[:1]], axis=0)


def point_along_polyline(polyline: np.ndarray, start: np.ndarray, distance: float) -> Tuple[np.ndarray, float]:
    """Walk up to ``distance`` metres along ``polyline`` from the vertex nearest ``start``.

    Returns the end point and the distance actually walked (shorter when the
    polyline ends first).
    """
    i0 = int(np.argmin(np.linalg.norm(polyline - start[None], axis=1)))
    traversed = 0.0
    for i in range(i0, len(polyline) - 1):
        seg = float(np.linalg.norm(polyline[i + 1] - polyline[i]))
        if seg > 0 and traversed + seg >= distance:
            ratio = (distance - traversed) / seg
            return polyline[i] + ratio * (polyline[i + 1] - polyline[i]), distance
        traversed += seg
    return polyline[-1], traversed


# -----------------------------------------------------------------------------
# Scenario-level containers
# -----------------------------------------------------------------------------
@dataclass
class WomdMap:
    """Renderer-ready map geometry in global coordinates, built once per scenario."""

    lanes: Dict[int, np.ndarray] = field(default_factory=dict)          # id -> closed polygon (N,2)
    lane_centerlines: Dict[int, np.ndarray] = field(default_factory=dict)  # all lanes, for lights
    lane_exits: Dict[int, List[int]] = field(default_factory=dict)
    boundaries: Dict[int, np.ndarray] = field(default_factory=dict)     # id -> polyline (N,2)
    crosswalks: Dict[int, np.ndarray] = field(default_factory=dict)     # id -> closed polygon (N,2)
    points_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # for slope checks
    half_width_measured: int = 0
    half_width_default: int = 0


def _side_half_width(seg_list, centerline: np.ndarray, boundary_lines: Dict[int, np.ndarray]) -> Optional[float]:
    dists = []
    for seg in seg_list:
        line = boundary_lines.get(int(seg.boundary_feature_id))
        if line is None or len(line) < 2:
            continue
        lo = max(int(seg.lane_start_index), 0)
        hi = min(int(seg.lane_end_index), len(centerline) - 1)
        if hi < lo:
            continue
        # Long freeway lanes and boundaries reach thousands of points; a median
        # only needs a few samples and the boundary vertices next to them.
        pts = centerline[lo:hi + 1:max(1, (hi - lo + 1) // _WIDTH_SAMPLES)]
        near = np.all((line >= pts.min(axis=0) - _WIDTH_CROP_M) & (line <= pts.max(axis=0) + _WIDTH_CROP_M), axis=1)
        keep = near[:-1] | near[1:]  # segments with an endpoint in the crop
        if not keep.any():
            continue
        dists.append(point_to_segments_distance(pts, line[:-1][keep], line[1:][keep]))
    if not dists:
        return None
    return float(np.median(np.concatenate(dists)))


def build_map(sc) -> WomdMap:
    out = WomdMap()
    road_lines: Dict[int, np.ndarray] = {}
    lane_msgs = []
    all_pts = []
    for mf in sc.map_features:
        fid = int(mf.id)
        if mf.HasField("lane"):
            pts = _xyz(mf.lane.polyline)
            if len(pts) >= 2:
                out.lane_centerlines[fid] = pts[:, :2]
                out.lane_exits[fid] = [int(x) for x in mf.lane.exit_lanes]
                lane_msgs.append((fid, mf.lane, pts[:, :2]))
                all_pts.append(pts)
        elif mf.HasField("road_edge"):
            pts = _xyz(mf.road_edge.polyline)
            if len(pts) >= 2:
                out.boundaries[fid] = pts[:, :2]
                all_pts.append(pts)
        elif mf.HasField("road_line"):
            pts = _xyz(mf.road_line.polyline)
            if len(pts) >= 2:
                road_lines[fid] = pts[:, :2]  # only used to measure lane widths
        elif mf.HasField("crosswalk"):
            pts = _xyz(mf.crosswalk.polygon)
            if len(pts) >= 3:
                ring = pts[:, :2]
                out.crosswalks[fid] = np.concatenate([ring, ring[:1]], axis=0)
                all_pts.append(pts)

    boundary_lines = {**road_lines, **out.boundaries}
    lo, hi = LANE_HALF_WIDTH_RANGE
    for fid, lane, center in lane_msgs:
        if lane.type == LANE_TYPE_BIKE:
            continue
        left = _side_half_width(lane.left_boundaries, center, boundary_lines)
        right = _side_half_width(lane.right_boundaries, center, boundary_lines)
        if left is None and right is None:
            left = right = DEFAULT_LANE_HALF_WIDTH
            out.half_width_default += 1
        else:
            left = right if left is None else left
            right = left if right is None else right
            out.half_width_measured += 1
        out.lanes[fid] = lane_polygon(center, float(np.clip(left, lo, hi)), float(np.clip(right, lo, hi)))

    if all_pts:
        out.points_xyz = np.concatenate(all_pts, axis=0)
    return out


@dataclass
class EgoTrack:
    center_xyz: np.ndarray   # (T,3) box center
    rear_xy: np.ndarray      # (T,2) rear axle
    ground_z: np.ndarray     # (T,)
    heading: np.ndarray      # (T,)
    velocity: np.ndarray     # (T,2) global
    valid: np.ndarray        # (T,) bool
    size: np.ndarray         # (3,) length, width, height


def ego_track(sc) -> EgoTrack:
    st = sc.tracks[sc.sdc_track_index].states
    center = np.array([(s.center_x, s.center_y, s.center_z) for s in st], dtype=np.float64)
    heading = np.array([s.heading for s in st], dtype=np.float64)
    height = np.array([s.height for s in st], dtype=np.float64)
    direction = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    valid = np.array([s.valid for s in st], dtype=bool)
    first = int(np.argmax(valid)) if valid.any() else 0
    return EgoTrack(
        center_xyz=center,
        rear_xy=center[:, :2] - REAR_AXLE_TO_CENTER * direction,
        ground_z=center[:, 2] - height / 2.0,
        heading=heading,
        velocity=np.array([(s.velocity_x, s.velocity_y) for s in st], dtype=np.float64),
        valid=valid,
        size=np.array([st[first].length, st[first].width, st[first].height], dtype=np.float64),
    )


def anchor_indices(valid: np.ndarray, horizon: int = 40, stride: int = 5, start: int = 1) -> List[int]:
    """Frames with t-1 (for acceleration), t and a full ``horizon`` all valid."""
    T = len(valid)
    return [t for t in range(start, T - horizon, stride) if valid[t - 1:t + horizon + 1].all()]


def slope_p95(points_xyz: np.ndarray, ego_xy: np.ndarray, ground_z: float,
              r_min: float = 30.0, r_max: float = 60.0) -> Optional[float]:
    """p95 of |map z - ego ground z| for map points in an annulus around the ego."""
    if len(points_xyz) == 0:
        return None
    d = np.linalg.norm(points_xyz[:, :2] - ego_xy[None], axis=1)
    sel = (d >= r_min) & (d < r_max)
    if not sel.any():
        return None
    return float(np.percentile(np.abs(points_xyz[sel, 2] - ground_z), 95))


# -----------------------------------------------------------------------------
# Per-anchor scenario
# -----------------------------------------------------------------------------
def _within(points: np.ndarray, center: np.ndarray, radius: float) -> bool:
    return bool(np.min(np.linalg.norm(points - center[None], axis=1)) <= radius)


def query_map(womd_map: WomdMap, ego_xy: np.ndarray, radius: float) -> Dict[str, Dict[str, Any]]:
    """Renderer ``map_features`` around the ego, translated to the ego origin."""
    out: Dict[str, Dict[str, Any]] = {}
    spec = (
        ("lane", "LANE", "polygon", womd_map.lanes),
        ("crosswalk", "CROSSWALK", "polygon", womd_map.crosswalks),
        ("boundary", "BOUNDARY", "polyline", womd_map.boundaries),
    )
    for prefix, ftype, key, layer in spec:
        for fid, pts in layer.items():
            if _within(pts, ego_xy, radius):
                out[f"{prefix}_{fid}"] = {"type": ftype, key: (pts - ego_xy[None]).astype(np.float32)}
    return out


def build_annotations(sc, t: int, ego_xy: np.ndarray, ego_ground_z: float, ego_yaw: float) -> Dict[str, Any]:
    """``info['anns']`` for agents valid at frame ``t`` (the SDC excluded)."""
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    rot_inv = np.array([[c, s], [-s, c]])
    rows_local, rows_world, names, velocities, track_ids = [], [], [], [], []
    for i, tr in enumerate(sc.tracks):
        if i == sc.sdc_track_index or tr.object_type not in TRACK_TYPE_TO_NAME:
            continue
        st = tr.states[t]
        if not st.valid:
            continue
        delta = np.array([st.center_x - ego_xy[0], st.center_y - ego_xy[1], st.center_z - ego_ground_z])
        local_xy = rot_inv @ delta[:2]
        dims = [float(st.length), float(st.width), float(st.height)]
        rows_local.append([local_xy[0], local_xy[1], delta[2], *dims, wrap_to_pi(st.heading - ego_yaw)])
        rows_world.append([*delta.tolist(), *dims, float(st.heading)])
        names.append(TRACK_TYPE_TO_NAME[tr.object_type])
        v_local = rot_inv @ np.array([st.velocity_x, st.velocity_y])
        velocities.append([v_local[0], v_local[1], 0.0])
        track_ids.append(int(tr.id))
    return dict(
        gt_boxes=np.asarray(rows_local, dtype=np.float32).reshape(-1, 7),
        gt_boxes_world=np.asarray(rows_world, dtype=np.float32).reshape(-1, 7),
        gt_names=np.asarray(names),
        gt_velocity_3d=np.asarray(velocities, dtype=np.float32).reshape(-1, 3),
        # Annotations needs per-frame box tokens; track id + frame is unique within a scenario.
        instance_tokens=[f"{x}_{t}" for x in track_ids],
        track_tokens=[str(x) for x in track_ids],
    )


def walk_lanes(womd_map: WomdMap, lane_id: int, start: np.ndarray, distance: float,
               max_hops: int = 3) -> Tuple[np.ndarray, int]:
    """Walk ``distance`` metres from ``start`` along a lane, continuing into exit lanes.

    WOMD sometimes attaches the signal to the approach lane, which ends at the
    stop point; the official nuPlan light then sits 8 m into the intersection,
    so the walk continues into the straightest exit lane.
    """
    lane = womd_map.lane_centerlines[lane_id]
    pos, walked = point_along_polyline(lane, start, distance)
    hops = 0
    while walked < distance - 1e-6 and hops < max_hops:
        exits = [e for e in womd_map.lane_exits.get(lane_id, []) if e in womd_map.lane_centerlines]
        if not exits:
            break
        end_dir = lane[-1] - lane[-2]

        def turn(e: int) -> float:
            nxt = womd_map.lane_centerlines[e]
            d = nxt[1] - nxt[0]
            return abs(wrap_to_pi(math.atan2(d[1], d[0]) - math.atan2(end_dir[1], end_dir[0])))

        lane_id = min(exits, key=turn)
        lane = womd_map.lane_centerlines[lane_id]
        pos, more = point_along_polyline(lane, lane[0], distance - walked)
        walked += more
        hops += 1
    return pos, hops


def traffic_lights_at(sc, t: int, womd_map: WomdMap, ego_xy: np.ndarray,
                      drop_unknown: bool = True) -> Tuple[List[Tuple[int, bool, List[float]]], Dict[str, int]]:
    """``(lane_id, is_red, [x, y])`` placed 8 m past the stop point along the lane."""
    lights: List[Tuple[int, bool, List[float]]] = []
    stats = {"red": 0, "not_red": 0, "unknown_dropped": 0, "missing_lane": 0, "into_exit_lane": 0}
    if t >= len(sc.dynamic_map_states):
        return lights, stats
    for ls in sc.dynamic_map_states[t].lane_states:
        state = int(ls.state)
        if state == SIGNAL_UNKNOWN and drop_unknown:
            stats["unknown_dropped"] += 1
            continue
        lane = womd_map.lane_centerlines.get(int(ls.lane))
        if lane is None:
            stats["missing_lane"] += 1
            continue
        stop = np.array([ls.stop_point.x, ls.stop_point.y]) if ls.HasField("stop_point") else lane[0]
        pos, hops = walk_lanes(womd_map, int(ls.lane), stop, TRAFFIC_LIGHT_OFFSET)
        pos = pos - ego_xy
        is_red = state in SIGNAL_RED
        stats["red" if is_red else "not_red"] += 1
        stats["into_exit_lane"] += hops > 0
        lights.append((int(ls.lane), bool(is_red), pos.tolist()))
    return lights, stats


def build_scenario(sc, t: int, womd_map: WomdMap, ego: EgoTrack,
                   map_radius: float = 200.0, drop_unknown_lights: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(scenario, extras)`` for anchor frame ``t``.

    ``scenario`` is the ``ScenarioRenderer.observe`` input; ``extras`` carries
    the ego pose, driving command and diagnostics for metadata and QA.
    """
    ego_xy = ego.rear_xy[t]
    ego_yaw = float(ego.heading[t])
    ground_z = float(ego.ground_z[t])
    lights, light_stats = traffic_lights_at(sc, t, womd_map, ego_xy, drop_unknown_lights)
    scenario = {
        "ego_pos": ego_xy.tolist(),
        "ego_heading": ego_yaw,
        "traffic_lights": lights,
        "map_features": query_map(womd_map, ego_xy, map_radius),
        "anns": build_annotations(sc, t, ego_xy, ground_z, ego_yaw),
    }
    valid_until = t + int(np.argmin(ego.valid[t:])) if not ego.valid[t:].all() else len(ego.valid)
    command = driving_command_from_track(
        ego.rear_xy[:valid_until], ego.heading[:valid_until], t,
        lookahead_frames=COMMAND_LOOKAHEAD_FRAMES,
    )
    extras = {
        "scenario_id": sc.scenario_id,
        "frame_index": t,
        "ego_rear_xy": ego_xy,
        "ego_heading": ego_yaw,
        "ego_ground_z": ground_z,
        "ego_speed": float(np.linalg.norm(ego.velocity[t])),
        "driving_command": command,
        "slope_p95_30_60m": slope_p95(womd_map.points_xyz, ego_xy, ground_z),
        "lights": light_stats,
    }
    return scenario, extras
