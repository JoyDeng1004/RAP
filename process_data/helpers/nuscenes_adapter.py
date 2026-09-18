"""Adapt nuScenes records to RAP/OpenScene metadata.

The module builds compact per-scene indexes from nuScenes JSON tables, then
reconstructs ``scenario`` dictionaries for ``ScenarioRenderer`` and
``frame_info`` dictionaries for the NAVSIM ``SceneLoader`` without requiring
the nuPlan, MetaDrive, ScenarioNet, or nuScenes development kits.

``ScenarioRenderer`` expects geometry in an ego-origin, globally aligned frame
and applies rotation through ``ego_heading``. This matches the nuPlan metadata
contract: ``gt_boxes_world`` is globally oriented after only subtracting the
ego translation.

nuScenes does not provide ``roadblock_ids`` or route-based driving commands.
Future lane tokens therefore serve as a route surrogate, and driving commands
are inferred from future ego trajectories with nuPlan-compatible thresholds.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Channel mapping based on optical-axis yaw alignment between the two rigs.
# NAVSIM CAM_L2 and CAM_R2 have no nuScenes counterpart.
NUSC_TO_NAVSIM: Dict[str, str] = {
    "CAM_FRONT": "CAM_F0",
    "CAM_FRONT_LEFT": "CAM_L0",
    "CAM_FRONT_RIGHT": "CAM_R0",
    "CAM_BACK": "CAM_B0",
    "CAM_BACK_LEFT": "CAM_L1",
    "CAM_BACK_RIGHT": "CAM_R1",
}
NAVSIM_TO_NUSC: Dict[str, str] = {v: k for k, v in NUSC_TO_NAVSIM.items()}

# Channels consumed by RAP-DINO.
RAP_DINO_CHANNELS: Tuple[str, ...] = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")

# Map fine-grained nuScenes categories to the nuPlan training vocabulary.
_CATEGORY_PREFIX_MAP: Tuple[Tuple[str, str], ...] = (
    ("human.pedestrian", "pedestrian"),
    ("vehicle.bicycle", "bicycle"),
    ("vehicle.motorcycle", "bicycle"),
    ("vehicle.", "vehicle"),
    ("movable_object.trafficcone", "traffic_cone"),
    ("movable_object.barrier", "barrier"),
)
# Keep the filtered classes aligned with ``create_openscene_metadata.py``.
FILTERED_CLASSES: Tuple[str, ...] = ("traffic_cone", "barrier", "czone_sign", "generic_object")


def map_category(nuscenes_category: str) -> str:
    for prefix, mapped in _CATEGORY_PREFIX_MAP:
        if nuscenes_category.startswith(prefix):
            return mapped
    return "generic_object"


# -----------------------------------------------------------------------------
# Basic geometry
# -----------------------------------------------------------------------------
def quaternion_to_rotation(q: Sequence[float]) -> np.ndarray:
    """Convert a nuScenes ``[w, x, y, z]`` quaternion to a 3x3 matrix."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_yaw(rotation: np.ndarray) -> float:
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def quaternion_to_yaw(q: Sequence[float]) -> float:
    return rotation_to_yaw(quaternion_to_rotation(q))


def wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


# =============================================================================
# Stage 1: index construction
# =============================================================================
_BIG_TABLES = ("sample_data", "ego_pose", "sample_annotation")


def _load_table(version_root: Path, name: str) -> List[Dict[str, Any]]:
    return json.loads((version_root / f"{name}.json").read_text())


def build_index(
    version_root: Path,
    index_dir: Path,
    version: str,
    progress: bool = True,
) -> Dict[str, Any]:
    """Split nuScenes tables into compact per-scene pickle indexes.

    This one-time step is memory-intensive; later render shards read only the
    compact indexes.
    """
    def say(msg: str) -> None:
        if progress:
            print(f"[index] {msg}", flush=True)

    index_dir.mkdir(parents=True, exist_ok=True)

    say("读取小表 ...")
    scenes = _load_table(version_root, "scene")
    logs = {r["token"]: r for r in _load_table(version_root, "log")}
    sensors = {r["token"]: r for r in _load_table(version_root, "sensor")}
    calibrated = {r["token"]: r for r in _load_table(version_root, "calibrated_sensor")}
    instances = {r["token"]: r for r in _load_table(version_root, "instance")}
    categories = {r["token"]: r for r in _load_table(version_root, "category")}
    samples = _load_table(version_root, "sample")

    # Materialize channels to avoid carrying the sensor table into render jobs.
    for cs in calibrated.values():
        cs["channel"] = sensors[cs["sensor_token"]]["channel"]

    samples_by_scene: Dict[str, List[str]] = {}
    sample_meta: Dict[str, Dict[str, Any]] = {}
    for rec in samples:
        sample_meta[rec["token"]] = {
            "timestamp": int(rec["timestamp"]),
            "scene_token": rec["scene_token"],
            "prev": rec["prev"],
            "next": rec["next"],
        }
    del samples

    # Follow each scene's ``next`` chain because ``sample.json`` is not ordered.
    for scene in scenes:
        chain: List[str] = []
        token = scene["first_sample_token"]
        while token:
            chain.append(token)
            token = sample_meta[token]["next"]
        samples_by_scene[scene["token"]] = chain

    say("读取 sample_data（大表，只保留 keyframe 的相机与 LIDAR_TOP）...")
    keyframe_sd: Dict[str, Dict[str, Dict[str, Any]]] = {}
    needed_ego_pose: set = set()
    for rec in _load_table(version_root, "sample_data"):
        if not rec["is_key_frame"]:
            continue
        cs = calibrated.get(rec["calibrated_sensor_token"])
        if cs is None:
            continue
        channel = cs["channel"]
        if not (channel.startswith("CAM") or channel == "LIDAR_TOP"):
            continue
        keyframe_sd.setdefault(rec["sample_token"], {})[channel] = {
            "filename": rec["filename"],
            "width": int(rec.get("width") or 0),
            "height": int(rec.get("height") or 0),
            "cs_token": rec["calibrated_sensor_token"],
            "ego_pose_token": rec["ego_pose_token"],
            "timestamp": int(rec["timestamp"]),
        }
        needed_ego_pose.add(rec["ego_pose_token"])

    say(f"  keyframe sample_data: {sum(len(v) for v in keyframe_sd.values())} 条")

    say("读取 ego_pose（大表，只保留被 keyframe 引用的）...")
    ego_poses: Dict[str, Dict[str, Any]] = {}
    for rec in _load_table(version_root, "ego_pose"):
        if rec["token"] in needed_ego_pose:
            ego_poses[rec["token"]] = {
                "translation": rec["translation"],
                "rotation": rec["rotation"],
                "timestamp": int(rec["timestamp"]),
            }
    del needed_ego_pose

    say("读取 sample_annotation（大表，顺带算速度）...")
    ann_raw = _load_table(version_root, "sample_annotation")
    ann_by_token = {r["token"]: r for r in ann_raw}

    def ann_velocity(rec: Dict[str, Any]) -> List[float]:
        """Estimate box velocity with the nuScenes-devkit finite difference."""
        prev = ann_by_token.get(rec["prev"]) if rec["prev"] else None
        nxt = ann_by_token.get(rec["next"]) if rec["next"] else None
        first = prev if prev is not None else rec
        last = nxt if nxt is not None else rec
        if first is last:
            return [0.0, 0.0, 0.0]
        t0 = sample_meta[first["sample_token"]]["timestamp"] * 1e-6
        t1 = sample_meta[last["sample_token"]]["timestamp"] * 1e-6
        dt = t1 - t0
        if dt <= 0:
            return [0.0, 0.0, 0.0]
        p0 = np.asarray(first["translation"], dtype=np.float64)
        p1 = np.asarray(last["translation"], dtype=np.float64)
        return ((p1 - p0) / dt).tolist()

    anns_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for rec in ann_raw:
        instance = instances[rec["instance_token"]]
        anns_by_sample.setdefault(rec["sample_token"], []).append(
            {
                "token": rec["token"],
                "instance_token": rec["instance_token"],
                "category": categories[instance["category_token"]]["name"],
                "translation": rec["translation"],
                "size": rec["size"],  # nuScenes: [width, length, height]
                "rotation": rec["rotation"],
                "velocity": ann_velocity(rec),  # Global-frame m/s.
                "num_pts": int(rec.get("num_lidar_pts", 0)) + int(rec.get("num_radar_pts", 0)),
            }
        )
    del ann_raw, ann_by_token, instances, categories

    say("写出每个 scene 的索引 ...")
    manifest: List[Dict[str, Any]] = []
    for scene in scenes:
        chain = samples_by_scene[scene["token"]]
        log = logs[scene["log_token"]]
        cs_tokens = {
            sd["cs_token"] for tok in chain for sd in keyframe_sd.get(tok, {}).values()
        }
        ep_tokens = {
            sd["ego_pose_token"] for tok in chain for sd in keyframe_sd.get(tok, {}).values()
        }
        payload = {
            "version": version,
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "description": scene.get("description", ""),
            "log_token": scene["log_token"],
            "log_name": log["logfile"],
            "location": log["location"],
            "vehicle": log.get("vehicle", ""),
            "date_captured": log.get("date_captured", ""),
            "sample_tokens": chain,
            "samples": {tok: sample_meta[tok] for tok in chain},
            "keyframe_sd": {tok: keyframe_sd.get(tok, {}) for tok in chain},
            "ego_poses": {tok: ego_poses[tok] for tok in ep_tokens if tok in ego_poses},
            "calibrated_sensor": {tok: calibrated[tok] for tok in cs_tokens},
            "anns": {tok: anns_by_sample.get(tok, []) for tok in chain},
        }
        out = index_dir / f"{scene['name']}.pkl"
        with open(out, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        manifest.append(
            {
                "scene_name": scene["name"],
                "scene_token": scene["token"],
                "location": log["location"],
                "log_name": log["logfile"],
                "num_samples": len(chain),
            }
        )

    manifest_payload = {"version": version, "num_scenes": len(manifest), "scenes": manifest}
    (index_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))
    say(f"完成：{len(manifest)} 个 scene -> {index_dir}")
    return manifest_payload


def load_scene_index(index_dir: Path, scene_name: str) -> Dict[str, Any]:
    with open(index_dir / f"{scene_name}.pkl", "rb") as fp:
        return pickle.load(fp)


def load_manifest(index_dir: Path) -> Dict[str, Any]:
    return json.loads((index_dir / "manifest.json").read_text())


# =============================================================================
# Map cache
# =============================================================================
@dataclass
class _MapLayer:
    """Vectorized cache for one map-feature layer.

    ``points`` concatenates global-coordinate geometries; ``offsets`` splits
    them, while centroids and radii support radius-based prefiltering.
    """

    ids: List[str]
    points: np.ndarray       # (M, 2) global float64 coordinates.
    offsets: np.ndarray      # (N + 1,) int64 offsets.
    centroids: np.ndarray    # (N, 2) float64 centroids.
    radii: np.ndarray        # (N,) float64 bounding radii.


MAP_LAYER_KEYS: Tuple[str, ...] = ("lane", "crosswalk", "boundary")

# Package format version. Version 3 uses drivable-area boundaries after shared
# interior seams have been removed; incompatible packages must not be reused.
MAP_PACK_VERSION = 3

# Split long boundaries so each segment has a useful centroid and bounding
# radius for ``query()`` prefiltering.
RING_CHUNK_POINTS = 50


def _ring_edges(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    """Return adjacent ring-node pairs, including the closing edge."""
    if len(tokens) < 2:
        return []
    closed = list(tokens)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    return list(zip(closed, closed[1:]))


def _chunk_polyline(points: np.ndarray, chunk: int = RING_CHUNK_POINTS) -> List[np.ndarray]:
    """Split a long open polyline into overlapping fixed-size segments.

    The caller controls closure; automatically closing seam-filtered runs
    would introduce spurious map-spanning boundary segments.
    """
    if len(points) < 2:
        return []
    if len(points) <= chunk:
        return [points]
    # Adjacent chunks share one endpoint to avoid a visible gap.
    step = chunk - 1
    return [points[i: i + chunk] for i in range(0, len(points) - 1, step)]


def _pack_layer(ids: List[str], polys: List[np.ndarray]) -> _MapLayer:
    if not polys:
        return _MapLayer([], np.zeros((0, 2)), np.zeros(1, np.int64), np.zeros((0, 2)), np.zeros(0))
    offsets = np.zeros(len(polys) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(p) for p in polys])
    points = np.concatenate(polys, axis=0)
    centroids = np.stack([p.mean(axis=0) for p in polys])
    radii = np.array([np.linalg.norm(p - c, axis=1).max() for p, c in zip(polys, centroids)])
    return _MapLayer(ids, points, offsets, centroids, radii)


class MapCache:
    """Cache vectorized nuScenes expansion maps by location.

    Layers mirror the renderer's nuPlan semantics: lane and lane-connector
    polygons, pedestrian-crossing polygons, and seam-free drivable-area
    boundaries. Packaged NumPy arrays avoid repeatedly parsing expansion JSON
    in parallel render workers.
    """

    def __init__(self, map_root: Path, pack_dir: Optional[Path] = None):
        self._map_root = Path(map_root)
        self._pack_dir = Path(pack_dir) if pack_dir is not None else None
        self._cache: Dict[str, Dict[str, _MapLayer]] = {}

    def _load_packed(self, location: str) -> Optional[Dict[str, _MapLayer]]:
        if self._pack_dir is None:
            return None
        path = self._pack_dir / f"{location}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            version = int(data["pack_version"][0]) if "pack_version" in data else 1
            if version != MAP_PACK_VERSION:
                raise SystemExit(
                    f"FAIL 地图包版本不符: {path} 是 v{version}，当前代码要 "
                    f"v{MAP_PACK_VERSION}。\n"
                    f"     v1: boundary = lane_divider/road_divider（分道线，在路中央）\n"
                    f"     v2: boundary = drivable_area 每块多边形的环（含内部接缝）\n"
                    f"     v3: boundary = 去掉接缝后的区域外轮廓（等价 unary_union）\n"
                    f"     语义不同，混用会让红线画在错的地方且【不报错】。\n"
                    f"     请重跑 build-index 重新打包，并重渲受影响的 raster。"
                )
            return {
                key: _MapLayer(
                    ids=[str(v) for v in data[f"{key}_ids"]],
                    points=data[f"{key}_points"],
                    offsets=data[f"{key}_offsets"],
                    centroids=data[f"{key}_centroids"],
                    radii=data[f"{key}_radii"],
                )
                for key in MAP_LAYER_KEYS
            }

    def pack_to_npz(self, location: str, pack_dir: Path) -> Path:
        """Package one map as an NPZ archive for direct worker loading."""
        pack_dir.mkdir(parents=True, exist_ok=True)
        layers = self._build(location)
        payload: Dict[str, np.ndarray] = {
            "pack_version": np.asarray([MAP_PACK_VERSION], dtype=np.int64)
        }
        for key in MAP_LAYER_KEYS:
            layer = layers[key]
            payload[f"{key}_ids"] = np.asarray(layer.ids, dtype=object).astype("U")
            payload[f"{key}_points"] = layer.points
            payload[f"{key}_offsets"] = layer.offsets
            payload[f"{key}_centroids"] = layer.centroids
            payload[f"{key}_radii"] = layer.radii
        out = pack_dir / f"{location}.npz"
        np.savez_compressed(out, **payload)
        return out

    def _build(self, location: str) -> Dict[str, _MapLayer]:
        data = json.loads((self._map_root / f"{location}.json").read_text())
        nodes = {r["token"]: (r["x"], r["y"]) for r in data["node"]}
        polygons = {r["token"]: r for r in data["polygon"]}
        # Dividers are lane separators, not drivable-area boundaries.

        def node_xy(tokens: Sequence[str]) -> Optional[np.ndarray]:
            pts = [nodes[t] for t in tokens if t in nodes]
            if len(pts) < 2:
                return None
            return np.asarray(pts, dtype=np.float64)

        lane_ids: List[str] = []
        lane_polys: List[np.ndarray] = []
        for layer in ("lane", "lane_connector"):
            for rec in data.get(layer, []):
                poly = polygons.get(rec.get("polygon_token"))
                if poly is None:
                    continue
                pts = node_xy(poly["exterior_node_tokens"])
                if pts is None:
                    continue
                lane_ids.append(f"{layer}:{rec['token']}")
                lane_polys.append(pts)

        cross_ids: List[str] = []
        cross_polys: List[np.ndarray] = []
        for rec in data.get("ped_crossing", []):
            poly = polygons.get(rec.get("polygon_token"))
            if poly is None:
                continue
            pts = node_xy(poly["exterior_node_tokens"])
            if pts is None:
                continue
            cross_ids.append(f"ped_crossing:{rec['token']}")
            cross_polys.append(pts)

        # Boundaries represent drivable-area outer rings and holes. Shared
        # polygon edges are internal seams, not boundaries.
        ring_specs: List[Tuple[str, List[str]]] = []
        for rec in data.get("drivable_area", []):
            for poly_token in rec.get("polygon_tokens", []):
                poly = polygons.get(poly_token)
                if poly is None:
                    continue
                ring_specs.append(
                    (f"drivable_area:{poly_token}", list(poly["exterior_node_tokens"]))
                )
                # Holes are also boundaries of the drivable area.
                for hole_index, hole in enumerate(poly.get("holes") or []):
                    ring_specs.append(
                        (f"drivable_area:{poly_token}:hole{hole_index}",
                         list(hole.get("node_tokens", [])))
                    )

        # Keep only edges owned by one polygon, equivalent to a union boundary
        # when adjacent polygons share node tokens.
        edge_count: Dict[Tuple[str, str], int] = {}
        for _, tokens in ring_specs:
            for a, b in _ring_edges(tokens):
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                edge_count[key] = edge_count.get(key, 0) + 1

        bound_ids: List[str] = []
        bound_polys: List[np.ndarray] = []
        for tag, tokens in ring_specs:
            # Split each ring into open runs separated by internal seams.
            runs: List[List[str]] = []
            current: List[str] = []
            for a, b in _ring_edges(tokens):
                key = (a, b) if a < b else (b, a)
                keep = (
                    a != b
                    and a in nodes
                    and b in nodes
                    and edge_count.get(key, 0) == 1
                )
                if keep:
                    if not current:
                        current = [a]
                    current.append(b)
                elif len(current) >= 2:
                    runs.append(current)
                    current = []
                else:
                    current = []
            if len(current) >= 2:
                runs.append(current)

            for run_index, run in enumerate(runs):
                pts = np.asarray([nodes[t] for t in run], dtype=np.float64)
                for part, chunk in enumerate(_chunk_polyline(pts)):
                    bound_ids.append(f"{tag}:run{run_index}#{part}")
                    bound_polys.append(chunk)

        return {
            "lane": _pack_layer(lane_ids, lane_polys),
            "crosswalk": _pack_layer(cross_ids, cross_polys),
            "boundary": _pack_layer(bound_ids, bound_polys),
        }

    def get(self, location: str) -> Dict[str, _MapLayer]:
        if location not in self._cache:
            packed = self._load_packed(location)
            self._cache[location] = packed if packed is not None else self._build(location)
        return self._cache[location]

    def query(
        self, location: str, center_xy: np.ndarray, radius: float
    ) -> Dict[str, Dict[str, Any]]:
        """Return renderer-compatible map features around an ego-origin center.

        The renderer recognizes only the ``LANE``, ``CROSSWALK``, and
        ``BOUNDARY`` type values used below. Boundaries denote the edge of the
        drivable area rather than lane dividers.
        """
        layers = self.get(location)
        out: Dict[str, Dict[str, Any]] = {}
        spec = (
            ("lane", "LANE", "polygon"),
            ("crosswalk", "CROSSWALK", "polygon"),
            ("boundary", "BOUNDARY", "polyline"),
        )
        for layer_key, ftype, geom_key in spec:
            layer = layers[layer_key]
            if len(layer.ids) == 0:
                continue
            dist = np.linalg.norm(layer.centroids - center_xy[None, :], axis=1) - layer.radii
            for i in np.nonzero(dist <= radius)[0]:
                pts = layer.points[layer.offsets[i]: layer.offsets[i + 1]]
                out[layer.ids[i]] = {
                    "type": ftype,
                    geom_key: (pts - center_xy[None, :]).astype(np.float32),
                }
        return out

    def lane_tokens_at(
        self, location: str, points_xy: np.ndarray, tolerance: float = 3.0
    ) -> List[str]:
        """Return unique nearby lane tokens in first-occurrence order.

        This approximate centroid-and-radius query supplies the
        ``roadblock_ids`` surrogate; ``has_route`` requires only non-emptiness.
        """
        layer = self.get(location)["lane"]
        if len(layer.ids) == 0 or len(points_xy) == 0:
            return []
        seen: Dict[str, None] = {}
        for pt in points_xy:
            dist = np.linalg.norm(layer.centroids - pt[None, :], axis=1) - layer.radii
            hit = np.nonzero(dist <= tolerance)[0]
            if len(hit) == 0:
                continue
            for i in hit[np.argsort(dist[hit])][:2]:
                seen.setdefault(layer.ids[i], None)
        return list(seen)


# =============================================================================
# Frame-level construction
# =============================================================================
def ego_pose_of(scene_index: Dict[str, Any], sample_token: str) -> Dict[str, Any]:
    """Return a stable ego pose for a sample.

    Prefer the LIDAR_TOP pose because camera sample-data timestamps can differ
    slightly within one sample.
    """
    sd = scene_index["keyframe_sd"][sample_token]
    anchor = sd.get("LIDAR_TOP") or sd.get("CAM_FRONT") or next(iter(sd.values()))
    return scene_index["ego_poses"][anchor["ego_pose_token"]]


def ego_track(scene_index: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time-ordered ``(xy(N, 2), yaw(N,), t_seconds(N,))`` arrays."""
    xs, ys, yaws, ts = [], [], [], []
    for tok in scene_index["sample_tokens"]:
        pose = ego_pose_of(scene_index, tok)
        xs.append(pose["translation"][0])
        ys.append(pose["translation"][1])
        yaws.append(quaternion_to_yaw(pose["rotation"]))
        ts.append(pose["timestamp"] * 1e-6)
    return (
        np.stack([np.asarray(xs), np.asarray(ys)], axis=1),
        np.asarray(yaws),
        np.asarray(ts),
    )


def ego_dynamics(xy: np.ndarray, yaw: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return ``[vx, vy, ax, ay, yaw_rate]`` in the ego frame.

    Transforming velocity and acceleration from global coordinates matches the
    nuPlan ``ego_dynamic_state`` contract.
    """
    n = len(xy)
    out = np.zeros((n, 5), dtype=np.float64)
    if n < 2:
        return out
    v_global = np.gradient(xy, t, axis=0)  # (N, 2) global-frame velocity.
    a_global = np.gradient(v_global, t, axis=0)
    yaw_rate = np.gradient(np.unwrap(yaw), t)
    c, s = np.cos(yaw), np.sin(yaw)
    out[:, 0] = c * v_global[:, 0] + s * v_global[:, 1]
    out[:, 1] = -s * v_global[:, 0] + c * v_global[:, 1]
    out[:, 2] = c * a_global[:, 0] + s * a_global[:, 1]
    out[:, 3] = -s * a_global[:, 0] + c * a_global[:, 1]
    out[:, 4] = yaw_rate
    return out


def driving_command_from_track(
    xy: np.ndarray,
    yaw: np.ndarray,
    idx: int,
    distance: float = 20.0,
    lateral_offset: float = 2.0,
    min_arc: float = 5.0,
    yaw_threshold: float = 0.15,
    lookahead_frames: int = 20,
    short_horizon_policy: str = "scaled",
) -> np.ndarray:
    """Infer a one-hot ``left/forward/right/unknown`` driving command.

    The function follows a fixed future-trajectory horizon because nuScenes
    lacks the nuPlan route centerline. Full horizons use ``lateral_offset``;
    partial horizons scale that threshold quadratically; near-stationary
    horizons use yaw change. ``short_horizon_policy='unknown'`` returns
    ``unknown`` whenever the full look-ahead distance is unavailable.

    Commands are inferred from future motion rather than map routes, so they
    may leak future trajectory information and differ from lane-centerline
    commands during lane offsets or lane changes.
    """
    command = np.zeros(4, dtype=int)
    # A fixed horizon prevents labels from depending on the remaining scene length.
    end = idx + max(int(lookahead_frames), 1) + 1
    future = xy[idx:end]
    future_yaw = yaw[idx:end]
    if len(future) < 2:
        command[3] = 1
        return command

    seg = np.linalg.norm(np.diff(future, axis=0), axis=1)
    acc = np.concatenate([[0.0], np.cumsum(seg)])
    arc = float(acc[-1])

    if arc < distance and short_horizon_policy == "unknown":
        # Do not extrapolate incomplete trajectories under the strict policy.
        command[3] = 1
        return command

    if arc < min_arc:
        # Near-stationary trajectories use heading change as the direction cue.
        delta_yaw = wrap_to_pi(float(future_yaw[-1] - future_yaw[0]))
        if abs(delta_yaw) < yaw_threshold:
            command[1] = 1
        elif delta_yaw > 0:
            command[0] = 1
        else:
            command[2] = 1
        return command

    if arc >= distance:
        effective = distance
        threshold = lateral_offset
    else:
        # For constant curvature, lateral displacement scales with arc squared.
        effective = arc
        threshold = lateral_offset * (arc / distance) ** 2

    j = int(np.searchsorted(acc, effective))
    j = min(max(j, 1), len(acc) - 1)
    span = acc[j] - acc[j - 1]
    ratio = 0.0 if span <= 0 else (effective - acc[j - 1]) / span
    target = future[j - 1] + ratio * (future[j] - future[j - 1])

    delta = target - xy[idx]
    c, s = math.cos(yaw[idx]), math.sin(yaw[idx])
    lateral = -s * delta[0] + c * delta[1]

    if lateral >= threshold:
        command[0] = 1
    elif lateral <= -threshold:
        command[2] = 1
    else:
        command[1] = 1
    return command


def build_annotations(
    scene_index: Dict[str, Any],
    sample_token: str,
    ego_translation: np.ndarray,
    ego_yaw: float,
    filter_instance: bool = True,
) -> Dict[str, Any]:
    """Build ``info['anns']`` fields compatible with OpenScene metadata.

    ``gt_boxes`` uses the rotated ego frame; ``gt_boxes_world`` is globally
    oriented around the ego origin for ``ScenarioRenderer``.
    """
    records = scene_index["anns"].get(sample_token, [])
    rows_local, rows_world, names, velocities = [], [], [], []
    inst_tokens, track_tokens = [], []

    rot_inv = quaternion_to_rotation(
        [math.cos(ego_yaw / 2), 0.0, 0.0, math.sin(ego_yaw / 2)]
    ).T

    for rec in records:
        name = map_category(rec["category"])
        if filter_instance and name in FILTERED_CLASSES:
            continue
        center = np.asarray(rec["translation"], dtype=np.float64)
        delta = center - ego_translation
        local = rot_inv @ delta
        width, length, height = (float(v) for v in rec["size"])
        box_yaw = quaternion_to_yaw(rec["rotation"])

        rows_local.append([*local.tolist(), length, width, height, wrap_to_pi(box_yaw - ego_yaw)])
        rows_world.append([*delta.tolist(), length, width, height, box_yaw])
        names.append(name)
        v_global = np.asarray(rec["velocity"], dtype=np.float64)
        velocities.append((rot_inv @ v_global).tolist())
        inst_tokens.append(rec["token"])
        track_tokens.append(rec["instance_token"])

    return dict(
        gt_boxes=np.asarray(rows_local, dtype=np.float32).reshape(-1, 7),
        gt_boxes_world=np.asarray(rows_world, dtype=np.float32).reshape(-1, 7),
        gt_names=np.asarray(names),
        gt_velocity_3d=np.asarray(velocities, dtype=np.float32).reshape(-1, 3),
        instance_tokens=inst_tokens,
        track_tokens=track_tokens,
    )


def build_can_bus(ego_translation: np.ndarray, rotation_q: Sequence[float], dyn: np.ndarray) -> np.ndarray:
    """Build the 18-element ``CanBus.tensor`` layout.

    Pose fields are global; acceleration, velocity, and angular-velocity fields
    use the ego frame to match the nuPlan convention.
    """
    vx, vy, ax, ay, yaw_rate = dyn
    return np.array(
        [
            ego_translation[0], ego_translation[1], ego_translation[2],
            rotation_q[0], rotation_q[1], rotation_q[2], rotation_q[3],
            ax, ay, 0.0,
            vx, vy, 0.0,
            0.0, 0.0, yaw_rate,
            0.0, 0.0,
        ],
        dtype=np.float64,
    )
