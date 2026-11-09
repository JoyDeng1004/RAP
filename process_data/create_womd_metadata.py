#!/usr/bin/env python3
"""Generate RAP metadata pkls and NAVSIM-rig rasters from WOMD scenarios.

WOMD has no camera images, so this is a raster-only source (like the nuScenes
``navsim`` rig with ``use_real=false``). One pkl per WOMD scenario, frames
resampled to 2 Hz, same frame-dict contract as ``create_nuscenes_metadata.py``
so ``SceneLoader`` / ``run_dataset_caching.py`` read it unchanged.

Anchor selection
----------------
A 2 Hz frame can be the current frame of a 4+10 window (scene filter
``num_history_frames=4``, ``num_future_frames=10``). Only *selected* current
frames are rendered and keep ``is_valid=True``; every other frame gets
``is_valid=False`` so ``filter_scenes`` (enable_filter=True, used by the first
pass of feature caching) never builds a sample whose raster does not exist.
Selection, in order:
  1. every ``--anchor-stride`` current frames (offset hashed from scenario id);
  2. hilly scenes dropped (``slope_p95_30_60m > --max-slope``, flat-ground
     renderer would mis-draw the map);
  3. stationary anchors kept with probability ``--stationary-keep``
     (deterministic hash of the token).

Subcommands
-----------
  run    process tfrecord files (sharded over files), write pkls + rasters
  audit  check every valid anchor has all rasters on disk

Cameras: the pkl lists all of ``--camera-channels`` (RAP's F0/L0/R0/B0), but
only ``--render-channels`` are rasterized. The default F0/L0/R0 is the official
``ScenarioRenderer`` default used for NAVSIM, whose B0 raster is therefore
missing and zero-filled by the loader; WOMD matches that.

Paths follow the ``data_split`` contract (``dataclasses.py:85``)::

    <ds-root>/navsim_logs/<split>/<log_name>.pkl
    <ds-root>/sensor_blobs/<split>/                 (empty; never read)
    <ds-root>/rendered_sensor_blobs/<split>/<log_name>/<CAM>/<token>.jpg

Example (pilot, 2 files, metadata only, for the anchor budget)::

    python process_data/create_womd_metadata.py run \
        --womd-root /gs/bs/tga-RLA/qdeng/womd_dataset/v_1_3_1 --womd-split training_20s \
        --ds-root /gs/bs/tga-RLA/qdeng/womd_dataset/rap_raster --limit-files 2 --no-render
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers import womd_adapter as wa  # noqa: E402
from process_data.helpers.nuscenes_adapter import (  # noqa: E402
    RAP_DINO_CHANNELS,
    build_can_bus,
    driving_command_from_track,
    ego_dynamics,
)
from process_data.helpers.renderer import ScenarioRenderer, camera_params  # noqa: E402

WOMD_HZ = 10
FRAME_STEP = 5            # 10 Hz -> 2 Hz (NAVSIM_INTERVAL_LENGTH = 0.5 s)
MAP_LOCATION = "womd"     # not in MAP_LOCATIONS -> scene.map_api is None
VEHICLE_NAME = "womd_sdc"
# ScenarioRenderer's own default, i.e. what the official NAVSIM rasters contain.
OFFICIAL_RENDER_CHANNELS: Tuple[str, ...] = ("CAM_F0", "CAM_L0", "CAM_R0")
RENDER_SIZE_WH = (1920, 1120)  # NAVSIM principal point (960, 560); loader crops [20:-20]


def split_name(womd_split: str) -> str:
    return f"womd_{womd_split}"


def log_name_of(womd_split: str, scenario_id: str) -> str:
    return f"womd-{womd_split}-{scenario_id}"


def stable_unit(text: str) -> float:
    """Deterministic value in [0, 1) so reruns select the same anchors."""
    return (zlib.crc32(text.encode()) & 0xFFFFFFFF) / 2 ** 32


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=np.float64)


# =============================================================================
# Per-scenario frames
# =============================================================================
def valid_run(valid: np.ndarray, anchor: int) -> Tuple[int, int]:
    """``[start, end)`` of the contiguous valid ego run that contains ``anchor``."""
    if not valid[anchor]:
        return anchor, anchor
    start = anchor
    while start > 0 and valid[start - 1]:
        start -= 1
    end = anchor + 1
    while end < len(valid) and valid[end]:
        end += 1
    return start, end


def grid_frames(sc, ego: wa.EgoTrack) -> List[int]:
    """10 Hz indices of the 2 Hz grid through ``current_time_index``.

    The grid lies inside one contiguous valid run, and its first and last
    points keep one valid neighbour each so finite differences stay central.
    """
    cur = int(sc.current_time_index)
    start, end = valid_run(ego.valid, cur)
    first = cur - ((cur - (start + 1)) // FRAME_STEP) * FRAME_STEP
    return list(range(first, end - 1, FRAME_STEP))


def frame_times(sc, num_states: int, stats: Dict[str, int]) -> np.ndarray:
    """Seconds per track state.

    Some training_20s scenarios carry fewer ``timestamps_seconds`` than track
    states (pilot job 9174861); WOMD is a fixed 10 Hz grid, so fall back to it.
    """
    ts = np.asarray(sc.timestamps_seconds, dtype=np.float64)
    if len(ts) == num_states:
        return ts
    stats["timestamps_synthesized"] += 1
    return np.arange(num_states, dtype=np.float64) / WOMD_HZ


def route_lane_ids(womd_map: wa.WomdMap, future_xy: np.ndarray, tolerance: float) -> List[str]:
    """Lanes the future ego path runs along: ``roadblock_ids`` surrogate.

    Like the nuScenes pipeline, only non-emptiness matters to the loader
    (``has_route``); caching runs with has_route=false anyway.
    """
    lane_ids = [i for i in womd_map.lanes if len(womd_map.lane_centerlines.get(i, ())) >= 2]
    if len(future_xy) == 0 or not lane_ids:
        return []
    # (lanes, points): nearest vehicle lane per future point, in path order.
    dist = np.stack([wa.point_to_polyline_distance(future_xy, womd_map.lane_centerlines[i])
                     for i in lane_ids])
    ids: List[str] = []
    for p in range(dist.shape[1]):
        j = int(np.argmin(dist[:, p]))
        if dist[j, p] <= tolerance and str(lane_ids[j]) not in ids:
            ids.append(str(lane_ids[j]))
    return ids


def cams_for(log_name: str, token: str, channels: Tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    """Camera dict: NAVSIM rig for both the (absent) real and the rendered image."""
    cams = {}
    for ch in channels:
        p = camera_params[ch]
        cams[ch] = {
            "data_path": f"{log_name}/{ch}/{token}.jpg",
            "sensor2lidar_rotation": np.asarray(p["sensor2lidar_rotation"], np.float64),
            "sensor2lidar_translation": np.asarray(p["sensor2lidar_translation"], np.float64),
            "cam_intrinsic": np.asarray(p["intrinsics"], np.float64),
            "distortion": np.asarray(p["distortion"], np.float64),
            "render_sensor2lidar_rotation": np.asarray(p["sensor2lidar_rotation"], np.float64),
            "render_sensor2lidar_translation": np.asarray(p["sensor2lidar_translation"], np.float64),
            "render_cam_intrinsic": np.asarray(p["intrinsics"], np.float64),
            "render_image_shape": [RENDER_SIZE_WH[1], RENDER_SIZE_WH[0]],
        }
    return cams


def select_anchors(
    sc, ego: wa.EgoTrack, frames: List[int], womd_map: wa.WomdMap, args: argparse.Namespace,
    stats: Dict[str, int],
) -> List[int]:
    """Indices into ``frames`` (2 Hz) that get rendered and stay valid."""
    h, f = args.num_history_frames, args.num_future_frames
    candidates = list(range(h - 1, len(frames) - f))
    stats["current_frames"] += len(candidates)
    if not candidates:
        return []
    offset = int(stable_unit(sc.scenario_id) * args.anchor_stride)
    candidates = [k for k in candidates if (k - (h - 1) - offset) % args.anchor_stride == 0]
    stats["stride_candidates"] += len(candidates)

    selected = []
    for k in candidates:
        t = frames[k]
        slope = wa.slope_p95(womd_map.points_xyz, ego.rear_xy[t], float(ego.ground_z[t]))
        if slope is not None and slope > args.max_slope:
            stats["dropped_hilly"] += 1
            continue
        t_end = frames[k + f]
        speed = float(np.linalg.norm(ego.velocity[t]))
        travel = float(np.linalg.norm(ego.rear_xy[t_end] - ego.rear_xy[t]))
        if speed < args.stationary_speed and travel < args.stationary_travel:
            stats["stationary"] += 1
            if stable_unit(f"{sc.scenario_id}_{t}") >= args.stationary_keep:
                stats["dropped_stationary"] += 1
                continue
        selected.append(k)
    return selected


def process_scenario(
    sc, womd_split: str, renderer: Optional[ScenarioRenderer], args: argparse.Namespace,
    stats: Dict[str, int],
) -> Optional[List[Dict[str, Any]]]:
    ego = wa.ego_track(sc)
    frames = grid_frames(sc, ego)
    if len(frames) < args.num_history_frames + args.num_future_frames:
        stats["scenarios_too_short"] += 1
        return None
    womd_map = wa.build_map(sc)
    log_name = log_name_of(womd_split, sc.scenario_id)
    anchors = set(select_anchors(sc, ego, frames, womd_map, args, stats))

    # Ego states on the valid run at 10 Hz, then sampled on the 2 Hz grid.
    lo, hi = frames[0] - 1, frames[-1] + 2
    run_xy, run_yaw = ego.rear_xy[lo:hi], ego.heading[lo:hi]
    times = frame_times(sc, len(ego.valid), stats)
    tsec = times[lo:hi]
    dynamics = ego_dynamics(run_xy, run_yaw, tsec)
    # Commands may look past the 2 Hz grid end, up to the end of the valid run.
    cmd_end = valid_run(ego.valid, frames[-1])[1]
    cmd_xy, cmd_yaw = ego.rear_xy[lo:cmd_end], ego.heading[lo:cmd_end]

    channels = tuple(args.camera_channels)
    render_root = Path(args.ds_root) / "rendered_sensor_blobs" / split_name(womd_split)
    tokens = [f"{sc.scenario_id}_{t:03d}" for t in frames]
    infos: List[Dict[str, Any]] = []
    for k, t in enumerate(frames):
        token = tokens[k]
        ego_xy = ego.rear_xy[t]
        ego_yaw = float(ego.heading[t])
        ground_z = float(ego.ground_z[t])
        translation = np.array([ego_xy[0], ego_xy[1], ground_z], dtype=np.float64)
        rotation_q = yaw_quaternion(ego_yaw)
        dyn = dynamics[t - lo]
        anns = wa.build_annotations(sc, t, ego_xy, ground_z, ego_yaw)
        lights, light_stats = wa.traffic_lights_at(
            sc, t, womd_map, ego_xy, drop_unknown=not args.keep_unknown_lights)
        command = driving_command_from_track(
            cmd_xy, cmd_yaw, t - lo, lookahead_frames=wa.COMMAND_LOOKAHEAD_FRAMES)
        future = ego.rear_xy[t:min(t + args.route_lookahead_frames, cmd_end):FRAME_STEP]
        roadblock_ids = route_lane_ids(womd_map, future, args.route_tolerance)

        is_anchor = k in anchors
        if is_anchor:
            stats["anchors"] += 1
            stats["anchors_no_route"] += int(not roadblock_ids)
            for key, value in light_stats.items():
                stats[f"lights_{key}"] += value
            stats[f"command_{('left', 'forward', 'right', 'unknown')[int(np.argmax(command))]}"] += 1
            if renderer is not None:
                scenario = {
                    "ego_pos": ego_xy.tolist(),
                    "ego_heading": ego_yaw,
                    "traffic_lights": lights,
                    "map_features": wa.query_map(womd_map, ego_xy, args.map_radius),
                    "anns": anns,
                }
                for ch, canvas in renderer.observe(scenario).items():
                    target = render_root / log_name / ch / f"{token}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target), canvas[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    stats["images"] += 1

        ego2global = np.eye(4)
        ego2global[:3, :3] = np.array([[math.cos(ego_yaw), -math.sin(ego_yaw), 0.0],
                                       [math.sin(ego_yaw), math.cos(ego_yaw), 0.0],
                                       [0.0, 0.0, 1.0]])
        ego2global[:3, 3] = translation
        # The renderer treats sensor2lidar extrinsics as relative to the ego
        # origin, so the lidar frame coincides with the ego frame here.
        lidar2ego = np.eye(4)
        infos.append({
            "token": token,
            "frame_idx": k,
            "timestamp": int(round(float(times[t]) * 1e6)),
            "log_name": log_name,
            "log_token": log_name,
            "scene_name": log_name,
            "scene_token": log_name,
            "map_location": MAP_LOCATION,
            "roadblock_ids": roadblock_ids,
            "vehicle_name": VEHICLE_NAME,
            "can_bus": build_can_bus(translation, rotation_q, dyn),
            "lidar_path": "",
            "lidar2ego_translation": np.zeros(3),
            "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego2global_translation": translation,
            "ego2global_rotation": rotation_q,
            "ego_dynamic_state": [float(dyn[0]), float(dyn[1]), float(dyn[2]), float(dyn[3])],
            "traffic_lights": lights,
            "driving_command": command,
            "cams": cams_for(log_name, token, channels),
            "camera_exists": False,  # WOMD ships no images; only rasters exist
            "is_valid": is_anchor,
            "ego2global": ego2global,
            "lidar2ego": lidar2ego,
            "lidar2global": ego2global @ lidar2ego,
            "sample_prev": tokens[k - 1] if k > 0 else None,
            "sample_next": tokens[k + 1] if k < len(frames) - 1 else None,
            "anns": anns,
            "rap_source": {
                "dataset": f"WOMD v1.3.1 {womd_split}",
                "scenario_id": sc.scenario_id,
                "womd_frame_index": t,
                "camera_mode": "navsim",
                "rap_viewpoint_shift_applied": True,
                "render_size_wh": list(RENDER_SIZE_WH),
                "rendered": bool(is_anchor and renderer is not None),
            },
        })
    stats["frames"] += len(infos)
    return infos


# =============================================================================
# Per-file worker
# =============================================================================
_RENDERER: Optional[ScenarioRenderer] = None


def _init_worker(channels: Tuple[str, ...], depth_max: float, render: bool) -> None:
    global _RENDERER
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    _RENDERER = (ScenarioRenderer(camera_channel_list=list(channels), width=RENDER_SIZE_WH[0],
                                  height=RENDER_SIZE_WH[1], depth_max=depth_max)
                 if render else None)


def new_stats() -> Counter:
    return Counter()


def process_file(payload: Tuple[str, argparse.Namespace]) -> Dict[str, Any]:
    path, args = payload
    started = time.time()
    stats = new_stats()
    log_dir = Path(args.ds_root) / "navsim_logs" / split_name(args.womd_split)
    log_dir.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    for i, raw in enumerate(wa.iter_tfrecord(Path(path))):
        if args.limit_scenarios and i >= args.limit_scenarios:
            break
        stats["scenarios"] += 1
        # Per-scenario counters merge only on success, so a failure mid-way
        # leaves no half-counted anchors (orphan rasters are harmless).
        local = new_stats()
        try:
            sc = wa.parse_scenario(raw)
            infos = process_scenario(sc, args.womd_split, _RENDERER, args, local)
        except Exception as exc:  # one bad scenario must not kill a 12 h array task
            stats["scenarios_failed"] += 1
            if len(failed) < 5:
                failed.append(f"#{i} {type(exc).__name__}: {exc}")
            continue
        stats.update(local)
        if infos is None or args.dry_run:
            continue
        out = log_dir / f"{infos[0]['log_name']}.pkl"
        tmp = out.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fp:
            pickle.dump(infos, fp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, out)
        stats["pkls"] += 1
    return {"file": Path(path).name, "seconds": round(time.time() - started, 1), "failed": failed, **stats}


# =============================================================================
# Subcommands
# =============================================================================
def list_files(args: argparse.Namespace) -> List[Path]:
    split_dir = Path(args.womd_root) / "scenario" / args.womd_split
    files = sorted(p for p in split_dir.iterdir() if ".tfrecord" in p.name)
    if not files:
        raise SystemExit(f"FAIL 没找到 tfrecord: {split_dir}")
    mine = [p for i, p in enumerate(files) if i % args.num_shards == args.shard - 1]
    if args.limit_files:
        mine = mine[: args.limit_files]
    return mine


def checkpoint_dir(args: argparse.Namespace) -> Path:
    tag = "dryrun" if args.dry_run else ("norender" if args.no_render else "render")
    return Path(args.ds_root) / "_progress" / split_name(args.womd_split) / tag


def cmd_run(args: argparse.Namespace) -> None:
    files = list_files(args)
    ckpt_dir = checkpoint_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in ckpt_dir.glob("*.jsonl"):
        done.update(json.loads(line)["file"] for line in f.read_text().splitlines() if line.strip())
    todo = [p for p in files if p.name not in done]
    print(f"[womd] {args.womd_split} shard {args.shard}/{args.num_shards}: {len(files)} files，"
          f"已完成 {len(files) - len(todo)}，待处理 {len(todo)}"
          f"{'（不渲染）' if args.no_render else ''}{'（dry-run）' if args.dry_run else ''}", flush=True)
    if not todo:
        return
    # Partial-file runs (pilots) must not mark files done.
    ckpt = ckpt_dir / f"shard{args.shard:04d}of{args.num_shards:04d}.jsonl" if not args.limit_scenarios else None

    totals = new_stats()
    started = time.time()
    payloads = [(str(p), args) for p in todo]
    with Pool(args.thread_num, initializer=_init_worker,
              initargs=(tuple(args.render_channels), args.depth_max, not args.no_render)) as pool:
        for n, result in enumerate(pool.imap_unordered(process_file, payloads), 1):
            for key, value in result.items():
                if isinstance(value, int):
                    totals[key] += value
            if ckpt is not None:
                with open(ckpt, "a") as fp:
                    fp.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"  [{n}/{len(todo)}] {result['file']}  scen={result.get('scenarios', 0)} "
                  f"anchors={result.get('anchors', 0)} imgs={result.get('images', 0)} {result['seconds']}s"
                  + (f"  FAILED {result.get('scenarios_failed', 0)}: {result['failed']}" if result["failed"] else ""),
                  flush=True)
    report_totals(totals, time.time() - started, args)


def report_totals(totals: Dict[str, int], seconds: float, args: argparse.Namespace) -> None:
    scen = max(totals["scenarios"], 1)
    print("[womd] 汇总: " + "  ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    print(f"[womd] 每场景 anchors={totals['anchors'] / scen:.2f}  "
          f"current={totals['current_frames'] / scen:.2f}  "
          f"墙钟 {seconds:.0f}s，{args.thread_num} 进程")
    if totals["anchors"]:
        print(f"[womd] 每 anchor 进程秒数 ≈ {seconds * args.thread_num / totals['anchors']:.2f}"
              "（含读文件、建图、非 anchor 帧的元数据）")


def cmd_audit(args: argparse.Namespace) -> None:
    """Every valid anchor must have all rendered cameras; exit 1 on gaps."""
    split = split_name(args.womd_split)
    log_dir = Path(args.ds_root) / "navsim_logs" / split
    render_root = Path(args.ds_root) / "rendered_sensor_blobs" / split
    pkls = sorted(log_dir.glob("*.pkl"))
    if args.limit_files:
        pkls = pkls[: args.limit_files]
    anchors = missing = 0
    examples: List[str] = []
    for pkl in pkls:
        with open(pkl, "rb") as fp:
            infos = pickle.load(fp)
        for info in infos:
            if not info["is_valid"]:
                continue
            anchors += 1
            for ch in args.render_channels:
                cam = info["cams"].get(ch)
                if cam is None or not (render_root / cam["data_path"]).exists():
                    missing += 1
                    if len(examples) < 5:
                        examples.append(cam["data_path"] if cam else f"{info['token']}: pkl 里没有 {ch}")
    print(f"[audit] {split}: {len(pkls)} pkls, {anchors} anchors, 缺图 {missing}")
    for e in examples:
        print(f"  缺: {e}")
    sys.exit(1 if missing else 0)


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--womd-root", type=Path, default=Path("/gs/bs/tga-RLA/qdeng/womd_dataset/v_1_3_1"))
        p.add_argument("--womd-split", default="training_20s", choices=("training_20s", "validation", "testing"))
        p.add_argument("--ds-root", type=Path, required=True, help="写 navsim_logs/、rendered_sensor_blobs/ 的根")
        p.add_argument("--limit-files", type=int, default=None)
        p.add_argument("--render-channels", default=",".join(OFFICIAL_RENDER_CHANNELS),
                       help="实际渲染的相机（默认同官方 NAVSIM：无 B0）")

    p_run = sub.add_parser("run", help="生成 pkl + 渲染 anchor（按 tfrecord 文件分片）")
    add_common(p_run)
    p_run.add_argument("--shard", type=int, default=1)
    p_run.add_argument("--num-shards", type=int, default=1)
    p_run.add_argument("--thread-num", type=int, default=16)
    p_run.add_argument("--limit-scenarios", type=int, default=None,
                       help="每个文件只处理前 N 个场景（pilot 用，不写 checkpoint）")
    p_run.add_argument("--no-render", action="store_true", help="只写 pkl（anchor 预算用）")
    p_run.add_argument("--dry-run", action="store_true", help="不写任何文件，只统计")
    p_run.add_argument("--camera-channels", default=",".join(RAP_DINO_CHANNELS))
    p_run.add_argument("--map-radius", type=float, default=200.0)
    p_run.add_argument("--depth-max", type=float, default=120.0)
    p_run.add_argument("--jpeg-quality", type=int, default=90)
    p_run.add_argument("--keep-unknown-lights", action="store_true",
                       help="UNKNOWN 信号按非红渲染（官方 nuPlan 行为）；默认丢弃")
    p_run.add_argument("--num-history-frames", type=int, default=4)
    p_run.add_argument("--num-future-frames", type=int, default=10)
    p_run.add_argument("--anchor-stride", type=int, default=4,
                       help="每隔几个 2 Hz current 帧取一个 anchor（4 = 2 s）")
    p_run.add_argument("--max-slope", type=float, default=1.5,
                       help="30–60 m 环带地图高差 p95 超过此值（m）的 anchor 丢弃")
    p_run.add_argument("--stationary-speed", type=float, default=0.5)
    p_run.add_argument("--stationary-travel", type=float, default=1.0,
                       help="未来 5 s 位移低于此值（m）且当前速度低于 --stationary-speed 视为静止")
    p_run.add_argument("--stationary-keep", type=float, default=0.2, help="静止 anchor 保留比例")
    p_run.add_argument("--route-lookahead-frames", type=int, default=100, help="10 Hz 帧数")
    p_run.add_argument("--route-tolerance", type=float, default=3.0)

    p_audit = sub.add_parser("audit", help="检查 valid anchor 的渲染图是否齐全")
    add_common(p_audit)

    args = parser.parse_args(argv)
    args.render_channels = tuple(c.strip() for c in args.render_channels.split(",") if c.strip())
    if hasattr(args, "camera_channels"):
        args.camera_channels = tuple(c.strip() for c in args.camera_channels.split(",") if c.strip())
        unknown = [c for c in args.camera_channels if c not in camera_params]
        if unknown:
            raise SystemExit(f"FAIL 未知相机通道: {unknown}")
        extra = [c for c in args.render_channels if c not in args.camera_channels]
        if extra:
            raise SystemExit(f"FAIL --render-channels 不在 --camera-channels 里: {extra}")
        if not 1 <= args.shard <= args.num_shards:
            raise SystemExit(f"FAIL --shard 须在 1..{args.num_shards}")
    return args


def main() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    args = parse_args()
    {"run": cmd_run, "audit": cmd_audit}[args.command](args)


if __name__ == "__main__":
    main()
