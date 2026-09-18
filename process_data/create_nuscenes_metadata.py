#!/usr/bin/env python3
"""Generate nuScenes RAP raster metadata with NAVSIM camera configurations.

Subcommands:
    build-index: Create per-scene indexes and package expansion maps. This is
        the only memory-intensive operation and should run on a compute node.
    render: Render scene shards from their compact indexes and write metadata.
    verify: Render one frame and report image coverage for cross-checking.

Camera modes:
    navsim: NAVSIM intrinsics and extrinsics.
    native: nuScenes calibrated-sensor intrinsics and extrinsics.
    hybrid: NAVSIM intrinsics with nuScenes extrinsics.

With ``--render-size auto``, NAVSIM and hybrid modes use 1920x1120 so the
NAVSIM principal point (960, 560) is centered. The extra 40 rows support the
loader's ``[20:-20]`` crop. Rendered paths must preserve the source image
extension and derive from a ``sensor_blobs`` path exactly once; otherwise the
loader cannot locate the rendered image.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers.nuscenes_adapter import (  # noqa: E402
    NAVSIM_TO_NUSC,
    NUSC_TO_NAVSIM,
    RAP_DINO_CHANNELS,
    MAP_LAYER_KEYS,
    MapCache,
    build_annotations,
    build_can_bus,
    build_index,
    driving_command_from_track,
    ego_dynamics,
    ego_pose_of,
    ego_track,
    load_manifest,
    load_scene_index,
    quaternion_to_rotation,
    quaternion_to_yaw,
)
from process_data.helpers.renderer import ScenarioRenderer, camera_params  # noqa: E402

# ScenarioRenderer applies this virtual-camera offset to every channel.
RAP_VIEWPOINT_SHIFT = np.array([-2.0, 0.0, 0.8], dtype=np.float64)


# =============================================================================
# NAVSIM canonical rig
# =============================================================================
def load_navsim_rig() -> Dict[str, Dict[str, np.ndarray]]:
    """Return the canonical NAVSIM rig used by ``ScenarioRenderer``.

    Reusing ``renderer.camera_params`` keeps preprocessing aligned with the
    renderer and avoids introducing nuplan-devkit dependencies.
    """
    return camera_params


def resolve_render_camera(
    mode: str,
    navsim_channel: str,
    nusc_calibration: Optional[Dict[str, Any]],
    rig: Dict[str, Dict[str, np.ndarray]],
    apply_viewpoint_shift: bool,
) -> Dict[str, np.ndarray]:
    """Resolve camera parameters for one ``ScenarioRenderer`` channel."""
    canonical = rig[navsim_channel]
    if mode == "navsim":
        intrinsics = np.asarray(canonical["intrinsics"], dtype=np.float64)
        rotation = np.asarray(canonical["sensor2lidar_rotation"], dtype=np.float64)
        translation = np.asarray(canonical["sensor2lidar_translation"], dtype=np.float64)
    else:
        if nusc_calibration is None:
            raise ValueError(f"{navsim_channel}: camera-mode={mode} 需要 nuScenes 标定但缺失")
        rotation = quaternion_to_rotation(nusc_calibration["rotation"])
        translation = np.asarray(nusc_calibration["translation"], dtype=np.float64)
        intrinsics = (
            np.asarray(canonical["intrinsics"], dtype=np.float64)
            if mode == "hybrid"
            else np.asarray(nusc_calibration["camera_intrinsic"], dtype=np.float64)
        )

    translation = translation.copy()
    if not apply_viewpoint_shift:
        # Compensate for the renderer's internal offset to preserve calibration.
        translation -= RAP_VIEWPOINT_SHIFT

    return {
        "intrinsics": intrinsics,
        "sensor2lidar_rotation": rotation,
        "sensor2lidar_translation": translation,
        "distortion": np.asarray(canonical["distortion"], dtype=np.float64),
    }


def effective_render_translation(translation: np.ndarray) -> np.ndarray:
    """Return the translation actually used by ``ScenarioRenderer``.

    ``ScenarioRenderer`` always adds ``RAP_VIEWPOINT_SHIFT``. Persisting this
    effective translation in ``render_*`` fields keeps downstream synthetic
    ``lidar2img`` projections geometrically aligned.
    """
    return np.asarray(translation, dtype=np.float64) + RAP_VIEWPOINT_SHIFT


def resolve_render_size(mode: str, spec: str, native_wh: Tuple[int, int]) -> Tuple[int, int]:
    """Return ``(width, height)`` including the 40-row crop margin."""
    if spec != "auto":
        w, h = (int(v) for v in spec.lower().split("x"))
        return w, h
    if mode in ("navsim", "hybrid"):
        return 1920, 1120
    return native_wh[0], native_wh[1] + 40


# =============================================================================
# preflight
# =============================================================================
def check_sensor_path(sensor_path: str) -> str:
    """Derive the rendered root while enforcing the ``sensor_blobs`` contract."""
    occurrences = sensor_path.count("sensor_blobs")
    if occurrences != 1:
        raise SystemExit(
            f"FAIL --sensor-path 必须含 'sensor_blobs' 字面量且恰好一次"
            f"（当前 {occurrences} 次）: {sensor_path}"
        )
    return sensor_path.replace("sensor_blobs", "rendered_sensor_blobs")


# =============================================================================
# Per-scene processing
# =============================================================================
def process_scene(payload: Tuple[str, argparse.Namespace]) -> Dict[str, Any]:
    scene_name, args = payload
    started = time.time()

    index_dir = Path(args.index_dir)
    scene = load_scene_index(index_dir, scene_name)
    location = scene["location"]
    sample_tokens: List[str] = scene["sample_tokens"]
    if len(sample_tokens) == 0:
        return {"scene": scene_name, "frames": 0, "images": 0, "skipped": "empty"}

    rig = load_navsim_rig()
    map_cache = MapCache(Path(args.map_root), pack_dir=index_dir / "_maps")
    render_root = Path(check_sensor_path(str(args.sensor_path)))
    nuscenes_root = Path(args.nuscenes_root)

    channels: Tuple[str, ...] = tuple(args.camera_channels)

    # Calibrated sensors are constant within a scene, so build models once.
    first_sd = scene["keyframe_sd"][sample_tokens[0]]
    native_wh = (1600, 900)
    for navsim_ch in channels:
        rec = first_sd.get(NAVSIM_TO_NUSC[navsim_ch])
        if rec and rec["width"] and rec["height"]:
            native_wh = (rec["width"], rec["height"])
            break
    width, height = resolve_render_size(args.camera_mode, args.render_size, native_wh)

    renderer = ScenarioRenderer(
        camera_channel_list=list(channels), width=width, height=height, depth_max=args.depth_max
    )
    # Override the renderer defaults so every requested channel is configured.
    renderer.camera_models = {}
    calib_by_channel: Dict[str, Dict[str, Any]] = {}
    for navsim_ch in channels:
        nusc_ch = NAVSIM_TO_NUSC[navsim_ch]
        rec = first_sd.get(nusc_ch)
        calibration = scene["calibrated_sensor"].get(rec["cs_token"]) if rec else None
        calib_by_channel[navsim_ch] = calibration
        renderer.camera_models[navsim_ch] = resolve_render_camera(
            args.camera_mode, navsim_ch, calibration, rig, args.apply_rap_viewpoint_shift
        )

    xy, yaw, tsec = ego_track(scene)
    dynamics = ego_dynamics(xy, yaw, tsec)

    frame_infos: List[Dict[str, Any]] = []
    images_written = 0

    for idx, sample_token in enumerate(sample_tokens):
        pose = ego_pose_of(scene, sample_token)
        ego_translation = np.asarray(pose["translation"], dtype=np.float64)
        ego_yaw = float(yaw[idx])
        meta = scene["samples"][sample_token]

        map_features = map_cache.query(location, ego_translation[:2], args.map_radius)
        anns = build_annotations(
            scene, sample_token, ego_translation, ego_yaw, filter_instance=args.filter_instance
        )

        # Use future lane tokens as a ``roadblock_ids`` surrogate; ``has_route``
        # requires only a non-empty sequence.
        future = xy[idx: idx + args.route_lookahead_frames]
        roadblock_ids = map_cache.lane_tokens_at(location, future, tolerance=args.route_tolerance)

        scenario = {
            "ego_pos": ego_translation[:2].tolist(),
            "ego_heading": ego_yaw,
            # nuScenes provides no traffic-light state labels.
            "traffic_lights": [],
            "map_features": map_features,
            "anns": anns,
        }

        cams: Dict[str, Dict[str, Any]] = {}
        camera_exists = True
        rendered = renderer.observe(scenario)

        for navsim_ch in channels:
            nusc_ch = NAVSIM_TO_NUSC[navsim_ch]
            sd = scene["keyframe_sd"][sample_token].get(nusc_ch)
            if sd is None:
                camera_exists = False
                continue
            data_path = sd["filename"]
            if not (nuscenes_root / data_path).exists():
                camera_exists = False

            canvas = rendered.get(navsim_ch)
            if canvas is not None and not args.dry_run:
                target = render_root / data_path
                target.parent.mkdir(parents=True, exist_ok=True)
                # OpenCV writes BGR images.
                cv2.imwrite(str(target), canvas[:, :, ::-1])
                images_written += 1

            calibration = calib_by_channel[navsim_ch]
            render_camera = renderer.camera_models[navsim_ch]
            cams[navsim_ch] = {
                "data_path": data_path,
                # Standard fields describe the corresponding real nuScenes image.
                "sensor2lidar_rotation": quaternion_to_rotation(calibration["rotation"]),
                "sensor2lidar_translation": np.asarray(calibration["translation"], np.float64),
                "cam_intrinsic": np.asarray(calibration["camera_intrinsic"], np.float64),
                "distortion": np.zeros(5, dtype=np.float64),  # nuScenes images are rectified.
                # ``render_*`` fields describe the rendered raster geometry.
                "render_sensor2lidar_rotation": render_camera["sensor2lidar_rotation"],
                "render_sensor2lidar_translation": effective_render_translation(
                    render_camera["sensor2lidar_translation"]
                ),
                "render_cam_intrinsic": render_camera["intrinsics"],
                "render_image_shape": [height, width],
                "nuscenes_channel": nusc_ch,
            }

        ego_rotation_q = pose["rotation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = quaternion_to_rotation(ego_rotation_q)
        ego2global[:3, 3] = ego_translation
        lidar_sd = scene["keyframe_sd"][sample_token].get("LIDAR_TOP")
        lidar_calibration = (
            scene["calibrated_sensor"].get(lidar_sd["cs_token"]) if lidar_sd else None
        )
        lidar2ego = np.eye(4)
        if lidar_calibration is not None:
            lidar2ego[:3, :3] = quaternion_to_rotation(lidar_calibration["rotation"])
            lidar2ego[:3, 3] = np.asarray(lidar_calibration["translation"], dtype=np.float64)

        vx, vy, ax, ay, _ = dynamics[idx]
        info = {
            "token": sample_token,
            "frame_idx": idx,
            "timestamp": meta["timestamp"],
            "log_name": scene["scene_name"],
            "log_token": scene["log_token"],
            "scene_name": scene["scene_name"],
            "scene_token": scene["scene_token"],
            "map_location": location,
            "roadblock_ids": roadblock_ids,
            "vehicle_name": scene["vehicle"],
            "can_bus": build_can_bus(ego_translation, ego_rotation_q, dynamics[idx]),
            "lidar_path": lidar_sd["filename"] if lidar_sd else "",
            "lidar2ego_translation": lidar2ego[:3, 3],
            "lidar2ego_rotation": list(lidar_calibration["rotation"]) if lidar_calibration else [1.0, 0.0, 0.0, 0.0],
            "ego2global_translation": ego_translation,
            "ego2global_rotation": np.asarray(ego_rotation_q, dtype=np.float64),
            "ego_dynamic_state": [float(vx), float(vy), float(ax), float(ay)],
            "traffic_lights": [],
            "driving_command": driving_command_from_track(
                xy, yaw, idx,
                distance=args.command_distance,
                lateral_offset=args.command_offset,
                min_arc=args.command_min_arc,
                yaw_threshold=args.command_yaw_threshold,
                lookahead_frames=args.command_lookahead_frames,
                short_horizon_policy=args.command_short_horizon,
            ),
            "cams": cams,
            "camera_exists": camera_exists,
            "is_valid": True,
            "ego2global": ego2global,
            "lidar2ego": lidar2ego,
            "lidar2global": ego2global @ lidar2ego,
            "sample_prev": sample_tokens[idx - 1] if idx > 0 else None,
            "sample_next": sample_tokens[idx + 1] if idx < len(sample_tokens) - 1 else None,
            "anns": anns,
            "rap_source": {
                "dataset": f"nuScenes {scene['version']}",
                "camera_mode": args.camera_mode,
                "rap_viewpoint_shift_applied": bool(args.apply_rap_viewpoint_shift),
                "render_size_wh": [width, height],
            },
        }
        frame_infos.append(info)

    if not args.dry_run:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{scene['scene_name']}.pkl", "wb") as fp:
            pickle.dump(frame_infos, fp, protocol=pickle.HIGHEST_PROTOCOL)

    commands = np.stack([info["driving_command"] for info in frame_infos])
    return {
        "scene": scene_name,
        "location": location,
        "frames": len(frame_infos),
        "images": images_written,
        "usable_tokens": max(0, len(frame_infos) - args.scene_window + 1),
        "no_route_frames": int(sum(1 for i in frame_infos if not i["roadblock_ids"])),
        "unknown_command_frames": int(commands[:, 3].sum()),
        "seconds": round(time.time() - started, 1),
    }


# =============================================================================
# Subcommands
# =============================================================================
def cmd_build_index(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)

    if args.maps_only:
        # Rebuild maps without rewriting scene indexes when map-layer logic changes.
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"FAIL --maps-only 需要已有的 {manifest_path}（它提供 location 清单）。\n"
                f"     首次建索引请不要加 --maps-only。"
            )
        manifest = load_manifest(index_dir)
    else:
        version_root = Path(args.nuscenes_root) / args.version
        manifest = build_index(version_root, index_dir, args.version)

    pack_dir = index_dir / "_maps"
    # Build from source JSON so incompatible packages can be regenerated.
    cache = MapCache(Path(args.map_root))
    locations = sorted({s["location"] for s in manifest["scenes"]})
    for location in locations:
        out = cache.pack_to_npz(location, pack_dir)
        print(f"[index] 地图打包 {location} -> {out}", flush=True)

    # Reload every package to validate packing and reading compatibility.
    verify_cache = MapCache(Path(args.map_root), pack_dir=pack_dir)
    for location in locations:
        layers = verify_cache.get(location)
        print(f"[index] 回读 {location}: "
              + "  ".join(f"{k}={len(layers[k].ids)}" for k in MAP_LAYER_KEYS),
              flush=True)

    if args.maps_only:
        print(f"[index] 完成（仅地图）：{len(locations)} maps，scene 索引未改动")
    else:
        print(f"[index] 完成：{manifest['num_scenes']} scenes / {len(locations)} maps")


def select_scenes(args: argparse.Namespace) -> List[str]:
    manifest = load_manifest(Path(args.index_dir))
    scenes = manifest["scenes"]
    if args.locations:
        wanted = set(args.locations)
        scenes = [s for s in scenes if s["location"] in wanted]
    # Group scenes by location to maximize worker-local map-cache reuse.
    scenes.sort(key=lambda s: (s["location"], s["scene_name"]))
    names = [s["scene_name"] for s in scenes]

    total = len(names)
    per = math.ceil(total / args.num_shards)
    start = (args.shard - 1) * per
    end = min(args.shard * per, total)
    selected = names[start:end] if start < total else []
    if args.limit_scenes:
        selected = selected[: args.limit_scenes]
    return selected


def cmd_plan(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.index_dir))
    total = len(manifest["scenes"])
    per = math.ceil(total / args.num_shards)
    print(f"index      : {args.index_dir}  ({total} scenes)")
    print(f"out-dir    : {args.out_dir}")
    print(f"sensor-path: {args.sensor_path}")
    print(f"渲染根     : {check_sensor_path(str(args.sensor_path))}")
    print(f"camera     : mode={args.camera_mode} channels={','.join(args.camera_channels)}")
    print(f"分片       : {args.num_shards} 片 x 约 {per} scenes，每片 {args.thread_num} 进程")
    for i in range(1, args.num_shards + 1):
        s, e = (i - 1) * per, min(i * per, total)
        if s >= total:
            break
        print(f"  shard {i:3d}: scenes[{s}:{e}]  ({e - s} 个)")


def warn_shape_mismatch(args: argparse.Namespace, native_wh: Tuple[int, int] = (1600, 900)) -> None:
    """Report when cropped rendered images differ from native image dimensions.

    NAVSIM and hybrid modes use a 1920x1120 canvas to center the NAVSIM
    principal point, producing a 1920x1080 crop rather than 1600x900.
    """
    width, height = resolve_render_size(args.camera_mode, args.render_size, native_wh)
    cropped = (width, height - 40)
    if cropped != native_wh:
        print(
            f"[render] 注意：渲染图裁剪后 {cropped[0]}x{cropped[1]}，"
            f"真实图 {native_wh[0]}x{native_wh[1]} —— 两者不同尺寸。\n"
            f"         camera-mode={args.camera_mode} 下这是必然的（NAVSIM 主点 960/560 要求 1920x1120 画布）。\n"
            f"         rendered 与 real 分支各自使用时没问题；要拼接/叠加请自行 resize。",
            flush=True,
        )


def cmd_render(args: argparse.Namespace) -> None:
    check_sensor_path(str(args.sensor_path))
    warn_shape_mismatch(args)
    selected = select_scenes(args)

    checkpoint = Path(args.checkpoint)
    done = set()
    if checkpoint.exists():
        done = {line.strip() for line in checkpoint.read_text().splitlines() if line.strip()}
    todo = [name for name in selected if name not in done]

    print(
        f"[render] shard {args.shard}/{args.num_shards}: "
        f"{len(selected)} scenes，已完成 {len(selected) - len(todo)}，待处理 {len(todo)}",
        flush=True,
    )
    if not todo:
        return

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    totals = {"frames": 0, "images": 0, "usable_tokens": 0, "no_route_frames": 0,
              "unknown_command_frames": 0}

    payloads = [(name, args) for name in todo]
    with Pool(processes=args.thread_num) as pool:
        for result in pool.imap_unordered(process_scene, payloads):
            for key in totals:
                totals[key] += result.get(key, 0)
            with open(checkpoint, "a") as fp:
                fp.write(result["scene"] + "\n")
            print(
                f"  {result['scene']:14s} {result.get('location','?'):24s} "
                f"frames={result['frames']:3d} imgs={result['images']:4d} "
                f"usable={result.get('usable_tokens',0):3d} "
                f"no_route={result.get('no_route_frames',0):2d} "
                f"unknown_cmd={result.get('unknown_command_frames',0):2d} "
                f"{result['seconds']}s",
                flush=True,
            )

    print("[render] 汇总: " + "  ".join(f"{k}={v}" for k, v in totals.items()))
    if totals["no_route_frames"]:
        print(
            "[render] 注意：有帧的 roadblock_ids 为空，这些帧在 has_route=true 下会被"
            " dataloader.py:120 丢掉。若比例偏高，调大 --route-tolerance。"
        )


def cmd_verify(args: argparse.Namespace) -> None:
    """Render one frame and report statistics for cross-checking."""
    args.dry_run = True
    scene = load_scene_index(Path(args.index_dir), args.scene_name)
    idx = args.frame_index
    token = scene["sample_tokens"][idx]

    rig = load_navsim_rig()
    map_cache = MapCache(Path(args.map_root), pack_dir=Path(args.index_dir) / "_maps")
    xy, yaw, _ = ego_track(scene)
    pose = ego_pose_of(scene, token)
    ego_translation = np.asarray(pose["translation"], dtype=np.float64)

    map_features = map_cache.query(scene["location"], ego_translation[:2], args.map_radius)
    anns = build_annotations(scene, token, ego_translation, float(yaw[idx]),
                             filter_instance=args.filter_instance)
    scenario = {
        "ego_pos": ego_translation[:2].tolist(),
        "ego_heading": float(yaw[idx]),
        "traffic_lights": [],
        "map_features": map_features,
        "anns": anns,
    }

    first_sd = scene["keyframe_sd"][scene["sample_tokens"][0]]
    native_wh = (1600, 900)
    rec = first_sd.get("CAM_FRONT")
    if rec and rec["width"]:
        native_wh = (rec["width"], rec["height"])
    width, height = resolve_render_size(args.camera_mode, args.render_size, native_wh)

    renderer = ScenarioRenderer(camera_channel_list=list(args.camera_channels),
                                width=width, height=height, depth_max=args.depth_max)
    renderer.camera_models = {}
    for navsim_ch in args.camera_channels:
        sd = first_sd.get(NAVSIM_TO_NUSC[navsim_ch])
        calibration = scene["calibrated_sensor"].get(sd["cs_token"]) if sd else None
        renderer.camera_models[navsim_ch] = resolve_render_camera(
            args.camera_mode, navsim_ch, calibration, rig, args.apply_rap_viewpoint_shift
        )

    rendered = renderer.observe(scenario)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scene": scene["scene_name"],
        "location": scene["location"],
        "sample_token": token,
        "camera_mode": args.camera_mode,
        "rap_viewpoint_shift_applied": bool(args.apply_rap_viewpoint_shift),
        "map_features": len(map_features),
        "annotation_boxes": int(anns["gt_boxes"].shape[0]),
        "renders": {},
    }
    for channel, canvas in rendered.items():
        path = out_dir / f"verify_{token}_{channel}_{args.camera_mode}.jpg"
        cv2.imwrite(str(path), canvas[:, :, ::-1])
        report["renders"][channel] = {
            "shape_hwc": list(canvas.shape),
            "nonzero_fraction": float(np.count_nonzero(canvas) / canvas.size),
            "path": str(path),
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--nuscenes-root", type=Path, required=True)
        p.add_argument("--version", default="v1.0-trainval")
        p.add_argument("--map-root", type=Path, default=None,
                       help="默认 <nuscenes-root>/maps/expansion")
        p.add_argument("--index-dir", type=Path, default=None,
                       help="默认 <out-dir>/_index；build-index 与 render 必须指向同一处")

    def add_render_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--camera-mode", choices=("navsim", "native", "hybrid"), default="navsim")
        p.add_argument("--camera-channels", default=",".join(RAP_DINO_CHANNELS),
                       help="NAVSIM 通道名，逗号分隔。默认 RAP-DINO 实际消费的四路。")
        p.add_argument("--render-size", default="auto", help="auto 或 WxH")
        p.add_argument("--apply-rap-viewpoint-shift", type=lambda v: v.lower() != "false",
                       default=True,
                       help="true(默认)=复现 RAP 的 +0.8m/-2m 虚拟视点；"
                            "false=预补偿掉，使有效标定等于真实标定")
        p.add_argument("--map-radius", type=float, default=200.0)
        p.add_argument("--depth-max", type=float, default=120.0)
        p.add_argument("--filter-instance", type=lambda v: v.lower() != "false", default=True)
        p.add_argument("--command-distance", type=float, default=20.0)
        p.add_argument("--command-offset", type=float, default=2.0)
        p.add_argument("--command-min-arc", type=float, default=5.0,
                       help="未来弧长低于此值视为近似静止，改用航向变化判方向")
        p.add_argument("--command-yaw-threshold", type=float, default=0.15,
                       help="近似静止时判 forward 的 |Δyaw| 阈值（弧度）")
        p.add_argument("--command-lookahead-frames", type=int, default=20,
                       help="driving_command 的固定前视帧数（2Hz 下 20 帧 = 10s）。"
                            "必须固定，否则标签会依赖 scene 剩余长度")
        p.add_argument("--command-short-horizon", choices=("scaled", "unknown"),
                       default="scaled",
                       help="前视弧长不足 --command-distance 时的口径："
                            "scaled=按平方缩放阈值（保覆盖率）；"
                            "unknown=直接给 unknown（与 nuPlan 的 unknown 语义最接近）")
        p.add_argument("--route-lookahead-frames", type=int, default=20)
        p.add_argument("--route-tolerance", type=float, default=3.0)
        p.add_argument("--scene-window", type=int, default=14,
                       help="仅用于统计 usable_tokens：num_history+num_future")

    p_index = sub.add_parser("build-index", help="拆表建索引 + 打包地图（跑一次，高内存）")
    add_common(p_index)
    p_index.add_argument("--out-dir", type=Path, required=True)
    p_index.add_argument(
        "--maps-only", action="store_true",
        help="只重打 _maps/*.npz，跳过 850 个 scene 索引的重建。"
             "改了 MapCache 的取层逻辑时用这个，秒级而不是分钟级。"
             "需要 <index-dir>/manifest.json 已存在。",
    )

    p_plan = sub.add_parser("plan", help="只打印分片计划")
    add_common(p_plan)
    add_render_opts(p_plan)
    p_plan.add_argument("--out-dir", type=Path, required=True)
    p_plan.add_argument("--sensor-path", type=Path, required=True)
    p_plan.add_argument("--num-shards", type=int, default=1)
    p_plan.add_argument("--thread-num", type=int, default=16)

    p_render = sub.add_parser("render", help="渲染 + 写 pkl（可分片）")
    add_common(p_render)
    add_render_opts(p_render)
    p_render.add_argument("--out-dir", type=Path, required=True)
    p_render.add_argument("--sensor-path", type=Path, required=True,
                          help="必须含 'sensor_blobs' 字面量且恰好一次")
    p_render.add_argument("--shard", type=int, default=1)
    p_render.add_argument("--num-shards", type=int, default=1)
    p_render.add_argument("--thread-num", type=int, default=16)
    p_render.add_argument("--limit-scenes", type=int, default=None, help="pilot 用")
    p_render.add_argument("--locations", nargs="*", default=None)
    p_render.add_argument("--checkpoint", type=Path, default=Path("checkpoint.txt"))
    p_render.add_argument("--dry-run", action="store_true", help="只跑不写盘")

    p_verify = sub.add_parser("verify", help="渲一帧自检")
    add_common(p_verify)
    add_render_opts(p_verify)
    p_verify.add_argument("--out-dir", type=Path, required=True)
    p_verify.add_argument("--scene-name", required=True)
    p_verify.add_argument("--frame-index", type=int, default=0)

    args = parser.parse_args()

    if args.map_root is None:
        args.map_root = Path(args.nuscenes_root) / "maps" / "expansion"
    if args.index_dir is None:
        args.index_dir = Path(args.out_dir) / "_index"
    if hasattr(args, "camera_channels") and isinstance(args.camera_channels, str):
        args.camera_channels = tuple(c.strip() for c in args.camera_channels.split(",") if c.strip())
        unknown = [c for c in args.camera_channels if c not in NAVSIM_TO_NUSC]
        if unknown:
            raise SystemExit(
                f"FAIL 这些 NAVSIM 通道在 nuScenes 里没有对应物: {unknown}。"
                f" 可用: {sorted(NAVSIM_TO_NUSC)}"
            )
    if not hasattr(args, "dry_run"):
        args.dry_run = False
    return args


def main() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    args = parse_args()
    {"build-index": cmd_build_index, "plan": cmd_plan,
     "render": cmd_render, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    main()
