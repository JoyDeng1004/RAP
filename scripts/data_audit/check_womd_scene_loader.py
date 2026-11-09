#!/usr/bin/env python3
"""Load WOMD pkls through the real SceneLoader path and check what RAP will see.

Runs in the rap env (needs navsim + nuplan-devkit). For a few logs:
filter_scenes(enable_filter=True) with the womd_all filter, then
Scene.from_scene_dict_list with RAP's sensor config, and checks the rendered
image (cropped to 1080x1920; rendered channels must decode, all-black is only a
warning), real image absent, 10-pose future,
driving command and map_api=None.

    python scripts/data_audit/check_womd_scene_loader.py \
        --ds-root /gs/bs/tga-RLA/qdeng/womd_dataset/rap_raster --split womd_training_20s --num-logs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import filter_scenes  # noqa: E402

RAP_SENSORS = SensorConfig(cam_f0=[3], cam_l0=[3], cam_l1=[], cam_l2=[], cam_r0=[3],
                           cam_r1=[], cam_r2=[], cam_b0=[3], lidar_pc=[])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ds-root", type=Path, required=True)
    p.add_argument("--split", default="womd_training_20s")
    p.add_argument("--num-logs", type=int, default=5)
    p.add_argument("--render-channels", default="CAM_F0,CAM_L0,CAM_R0",
                   help="channels that must have a raster; others are zero-filled like NAVSIM's B0")
    args = p.parse_args()
    rendered = {c.strip().lower() for c in args.render_channels.split(",") if c.strip()}

    log_dir = args.ds_root / "navsim_logs" / args.split
    logs = sorted(x.stem for x in log_dir.glob("*.pkl"))[: args.num_logs]
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                               has_route=False, log_names=logs)
    scenes = filter_scenes(log_dir, scene_filter, enable_filter=True)
    print(f"{len(logs)} logs -> {len(scenes)} samples")
    if not scenes:
        print("FAIL 没有 is_valid 的 anchor")
        return 1

    failures = n_warn = 0
    for token, frames in scenes.items():
        scene = Scene.from_scene_dict_list(frames, args.ds_root / "sensor_blobs" / args.split,
                                           num_history_frames=4, num_future_frames=10,
                                           sensor_config=RAP_SENSORS)
        agent_input = scene.get_agent_input()
        cams = agent_input.cameras[-1]
        future = scene.get_future_trajectory(10).poses
        problems: list = []
        warnings: list = []
        if scene.map_api is not None:
            problems.append("map_api 不是 None")
        for name in ("cam_f0", "cam_l0", "cam_r0", "cam_b0"):
            cam = getattr(cams, name)
            if cam.rendered_image is None or cam.rendered_image.shape != (1080, 1920, 3):
                problems.append(f"{name} 渲染图形状 {None if cam.rendered_image is None else cam.rendered_image.shape}")
            elif name in rendered and not np.asarray(cam.rendered_image).any():
                # The loader zero-fills unreadable files, so only a black image that
                # decodes is a genuinely empty view (NAVSIM's own CAM_R0 is ~3% black).
                path = (args.ds_root / "rendered_sensor_blobs" / args.split
                        / frames[3]["cams"][name.upper()]["data_path"])
                try:
                    Image.open(path).load()
                    warnings.append(f"{name} 视野内无要素（全黑）")
                except Exception:  # noqa: BLE001 - mirror the loader's bare except
                    problems.append(f"{name} 渲染图读不出来，loader 填了零: {path}")
            if cam.real_valid:
                problems.append(f"{name} 居然读到了真图")
        if future.shape != (10, 3):
            problems.append(f"未来轨迹形状 {future.shape}")
        status = agent_input.ego_statuses[-1]
        print(f"{token}: future_end=({future[-1, 0]:.1f},{future[-1, 1]:.1f}) "
              f"v=({status.ego_velocity[0]:.1f},{status.ego_velocity[1]:.1f}) "
              f"cmd={list(status.driving_command)}" + (f"  FAIL {problems}" if problems else "")
              + (f"  WARN {warnings}" if warnings else ""))
        failures += bool(problems)
        n_warn += bool(warnings)
    print(f"{'PASS' if not failures else 'FAIL'} {len(scenes) - failures}/{len(scenes)}"
          f"（其中 {n_warn} 个有全黑视角，属正常）")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
