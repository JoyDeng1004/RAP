#!/usr/bin/env python3
"""Render a handful of WOMD anchors with RAP's rasterizer for visual QA.

Each scenario becomes one contact sheet (``.webp`` so mutagen syncs it back):
the NAVSIM-rig rasters of the RAP-DINO cameras plus a top-down view of the
adapted map, agents, lights and the 4 s ego future, and a text panel with the
driving command and diagnostics. Nothing is written into dataset folders.

    python tools/render_womd_samples.py \
        --root /gs/bs/tga-RLA/qdeng/womd_dataset/v_1_3_1 \
        --splits training_20s validation --num 20 \
        --out /gs/bs/tga-RLA/qdeng/RAP/eval/womd_render
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers.nuscenes_adapter import RAP_DINO_CHANNELS  # noqa: E402
from process_data.helpers.renderer import COLOR_TABLE, ScenarioRenderer  # noqa: E402
from process_data.helpers import womd_adapter as wa  # noqa: E402

COMMAND_NAMES = ("LEFT", "FORWARD", "RIGHT", "UNKNOWN")
FUTURE_FRAMES = 40  # 4 s at 10 Hz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=Path("/gs/bs/tga-RLA/qdeng/womd_dataset/v_1_3_1"))
    p.add_argument("--splits", nargs="+", default=["training_20s", "validation"])
    p.add_argument("--num", type=int, default=20, help="total scenarios, split evenly across --splits")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--channels", default=",".join(RAP_DINO_CHANNELS))
    p.add_argument("--map-radius", type=float, default=200.0)
    p.add_argument("--depth-max", type=float, default=120.0)
    p.add_argument("--thumb-width", type=int, default=640, help="width of each camera tile")
    p.add_argument("--keep-unknown-lights", action="store_true",
                   help="render UNKNOWN signals as non-red (official nuPlan behaviour)")
    return p.parse_args()


def pick_scenarios(split_dir: Path, k: int) -> List[Tuple[Path, int]]:
    """First scenario of ``k`` evenly spaced shards."""
    shards = sorted(p for p in split_dir.iterdir() if ".tfrecord" in p.name)
    if not shards:
        raise FileNotFoundError(f"no tfrecord under {split_dir}")
    idx = np.linspace(0, len(shards) - 1, min(k, len(shards))).round().astype(int)
    return [(shards[i], 0) for i in sorted(set(idx.tolist()))]


def pick_anchor(sc, ego: wa.EgoTrack) -> int:
    cur = int(sc.current_time_index)
    if cur >= 1 and ego.valid[cur - 1:cur + FUTURE_FRAMES + 1].all() and cur + FUTURE_FRAMES < len(ego.valid):
        return cur  # validation: the official "current" frame (off the 0.5 s grid)
    anchors = wa.anchor_indices(ego.valid, horizon=FUTURE_FRAMES)
    return anchors[len(anchors) // 2] if anchors else -1


# -----------------------------------------------------------------------------
# Top-down view
# -----------------------------------------------------------------------------
def draw_bev(scenario: Dict, future_xy: np.ndarray, size: int, extent: float = 60.0) -> np.ndarray:
    """Ego-up BEV (RGB). ``future_xy`` is ego-origin, globally aligned."""
    canvas = np.full((size, size, 3), 25, np.uint8)
    yaw = scenario["ego_heading"]
    c, s = math.cos(yaw), math.sin(yaw)
    scale = size / (2 * extent)

    def to_px(xy: np.ndarray) -> np.ndarray:
        local = xy @ np.array([[c, -s], [s, c]])  # global-aligned -> ego frame
        u = size / 2 - local[:, 1] * scale        # left of ego is left in image
        v = size * 0.7 - local[:, 0] * scale      # forward is up; ego sits low
        return np.stack([u, v], axis=1).round().astype(np.int32)

    def col(name):
        return tuple(int(x) for x in COLOR_TABLE[name])

    for feat in scenario["map_features"].values():
        if feat["type"] == "CROSSWALK":
            cv2.fillPoly(canvas, [to_px(feat["polygon"])], col("crosswalks"))
    for feat in scenario["map_features"].values():
        if feat["type"] == "LANE":
            cv2.polylines(canvas, [to_px(feat["polygon"])], False, col("lanelines"), 1, cv2.LINE_AA)
        elif feat["type"] == "BOUNDARY":
            cv2.polylines(canvas, [to_px(feat["polyline"])], False, col("road_boundaries"), 2, cv2.LINE_AA)

    for row in scenario["anns"]["gt_boxes_world"]:
        x, y, _, L, W, _, h = row
        corners = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
        rot = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
        cv2.polylines(canvas, [to_px(corners @ rot.T + [x, y])], True, col("vehicle"), 1, cv2.LINE_AA)

    for _, is_red, xy in scenario["traffic_lights"]:
        p = to_px(np.array([xy]))[0]
        cv2.circle(canvas, tuple(int(v) for v in p), 4,
                   col("traffic_light_red" if is_red else "traffic_light_green"), -1)

    cv2.polylines(canvas, [to_px(future_xy)], False, (0, 255, 0), 2, cv2.LINE_AA)
    ego = np.array([[4.0, 1.0], [4.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])  # rear-axle origin
    ego_rot = ego @ np.array([[c, s], [-s, c]])
    cv2.polylines(canvas, [to_px(ego_rot)], True, (255, 255, 255), 2)
    return canvas


def text_panel(lines: List[str], w: int, h: int) -> np.ndarray:
    panel = np.full((h, w, 3), 15, np.uint8)
    y = 28
    for line in lines:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        y += 24
    return panel


def tile(img: np.ndarray, w: int, label: str) -> np.ndarray:
    h = int(round(img.shape[0] * w / img.shape[1]))
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    renderer = ScenarioRenderer(camera_channel_list=channels, depth_max=args.depth_max)

    per_split = math.ceil(args.num / len(args.splits))
    summary = []
    for split in args.splits:
        for shard, rec_idx in pick_scenarios(args.root / "scenario" / split, per_split):
            started = time.time()
            raw = next(r for i, r in enumerate(wa.iter_tfrecord(shard)) if i == rec_idx)
            t_read = time.time() - started
            started = time.time()
            sc = wa.parse_scenario(raw)
            ego = wa.ego_track(sc)
            t = pick_anchor(sc, ego)
            if t < 0:
                print(f"[skip] {shard.name}: no valid anchor", file=sys.stderr)
                continue
            womd_map = wa.build_map(sc)
            scenario, extras = wa.build_scenario(
                sc, t, womd_map, ego, map_radius=args.map_radius,
                drop_unknown_lights=not args.keep_unknown_lights)
            t_adapt = time.time() - started
            rendered = renderer.observe(scenario)
            t_render = time.time() - started - t_adapt

            future = ego.rear_xy[t:t + FUTURE_FRAMES + 1] - ego.rear_xy[t]
            # Front-facing trio on top in left-to-right order, everything else below.
            order = [c for c in ("CAM_L0", "CAM_F0", "CAM_R0") if c in channels]
            order += [c for c in channels if c not in order]
            tiles = [tile(rendered[ch], args.thumb_width, ch) for ch in order]
            th, tw = tiles[0].shape[:2]
            bev = draw_bev(scenario, future, th)
            n_types = {k: sum(f["type"] == k for f in scenario["map_features"].values())
                       for k in ("LANE", "BOUNDARY", "CROSSWALK")}
            command = COMMAND_NAMES[int(np.argmax(extras["driving_command"]))]
            slope = extras["slope_p95_30_60m"]
            info = [
                f"{split}  {extras['scenario_id']}  t={t}",
                f"command={command}  speed={extras['ego_speed']:.1f} m/s",
                f"lanes={n_types['LANE']} boundaries={n_types['BOUNDARY']} crosswalks={n_types['CROSSWALK']}",
                f"lane width measured/default={womd_map.half_width_measured}/{womd_map.half_width_default}",
                f"agents={len(scenario['anns']['gt_names'])}  lights {extras['lights']}",
                f"slope p95 30-60m={'n/a' if slope is None else f'{slope:.2f} m'}"
                + ("  [HILLY]" if slope is not None and slope > 1.5 else ""),
                f"read {t_read:.2f}s  adapt {t_adapt:.2f}s  render {t_render:.2f}s",
            ]
            row1 = np.concatenate(tiles[:3], axis=1)
            below = tiles[3:] + [bev]
            used = sum(im.shape[1] for im in below)
            panel = text_panel(info, max(row1.shape[1] - used, 560), th)
            row2 = np.concatenate(below + [panel], axis=1)
            width = max(row1.shape[1], row2.shape[1])
            pad = lambda im: np.pad(im, ((0, 0), (0, width - im.shape[1]), (0, 0)))
            sheet = np.concatenate([pad(row1), pad(row2)], axis=0)

            name = f"{split}_{shard.name.split('.')[1]}_{extras['scenario_id']}_t{t:03d}.webp"
            cv2.imwrite(str(args.out / name), sheet[:, :, ::-1], [cv2.IMWRITE_WEBP_QUALITY, 85])
            record = {
                "file": name, "split": split, "shard": shard.name, "frame": t,
                "command": command, "speed": round(extras["ego_speed"], 2),
                "map_counts": n_types,
                "lane_width_measured": womd_map.half_width_measured,
                "lane_width_default": womd_map.half_width_default,
                "agents": int(len(scenario["anns"]["gt_names"])),
                "lights": extras["lights"],
                "slope_p95_30_60m": None if slope is None else round(slope, 3),
                "seconds_read": round(t_read, 3), "seconds_adapt": round(t_adapt, 3), "seconds_render": round(t_render, 3),
            }
            summary.append(record)
            print(json.dumps(record, ensure_ascii=False), file=sys.stderr)

    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {args.out} ({len(summary)} sheets)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
