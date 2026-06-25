"""Batch canonical-BEV visualization over many scenes.

Opens the dataset resources ONCE (NuScenes / SceneLoader + map caches) and loops
over tokens, dumping a composite (and optionally per-channel) PNG per scene.

Examples:
  # nuScenes: every 10th sample, up to 50 scenes
  python -m tools.canonical_bev.batch_visualize --dataset nuscenes \
      --dataroot /gs/bs/tga-RLA/qdeng/data/nuscenes --version v1.0-trainval \
      --limit 50 --stride 10 --out out/canonical_bev/nuscenes

  # NAVSIM (OpenScene): needs the navsim_logs layout (NOT raw nuplan)
  python -m tools.canonical_bev.batch_visualize --dataset navsim \
      --data-path $OPENSCENE_DATA_ROOT/navsim_logs/trainval \
      --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/trainval \
      --limit 50 --out out/canonical_bev/navsim
"""

from __future__ import annotations

import argparse
import os
import traceback

from .canonical_bev import build_canonical_bev
from .config import CanonBEVConfig
from .visualize import save_visualizations


def _run_nuscenes(args, cfg):
    """Group by scene: <out>/<scene_token>/<sample_token>_composite.png."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from .adapters.nuscenes_adapter import sample_to_meta

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    map_cache = {}

    def get_map(location):
        if location not in map_cache:
            map_cache[location] = NuScenesMap(dataroot=args.dataroot, map_name=location)
        return map_cache[location]

    scenes = nusc.scene[args.start:: args.stride]
    if args.limit:
        scenes = scenes[: args.limit]
    print(f"[nuscenes] {len(scenes)} scenes to render")

    for si, scene in enumerate(scenes):
        scene_tok = scene["token"]
        out_scene = os.path.join(args.out, scene_tok)
        log = nusc.get("log", scene["log_token"])
        nmap = get_map(log["location"])
        sample_tok = scene["first_sample_token"]
        k = 0
        while sample_tok:
            s = nusc.get("sample", sample_tok)
            nxt = s["next"]
            if k % args.sample_stride == 0:
                try:
                    sm = sample_to_meta(nusc, nmap, sample_tok, cfg)
                    raster = build_canonical_bev(sm, cfg)
                    save_visualizations(raster, cfg, out_scene, name=sample_tok,
                                        channels=args.channels, composite=True)
                except Exception:
                    print(f"  [skip {sample_tok}] {traceback.format_exc().splitlines()[-1]}")
            k += 1
            sample_tok = nxt
        print(f"  scene {si + 1}/{len(scenes)} ({scene['name']}): {k} samples -> {out_scene}")


def _run_navsim(args, cfg):
    """Group by scene_token: <out>/<scene_token>/<frame_token>_composite.png."""
    from pathlib import Path
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from .adapters.navsim_adapter import scene_to_meta

    max_scenes = None if args.limit is None else args.start + args.stride * args.limit
    scene_filter = SceneFilter(max_scenes=max_scenes)
    loader = SceneLoader(
        data_path=Path(args.data_path), sensor_blobs_path=Path(args.sensor_blobs_path),
        scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors(),
        enable_filter=True,
    )
    tokens = loader.tokens[args.start:: args.stride]
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"[navsim] {len(tokens)} scene-windows to render")
    rendered = 0
    for i, tok in enumerate(tokens):
        try:
            scene = loader.get_scene_from_token(tok)
            scene_tok = getattr(scene.scene_metadata, "scene_token", None) \
                or getattr(scene.scene_metadata, "log_name", "scenes")
            out_scene = os.path.join(args.out, str(scene_tok))
            start_frame_idx = max(0, scene.scene_metadata.num_history_frames - 1)
            for frame_idx in range(start_frame_idx, len(scene.frames)):
                frame = scene.frames[frame_idx]
                if frame is None:
                    continue
                sm = scene_to_meta(scene, frame_idx=frame_idx, cfg=cfg)
                raster = build_canonical_bev(sm, cfg)
                save_visualizations(raster, cfg, out_scene, name=frame.token,
                                    channels=args.channels, composite=True)
                rendered += 1
        except Exception:
            print(f"  [skip {tok}] {traceback.format_exc().splitlines()[-1]}")
            continue
        if (i + 1) % args.log_every == 0:
            print(f"  {i + 1}/{len(tokens)} scene-windows done, {rendered} frames rendered")


def main():
    ap = argparse.ArgumentParser(description="Batch canonical BEV visualization.")
    ap.add_argument("--dataset", choices=["nuscenes", "navsim"], required=True)
    ap.add_argument("--out", default="out/canonical_bev/batch")
    ap.add_argument("--limit", type=int, default=50, help="max number of SCENES (0 = all)")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth scene")
    ap.add_argument("--start", type=int, default=0, help="start offset into scene list")
    ap.add_argument("--sample-stride", type=int, default=1,
                    help="within a scene, render every Nth sample/frame (nuscenes)")
    ap.add_argument("--channels", action="store_true",
                    help="also dump the per-channel grid (default: composite only)")
    ap.add_argument("--log-every", type=int, default=10)
    # nuscenes
    ap.add_argument("--dataroot", default=os.environ.get("NUSCENES_DATAROOT", ""))
    ap.add_argument("--version", default="v1.0-trainval")
    # navsim
    ap.add_argument("--data-path", default=os.environ.get("OPENSCENE_DATA_ROOT", ""))
    ap.add_argument("--sensor-blobs-path", default="")
    args = ap.parse_args()
    if args.limit == 0:
        args.limit = None

    cfg = CanonBEVConfig()
    os.makedirs(args.out, exist_ok=True)
    if args.dataset == "nuscenes":
        _run_nuscenes(args, cfg)
    else:
        _run_navsim(args, cfg)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
