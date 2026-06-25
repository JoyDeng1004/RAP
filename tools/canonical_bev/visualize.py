"""Visualize the canonical BEV raster (Stage 0 sanity check).

Outputs (metadata-BEV only; no 3-way panel):
  <name>_channels.png   per-channel grayscale grid, ego marker + forward arrow
  <name>_composite.png  all channels alpha-blended into one RGB image

Usage:
  python -m tools.canonical_bev.visualize --dataset synthetic --out out/
  python -m tools.canonical_bev.visualize --dataset navsim --token <tok> \
      --data-path <pkls> --sensor-blobs-path <blobs> --out out/
  python -m tools.canonical_bev.visualize --dataset nuscenes --token <sample> \
      --dataroot <root> --version v1.0-trainval --out out/
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .canonical_bev import build_canonical_bev
from .config import CanonBEVConfig
from .scene_meta import SceneMeta


# --------------------------------------------------------------------------- #
# synthetic scene (no dataset needed -- proves rasterizer + viz end to end)
# --------------------------------------------------------------------------- #
def synthetic_scene_meta(cfg: CanonBEVConfig) -> SceneMeta:
    R = cfg.canon_bev_range
    # A straight road corridor along +x, plus a cross street, as drivable polygons.
    road_main = np.array([[-R, -7], [R, -7], [R, 7], [-R, 7]], np.float32)
    cross = np.array([[8, -R], [16, -R], [16, R], [8, R]], np.float32)
    walk = np.array([[-R, 7], [R, 7], [R, 10], [-R, 10]], np.float32)
    # lane centerlines
    lane_l = np.array([[-R, 3.5], [R, 3.5]], np.float32)
    lane_r = np.array([[-R, -3.5], [R, -3.5]], np.float32)
    # agents: 10m ahead (sanity), oncoming, side-parked, a pedestrian
    boxes = np.array([
        [10.0, 0.0, 0, 4.5, 2.0, 1.6, 0.0],          # lead vehicle straight ahead
        [22.0, 3.5, 0, 4.5, 2.0, 1.6, np.pi],         # oncoming in left lane
        [6.0, -5.5, 0, 4.5, 2.0, 1.6, 0.2],           # parked to the right
        [14.0, -6.0, 0, 0.7, 0.7, 1.7, 0.0],          # pedestrian on sidewalk
    ], np.float32)
    names = ["vehicle", "vehicle", "vehicle", "pedestrian"]
    return SceneMeta(
        boxes=boxes, names=names,
        map_polygons={"drivable": [road_main, cross], "walkway": [walk]},
        map_polylines={"centerline": [lane_l, lane_r]},
        ego_size=cfg.default_ego_size,
        meta={"dataset": "synthetic"},
    )


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def composite_rgb(raster: np.ndarray, cfg: CanonBEVConfig) -> np.ndarray:
    """Alpha-blend channels (registry order; ego last = on top) onto white."""
    H, W = raster.shape[1:]
    img = np.full((H, W, 3), 255.0, np.float32)
    for i, spec in enumerate(cfg.channels):
        m = raster[i] > 0.5
        if not m.any():
            continue
        color = np.array(spec.color, np.float32)
        img[m] = 0.4 * img[m] + 0.6 * color
    return img.clip(0, 255).astype(np.uint8)


def save_visualizations(raster: np.ndarray, cfg: CanonBEVConfig, out_dir: str, name: str,
                        channels: bool = True, composite: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)
    H = cfg.canon_raster_size
    ctr = H / 2.0
    saved = []

    # 1) per-channel grid
    if channels:
        n = cfg.num_channels
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i, spec in enumerate(cfg.channels):
            ax = axes[i]
            ax.imshow(raster[i], cmap="gray", origin="upper", vmin=0, vmax=1)
            ax.plot(ctr, ctr, "r+", ms=12, mew=2)
            ax.annotate("", xy=(ctr, ctr - H * 0.18), xytext=(ctr, ctr),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.set_title(f"[{i}] {spec.name}")
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"canonical BEV channels — {name}  (forward=up, ego=+)")
        fig.tight_layout()
        p1 = os.path.join(out_dir, f"{name}_channels.png")
        fig.savefig(p1, dpi=110); plt.close(fig)
        saved.append(p1)

    # 2) composite
    if composite:
        img = composite_rgb(raster, cfg)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, origin="upper")
        ax.plot(ctr, ctr, "k+", ms=14, mew=2)
        ax.annotate("", xy=(ctr, ctr - H * 0.16), xytext=(ctr, ctr),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))
        ax.set_title(f"canonical BEV composite — {name}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(handles=[Patch(color=np.array(s.color) / 255.0, label=s.name)
                           for s in cfg.channels],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        fig.tight_layout()
        p2 = os.path.join(out_dir, f"{name}_composite.png")
        fig.savefig(p2, dpi=110, bbox_inches="tight"); plt.close(fig)
        saved.append(p2)
    return saved


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_scene_meta(args, cfg):
    if args.dataset == "synthetic":
        return synthetic_scene_meta(cfg)
    if args.dataset == "navsim":
        from .adapters.navsim_adapter import load_scene_meta
        return load_scene_meta(args.token, args.data_path, args.sensor_blobs_path, cfg=cfg)
    if args.dataset == "nuscenes":
        from .adapters.nuscenes_adapter import load_sample_meta
        return load_sample_meta(args.dataroot, args.token, version=args.version, cfg=cfg)
    raise ValueError(args.dataset)


def main():
    ap = argparse.ArgumentParser(description="Visualize canonical BEV from metadata.")
    ap.add_argument("--dataset", choices=["synthetic", "navsim", "nuscenes"], required=True)
    ap.add_argument("--token", default=None, help="navsim scene token / nuscenes sample token")
    ap.add_argument("--out", default="out/canonical_bev")
    ap.add_argument("--name", default=None, help="output filename stem")
    # navsim
    ap.add_argument("--data-path", default=os.environ.get("OPENSCENE_DATA_ROOT", ""))
    ap.add_argument("--sensor-blobs-path", default="")
    # nuscenes
    ap.add_argument("--dataroot", default=os.environ.get("NUSCENES_DATAROOT", ""))
    ap.add_argument("--version", default="v1.0-trainval")
    args = ap.parse_args()

    cfg = CanonBEVConfig()
    scene_meta = _build_scene_meta(args, cfg)
    raster = build_canonical_bev(scene_meta, cfg)
    name = args.name or f"{args.dataset}_{args.token or 'demo'}"
    saved = save_visualizations(raster, cfg, args.out, name)
    nz = [f"{cfg.channels[i].name}:{int((raster[i] > 0.5).sum())}" for i in range(cfg.num_channels)]
    print(f"raster {raster.shape} dtype={raster.dtype}")
    print("nonzero px/channel:", "  ".join(nz))
    for p in saved:
        print("saved:", p)


if __name__ == "__main__":
    main()
