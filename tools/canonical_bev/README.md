# canonical_bev — metadata → canonical BEV (Stage 0)

Standalone toolkit (independent of the RAP training pipeline) that builds an
ego-centric, **camera-agnostic** multi-channel BEV raster purely from dataset
metadata (3D boxes + map + ego) for **nuScenes** and **NAVSIM**, and visualizes
it. This is the front-end of the Sekikawa camera-agnostic-planner idea; see
`docs/plan_v1_0624.md` for the full A–J plan.

## Layout
```
config.py            CanonBEVConfig + ChannelSpec registry (the channel set is data, not code)
scene_meta.py        SceneMeta — ego-frame intermediate representation
canonical_bev.py     dataset-agnostic rasterizer (_xy_to_pixel / _paint_* / build_canonical_bev)
adapters/
  navsim_adapter.py  navsim Scene  -> SceneMeta
  nuscenes_adapter.py nuScenes sample -> SceneMeta
visualize.py         CLI: per-channel grid + composite RGB PNG
```

## Coordinate convention (pinned by `tests/canonical_bev/test_coord.py`, do NOT change)
- ego frame: **+x = forward, +y = left**, ego at raster center.
- `col = size/2 + (y/bev_range)*(size/2)`, `row = size/2 - (x/bev_range)*(size/2)`.
- `size=128, bev_range=32` ⇒ `(x=10,y=0) → (col=64,row=44)`; `(0,0) → (64,64)`.
- **forward = up** in the image.
- ⚠️ **+y (ego-left) maps to higher column ⇒ it appears on the RIGHT of the
  rendered image** (a mirror of the usual god's-eye view). This is intentional:
  it matches the `grid_sample` lockstep (`gx = y/bev_range`, `gy = -x/bev_range`,
  `align_corners=True`) that `BEVGridRefiner` uses downstream. Keep raster and
  sampler in lockstep; do not "fix" the mirror.

## Channels (default v1 — 6, extensible)
`drivable_area`, `lane_centerline`, `walkway`, `vehicle`, `pedestrian`, `ego`.
To add `velocity_x/_y`, `yaw_sin/cos`, multi-frame occupancy, or `route`: append a
`ChannelSpec` in `config.py`. New `kind`s also need a branch in
`build_canonical_bev`. Box `yaw` is already encoded implicitly in the footprint.

## Usage
Synthetic demo (no dataset; renders on any machine with numpy+cv2+matplotlib):
```
python -m tools.canonical_bev.visualize --dataset synthetic --out out/canonical_bev
```
NAVSIM (run where OpenScene data + nuPlan maps live; needs `NUPLAN_MAPS_ROOT`,
`OPENSCENE_DATA_ROOT`):
```
python -m tools.canonical_bev.visualize --dataset navsim --token <scene_token> \
    --data-path <navsim_logs_dir> --sensor-blobs-path <blobs_dir> --out out/canonical_bev
```
nuScenes (needs `pip install nuscenes-devkit` + data root with map expansion):
```
python -m tools.canonical_bev.visualize --dataset nuscenes --token <sample_token> \
    --dataroot <nuscenes_root> --version v1.0-trainval --out out/canonical_bev
```
Outputs `<name>_channels.png` (per-channel grid) and `<name>_composite.png`.

## Verification status
- `tests/canonical_bev/test_coord.py` pins the convention; the rasterizer math +
  grid_sample lockstep are verified (forward box → row 44 / col 64, hit=1.0).
  The pytest itself imports `navsim.common.dataclasses.Annotations`, so run it in
  an env with the full navsim stack (cluster).
- Synthetic render verified locally (`out/canonical_bev/synthetic_demo_*.png`).
- NAVSIM / nuScenes adapters require their datasets; not runnable on the dev
  laptop (no data root / no `nuscenes-devkit`). Per-dataset sanity: place a box
  10 m ahead and confirm it lands at row≈44, col≈64 to validate global→ego.

## Notes / known gaps
- nuScenes ego origin (IMU/baselink) vs NAVSIM (rear axle) differ — confirm yaw
  sign per dataset with the forward-box check.
- nuScenes true lane centerlines (arcline discretization) are best-effort; the
  `centerline` channel falls back to lane/road dividers if unavailable.
- velocity channel intentionally deferred (avoids global/ego velocity-frame bug).
