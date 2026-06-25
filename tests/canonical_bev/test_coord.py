"""P0 coordinate-frame test for the canonical BEV raster.

Pins that:
  1. a box 10 m directly ahead lands above-center (forward = up) in the raster, and
  2. the ``grid_sample`` normalization used by ``BEVGridRefiner`` samples the SAME
     location the raster painted it (raster <-> sampler lockstep).

If this test fails, alignment will never converge and "F-zeroing" ablations are
meaningless -- fix the coordinate convention before training anything.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# Make the repo root importable regardless of pytest's rootdir insertion mode,
# so `tools.canonical_bev` resolves when running `pytest tests/...`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.canonical_bev import canonical_bev as cb


class _Cfg:
    canon_raster_size = 128
    point_cloud_range = [-32, -32, -2.0, 32, 32, 6.0]


def _fake_annotations(x, y, yaw=0.0, vx=0.0, vy=0.0):
    from navsim.common.dataclasses import Annotations

    box = np.array([[x, y, 0.0, 4.0, 2.0, 1.5, yaw]], dtype=np.float32)  # x,y,z,l,w,h,yaw
    return Annotations(
        boxes=box,
        names=["vehicle"],
        velocity_3d=np.array([[vx, vy, 0.0]], dtype=np.float32),
        instance_tokens=["i0"],
        track_tokens=["t0"],
    )


def _paint_only_vehicle(annotations, size=128, bev_range=32.0):
    raster = np.zeros((cb.CANON_RASTER_CH, size, size), dtype=np.float32)
    cb._paint_boxes(raster, annotations, cb._VEHICLE_TYPES, cb.CH_VEHICLE, size, bev_range)
    return raster


def test_xy_to_pixel_forward_is_up_and_centered():
    size, bev_range = 128, 32.0
    # 10 m ahead, on centerline
    px = cb._xy_to_pixel(np.array([[10.0, 0.0]], dtype=np.float32), size, bev_range)[0]
    col, row = int(px[0]), int(px[1])
    assert col == 64, f"y=0 should map to center column, got {col}"
    # row = (32 - 10)/64 * 128 = 44  -> above center row (64) => forward is up
    assert row == 44, f"x=10 should map to row 44, got {row}"

    # ego (0,0) is the center of the frame
    px0 = cb._xy_to_pixel(np.array([[0.0, 0.0]], dtype=np.float32), size, bev_range)[0]
    assert (int(px0[0]), int(px0[1])) == (64, 64)

    # +y (left) moves to higher column
    px_left = cb._xy_to_pixel(np.array([[0.0, 10.0]], dtype=np.float32), size, bev_range)[0]
    assert int(px_left[0]) > 64


def test_box_centroid_lands_ahead_of_center():
    raster = _paint_only_vehicle(_fake_annotations(10.0, 0.0))
    veh = raster[cb.CH_VEHICLE]
    assert veh.sum() > 0, "vehicle box should paint some pixels"
    rows, cols = np.nonzero(veh)
    assert abs(cols.mean() - 64) < 3, f"box should be horizontally centered, got col {cols.mean():.1f}"
    assert rows.mean() < 64, f"box 10m ahead should be above center row, got row {rows.mean():.1f}"
    assert abs(rows.mean() - 44) < 6, f"box centroid row should be ~44, got {rows.mean():.1f}"


def test_raster_gridsample_lockstep():
    """The (gx, gy) used by BEVGridRefiner must sample the painted box."""
    size, bev_range = 128, 32.0
    raster = _paint_only_vehicle(_fake_annotations(10.0, 0.0), size, bev_range)
    F_feat = torch.from_numpy(raster[cb.CH_VEHICLE])[None, None]  # (1,1,H,W)

    x, y = 10.0, 0.0
    gx = y / bev_range          # -> width / col
    gy = -x / bev_range         # -> height / row   (see canonical_bev docstring)
    grid = torch.tensor([[[[gx, gy]]]], dtype=torch.float32)  # (1,1,1,2)
    sampled = F.grid_sample(F_feat, grid, align_corners=True, mode="nearest")
    assert sampled.item() > 0.5, "grid_sample at the box's (x,y) must hit the painted box"

    # a clearly empty location (20 m behind) must sample ~0
    gx2, gy2 = 0.0 / bev_range, -(-20.0) / bev_range
    grid2 = torch.tensor([[[[gx2, gy2]]]], dtype=torch.float32)
    sampled2 = F.grid_sample(F_feat, grid2, align_corners=True, mode="nearest")
    assert sampled2.item() < 0.5, "empty rear location must not hit the box"
