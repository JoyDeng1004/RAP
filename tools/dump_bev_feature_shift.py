#!/usr/bin/env python3
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navsim.agents.rap_dino.bevformer.encoder import BEVFormerEncoder
from navsim.agents.rap_dino.navsim_config import RAPConfig


def _minimal_encoder(config):
    encoder = BEVFormerEncoder.__new__(BEVFormerEncoder)
    encoder.return_intermediate = False
    encoder.num_points_in_pillar = int(config.num_points_in_pillar)
    encoder.pc_range = list(config.point_cloud_range)
    encoder.half_width = float(config.half_width)
    encoder.half_length = float(config.half_length)
    encoder.rear_axle_to_center = float(config.rear_axle_to_center)
    encoder.lidar_height = float(config.lidar_height)
    encoder.layers = []

    def fake_point_sampling(self, reference_points, img_metas):
        metric_points = reference_points.permute(1, 2, 0, 3)[None, ..., :2]
        projected = torch.stack([metric_points[..., 1], metric_points[..., 1]], dim=-1)
        bev_mask = projected[..., 1] > 0.75
        return projected, bev_mask

    encoder.point_sampling = types.MethodType(fake_point_sampling, encoder)
    return encoder


def _make_ref_2d(proposal_num, num_poses):
    x = torch.linspace(2.0, 2.0 * num_poses, num_poses)
    y = torch.linspace(-6.0, 6.0, proposal_num)
    headings = torch.zeros(proposal_num, num_poses)
    ref_2d = torch.stack(
        [
            x[None, :].expand(proposal_num, num_poses),
            y[:, None].expand(proposal_num, num_poses),
            headings,
        ],
        dim=-1,
    )
    return ref_2d.reshape(1, proposal_num * num_poses, 3)


def _run_encoder(encoder, ref_2d, channels, shift_y):
    batch_size, queries = ref_2d.shape[:2]
    base = torch.arange(queries * channels, dtype=torch.float32).reshape(queries, batch_size, channels)
    bev_query = base / max(float(queries * channels), 1.0)
    bev_pos = torch.zeros_like(bev_query)
    lidar2img = torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4)
    features = {
        "lidar2img": lidar2img,
        "img_shape": [[(256, 1024, 3)]],
        "ref2d_aug_shift_y": torch.tensor([shift_y], dtype=torch.float32),
        "ref2d_debug": {},
    }

    with torch.no_grad():
        encoder.forward(
            bev_query,
            None,
            None,
            bev_h=1,
            bev_w=queries,
            bev_pos=bev_pos,
            spatial_shapes=None,
            level_start_index=None,
            ref_2d=ref_2d,
            img_metas=features,
            features=features,
        )
    return features["ref2d_debug"]


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_batch(value):
    array = _to_numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def _assert_no_object_arrays(arrays):
    object_keys = [key for key, value in arrays.items() if np.asarray(value).dtype == object]
    if object_keys:
        raise TypeError(f"Refusing to write object arrays: {object_keys}")


def dump_synthetic(out_path, shift_y=None, proposal_num=None, num_poses=None, channels=8):
    config = RAPConfig()
    if shift_y is None:
        shift_y = float(config.ref2d_aug_y_range[1])
    if proposal_num is None:
        proposal_num = int(config.proposal_num)
    if num_poses is None:
        num_poses = int(config.trajectory_sampling.num_poses)

    encoder = _minimal_encoder(config)
    ref_2d = _make_ref_2d(proposal_num, num_poses)
    pre = _run_encoder(encoder, ref_2d, channels, shift_y=0.0)
    post = _run_encoder(encoder, ref_2d, channels, shift_y=shift_y)

    with torch.no_grad():
        corners = encoder.compute_corners(ref_2d.reshape(-1, 3)).reshape(-1, 4, 2)

    arrays = {
        "bev_feature_pre": _squeeze_batch(pre["bev_feature"]).astype(np.float32),
        "bev_feature_post": _squeeze_batch(post["bev_feature"]).astype(np.float32),
        "shift_y": np.asarray(shift_y, dtype=np.float32),
        "ref_2d": _squeeze_batch(ref_2d).astype(np.float32),
        "ref_pos": _squeeze_batch(pre["ref_pos"]).astype(np.float32),
        "corners": _to_numpy(corners).astype(np.float32),
        "proposal_num": np.asarray(proposal_num, dtype=np.int64),
        "num_poses": np.asarray(num_poses, dtype=np.int64),
        "pc_range": np.asarray(config.point_cloud_range, dtype=np.float32),
        "half_length": np.asarray(config.half_length, dtype=np.float32),
        "half_width": np.asarray(config.half_width, dtype=np.float32),
        "rear_axle_to_center": np.asarray(config.rear_axle_to_center, dtype=np.float32),
        "Q": np.asarray(proposal_num * num_poses, dtype=np.int64),
        "D": np.asarray(pre["reference_points_cam"].shape[3], dtype=np.int64),
        "num_cam": np.asarray(pre["reference_points_cam"].shape[0], dtype=np.int64),
        "reference_points_cam": _squeeze_batch(pre["reference_points_cam"]).astype(np.float32),
        "bev_mask": _squeeze_batch(pre["bev_mask"]).astype(bool),
    }
    _assert_no_object_arrays(arrays)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    with np.load(out_path, allow_pickle=False) as data:
        missing = [key for key in arrays if key not in data]
        if missing:
            raise RuntimeError(f"Dump verification failed, missing keys: {missing}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Dump BEV feature shift debug arrays for viz_ref2d.py.")
    parser.add_argument("--out", default="outputs/bev_feature_shift_synthetic.npz", help="Output .npz path")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use the lightweight synthetic encoder path; this is the current default")
    parser.add_argument("--shift-y", type=float, default=None,
                        help="Metric y shift. Defaults to RAPConfig.ref2d_aug_y_range upper bound.")
    parser.add_argument("--proposal-num", type=int, default=None, help="Synthetic proposal count")
    parser.add_argument("--num-poses", type=int, default=None, help="Synthetic poses per proposal")
    parser.add_argument("--channels", type=int, default=8, help="Synthetic BEV feature channels")
    args = parser.parse_args()

    out_path = dump_synthetic(
        args.out,
        shift_y=args.shift_y,
        proposal_num=args.proposal_num,
        num_poses=args.num_poses,
        channels=args.channels,
    )
    print(out_path)


if __name__ == "__main__":
    main()
