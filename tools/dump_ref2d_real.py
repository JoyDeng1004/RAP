#!/usr/bin/env python3
"""
    python tools/dump_ref2d_real.py \
    --data-root dataset_norm --split mini \
    --checkpoint ckpts/RAP_DINO_navsimv2.ckpt --shift-y 1.0 \
    --scene-index 0 --num-poses 10 --image-source auto \
    --out outputs/ref2d_real_shift.npz
    
    python tools/viz_ref2d.py outputs/ref2d_real_shift.npz

"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for extra_path in (
    REPO_ROOT.parent / "mmdetection3d",
    REPO_ROOT.parent / "mmcv" / "mmdetection3d",
):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.append(str(extra_path))

import numpy as np
import torch

# torch._pytree compatibility shim (same as rollout_recovery_trajectory.py).
if (
    not hasattr(torch.utils._pytree, "register_pytree_node")
    and hasattr(torch.utils._pytree, "_register_pytree_node")
):
    def _register_pytree_node_compat(typ, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return torch.utils._pytree._register_pytree_node(typ, flatten_fn, unflatten_fn, **kwargs)

    torch.utils._pytree.register_pytree_node = _register_pytree_node_compat

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.agents.rap_dino.navsim_config import RAPConfig
from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder
from navsim.agents.rap_dino.bevformer.bev_feature_build import (
    CAMERA_ORDER,
    LoadMultiViewImageFromFiles,
)
from navsim.agents.rap_dino.bevformer.encoder import BEVFormerEncoder
from navsim.agents.rap_dino.bevformer.spatial_cross_attention import SpatialCrossAttention


# --------------------------------------------------------------------------- #
# Loading helpers (mirrors navsim/planning/script/tools/rollout_recovery_trajectory.py)
# --------------------------------------------------------------------------- #
def _build_sensor_config() -> SensorConfig:
    return SensorConfig(
        cam_f0=[3], cam_l0=[3], cam_l1=[], cam_l2=[],
        cam_r0=[3], cam_r1=[], cam_r2=[], cam_b0=[3], lidar_pc=[],
    )


def _load_model(checkpoint_path: Path, device: torch.device, config):
    os.environ.setdefault("RAP_DINO_OFFLINE_INIT", "1")
    from navsim.agents.rap_dino.rap_model import RAPModel

    model = RAPModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("agent._rap_model."):
            model_state[key[len("agent._rap_model.") :]] = value
        elif key.startswith("_rap_model."):
            model_state[key[len("_rap_model.") :]] = value
    if not model_state:
        raise ValueError(f"No RAP model weights found in {checkpoint_path}")

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if unexpected:
        print(f"Unexpected checkpoint keys ignored: {len(unexpected)}")
    if missing:
        print(f"Missing model keys: {len(missing)}")
    model.to(device)
    model.eval()
    model.progress = 1.0
    return model


def _select_pkl(log_dir: Path, pkl_glob, scene_index: int, sample_token):
    if pkl_glob:
        paths = sorted(glob.glob(pkl_glob))
    else:
        paths = sorted(str(p) for p in log_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No scene pickles found (log_dir={log_dir}, glob={pkl_glob}).")
    if sample_token:
        for path in paths:
            frames = pickle.load(open(path, "rb"))
            for frame in frames:
                if frame.get("token") == sample_token:
                    return path
        raise ValueError(f"sample_token {sample_token} not found in any pickle.")
    return paths[min(scene_index, len(paths) - 1)]


# --------------------------------------------------------------------------- #
# Module discovery + debug capture
# --------------------------------------------------------------------------- #
def _find_sca_modules(model):
    return [m for m in model.modules() if isinstance(m, SpatialCrossAttention)]


def _find_encoder(model):
    for m in model.modules():
        if isinstance(m, BEVFormerEncoder):
            return m
    raise RuntimeError("No BEVFormerEncoder found in model.")


# --------------------------------------------------------------------------- #
# Numpy / npz helpers (mirrors tools/dump_bev_feature_shift.py)
# --------------------------------------------------------------------------- #
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
    object_keys = [k for k, v in arrays.items() if np.asarray(v).dtype == object]
    if object_keys:
        raise TypeError(f"Refusing to save object arrays (not npz-safe): {object_keys}")


# --------------------------------------------------------------------------- #
# C6 densification: scatter rebatched sampling points back to BEV queries
# --------------------------------------------------------------------------- #
def _densify_sampling(sink, num_query):
    """Return (sampling_locations, attention_weights, valid) per BEV query.

    sink["sampling_locations"]: (num_cam, max_len, num_heads, num_levels, num_all_points, 2)
    sink["attention_weights"]:  (num_cam, max_len, num_heads, num_levels, num_points)
    sink["indexes"]: list[num_cam] of LongTensor of selected BEV-query indices.
    """
    sl = _to_numpy(sink["sampling_locations"])
    aw = _to_numpy(sink["attention_weights"])
    indexes = sink["indexes"]
    num_cam = sl.shape[0]
    num_heads = sl.shape[2]
    # Collapse the (num_levels, points) axes into a single "all points" axis.
    all_pts = int(np.prod(sl.shape[3:-1]))
    sl = sl.reshape(num_cam, sl.shape[1], num_heads, all_pts, 2)
    aw = aw.reshape(num_cam, aw.shape[1], num_heads, all_pts)

    dense_sl = np.full((num_cam, num_query, num_heads, all_pts, 2), np.nan, dtype=np.float32)
    dense_aw = np.zeros((num_cam, num_query, num_heads, all_pts), dtype=np.float32)
    valid = np.zeros((num_cam, num_query), dtype=bool)
    for cam in range(num_cam):
        idx = _to_numpy(indexes[cam]).astype(np.int64).reshape(-1)
        n = idx.shape[0]
        if n == 0:
            continue
        dense_sl[cam, idx] = sl[cam, :n].astype(np.float32)
        dense_aw[cam, idx] = aw[cam, :n].astype(np.float32)
        valid[cam, idx] = True
    return dense_sl, dense_aw, valid


# --------------------------------------------------------------------------- #
# Optional clean geometric isolation: reproject a fixed ref_2d with/without shift
# --------------------------------------------------------------------------- #
def _project_ref2d(encoder: BEVFormerEncoder, ref_2d, img_metas, shift_y, device):
    """Mirror BEVFormerEncoder.forward's SCA projection for a fixed ref_2d.

    Returns (reference_points_cam, bev_mask, ref_2d_sca) for the given shift.
    """
    bs, len_bev, _ = ref_2d.shape
    feats = {"ref2d_aug_shift_y": torch.tensor([float(shift_y)] * bs, device=device)}
    ref_2d_sca = encoder._sca_ref_2d(ref_2d, feats)

    zs = torch.linspace(
        encoder.pc_range[2] - encoder.lidar_height,
        encoder.pc_range[5] - encoder.lidar_height,
        encoder.num_points_in_pillar,
        dtype=torch.float32, device=device,
    )
    zs = zs[None, None, :, None].repeat(bs, len_bev, 1, 1)
    ref_pos = encoder.compute_corners(ref_2d_sca.reshape(-1, 3)).reshape(-1, len_bev, 4, 2)
    zs = zs.repeat(1, 1, 4, 1)
    ref_3d = torch.cat(
        [ref_pos.repeat(1, 1, encoder.num_points_in_pillar, 1), zs], dim=-1
    ).permute(0, 2, 1, 3)
    reference_points = ref_3d.to(torch.float32).clone()
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1
    ).permute(1, 0, 2, 3)
    reference_points_cam, bev_mask = encoder.point_sampling(reference_points, img_metas)
    return reference_points_cam, bev_mask, ref_2d_sca


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="ckpts/RAP_DINO_navsimv1.ckpt")
    parser.add_argument("--data-root", default="dataset_perturbed")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--pkl-glob", default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--sample-token", default=None)
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-poses", type=int, default=8,
                        help="Must match the checkpoint trajectory head.")
    parser.add_argument("--interval-length", type=float, default=0.5)
    parser.add_argument("--shift-y", type=float, default=1.0)
    parser.add_argument("--proposal", type=int, default=-1,
                        help="Proposal index to highlight. -1 = best by pdm_score.")
    parser.add_argument("--refine-stage-layer", type=int, default=-1,
                        help="Which SpatialCrossAttention module to capture for C6 (-1 = last).")
    parser.add_argument("--isolate-geometry", action="store_true",
                        help="Also emit a fixed-ref_2d reprojection (clean geometric isolation).")
    parser.add_argument("--image-source", default="auto", choices=["auto", "real", "rendered"],
                        help="Which camera images to overlay. 'auto' uses the real slot per camera "
                             "and falls back to the rendered image when the real slot is black "
                             "(common in the perturbed dataset).")
    parser.add_argument("--sensor-root", default=None,
                        help="Directory holding the camera sensor blobs (jpg). If omitted, auto-detect "
                             "under --data-root among: sensor_blobs_perturbed, rendered_sensor_blobs, "
                             "sensor_blobs. A missing directory yields all-black images, so this must "
                             "point at the real blobs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default="outputs/ref2d_real_shift.npz")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_root = Path(args.data_root)
    log_dir = data_root / "navsim_logs" / args.split
    checkpoint_path = Path(args.checkpoint)

    # Resolve the sensor-blob directory. A missing directory makes the loader return
    # all-black images silently, so we resolve it explicitly and fail loudly if absent.
    if args.sensor_root is not None:
        sensor_root = Path(args.sensor_root)
    else:
        candidates = [
            data_root / "sensor_blobs_perturbed",
            data_root / "rendered_sensor_blobs",
            data_root / "sensor_blobs",
        ]
        sensor_root = next((c for c in candidates if c.exists()), None)
        if sensor_root is None:
            raise FileNotFoundError(
                f"No sensor-blob directory found under {data_root} "
                f"(looked for {[c.name for c in candidates]}). Pass --sensor-root explicitly."
            )
    if not sensor_root.exists():
        raise FileNotFoundError(f"--sensor-root does not exist: {sensor_root}")
    print(f"Sensor blobs: {sensor_root}")

    # --- load one real scene ------------------------------------------------ #
    pkl_path = _select_pkl(log_dir, args.pkl_glob, args.scene_index, args.sample_token)
    print(f"Loading scene: {pkl_path}")
    frames = pickle.load(open(pkl_path, "rb"))
    if len(frames) < args.num_history_frames:
        raise ValueError("Not enough frames for the requested history length.")
    history_frames = frames[: args.num_history_frames]
    sample_token = history_frames[-1].get("token", "")

    sensor_config = _build_sensor_config()
    agent_input = AgentInput.from_scene_dict_list(
        history_frames,
        sensor_root,
        num_history_frames=len(history_frames),
        sensor_config=sensor_config,
    )

    plan_sampling = TrajectorySampling(num_poses=args.num_poses, interval_length=args.interval_length)
    config = RAPConfig(trajectory_sampling=plan_sampling)

    # Original-resolution RGB at the SAME resolution point_sampling normalizes by.
    # In the perturbed dataset the real ``cam.image`` slot is often a black placeholder
    # and the visible pixels live in ``rendered_image`` (synthetic=True), so we load both
    # and choose per --image-source (default: per-camera auto = whichever has signal).
    def _to_uint8(img):
        return np.clip(img, 0, 255).astype(np.uint8)

    real_imgs = [_to_uint8(i) for i in LoadMultiViewImageFromFiles(agent_input, synthetic=False)["img"]]
    rend_imgs = [_to_uint8(i) for i in LoadMultiViewImageFromFiles(agent_input, synthetic=True)["img"]]
    print("real    per-cam pixel max:", [int(i.max()) for i in real_imgs])
    print("rendered per-cam pixel max:", [int(i.max()) for i in rend_imgs])

    if args.image_source == "real":
        chosen = real_imgs
    elif args.image_source == "rendered":
        chosen = rend_imgs
    else:  # auto: per camera, fall back to rendered when the real slot is (near-)black
        chosen = [r if int(r.max()) > 5 else d for r, d in zip(real_imgs, rend_imgs)]
    print("image source per cam:", [
        ("real" if c is r else "rendered") for c, r in zip(chosen, real_imgs)
    ])

    if max(int(i.max()) for i in chosen) <= 5:
        raise RuntimeError(
            f"All camera images are black (max pixel <= 5) from sensor_root={sensor_root}. "
            "The blobs for this scene were not found/loaded. Check that --sensor-root points at the "
            "directory that actually contains this scene's camera jpgs (e.g. the *_sensor_blobs dir), "
            "and that the pkl's image filenames resolve under it."
        )

    # point_sampling normalizes EVERY camera by cam0's (real) image size
    # (img_metas['img_shape'][0][0]); cam0 == CAMERA_ORDER[0]. So normalized coords are
    # fractions of cam0's size, and normalized * cam0_size == each camera's own pixel.
    # Canvas every image (top-left, no resize -> no coordinate distortion) to cam0's size so
    # (a) they stack, and (b) viz's per-camera `pts * [W, H]` uses cam0_size and lands right.
    target_h, target_w = real_imgs[0].shape[:2]

    def _canvas_to(img, th, tw):
        out = np.zeros((th, tw, 3), dtype=np.uint8)
        h, w = min(img.shape[0], th), min(img.shape[1], tw)
        out[:h, :w] = img[:h, :w]
        return out

    if any(c.shape[:2] != (target_h, target_w) for c in chosen):
        print(f"Canvassing cameras to cam0 normalization size {(target_h, target_w)} "
              f"(differing sizes: {sorted({c.shape[:2] for c in chosen})}).")
    camera_images = np.stack([_canvas_to(c, target_h, target_w) for c in chosen], axis=0)

    # Model input features.
    feature_builder = RAPFeatureBuilder(config)
    features_cpu = feature_builder.compute_features(agent_input)
    base_features = {
        k: v.unsqueeze(0).to(device)
        for k, v in features_cpu.items()
        if isinstance(v, torch.Tensor)
    }

    # Feed the model whichever source actually has pixels, so proposals aren't computed on
    # black images. For 'auto', pick the source with more total signal (in this dataset the
    # rendered slot is empty and the real slot holds the pixels, or vice-versa).
    real_signal = sum(int(i.max()) for i in real_imgs)
    rend_signal = sum(int(i.max()) for i in rend_imgs)
    if args.image_source == "rendered":
        use_rendered_input = True
    elif args.image_source == "real":
        use_rendered_input = False
    else:  # auto
        use_rendered_input = rend_signal > real_signal
    if use_rendered_input and "rendered_camera_feature" in base_features:
        base_features["camera_feature"] = base_features["rendered_camera_feature"]
        print(f"Model input: rendered_camera_feature (signal real={real_signal} rendered={rend_signal}).")
    else:
        print(f"Model input: real camera_feature (signal real={real_signal} rendered={rend_signal}).")

    # --- build + load model ------------------------------------------------- #
    model = _load_model(checkpoint_path, device, config)
    model.batch_size = 1
    encoder = _find_encoder(model)
    sca_modules = _find_sca_modules(model)
    if not sca_modules:
        raise RuntimeError("No SpatialCrossAttention modules found in model.")
    target_sca = sca_modules[args.refine_stage_layer]
    print(f"Capturing C6 from SpatialCrossAttention #{args.refine_stage_layer} of {len(sca_modules)}.")

    def run_forward(shift_y):
        features = dict(base_features)
        features["ref2d_aug_shift_y"] = torch.tensor([float(shift_y)], device=device)
        ref2d_debug = {}
        features["ref2d_debug"] = ref2d_debug
        sink = {}
        target_sca.debug_sink = sink
        try:
            with torch.no_grad():
                output = model(features, targets=None)
        finally:
            # Clear both the wrapper sink and the nested deformable-attention sink it
            # routed into, so a reused model never keeps writing into a stale dict.
            target_sca.debug_sink = None
            if getattr(target_sca, "deformable_attention", None) is not None:
                target_sca.deformable_attention.debug_sink = None
        return output, ref2d_debug, sink

    base_out, base_dbg, base_sink = run_forward(0.0)
    shift_out, shift_dbg, shift_sink = run_forward(args.shift_y)

    # --- select highlighted proposal/query --------------------------------- #
    pdm = base_out.get("pdm_score", base_out.get("score"))
    proposal_num = int(config.proposal_num)
    num_poses = int(args.num_poses)
    if args.proposal >= 0:
        p_star = int(args.proposal)
    else:
        p_star = int(torch.argmax(pdm.reshape(-1)).item()) if pdm is not None else 0
    q_star = p_star * num_poses + (num_poses - 1)  # farthest pose: largest lateral travel

    # --- B4 fields from ref2d_debug ---------------------------------------- #
    ref_2d = _squeeze_batch(base_dbg["ref_2d_sca"])          # (Q, 3) baseline == shift 0
    ref_2d_sca = _squeeze_batch(shift_dbg["ref_2d_sca"])     # (Q, 3) shifted
    ref_pos = _squeeze_batch(base_dbg["ref_pos"])            # (Q, 2)
    num_query = ref_2d.shape[0]
    corners = _to_numpy(
        encoder.compute_corners(torch.as_tensor(ref_2d, device=device).reshape(-1, 3))
    ).reshape(num_query, 4, 2)

    # reference_points_cam: (num_cam, B, Q, D, 2) -> (num_cam, Q, D, 2)
    ref_cam_base = _to_numpy(base_dbg["reference_points_cam"])[:, 0]
    ref_cam_shift = _to_numpy(shift_dbg["reference_points_cam"])[:, 0]
    bev_mask_base = _to_numpy(base_dbg["bev_mask"])[:, 0]
    bev_mask_shift = _to_numpy(shift_dbg["bev_mask"])[:, 0]
    num_cam, _, D, _ = ref_cam_base.shape

    # --- C6 fields (densified) --------------------------------------------- #
    sl_base, aw_base, valid_base = _densify_sampling(base_sink, num_query)
    sl_shift, aw_shift, valid_shift = _densify_sampling(shift_sink, num_query)

    arrays = {
        # B4
        "camera_images": camera_images,
        "camera_order": np.array(list(CAMERA_ORDER)),
        "camera_image_color_order": "RGB",
        "reference_points_cam_baseline": ref_cam_base.astype(np.float32),
        "bev_mask_baseline": bev_mask_base.astype(bool),
        "reference_points_cam_shifted": ref_cam_shift.astype(np.float32),
        "bev_mask_shifted": bev_mask_shift.astype(bool),
        "reference_points_cam": ref_cam_base.astype(np.float32),
        "bev_mask": bev_mask_base.astype(bool),
        "ref_2d": ref_2d.astype(np.float32),
        "ref_2d_sca": ref_2d_sca.astype(np.float32),
        "ref_pos": ref_pos.astype(np.float32),
        "corners": corners.astype(np.float32),
        "shift_y": np.float32(args.shift_y),
        # C6
        "sampling_locations_baseline": sl_base,
        "attention_weights_baseline": aw_base,
        "sampling_valid_baseline": valid_base,
        "sampling_locations_shifted": sl_shift,
        "attention_weights_shifted": aw_shift,
        "sampling_valid_shifted": valid_shift,
        # scalars / metadata
        "Q": np.int64(num_query),
        "proposal_num": np.int64(proposal_num),
        "num_poses": np.int64(num_poses),
        "D": np.int64(D),
        "num_cam": np.int64(num_cam),
        "npp": np.int64(config.num_points_in_pillar),
        "pc_range": np.asarray(config.point_cloud_range, dtype=np.float32),
        "half_length": np.float32(config.half_length),
        "half_width": np.float32(config.half_width),
        "rear_axle_to_center": np.float32(config.rear_axle_to_center),
        "lidar_height": np.float32(config.lidar_height),
        "q_star": np.int64(q_star),
        "p_star": np.int64(p_star),
        "sample_token": str(sample_token),
    }

    if args.isolate_geometry:
        img_metas = {"lidar2img": base_features["lidar2img"], "img_shape": base_features["img_shape"]}
        ref_2d_t = torch.as_tensor(ref_2d, device=device).reshape(1, num_query, 3)
        cam_fixed_base, mask_fixed_base, _ = _project_ref2d(encoder, ref_2d_t, img_metas, 0.0, device)
        cam_fixed_shift, mask_fixed_shift, _ = _project_ref2d(encoder, ref_2d_t, img_metas, args.shift_y, device)
        arrays["reference_points_cam_baseline_fixed"] = _to_numpy(cam_fixed_base)[:, 0].astype(np.float32)
        arrays["reference_points_cam_shifted_fixed"] = _to_numpy(cam_fixed_shift)[:, 0].astype(np.float32)
        arrays["bev_mask_baseline_fixed"] = _to_numpy(mask_fixed_base)[:, 0].astype(bool)
        arrays["bev_mask_shifted_fixed"] = _to_numpy(mask_fixed_shift)[:, 0].astype(bool)

    _assert_no_object_arrays({k: v for k, v in arrays.items() if not isinstance(v, str)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print(
        f"Wrote {out_path}  | Q={num_query} num_cam={num_cam} D={D} "
        f"p*={p_star} q*={q_star} shift_y={args.shift_y}"
    )


if __name__ == "__main__":
    main()
