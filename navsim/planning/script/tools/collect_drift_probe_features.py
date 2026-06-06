"""Collect RAP drift-representation probe features.

Example:
python navsim/planning/script/tools/collect_drift_probe_features.py \
    --checkpoint ckpts/RAP_DINO_navsimv1.ckpt \
    --data-root dataset_perturbed \
    --original-data-root dataset_norm \
    --split mini \
    --drift-source perturbed \
    --output-dir outputs/drift_probe_features_perturbed \
    --device cuda \
    --image-source rendered
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
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
from pyquaternion import Quaternion
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# transformers>=4.56 calls the public pytree API added after torch 2.1.
if (
    not hasattr(torch.utils._pytree, "register_pytree_node")
    and hasattr(torch.utils._pytree, "_register_pytree_node")
):
    def _register_pytree_node_compat(typ, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return torch.utils._pytree._register_pytree_node(
            typ, flatten_fn, unflatten_fn, **kwargs
        )

    torch.utils._pytree.register_pytree_node = _register_pytree_node_compat

from navsim.agents.rap_dino.navsim_config import RAPConfig
from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder
from navsim.agents.rap_dino.rap_model import RAPModel
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)


PRIMARY_HOOKS = [
    "_trajectory_head.0.Bev_refiner",
    "_trajectory_head.0.traj_decoder.mlp.5",
    "_backbone",
]


def _parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze RAP and collect pooled intermediate features for drift probe training."
    )
    parser.add_argument("--checkpoint", default="ckpts/RAP_DINO_navsimv1.ckpt")
    parser.add_argument("--data-root", default="dataset_norm")
    parser.add_argument(
        "--original-data-root",
        default=None,
        help="Original/non-perturbed dataset root used to label dataset_perturbed samples.",
    )
    parser.add_argument("--split", default="mini")
    parser.add_argument("--pkl-glob", default=None)
    parser.add_argument("--output-dir", default="outputs/drift_probe_features")
    parser.add_argument("--max-scenes", type=int, default=128, help="Use <=0 for all scenes.")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-poses", type=int, default=10)
    parser.add_argument("--interval-length", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--delta-lat-values",
        default="-0.5,-0.25,0,0.25,0.5",
        help="Controlled lateral drift grid in meters. Default matches perturbed script xy +/-0.5m scale.",
    )
    parser.add_argument(
        "--delta-yaw-values-deg",
        default="-15,-7.5,0,7.5,15",
        help="Controlled yaw drift grid in degrees. Default matches perturbed script yaw +/-15deg scale.",
    )
    parser.add_argument(
        "--grid-mode",
        default="factorial",
        choices=["factorial", "axis"],
        help="factorial uses all lat*yaw combinations; axis uses lat-only plus yaw-only perturbations.",
    )
    parser.add_argument(
        "--drift-source",
        default="auto",
        choices=["auto", "controlled", "perturbed"],
        help=(
            "controlled dynamically modifies ego pose on data-root scenes; "
            "perturbed uses existing dataset_perturbed logs/images and labels them against original-data-root."
        ),
    )
    parser.add_argument(
        "--hooks",
        default=",".join(PRIMARY_HOOKS),
        help="Comma-separated module names relative to RAPModel. RAPAgent/_rap_model/module prefixes are handled.",
    )
    parser.add_argument(
        "--image-source",
        default="rendered",
        choices=["real", "rendered"],
        help="Use camera_feature or rendered_camera_feature when available.",
    )
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--feature-dtype", default="float32", choices=["float32", "float16"])
    return parser.parse_args()


def _build_sensor_config() -> SensorConfig:
    return SensorConfig(
        cam_f0=[3],
        cam_l0=[3],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[3],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[3],
        lidar_pc=[],
    )


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def _load_model(checkpoint_path: Path, device: torch.device, config: RAPConfig) -> RAPModel:
    os.environ.setdefault("RAP_DINO_OFFLINE_INIT", "1")
    model = RAPModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("agent._rap_model."):
            model_state[key[len("agent._rap_model.") :]] = value
        elif key.startswith("_rap_model."):
            model_state[key[len("_rap_model.") :]] = value
        elif not key.startswith("agent."):
            model_state[key] = value

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"Missing model keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected checkpoint keys ignored: {len(unexpected)}")

    model.to(device)
    model.eval()
    model.progress = 1.0
    model.batch_size = 1
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _module_name_candidates(name: str) -> List[str]:
    names = [name]
    for prefix in ("module.", "_rap_model.", "agent._rap_model."):
        if name.startswith(prefix):
            names.append(name[len(prefix) :])
        else:
            names.append(prefix + name)
    return list(dict.fromkeys(names))


def _resolve_module(model: torch.nn.Module, name: str) -> Tuple[str, torch.nn.Module]:
    unwrapped = _unwrap_model(model)
    modules = dict(unwrapped.named_modules())
    for candidate in _module_name_candidates(name):
        if candidate in modules:
            return candidate, modules[candidate]
    raise KeyError(
        f"Could not find hook module '{name}'. Available examples: "
        f"{list(modules.keys())[:20]}"
    )


def _first_tensor_like(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    if isinstance(value, dict):
        for key in ("last_hidden_state", "logits", "hidden_states"):
            if key in value:
                return _first_tensor_like(value[key])
        for item in value.values():
            found = _first_tensor_like(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return _first_tensor_like(value[0])
    return None


def _pool_tensor(tensor: torch.Tensor, hook_name: str) -> Tuple[np.ndarray, str]:
    x = tensor.detach().float().cpu()
    pooling_name = "unknown"

    if x.ndim == 4:
        # _backbone returns image memory as [num_cam, num_tokens, B, D].
        if hook_name.endswith("_backbone") and x.shape[2] <= 64:
            pooled = x.mean(dim=(0, 1))
            pooling_name = "cam_token_mean"
        # Convolutional feature maps are [B, C, H, W].
        elif x.shape[1] >= x.shape[-1] and x.shape[1] >= x.shape[-2]:
            pooled = x.mean(dim=(2, 3))
            pooling_name = "global_avg_hw"
        else:
            pooled = x.reshape(x.shape[0], -1, x.shape[-1]).mean(dim=1)
            pooling_name = "token_mean_4d"
    elif x.ndim == 3:
        # Planning tokens: [B, 64*8, D] in the current config.
        if x.shape[1] == 64 * 8:
            pooled = x.reshape(x.shape[0], 64, 8, x.shape[2]).mean(dim=(1, 2))
            pooling_name = "proposal_time_mean"
        # DINO output is [B*num_cam, tokens, D].
        else:
            pooled = x.mean(dim=1)
            pooling_name = "token_mean"
    elif x.ndim == 2:
        pooled = x
        pooling_name = "identity"
    else:
        pooled = x.reshape(x.shape[0], -1)
        pooling_name = "flatten"

    return pooled.numpy(), pooling_name


class HookCollector:
    def __init__(self, model: torch.nn.Module, hook_names: List[str]):
        self.records: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.handles: List[Any] = []
        self.resolved_names: List[str] = []
        for requested_name in hook_names:
            resolved_name, module = _resolve_module(model, requested_name)
            self.resolved_names.append(resolved_name)
            self.handles.append(module.register_forward_hook(self._make_hook(resolved_name)))

    def _make_hook(self, hook_name: str):
        def hook(_module, _inputs, output):
            value = _first_tensor_like(output)
            if value is None:
                return
            pooled, pooling_name = _pool_tensor(value, hook_name)
            call_idx = len(self.records[hook_name])
            self.records[hook_name].append(
                {
                    "call_idx": call_idx,
                    "pooling_name": pooling_name,
                    "feature": pooled,
                    "raw_shape": list(value.shape),
                }
            )

        return hook

    def clear(self) -> None:
        self.records.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _iter_scene_pickles(log_dir: Path, max_scenes: Optional[int]) -> Iterable[Path]:
    count = 0
    for path in sorted(log_dir.glob("*.pkl")):
        if max_scenes is not None and count >= max_scenes:
            break
        count += 1
        yield path


def _infer_drift_source(data_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if "perturbed" in data_root.name or (data_root / "rendered_sensor_blobs_perturbed").exists():
        return "perturbed"
    return "controlled"


def _default_original_data_root(data_root: Path) -> Path:
    if data_root.name == "dataset_perturbed":
        return data_root.with_name("dataset_norm")
    if "perturbed" in data_root.name:
        return data_root.parent / data_root.name.replace("perturbed", "norm")
    return data_root


def _source_log_name(log_name: str) -> str:
    if log_name.endswith(".pkl"):
        log_name = log_name[:-4]
    # dataset_perturbed appends the current sample token/hash to the source log.
    return log_name.rsplit("_", 1)[0]


def _find_frame_index_by_timestamp(frames: List[Dict], timestamp: Any) -> Optional[int]:
    for idx, frame in enumerate(frames):
        if frame.get("timestamp") == timestamp:
            return idx
    return None


def _load_original_context(
    cache: Dict[str, Optional[List[Dict]]],
    original_data_root: Path,
    split: str,
    current_frame: Dict,
) -> Tuple[List[Dict], int]:
    source_log = _source_log_name(str(current_frame["log_name"]))
    if source_log not in cache:
        path = original_data_root / "navsim_logs" / split / f"{source_log}.pkl"
        cache[source_log] = pickle.load(open(path, "rb")) if path.exists() else None
    frames = cache[source_log]
    if frames is None:
        raise FileNotFoundError(f"Missing original log for perturbed sample: {source_log}")
    idx = _find_frame_index_by_timestamp(frames, current_frame.get("timestamp"))
    if idx is None:
        raise KeyError(f"Could not match original frame by timestamp: {current_frame.get('timestamp')}")
    return frames, idx


def _frame_pose(frame: Dict) -> np.ndarray:
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _angle_abs_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(_wrap_angle(pred - target))


def _future_label_from_index(frames: List[Dict], start_idx: int, num_poses: int) -> np.ndarray:
    if len(frames) <= start_idx + num_poses:
        raise ValueError("scene does not have enough future frames")
    global_poses = [_frame_pose(frames[idx]) for idx in range(start_idx, start_idx + num_poses + 1)]
    local_poses = convert_absolute_to_relative_se2_array(
        StateSE2(*global_poses[0]), np.asarray(global_poses[1:], dtype=np.float64)
    )
    return local_poses.astype(np.float32)


def _transform_trajectory_between_ego_frames(
    trajectory: np.ndarray,
    from_ego_pose: np.ndarray,
    to_ego_pose: np.ndarray,
) -> np.ndarray:
    from_yaw = from_ego_pose[2]
    to_yaw = to_ego_pose[2]
    c_from, s_from = np.cos(from_yaw), np.sin(from_yaw)
    c_to, s_to = np.cos(-to_yaw), np.sin(-to_yaw)
    r_from = np.array([[c_from, -s_from], [s_from, c_from]], dtype=np.float64)
    r_to_inv = np.array([[c_to, -s_to], [s_to, c_to]], dtype=np.float64)
    global_xy = trajectory[:, :2] @ r_from.T + from_ego_pose[:2]
    target_xy = (global_xy - to_ego_pose[:2]) @ r_to_inv.T
    target_heading = _wrap_angle(trajectory[:, 2] + from_yaw - to_yaw)
    return np.column_stack([target_xy, target_heading]).astype(np.float32)


def _drift_current_frame(frames: List[Dict], current_idx: int, delta_lat: float, delta_yaw: float) -> List[Dict]:
    drifted = copy.deepcopy(frames[: current_idx + 1])
    frame = drifted[current_idx]
    pose = _frame_pose(frame)
    yaw = pose[2]

    # Positive delta_lat means ego is shifted to its left in the original ego frame.
    left = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float64).copy()
    translation[:2] += delta_lat * left
    new_yaw = float(_wrap_angle(np.array([yaw + delta_yaw], dtype=np.float64))[0])
    quat = Quaternion(axis=[0.0, 0.0, 1.0], radians=new_yaw)

    frame["ego2global_translation"] = translation
    frame["ego2global_rotation"] = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64)
    frame["token"] = f"{frame.get('token', 'sample')}_dlat{delta_lat:+.2f}_dyaw{math.degrees(delta_yaw):+.1f}"
    return drifted


def _drift_label_from_poses(original_pose: np.ndarray, drifted_pose: np.ndarray) -> Tuple[float, float, float]:
    local = convert_absolute_to_relative_se2_array(
        StateSE2(*original_pose), np.asarray([drifted_pose], dtype=np.float64)
    )[0]
    delta_long = float(local[0])
    delta_lat = float(local[1])
    delta_yaw = float(local[2])
    return delta_long, delta_lat, delta_yaw


def _make_drift_grid(delta_lats: List[float], delta_yaws: List[float], mode: str) -> List[Tuple[float, float]]:
    if mode == "factorial":
        return [(lat, yaw) for lat in delta_lats for yaw in delta_yaws]
    values = {(0.0, 0.0)}
    values.update((lat, 0.0) for lat in delta_lats)
    values.update((0.0, yaw) for yaw in delta_yaws)
    return sorted(values)


def _constant_velocity_baseline(current_frame: Dict, num_poses: int, interval_length: float) -> np.ndarray:
    velocity = np.asarray(current_frame["ego_dynamic_state"][:2], dtype=np.float32)
    steps = np.arange(1, num_poses + 1, dtype=np.float32)[:, None]
    xy = steps * interval_length * velocity[None]
    heading = np.zeros((num_poses, 1), dtype=np.float32)
    return np.concatenate([xy, heading], axis=-1)


def _trajectory_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    xy_err = np.linalg.norm(pred[:, :2] - target[:, :2], axis=-1)
    heading_err = _angle_abs_error(pred[:, 2], target[:, 2])
    return {
        "ade_to_gt": float(xy_err.mean()),
        "fde_to_gt": float(xy_err[-1]),
        "ahe_to_gt": float(heading_err.mean()),
        "fhe_to_gt": float(heading_err[-1]),
    }


def _recovery_metrics(
    pred_original: np.ndarray,
    original_reference: np.ndarray,
    delta_lat: float,
    delta_yaw: float,
) -> Dict[str, float]:
    lateral_errors = np.abs(pred_original[:, 1] - original_reference[:, 1])
    initial = float(abs(delta_lat))
    final = float(lateral_errors[-1])
    reduction = float(initial - final)
    recovery_ratio = float(reduction / initial) if initial > 1e-6 else float("nan")

    pred_lateral_delta = float(pred_original[-1, 1] - pred_original[0, 1])
    correction_direction_sign = float(np.sign(pred_lateral_delta))
    expected_sign = float(-np.sign(delta_lat))
    correction_matches = (
        float(correction_direction_sign == expected_sign)
        if abs(delta_lat) > 1e-6 and correction_direction_sign != 0
        else float("nan")
    )

    heading_errors = _angle_abs_error(pred_original[:, 2], original_reference[:, 2])
    return {
        "initial_lateral_error_to_gt_path": initial,
        "final_lateral_error_to_gt_path": final,
        "lateral_error_reduction": reduction,
        "recovery_ratio": recovery_ratio,
        "correction_direction_sign": correction_direction_sign,
        "correction_matches_expected_direction": correction_matches,
        "initial_heading_error_to_gt_path": float(abs(delta_yaw)),
        "final_heading_error_to_gt_path": float(heading_errors[-1]),
    }


def _to_jsonable_trajectory(traj: np.ndarray) -> str:
    return json.dumps(np.round(traj.astype(float), 4).tolist(), separators=(",", ":"))


def _feature_key(hook_name: str, call_idx: int, pooling_name: str) -> str:
    safe_hook = hook_name.replace(".", "__").replace("/", "_")
    return f"{safe_hook}__call{call_idx}__{pooling_name}"


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    trajectories_dir = output_dir / "trajectories"
    features_dir.mkdir(parents=True, exist_ok=True)
    if args.save_trajectories:
        trajectories_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = RAPConfig(
        trajectory_sampling=TrajectorySampling(
            num_poses=args.num_poses,
            interval_length=args.interval_length,
        )
    )
    model = _load_model(checkpoint_path, device, config)
    hook_names = [item.strip() for item in args.hooks.split(",") if item.strip()]
    hook_collector = HookCollector(model, hook_names)

    data_root = Path(args.data_root)
    log_dir = data_root / "navsim_logs" / args.split
    sensor_root = data_root / "sensor_blobs"
    if not sensor_root.exists() and (data_root / "sensor_blobs_perturbed").exists():
        sensor_root = data_root / "sensor_blobs_perturbed"
    elif data_root.name.endswith("perturbed"):
        sensor_root = data_root / "sensor_blobs_perturbed"
    if not log_dir.exists():
        raise FileNotFoundError(f"Missing NAVSIM log directory: {log_dir}")

    max_scenes = None if args.max_scenes <= 0 else args.max_scenes
    if args.pkl_glob:
        pkl_paths = [Path(path) for path in sorted(glob.glob(args.pkl_glob))]
        if max_scenes is not None:
            pkl_paths = pkl_paths[:max_scenes]
    else:
        pkl_paths = list(_iter_scene_pickles(log_dir, max_scenes))

    drift_source = _infer_drift_source(data_root, args.drift_source)
    original_data_root = Path(args.original_data_root) if args.original_data_root else _default_original_data_root(data_root)

    delta_lats = _parse_float_list(args.delta_lat_values)
    delta_yaws = [math.radians(value) for value in _parse_float_list(args.delta_yaw_values_deg)]
    drift_grid = _make_drift_grid(delta_lats, delta_yaws, args.grid_mode)
    if drift_source == "perturbed":
        drift_grid = [(float("nan"), float("nan"))]

    sensor_config = _build_sensor_config()
    feature_builder = RAPFeatureBuilder(config)

    metadata_rows: List[Dict[str, Any]] = []
    behavior_rows: List[Dict[str, Any]] = []
    feature_store: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
    hook_summary: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, str]] = []
    original_cache: Dict[str, Optional[List[Dict]]] = {}

    sample_index = 0
    iterator = tqdm(pkl_paths, desc="Collecting drift probe features")
    for pkl_path in iterator:
        try:
            frames = pickle.load(open(pkl_path, "rb"))
            current_idx = args.num_history_frames - 1
            if len(frames) <= current_idx + args.num_poses:
                failures.append({"pkl_path": str(pkl_path), "error": "not enough frames"})
                continue
            if drift_source == "perturbed":
                perturbed_frame = frames[current_idx]
                original_frames, original_idx = _load_original_context(
                    original_cache,
                    original_data_root,
                    args.split,
                    perturbed_frame,
                )
                original_frame = original_frames[original_idx]
                original_pose = _frame_pose(original_frame)
                original_reference = _future_label_from_index(original_frames, original_idx, args.num_poses)
            else:
                original_frame = frames[current_idx]
                original_pose = _frame_pose(original_frame)
                original_reference = _future_label_from_index(frames, current_idx, args.num_poses)
        except Exception as exc:
            failures.append({"pkl_path": str(pkl_path), "error": repr(exc)})
            continue

        for grid_delta_lat, grid_delta_yaw in drift_grid:
            try:
                if drift_source == "perturbed":
                    drifted_frames = frames
                else:
                    drifted_frames = _drift_current_frame(frames, current_idx, grid_delta_lat, grid_delta_yaw)
                current_frame = drifted_frames[current_idx]
                drifted_pose = _frame_pose(current_frame)
                delta_long, delta_lat, delta_yaw = _drift_label_from_poses(original_pose, drifted_pose)
                if drift_source == "controlled":
                    delta_long = 0.0
                    delta_lat = float(grid_delta_lat)
                    delta_yaw = float(grid_delta_yaw)
                target_in_drifted = _transform_trajectory_between_ego_frames(
                    original_reference,
                    original_pose,
                    drifted_pose,
                )

                agent_input = AgentInput.from_scene_dict_list(
                    drifted_frames,
                    sensor_root,
                    num_history_frames=args.num_history_frames,
                    sensor_config=sensor_config,
                )
                features = feature_builder.compute_features(agent_input)
                if args.image_source == "rendered" and "rendered_camera_feature" in features:
                    features["camera_feature"] = features["rendered_camera_feature"]
                if args.image_source == "real" and not args.allow_missing_images and "camera_valid" in features:
                    valid = bool(features["camera_valid"].item())
                    if not valid:
                        raise RuntimeError("missing one or more real camera images")

                features = {
                    key: value.unsqueeze(0).to(device)
                    for key, value in features.items()
                    if isinstance(value, torch.Tensor)
                }
                model.batch_size = features["ego_status"].shape[0]

                hook_collector.clear()
                with torch.no_grad():
                    prediction = model(features, targets=None)

                pred = prediction["trajectory"].squeeze(0).detach().cpu().numpy()[: args.num_poses]
                pred_original = _transform_trajectory_between_ego_frames(pred, drifted_pose, original_pose)
                target_original = original_reference
                top_score = float(prediction["pdm_score"].max().detach().cpu().item())
                cv = _constant_velocity_baseline(current_frame, args.num_poses, args.interval_length)
                cv_original = _transform_trajectory_between_ego_frames(cv, drifted_pose, original_pose)

                row = {
                    "sample_index": sample_index,
                    "sample_id": f"{original_frame['token']}__{sample_index:08d}",
                    "scene_token": original_frame.get("scene_token", ""),
                    "token": original_frame.get("token", ""),
                    "log_name": original_frame.get("log_name", ""),
                    "pkl_path": str(pkl_path),
                    "delta_lat": float(delta_lat),
                    "delta_long": float(delta_long),
                    "delta_yaw": float(delta_yaw),
                    "delta_yaw_deg": float(math.degrees(delta_yaw)),
                    "top_score": top_score,
                    "image_source": args.image_source,
                    "drift_source": drift_source,
                    "original_log_name": original_frame.get("log_name", ""),
                    "original_token": original_frame.get("token", ""),
                }
                metadata_rows.append(row)

                behavior = dict(row)
                behavior.update(_recovery_metrics(pred_original, target_original, delta_lat, delta_yaw))
                behavior.update(_trajectory_metrics(pred_original, target_original))
                behavior.update(
                    {
                        "cv_ade_to_gt": _trajectory_metrics(cv_original, target_original)["ade_to_gt"],
                        "cv_fde_to_gt": _trajectory_metrics(cv_original, target_original)["fde_to_gt"],
                    }
                )
                behavior_rows.append(behavior)

                if args.save_trajectories:
                    traj_path = trajectories_dir / f"{sample_index:08d}.json"
                    traj_path.write_text(
                        json.dumps(
                            {
                                "sample_index": sample_index,
                                "pred_drifted": json.loads(_to_jsonable_trajectory(pred)),
                                "pred_original": json.loads(_to_jsonable_trajectory(pred_original)),
                                "target_original": json.loads(_to_jsonable_trajectory(target_original)),
                                "target_drifted": json.loads(_to_jsonable_trajectory(target_in_drifted)),
                            },
                            indent=2,
                        )
                    )

                for hook_name, records in hook_collector.records.items():
                    for record in records:
                        feature = record["feature"]
                        if feature.shape[0] != 1:
                            raise RuntimeError(f"Expected batch size 1 from hook {hook_name}, got {feature.shape}")
                        key = _feature_key(hook_name, int(record["call_idx"]), str(record["pooling_name"]))
                        feature_store[key].append(feature[0].astype(args.feature_dtype))
                        hook_summary.setdefault(
                            key,
                            {
                                "hook_name": hook_name,
                                "call_idx": int(record["call_idx"]),
                                "pooling_name": str(record["pooling_name"]),
                                "raw_shape": record["raw_shape"],
                                "feature_dim": int(feature.shape[-1]),
                            },
                        )

                sample_index += 1
            except Exception as exc:
                failures.append(
                    {
                        "pkl_path": str(pkl_path),
                        "token": str(original_frame.get("token", "")),
                        "delta_lat": str(grid_delta_lat),
                        "delta_yaw": str(grid_delta_yaw),
                        "error": repr(exc),
                    }
                )

    hook_collector.close()

    if not metadata_rows:
        raise RuntimeError(f"No samples collected. First failures: {failures[:5]}")

    for key, values in feature_store.items():
        matrix = np.stack(values, axis=0)
        np.save(features_dir / f"{key}.npy", matrix)
        hook_summary[key]["num_samples"] = int(matrix.shape[0])
        hook_summary[key]["path"] = str(features_dir / f"{key}.npy")

    metadata_fields = [
        "sample_index",
        "sample_id",
        "scene_token",
        "token",
        "log_name",
        "pkl_path",
        "delta_lat",
        "delta_long",
        "delta_yaw",
        "delta_yaw_deg",
        "top_score",
        "image_source",
        "drift_source",
        "original_log_name",
        "original_token",
    ]
    behavior_fields = metadata_fields + [
        "initial_lateral_error_to_gt_path",
        "final_lateral_error_to_gt_path",
        "lateral_error_reduction",
        "recovery_ratio",
        "correction_direction_sign",
        "correction_matches_expected_direction",
        "initial_heading_error_to_gt_path",
        "final_heading_error_to_gt_path",
        "ade_to_gt",
        "fde_to_gt",
        "ahe_to_gt",
        "fhe_to_gt",
        "cv_ade_to_gt",
        "cv_fde_to_gt",
    ]
    _write_csv(output_dir / "metadata.csv", metadata_rows, metadata_fields)
    _write_csv(output_dir / "behavior_metrics.csv", behavior_rows, behavior_fields)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "original_data_root": str(original_data_root),
        "split": args.split,
        "drift_source": drift_source,
        "num_samples": len(metadata_rows),
        "num_failures": len(failures),
        "drift_grid": {
            "delta_lat_values": delta_lats,
            "delta_yaw_values_deg": [math.degrees(v) for v in delta_yaws],
            "grid_mode": args.grid_mode,
        },
        "requested_hooks": hook_names,
        "resolved_hooks": hook_collector.resolved_names,
        "features": hook_summary,
        "failures": failures[:100],
        "notes": [
            "RAP parameters are frozen; only features are collected.",
            "Controlled drift modifies ego pose/history frame geometry. It does not render new photorealistic camera images.",
            "Perturbed drift uses existing dataset_perturbed rendered_sensor_blobs_perturbed images and labels each sample by matching timestamp to original-data-root.",
            "_trajectory_head.0 modules are shared across refinement iterations; call_idx identifies each forward invocation.",
        ],
    }
    (output_dir / "hook_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Collected {len(metadata_rows)} samples into {output_dir}")
    if failures:
        print(f"Failures: {len(failures)}; see hook_summary.json for first 100.")


if __name__ == "__main__":
    main()
