"""Usage:
python navsim/planning/script/tools/validate_recovery_trajectory.py \
    --checkpoint ckpts/RAP_DINO_navsimv1.ckpt \
    --pkl-glob "dataset_perturbed/navsim_logs/mini/*.pkl" \
    --output-dir outputs/recovery_validation_vis \
    --device cuda \
    --save-vis \
    --vis-debug \
    --make-html
"""

import argparse
import csv
import glob
import json
import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# transformers>=4.56 calls the public pytree API added after torch 2.1.
# This repository's navsim env currently has torch 2.1, where the same
# implementation is still exposed as a private helper.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an official RAP checkpoint on recovery-oriented perturbed "
            "NAVSIM logs and compare predictions with the perturbed recovery label."
        )
    )
    parser.add_argument("--checkpoint", default="ckpts/RAP_DINO_navsimv1.ckpt")
    parser.add_argument("--data-root", default="dataset_perturbed")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--pkl-glob", default=None)
    parser.add_argument("--output-dir", default="outputs/recovery_validation")
    parser.add_argument("--max-scenes", type=int, default=128, help="Use <=0 to run all scenes.")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-poses", type=int, default=10)
    parser.add_argument("--interval-length", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--use-rendered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rendered perturbed camera images as model input.",
    )
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--vis-max-samples", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=1)
    parser.add_argument("--vis-debug", action="store_true")
    parser.add_argument("--make-html", action="store_true")
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


def _frame_pose(frame: Dict) -> np.ndarray:
    translation = frame["ego2global_translation"]
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def _future_recovery_label(
    frames: List[Dict],
    num_history_frames: int,
    num_poses: int,
    interval_length: float,
) -> np.ndarray:
    start_idx = num_history_frames - 1
    end_idx = start_idx + num_poses
    if len(frames) <= end_idx:
        raise ValueError(f"Need at least {end_idx + 1} frames, got {len(frames)}")

    global_poses = [_frame_pose(frames[idx]) for idx in range(start_idx, end_idx + 1)]
    local_poses = convert_absolute_to_relative_se2_array(
        StateSE2(*global_poses[0]), np.array(global_poses[1:], dtype=np.float64)
    )
    if len(local_poses) != num_poses:
        raise ValueError(f"Expected {num_poses} target poses, got {len(local_poses)}")
    return local_poses.astype(np.float32)


def _constant_velocity_baseline(
    current_frame: Dict,
    num_poses: int,
    interval_length: float,
) -> np.ndarray:
    # Constant-velocity sanity baseline: ego keeps its current ego-frame
    # velocity and heading. This is not a recovery method; it only tests
    # whether RAP is clearly better than "do not recover, keep going".
    velocity = np.asarray(current_frame["ego_dynamic_state"][:2], dtype=np.float32)
    steps = np.arange(1, num_poses + 1, dtype=np.float32)[:, None]
    xy = steps * interval_length * velocity[None]
    heading = np.zeros((num_poses, 1), dtype=np.float32)
    return np.concatenate([xy, heading], axis=-1)


def _angle_abs_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = pred - target
    return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _metrics(pred: np.ndarray, target: np.ndarray, cv: np.ndarray) -> Dict[str, float]:
    xy_err = np.linalg.norm(pred[:, :2] - target[:, :2], axis=-1)
    cv_xy_err = np.linalg.norm(cv[:, :2] - target[:, :2], axis=-1)
    heading_err = _angle_abs_error(pred[:, 2], target[:, 2])
    return {
        "ade": float(xy_err.mean()),
        "fde": float(xy_err[-1]),
        "ahe": float(heading_err.mean()),
        "fhe": float(heading_err[-1]),
        "cv_ade": float(cv_xy_err.mean()),
        "cv_fde": float(cv_xy_err[-1]),
        "beats_cv": float(xy_err.mean() < cv_xy_err.mean()),
        "first_step_cos": _cosine(pred[0, :2], target[0, :2]),
        "final_step_cos": _cosine(pred[-1, :2], target[-1, :2]),
        "pred_final_norm": float(np.linalg.norm(pred[-1, :2])),
        "target_final_norm": float(np.linalg.norm(target[-1, :2])),
    }


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _transform_trajectory_between_ego_frames(
    trajectory: np.ndarray,
    from_ego_pose: np.ndarray,
    to_ego_pose: np.ndarray,
) -> np.ndarray:
    """Transform SE(2) trajectory from one ego-local frame into another."""
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


def _source_log_name(current_frame: Dict) -> str:
    return current_frame["log_name"].rsplit("_", 1)[0]


def _default_original_data_root(data_root: Path) -> Path:
    if data_root.name == "dataset_perturbed":
        return data_root.with_name("dataset_norm")
    return data_root.parent / "dataset_norm"


def _find_frame_index(frames: List[Dict], token: str) -> Optional[int]:
    for idx, frame in enumerate(frames):
        if frame.get("token") == token:
            return idx
    return None


def _future_label_from_index(frames: List[Dict], start_idx: int, num_poses: int) -> np.ndarray:
    if len(frames) <= start_idx + num_poses:
        raise ValueError("original reference does not have enough future frames")
    global_poses = [_frame_pose(frames[idx]) for idx in range(start_idx, start_idx + num_poses + 1)]
    local_poses = convert_absolute_to_relative_se2_array(
        StateSE2(*global_poses[0]), np.array(global_poses[1:], dtype=np.float64)
    )
    return local_poses.astype(np.float32)


def _load_original_context(
    original_cache: Dict[str, Optional[List[Dict]]],
    original_data_root: Path,
    split: str,
    current_frame: Dict,
    num_poses: int,
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[str]]:
    source_log = _source_log_name(current_frame)
    if source_log not in original_cache:
        original_path = original_data_root / "navsim_logs" / split / f"{source_log}.pkl"
        if original_path.exists():
            original_cache[source_log] = pickle.load(open(original_path, "rb"))
        else:
            original_cache[source_log] = None

    original_frames = original_cache[source_log]
    if original_frames is None:
        return None, None, f"original log unavailable: {source_log}"

    original_idx = _find_frame_index(original_frames, current_frame["token"])
    if original_idx is None:
        return None, None, f"original token unavailable: {current_frame['token']}"

    try:
        original_reference = _future_label_from_index(original_frames, original_idx, num_poses)
    except Exception as exc:
        return original_frames[original_idx], None, f"original reference unavailable: {exc}"

    return original_frames[original_idx], original_reference, None


def compute_recovery_metrics(
    pred: Optional[np.ndarray],
    target: Optional[np.ndarray],
    original_reference: Optional[np.ndarray],
) -> Tuple[Dict[str, Optional[float]], Optional[str]]:
    keys = [
        "lateral_error_initial",
        "lateral_error_final",
        "lateral_error_reduction_ratio",
        "heading_error_initial",
        "heading_error_final",
        "heading_error_reduction_ratio",
        "time_to_recover",
    ]
    empty = {key: None for key in keys}
    if pred is None or original_reference is None:
        return empty, "recovery metrics unavailable: missing original_reference"

    lateral_errors = np.abs(pred[:, 1] - original_reference[:, 1])
    heading_errors = _angle_abs_error(pred[:, 2], original_reference[:, 2])

    lateral_initial = float(lateral_errors[0])
    lateral_final = float(lateral_errors[-1])
    heading_initial = float(heading_errors[0])
    heading_final = float(heading_errors[-1])

    recovery_mask = (lateral_errors < 0.5) & (heading_errors < math.radians(5.0))
    recover_indices = np.flatnonzero(recovery_mask)
    time_to_recover = float((recover_indices[0] + 1) * 0.5) if len(recover_indices) else None

    return {
        "lateral_error_initial": lateral_initial,
        "lateral_error_final": lateral_final,
        "lateral_error_reduction_ratio": (
            float((lateral_initial - lateral_final) / lateral_initial)
            if lateral_initial > 1e-6
            else None
        ),
        "heading_error_initial": heading_initial,
        "heading_error_final": heading_final,
        "heading_error_reduction_ratio": (
            float((heading_initial - heading_final) / heading_initial)
            if heading_initial > 1e-6
            else None
        ),
        "time_to_recover": time_to_recover,
    }, None


def _load_image_or_none(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


def _candidate_original_camera_path(original_data_root: Path, split: str, rel_path: Path) -> Optional[Path]:
    openscene_data_root = os.environ.get("OPENSCENE_DATA_ROOT")
    nuplan_sensor_root = os.environ.get("NUPLAN_SENSOR_PATH")
    candidates = [
        *([Path(nuplan_sensor_root) / rel_path] if nuplan_sensor_root else []),
        *([Path(openscene_data_root) / "sensor_blobs" / split / rel_path] if openscene_data_root else []),
        *([Path(openscene_data_root) / "sensor_blobs" / rel_path] if openscene_data_root else []),
        original_data_root / "sensor_blobs" / split / rel_path,
        original_data_root / "sensor_blobs" / rel_path,
        REPO_ROOT.parent / "nuplan_dataset" / "nuplan-v1.1" / "sensor_blobs" / rel_path,
    ]
    return next((path for path in candidates if path.exists()), None)


def _lidar_to_image_matrix(cam: Dict) -> np.ndarray:
    lidar2cam_r = np.linalg.inv(np.asarray(cam["sensor2lidar_rotation"]))
    lidar2cam_t = np.asarray(cam["sensor2lidar_translation"]) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam["cam_intrinsic"])
    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


def _project_trajectory_to_image(
    trajectory: np.ndarray,
    cam: Dict,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    lidar2img = _lidar_to_image_matrix(cam)
    xyz1 = np.column_stack(
        [
            trajectory[:, 0],
            trajectory[:, 1],
            np.zeros(len(trajectory), dtype=np.float32),
            np.ones(len(trajectory), dtype=np.float32),
        ]
    )
    proj = xyz1 @ lidar2img.T
    depth = proj[:, 2]
    uv = proj[:, :2] / np.maximum(depth[:, None], 1e-6)
    height, width = image_shape[:2]
    valid = (
        (depth > 1e-5)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, valid


def _draw_projected_trajectory(
    ax,
    trajectory: np.ndarray,
    cam: Dict,
    image_shape: Tuple[int, int],
    color: str,
    label: str,
    linestyle: str = "-",
) -> Optional[str]:
    uv, valid = _project_trajectory_to_image(trajectory, cam, image_shape)
    if not valid.any():
        return f"{label} outside image"
    ax.plot(uv[valid, 0], uv[valid, 1], color=color, linestyle=linestyle, linewidth=2, label=label)
    first = np.flatnonzero(valid)[0]
    last = np.flatnonzero(valid)[-1]
    ax.scatter(uv[first, 0], uv[first, 1], color=color, marker="o", s=36)
    ax.scatter(uv[last, 0], uv[last, 1], color=color, marker="X", s=64)
    return None


def _draw_panel(
    ax,
    image: Optional[np.ndarray],
    title: str,
    frame_text: str,
    trajectories: Optional[List[Tuple[str, np.ndarray, str, str]]],
    cam: Optional[Dict],
) -> List[str]:
    warnings = []
    if image is None:
        image = np.full((360, 640, 3), 180, dtype=np.uint8)
        ax.imshow(image)
        ax.text(0.5, 0.5, "missing image", transform=ax.transAxes, ha="center", va="center", fontsize=14)
        warnings.append(f"{title}: missing image")
    else:
        ax.imshow(image)

    ax.set_title(f"{title}\n{frame_text}", fontsize=10)
    ax.axis("off")
    if trajectories is None:
        ax.text(0.5, 0.42, "transform unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=13)
        warnings.append(f"{title}: transform unavailable")
    elif cam is None:
        ax.text(0.5, 0.42, "camera calibration unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=13)
        warnings.append(f"{title}: camera calibration unavailable")
    else:
        drew_trajectory = False
        for label, traj, color, linestyle in trajectories:
            warning = _draw_projected_trajectory(ax, traj, cam, image.shape, color, label, linestyle)
            if warning:
                warnings.append(f"{title}: {warning}")
            else:
                drew_trajectory = True
        if drew_trajectory:
            ax.legend(loc="lower left", fontsize=8)
    return warnings


def _bev_display_xy(xy: np.ndarray) -> np.ndarray:
    return np.column_stack([-xy[:, 1], xy[:, 0]])


def _ego_speed_mps(frame: Optional[Dict]) -> Optional[float]:
    if frame is None or "ego_dynamic_state" not in frame:
        return None
    velocity = np.asarray(frame["ego_dynamic_state"][:2], dtype=np.float64)
    return float(np.linalg.norm(velocity))


def _format_speed(speed: Optional[float]) -> str:
    return "N/A" if speed is None else f"{speed:.2f} m/s"


def _format_ego_motion_text(original_frame: Optional[Dict], perturbed_frame: Dict) -> str:
    perturbed_pose = _frame_pose(perturbed_frame)
    perturbed_speed = _ego_speed_mps(perturbed_frame)
    lines = [
        "ego motion",
        f"pert: v={_format_speed(perturbed_speed)}, yaw={math.degrees(perturbed_pose[2]):.1f} deg",
    ]
    if original_frame is None:
        lines.insert(1, "orig: unavailable")
        return "\n".join(lines)

    original_pose = _frame_pose(original_frame)
    original_speed = _ego_speed_mps(original_frame)
    delta_yaw = float(_wrap_angle(np.array([perturbed_pose[2] - original_pose[2]], dtype=np.float64))[0])
    lines.insert(
        1,
        f"orig: v={_format_speed(original_speed)}, yaw={math.degrees(original_pose[2]):.1f} deg",
    )
    lines.append(f"pert-orig dyaw={math.degrees(delta_yaw):.1f} deg")
    return "\n".join(lines)


def _draw_ego_frame(ax, frame_pose: np.ndarray, color: str, label: str, axis_len: float = 2.5) -> None:
    x, y, yaw = frame_pose
    display_x, display_y = -y, x
    forward_dx = -axis_len * np.sin(yaw)
    forward_dy = axis_len * np.cos(yaw)
    left_dx = -axis_len * np.cos(yaw)
    left_dy = -axis_len * np.sin(yaw)
    ax.scatter([display_x], [display_y], color=color, s=32, marker="s", label=label)
    ax.arrow(
        display_x,
        display_y,
        forward_dx,
        forward_dy,
        color=color,
        width=0.035,
        head_width=0.35,
        length_includes_head=True,
    )
    ax.arrow(
        display_x,
        display_y,
        left_dx,
        left_dy,
        color=color,
        width=0.025,
        head_width=0.28,
        alpha=0.7,
        length_includes_head=True,
    )


def _draw_bev_panel(
    ax,
    title: str,
    trajectories: Optional[List[Tuple[str, np.ndarray, str, str]]],
    perturbed_pose_in_original: Optional[np.ndarray],
    ego_motion_text: str,
) -> List[str]:
    warnings = []
    ax.set_title(f"{title}\ntrajectories aligned to original ego", fontsize=10)
    ax.set_xlabel("lateral (m, left <- / right ->)")
    ax.set_ylabel("forward (m)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.axhline(0.0, color="0.85", linewidth=1.0)
    ax.axvline(0.0, color="0.85", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02,
        0.98,
        ego_motion_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    _draw_ego_frame(ax, np.array([0.0, 0.0, 0.0], dtype=np.float32), "black", "original ego frame")
    if perturbed_pose_in_original is None:
        ax.text(0.5, 0.5, "transform unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=13)
        warnings.append(f"{title}: transform unavailable")
    else:
        _draw_ego_frame(ax, perturbed_pose_in_original, "orange", "perturbed ego frame")

    if trajectories is None:
        ax.text(0.5, 0.42, "trajectory transform unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=13)
        warnings.append(f"{title}: trajectory transform unavailable")
    else:
        all_xy = [np.array([[0.0, 0.0]], dtype=np.float32)]
        if perturbed_pose_in_original is not None:
            all_xy.append(_bev_display_xy(perturbed_pose_in_original[None, :2]))
        for label, traj, color, linestyle in trajectories:
            display_xy = _bev_display_xy(traj[:, :2])
            ax.plot(display_xy[:, 0], display_xy[:, 1], color=color, linestyle=linestyle, linewidth=2.2, label=label)
            ax.scatter(display_xy[0, 0], display_xy[0, 1], color=color, marker="o", s=32)
            ax.scatter(display_xy[-1, 0], display_xy[-1, 1], color=color, marker="X", s=56)
            all_xy.append(display_xy)
        xy = np.concatenate(all_xy, axis=0)
        margin = 5.0
        ax.set_xlim(float(xy[:, 0].min() - margin), float(xy[:, 0].max() + margin))
        ax.set_ylim(float(xy[:, 1].min() - margin), float(xy[:, 1].max() + margin))
    ax.legend(loc="lower right", fontsize=8)
    return warnings


def _save_visualization(
    vis_path: Path,
    original_data_root: Path,
    data_root: Path,
    split: str,
    frames: List[Dict],
    current_frame: Dict,
    original_frame: Optional[Dict],
    pred: np.ndarray,
    target: np.ndarray,
    cv: np.ndarray,
    pred_original: Optional[np.ndarray],
    target_original: Optional[np.ndarray],
    cv_original: Optional[np.ndarray],
    metric_row: Dict[str, object],
    vis_debug: bool,
) -> List[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings = []
    rel_path = Path(current_frame["cams"]["CAM_F0"]["data_path"])

    original_camera_path = _candidate_original_camera_path(original_data_root, split, rel_path)
    original_raster_path = original_data_root / "rendered_sensor_blobs" / rel_path
    perturbed_raster_path = data_root / "rendered_sensor_blobs_perturbed" / rel_path

    original_camera = _load_image_or_none(original_camera_path)
    perturbed_raster = _load_image_or_none(perturbed_raster_path)

    original_cam = original_frame["cams"]["CAM_F0"] if original_frame is not None else None
    perturbed_cam = current_frame["cams"]["CAM_F0"]
    perturbed_pose_in_original = None
    if original_frame is not None:
        try:
            perturbed_pose_in_original = _transform_trajectory_between_ego_frames(
                np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                _frame_pose(current_frame),
                _frame_pose(original_frame),
            )[0]
        except Exception as transform_exc:
            warnings.append(f"frame transform unavailable: {repr(transform_exc)}")

    original_trajs = None
    if pred_original is not None and target_original is not None and cv_original is not None:
        original_trajs = [
            ("GT recovery", target_original, "lime", "-"),
            ("RAP pred", pred_original, "red", "-"),
            ("CV baseline", cv_original, "dodgerblue", "--"),
        ]
    ego_motion_text = _format_ego_motion_text(original_frame, current_frame)

    perturbed_trajs = [
        ("GT recovery", target, "lime", "-"),
        ("RAP pred", pred, "red", "-"),
        ("CV baseline", cv, "dodgerblue", "--"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    suptitle = (
        f"token={metric_row['token']} | log={metric_row['log_name']}\n"
        f"ADE/FDE={metric_row['ade']:.3f}/{metric_row['fde']:.3f} | "
        f"CV_ADE/CV_FDE={metric_row['cv_ade']:.3f}/{metric_row['cv_fde']:.3f} | "
        f"beats_cv={int(metric_row['beats_cv'])}"
    )
    fig.suptitle(suptitle, fontsize=12)
    warnings.extend(
        _draw_panel(
            axes[0],
            original_camera,
            "original camera",
            "original ego frame",
            original_trajs,
            original_cam,
        )
    )
    warnings.extend(
        _draw_bev_panel(
            axes[1],
            "BEV trajectory overlay",
            original_trajs,
            perturbed_pose_in_original,
            ego_motion_text,
        )
    )
    warnings.extend(
        _draw_panel(
            axes[2],
            perturbed_raster,
            "perturbed raster",
            "perturbed ego frame",
            perturbed_trajs,
            perturbed_cam,
        )
    )
    if vis_debug:
        fig.text(
            0.01,
            0.01,
            f"original_camera={original_camera_path}\noriginal_raster={original_raster_path}\nperturbed_raster={perturbed_raster_path}\n",
            fontsize=7,
        )
    fig.tight_layout()
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_path, dpi=150)
    plt.close(fig)
    return warnings


def _write_html(vis_rows: List[Dict[str, object]], html_path: Path) -> None:
    lines = ["<html><body><h1>RAP Recovery Validation</h1>"]
    for row in vis_rows:
        vis_path = row.get("vis_path")
        if not vis_path:
            continue
        rel = os.path.relpath(str(vis_path), start=str(html_path.parent))
        lines.append(
            f"<h3>{row['token']} | ADE={row['ade']:.3f} FDE={row['fde']:.3f} "
            f"beats_cv={int(row['beats_cv'])}</h3>"
        )
        if row.get("vis_warning"):
            lines.append(f"<pre>{row['vis_warning']}</pre>")
        lines.append(f'<img src="{rel}" style="max-width: 100%;"><hr>')
    lines.append("</body></html>")
    html_path.write_text("\n".join(lines))


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


def _iter_scene_pickles(log_dir: Path, max_scenes: int) -> Iterable[Path]:
    count = 0
    for path in sorted(log_dir.glob("*.pkl")):
        if max_scenes is not None and count >= max_scenes:
            break
        count += 1
        yield path


def _to_jsonable_trajectory(traj: np.ndarray) -> str:
    rounded = np.round(traj.astype(float), 4).tolist()
    return json.dumps(rounded, separators=(",", ":"))


def _summarize(rows: List[Dict[str, object]]) -> Dict[str, float]:
    numeric_keys = [
        "ade",
        "fde",
        "ahe",
        "fhe",
        "cv_ade",
        "cv_fde",
        "beats_cv",
        "first_step_cos",
        "final_step_cos",
        "pred_final_norm",
        "target_final_norm",
        "lateral_error_initial",
        "lateral_error_final",
        "lateral_error_reduction_ratio",
        "heading_error_initial",
        "heading_error_final",
        "heading_error_reduction_ratio",
        "time_to_recover",
    ]
    summary = {"num_success": len(rows)}
    for key in numeric_keys:
        values = np.asarray(
            [row[key] for row in rows if row.get(key) is not None],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if len(values) == 0:
            summary[f"mean_{key}"] = float("nan")
            summary[f"median_{key}"] = float("nan")
        else:
            summary[f"mean_{key}"] = float(values.mean())
            summary[f"median_{key}"] = float(np.median(values))
    summary["mean_ade_improvement_vs_cv"] = summary["mean_cv_ade"] - summary["mean_ade"]
    summary["mean_fde_improvement_vs_cv"] = summary["mean_cv_fde"] - summary["mean_fde"]
    return summary


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root)
    log_dir = data_root / "navsim_logs" / args.split
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not log_dir.exists():
        raise FileNotFoundError(f"Missing NAVSIM log directory: {log_dir}")
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
    sensor_config = _build_sensor_config()
    feature_builder = RAPFeatureBuilder(config)
    synthetic_sensor_root = data_root / "sensor_blobs_perturbed"
    original_data_root = _default_original_data_root(data_root)
    original_cache: Dict[str, Optional[List[Dict]]] = {}
    vis_dir = Path(args.vis_dir) if args.vis_dir else output_dir / "visualizations"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"recovery_validation_{timestamp}.csv"
    summary_path = output_dir / f"recovery_validation_{timestamp}.json"

    fieldnames = [
        "token",
        "log_name",
        "pkl_path",
        "ade",
        "fde",
        "ahe",
        "fhe",
        "cv_ade",
        "cv_fde",
        "beats_cv",
        "first_step_cos",
        "final_step_cos",
        "pred_final_norm",
        "target_final_norm",
        "lateral_error_initial",
        "lateral_error_final",
        "lateral_error_reduction_ratio",
        "heading_error_initial",
        "heading_error_final",
        "heading_error_reduction_ratio",
        "time_to_recover",
        "top_score",
        "vis_path",
        "vis_warning",
        "pred_trajectory",
        "target_trajectory",
    ]

    rows: List[Dict[str, float]] = []
    vis_rows: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []

    max_scenes = None if args.max_scenes <= 0 else args.max_scenes
    if args.pkl_glob:
        pkl_paths = [Path(path) for path in sorted(glob.glob(args.pkl_glob))]
        if max_scenes is not None:
            pkl_paths = pkl_paths[:max_scenes]
    else:
        pkl_paths = list(_iter_scene_pickles(log_dir, max_scenes))
    vis_saved = 0
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample_idx, pkl_path in enumerate(tqdm(pkl_paths, desc="Validating perturbed scenes")):
            try:
                frames = pickle.load(open(pkl_path, "rb"))
                current_frame = frames[args.num_history_frames - 1]
                target = _future_recovery_label(
                    frames,
                    args.num_history_frames,
                    args.num_poses,
                    args.interval_length,
                )
                cv = _constant_velocity_baseline(
                    current_frame,
                    args.num_poses,
                    args.interval_length,
                )

                agent_input = AgentInput.from_scene_dict_list(
                    frames,
                    synthetic_sensor_root,
                    num_history_frames=args.num_history_frames,
                    sensor_config=sensor_config,
                )
                features = feature_builder.compute_features(agent_input)
                if args.use_rendered:
                    features["camera_feature"] = features["rendered_camera_feature"]
                features = {
                    key: value.unsqueeze(0).to(device)
                    for key, value in features.items()
                    if isinstance(value, torch.Tensor)
                }

                model.batch_size = features["ego_status"].shape[0]
                with torch.no_grad():
                    prediction = model(features, targets=None)
                pred = prediction["trajectory"].squeeze(0).detach().cpu().numpy()[: args.num_poses]
                top_score = float(prediction["pdm_score"].max().detach().cpu().item())

                vis_warnings: List[str] = []
                original_frame, original_reference, original_warning = _load_original_context(
                    original_cache,
                    original_data_root,
                    args.split,
                    current_frame,
                    args.num_poses,
                )
                if original_warning:
                    vis_warnings.append(original_warning)

                pred_original = target_original = cv_original = None
                if original_frame is not None:
                    try:
                        perturbed_pose = _frame_pose(current_frame)
                        original_pose = _frame_pose(original_frame)
                        pred_original = _transform_trajectory_between_ego_frames(
                            pred, perturbed_pose, original_pose
                        )
                        target_original = _transform_trajectory_between_ego_frames(
                            target, perturbed_pose, original_pose
                        )
                        cv_original = _transform_trajectory_between_ego_frames(
                            cv, perturbed_pose, original_pose
                        )
                    except Exception as transform_exc:
                        vis_warnings.append(f"transform unavailable: {repr(transform_exc)}")
                else:
                    vis_warnings.append("transform unavailable: missing original ego pose")

                recovery_metrics, recovery_warning = compute_recovery_metrics(
                    pred_original,
                    target_original,
                    original_reference,
                )
                if recovery_warning:
                    vis_warnings.append(recovery_warning)

                metric_row = _metrics(pred, target, cv)
                metric_row.update(
                    {
                        "token": current_frame["token"],
                        "log_name": current_frame["log_name"],
                        "pkl_path": str(pkl_path),
                        **recovery_metrics,
                        "top_score": top_score,
                        "vis_path": "",
                        "vis_warning": "",
                        "pred_trajectory": _to_jsonable_trajectory(pred),
                        "target_trajectory": _to_jsonable_trajectory(target),
                    }
                )

                should_save_vis = (
                    args.save_vis
                    and (args.vis_max_samples <= 0 or vis_saved < args.vis_max_samples)
                    and args.vis_every > 0
                    and (sample_idx % args.vis_every == 0)
                )
                if should_save_vis:
                    vis_path = vis_dir / f"{current_frame['token']}.png"
                    try:
                        vis_warnings.extend(
                            _save_visualization(
                                vis_path,
                                original_data_root,
                                data_root,
                                args.split,
                                frames,
                                current_frame,
                                original_frame,
                                pred,
                                target,
                                cv,
                                pred_original,
                                target_original,
                                cv_original,
                                metric_row,
                                args.vis_debug,
                            )
                        )
                        metric_row["vis_path"] = str(vis_path)
                        vis_saved += 1
                    except Exception as vis_exc:
                        vis_warnings.append(f"visualization failed: {repr(vis_exc)}")

                metric_row["vis_warning"] = "; ".join(dict.fromkeys(w for w in vis_warnings if w))
                rows.append(metric_row)
                if metric_row["vis_path"]:
                    vis_rows.append(metric_row)
                writer.writerow(metric_row)
            except Exception as exc:
                failures.append((str(pkl_path), repr(exc)))

    summary = _summarize(rows) if rows else {"num_success": 0}
    summary.update(
        {
            "num_requested": len(pkl_paths),
            "num_failed": len(failures),
            "checkpoint": str(checkpoint_path),
            "data_root": str(data_root),
            "split": args.split,
            "use_rendered": args.use_rendered,
            "csv_path": str(csv_path),
            "vis_dir": str(vis_dir) if args.save_vis else "",
            "samples": rows,
            "failures": failures[:20],
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if args.make_html and args.save_vis:
        _write_html(vis_rows, output_dir / "index.html")

    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
