"""Usage:
python navsim/planning/script/tools/rollout_recovery_trajectory.py \
--save-vis --save-gif --make-html
"""

import argparse
import copy
import csv
import glob
import html
import io
import json
import math
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

if (
    not hasattr(torch.utils._pytree, "register_pytree_node")
    and hasattr(torch.utils._pytree, "_register_pytree_node")
):
    def _register_pytree_node_compat(typ, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return torch.utils._pytree._register_pytree_node(
            typ,
            flatten_fn,
            unflatten_fn,
            **kwargs,
        )

    torch.utils._pytree.register_pytree_node = _register_pytree_node_compat

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimeDuration, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pyquaternion import Quaternion
from tqdm import tqdm

from navsim.common.dataclasses import AgentInput, NUPLAN_MAPS_ROOT, SensorConfig, Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.visualization.bev import add_map_to_bev_ax, add_oriented_box_to_bev_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG, ELLIS_5, NEW_TAB_10, TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RAP in a receding-horizon recovery loop from perturbed ego states: "
            "plan locally, execute with NAVSIM LQR/kinematics, update ego, and re-plan."
        )
    )
    parser.add_argument("--checkpoint", default="ckpts/RAP_DINO_navsimv1.ckpt")
    parser.add_argument("--data-root", default="dataset_perturbed")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--pkl-glob", default=None)
    parser.add_argument("--output-dir", default="outputs/recovery_rollout_vis")
    parser.add_argument("--max-scenes", type=int, default=16, help="Use <=0 to run all scenes.")
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument(
        "--num-poses",
        type=int,
        default=10,
        help="Number of future trajectory poses predicted per RAP call. Must match the checkpoint trajectory head.",
    )
    parser.add_argument("--interval-length", type=float, default=0.5)
    parser.add_argument("--sim-interval-length", type=float, default=0.1)
    parser.add_argument("--execute-interval-length", type=float, default=0.5)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--use-rendered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rendered perturbed camera images as model input.",
    )
    parser.add_argument(
        "--observation-policy",
        default="reuse_latest_rendered",
        choices=["reuse_latest_rendered"],
        help=(
            "How to build observations after the executed ego state changes. "
            "This repo currently ships a state-rollout harness; plug an official NAVSIM v2 "
            "synthetic renderer here for full visual closed-loop observations."
        ),
    )
    parser.add_argument("--save-vis", action="store_true", help="Save rollout visualization images.")
    parser.add_argument("--vis-dir", default=None, help="Visualization directory. Defaults to output_dir/visualizations.")
    parser.add_argument("--vis-max-samples", type=int, default=100, help="Maximum samples to visualize. Use <=0 for all.")
    parser.add_argument("--vis-every", type=int, default=1, help="Visualize every Nth input sample when --save-vis is set.")
    parser.add_argument("--make-html", action="store_true", help="Write an index.html for saved visualizations.")
    parser.add_argument("--save-gif", action="store_true", help="Save ego-centered rollout GIFs.")
    parser.add_argument("--gif-duration-ms", type=int, default=500, help="GIF frame duration in milliseconds.")
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


def _to_jsonable_trajectory(traj: np.ndarray) -> str:
    rounded = np.round(traj.astype(float), 4).tolist()
    return json.dumps(rounded, separators=(",", ":"))


def _iter_scene_pickles(log_dir: Path, max_scenes: Optional[int]) -> Iterable[Path]:
    count = 0
    for path in sorted(log_dir.glob("*.pkl")):
        if max_scenes is not None and count >= max_scenes:
            break
        count += 1
        yield path


def _default_original_data_root(data_root: Path) -> Path:
    if data_root.name == "dataset_perturbed":
        return data_root.with_name("dataset_norm")
    return data_root.parent / "dataset_norm"


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


def _frame_timestamp(frame: Dict, fallback_us: int = 0) -> int:
    timestamp = frame.get("timestamp", fallback_us)
    return int(0 if timestamp is None else timestamp)


def _ego_state_from_frame(frame: Dict, fallback_timestamp_us: int = 0) -> EgoState:
    pose = _frame_pose(frame)
    dynamics = np.asarray(frame.get("ego_dynamic_state", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    if dynamics.shape[0] < 4:
        dynamics = np.pad(dynamics, (0, 4 - dynamics.shape[0]), constant_values=0.0)
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(float(pose[0]), float(pose[1]), float(pose[2])),
        rear_axle_velocity_2d=StateVector2D(float(dynamics[0]), float(dynamics[1])),
        rear_axle_acceleration_2d=StateVector2D(float(dynamics[2]), float(dynamics[3])),
        tire_steering_angle=0.0,
        time_point=TimePoint(_frame_timestamp(frame, fallback_timestamp_us)),
        vehicle_parameters=get_pacifica_parameters(),
        is_in_auto_mode=True,
        angular_vel=0.0,
        angular_accel=0.0,
        tire_steering_rate=0.0,
    )


def _state_array_pose(state_array: np.ndarray) -> np.ndarray:
    return state_array[StateIndex.STATE_SE2].astype(np.float64)


def _update_frame_from_ego_state(template_frame: Dict, ego_state: EgoState, token_suffix: str) -> Dict:
    frame = copy.deepcopy(template_frame)
    rear_axle = ego_state.rear_axle
    yaw = float(rear_axle.heading)
    quat = Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw)
    z = float(np.asarray(template_frame.get("ego2global_translation", [0.0, 0.0, 0.0]))[2])
    frame["ego2global_translation"] = np.array([rear_axle.x, rear_axle.y, z], dtype=np.float64)
    frame["ego2global_rotation"] = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64)
    frame["ego_dynamic_state"] = np.array(
        [
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
        ],
        dtype=np.float64,
    )
    frame["timestamp"] = int(ego_state.time_point.time_us)
    frame["token"] = f"{template_frame.get('token', 'rollout')}_{token_suffix}"
    frame["log_name"] = f"{template_frame.get('log_name', 'rollout')}_rollout"
    return frame


def _reference_global_poses(frames: List[Dict]) -> np.ndarray:
    return np.asarray([_frame_pose(frame) for frame in frames], dtype=np.float64)


def _reference_segment_local(
    reference_global: np.ndarray,
    current_pose: np.ndarray,
    start_index: int,
    num_poses: int,
) -> Optional[np.ndarray]:
    end_index = start_index + num_poses
    if end_index >= len(reference_global):
        return None
    future = reference_global[start_index + 1 : end_index + 1]
    return convert_absolute_to_relative_se2_array(
        StateSE2(*current_pose),
        future.astype(np.float64),
    ).astype(np.float32)


def _local_prediction_to_state_array(
    pred_local: np.ndarray,
    current_ego_state: EgoState,
    plan_sampling: TrajectorySampling,
    sim_sampling: TrajectorySampling,
) -> np.ndarray:
    trajectory = Trajectory(pred_local.astype(np.float32), plan_sampling)
    interpolated = transform_trajectory(trajectory, current_ego_state)
    return get_trajectory_as_array(
        interpolated,
        sim_sampling,
        current_ego_state.time_point,
    )[None, ...]


def _build_features(
    history_frames: List[Dict],
    synthetic_sensor_root: Path,
    sensor_config: SensorConfig,
    feature_builder,
    device: torch.device,
    use_rendered: bool,
) -> Dict[str, torch.Tensor]:
    agent_input = AgentInput.from_scene_dict_list(
        history_frames,
        synthetic_sensor_root,
        num_history_frames=len(history_frames),
        sensor_config=sensor_config,
    )
    features = feature_builder.compute_features(agent_input)
    if use_rendered:
        features["camera_feature"] = features["rendered_camera_feature"]
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in features.items()
        if isinstance(value, torch.Tensor)
    }


def _run_rap(
    model,
    history_frames: List[Dict],
    synthetic_sensor_root: Path,
    sensor_config: SensorConfig,
    feature_builder,
    device: torch.device,
    use_rendered: bool,
    num_poses: int,
) -> Tuple[np.ndarray, float]:
    features = _build_features(
        history_frames,
        synthetic_sensor_root,
        sensor_config,
        feature_builder,
        device,
        use_rendered,
    )
    model.batch_size = features["ego_status"].shape[0]
    with torch.no_grad():
        prediction = model(features, targets=None)
    pred = prediction["trajectory"].squeeze(0).detach().cpu().numpy()[:num_poses]
    top_score = float(prediction["pdm_score"].max().detach().cpu().item())
    return pred.astype(np.float32), top_score


def _pose_error(pred_pose: np.ndarray, ref_pose: np.ndarray) -> Dict[str, float]:
    xy_error = float(np.linalg.norm(pred_pose[:2] - ref_pose[:2]))
    heading_error = float(_angle_abs_error(np.array([pred_pose[2]]), np.array([ref_pose[2]]))[0])
    return {
        "executed_xy_error": xy_error,
        "executed_heading_error": heading_error,
        "executed_lateral_like_error": float(abs(pred_pose[1] - ref_pose[1])),
    }


def _summarize(rows: List[Dict[str, object]], failures: List[Tuple[str, str]]) -> Dict[str, object]:
    numeric_keys = [
        "step_ade",
        "step_fde",
        "step_cv_ade",
        "step_cv_fde",
        "executed_xy_error",
        "executed_heading_error",
        "executed_lateral_like_error",
    ]
    summary: Dict[str, object] = {
        "num_rows": len(rows),
        "num_failures": len(failures),
    }
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
        values = values[np.isfinite(values)]
        summary[f"mean_{key}"] = float(values.mean()) if len(values) else float("nan")
        summary[f"final_{key}"] = float(values[-1]) if len(values) else float("nan")
    return summary


def _safe_filename(value: str, fallback: str = "sample") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or fallback


def _trajectory_config(base_name: str, color: Optional[str] = None, line_style: Optional[str] = None) -> Dict[str, object]:
    config = dict(TRAJECTORY_CONFIG[base_name])
    if color is not None:
        config["fill_color"] = color
        config["line_color"] = color
    if line_style is not None:
        config["line_style"] = line_style
    return config


def _local_trajectory_to_global(local_trajectory: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    c = math.cos(float(current_pose[2]))
    s = math.sin(float(current_pose[2]))
    global_trajectory = np.asarray(local_trajectory, dtype=np.float64).copy()
    x_local = global_trajectory[:, 0].copy()
    y_local = global_trajectory[:, 1].copy()
    global_trajectory[:, 0] = float(current_pose[0]) + x_local * c - y_local * s
    global_trajectory[:, 1] = float(current_pose[1]) + x_local * s + y_local * c
    global_trajectory[:, 2] = np.arctan2(
        np.sin(global_trajectory[:, 2] + float(current_pose[2])),
        np.cos(global_trajectory[:, 2] + float(current_pose[2])),
    )
    return global_trajectory


def _global_pose_to_local(global_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    return convert_absolute_to_relative_se2_array(
        StateSE2(float(current_pose[0]), float(current_pose[1]), float(current_pose[2])),
        np.asarray([global_pose], dtype=np.float64),
    )[0]


def _build_map_api(map_location: str, map_api_cache: Dict[str, object]) -> Optional[object]:
    map_name = "us-nv-las-vegas-strip" if map_location == "las_vegas" else map_location
    if map_name in map_api_cache:
        return map_api_cache[map_name]
    if map_name not in MAP_LOCATIONS or NUPLAN_MAPS_ROOT is None:
        map_api_cache[map_name] = None
        return None
    try:
        map_api_cache[map_name] = get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_name)
    except Exception as exc:
        print(f"Warning: failed to load map API for {map_name}: {exc}")
        map_api_cache[map_name] = None
    return map_api_cache[map_name]


def _plot_rollout_step_bev(
    step_data: Dict[str, object],
    map_api: Optional[object],
    plan_sampling: TrajectorySampling,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    current_pose = np.asarray(step_data["current_pose"], dtype=np.float64)

    if map_api is not None:
        try:
            add_map_to_bev_ax(
                ax,
                map_api,
                StateSE2(float(current_pose[0]), float(current_pose[1]), float(current_pose[2])),
            )
        except Exception as exc:
            ax.text(0.02, 0.02, f"map unavailable: {exc}", transform=ax.transAxes, fontsize=7)

    ego_box = CarFootprint.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
        vehicle_parameters=get_pacifica_parameters(),
    ).oriented_box
    add_oriented_box_to_bev_ax(ax, ego_box, AGENT_CONFIG[TrackedObjectType.EGO], add_heading=True)

    pred_local = np.asarray(step_data["pred_local"], dtype=np.float32)
    target_local = np.asarray(step_data["target_local"], dtype=np.float32)
    add_trajectory_to_bev_ax(
        ax,
        Trajectory(pred_local, plan_sampling),
        _trajectory_config("agent", ELLIS_5[0]),
    )
    add_trajectory_to_bev_ax(
        ax,
        Trajectory(target_local, plan_sampling),
        _trajectory_config("human", NEW_TAB_10[4]),
    )

    executed_local = _global_pose_to_local(np.asarray(step_data["executed_pose"], dtype=np.float64), current_pose)
    executed_box = CarFootprint.build_from_rear_axle(
        rear_axle_pose=StateSE2(float(executed_local[0]), float(executed_local[1]), float(executed_local[2])),
        vehicle_parameters=get_pacifica_parameters(),
    ).oriented_box
    executed_config = dict(AGENT_CONFIG[TrackedObjectType.EGO])
    executed_config.update({"fill_color": NEW_TAB_10[2], "fill_color_alpha": 0.55, "line_color": NEW_TAB_10[2]})
    add_oriented_box_to_bev_ax(ax, executed_box, executed_config, add_heading=True)

    ax.set_title(
        f"{step_data['sample_token']} | step {step_data['rollout_step']} | "
        f"score={float(step_data['top_score']):.3f} | xy_err={float(step_data['executed_xy_error']):.3f}",
        fontsize=8,
    )
    configure_bev_ax(ax)
    configure_ax(ax)
    fig.tight_layout()
    return fig, ax


def _save_step_pngs_and_gif(
    step_data_list: List[Dict[str, object]],
    output_prefix: Path,
    map_api: Optional[object],
    plan_sampling: TrajectorySampling,
    save_gif: bool,
    gif_duration_ms: int,
) -> Tuple[List[Path], Optional[Path]]:
    step_paths: List[Path] = []
    images: List[Image.Image] = []
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    for step_data in step_data_list:
        fig, _ = _plot_rollout_step_bev(step_data, map_api, plan_sampling)
        step_path = output_prefix.parent / f"{output_prefix.name}_step{int(step_data['rollout_step']):02d}.png"
        fig.savefig(step_path, dpi=150)
        step_paths.append(step_path)

        if save_gif:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            images.append(Image.open(buf).copy())
            buf.close()
        plt.close(fig)

    gif_path: Optional[Path] = None
    if save_gif and images:
        gif_path = output_prefix.parent / f"{output_prefix.name}_ego.gif"
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=gif_duration_ms, loop=0)
    return step_paths, gif_path


def _save_global_summary(
    step_data_list: List[Dict[str, object]],
    global_path: Path,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    current = np.asarray([step["current_pose"] for step in step_data_list], dtype=np.float64)
    executed = np.asarray([step["executed_pose"] for step in step_data_list], dtype=np.float64)
    reference = np.asarray([step["reference_pose"] for step in step_data_list], dtype=np.float64)

    for idx, step in enumerate(step_data_list):
        current_pose = np.asarray(step["current_pose"], dtype=np.float64)
        pred_global = _local_trajectory_to_global(np.asarray(step["pred_local"], dtype=np.float64), current_pose)
        target_global = _local_trajectory_to_global(np.asarray(step["target_local"], dtype=np.float64), current_pose)
        pred_label = "pred rollout trajectories" if idx == 0 else None
        target_label = "target trajectories" if idx == 0 else None
        ax.plot(pred_global[:, 0], pred_global[:, 1], color=ELLIS_5[0], alpha=0.28, linewidth=1.2, label=pred_label)
        ax.plot(
            target_global[:, 0],
            target_global[:, 1],
            color=NEW_TAB_10[4],
            alpha=0.28,
            linewidth=1.2,
            linestyle="--",
            label=target_label,
        )
        ax.text(current_pose[0], current_pose[1], str(step["rollout_step"]), fontsize=7, color="#333333")

    ax.plot(current[:, 0], current[:, 1], color=NEW_TAB_10[0], marker="o", linewidth=2.0, label="current path")
    ax.plot(executed[:, 0], executed[:, 1], color=NEW_TAB_10[2], marker="o", linewidth=2.0, label="executed path")
    ax.plot(reference[:, 0], reference[:, 1], color=NEW_TAB_10[4], marker="o", linewidth=2.0, label="reference path")
    ax.scatter(current[0, 0], current[0, 1], color="#111111", marker="s", s=60, label="start", zorder=5)
    ax.scatter(executed[-1, 0], executed[-1, 1], color="#111111", marker="*", s=100, label="end", zorder=5)
    ax.set_title(f"{step_data_list[0]['sample_token']} global rollout summary", fontsize=10)
    ax.set_xlabel("global x [m]")
    ax.set_ylabel("global y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(global_path, dpi=150)
    plt.close(fig)
    return global_path


def _render_sample_visualizations(
    step_data_list: List[Dict[str, object]],
    vis_dir: Path,
    map_api_cache: Dict[str, object],
    plan_sampling: TrajectorySampling,
    save_gif: bool,
    gif_duration_ms: int,
) -> Dict[str, object]:
    sample_token = str(step_data_list[0]["sample_token"])
    prefix = vis_dir / _safe_filename(sample_token)
    map_api = _build_map_api(str(step_data_list[0].get("map_location", "")), map_api_cache)

    global_path = _save_global_summary(step_data_list, prefix.parent / f"{prefix.name}_global.png")
    step_paths, gif_path = _save_step_pngs_and_gif(
        step_data_list,
        prefix,
        map_api,
        plan_sampling,
        save_gif=save_gif,
        gif_duration_ms=gif_duration_ms,
    )

    xy_errors = np.asarray([step["executed_xy_error"] for step in step_data_list], dtype=np.float64)
    return {
        "sample_token": sample_token,
        "source_log_name": step_data_list[0]["source_log_name"],
        "global_path": str(global_path),
        "step_paths": [str(path) for path in step_paths],
        "gif_path": str(gif_path) if gif_path is not None else None,
        "num_steps": len(step_data_list),
        "final_executed_xy_error": float(xy_errors[-1]) if len(xy_errors) else float("nan"),
        "mean_executed_xy_error": float(xy_errors.mean()) if len(xy_errors) else float("nan"),
    }


def _write_html(vis_rows: List[Dict[str, object]], html_path: Path, observation_policy: str) -> None:
    lines = [
        "<html><head><meta charset=\"utf-8\"><title>RAP Rollout Recovery</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;}"
        "section{border-bottom:1px solid #ddd;margin-bottom:28px;padding-bottom:24px;}"
        "img{max-width:100%;height:auto;} .media{display:grid;grid-template-columns:1fr 1fr;gap:16px;}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:12px;}</style></head><body>",
        "<h1>RAP Rollout Recovery</h1>",
        f"<p>observation_policy={html.escape(observation_policy)}</p>",
    ]
    for row in vis_rows:
        global_path = row.get("global_path")
        gif_path = row.get("gif_path")
        step_paths = row.get("step_paths") or []
        lines.append("<section>")
        lines.append(
            f"<h2>{html.escape(str(row['sample_token']))}</h2>"
            f"<pre>source_log_name={html.escape(str(row['source_log_name']))}\n"
            f"num_steps={int(row['num_steps'])}\n"
            f"final_executed_xy_error={float(row['final_executed_xy_error']):.4f}\n"
            f"mean_executed_xy_error={float(row['mean_executed_xy_error']):.4f}</pre>"
        )
        lines.append("<div class=\"media\">")
        if global_path:
            rel_global = os.path.relpath(str(global_path), start=str(html_path.parent))
            lines.append(f"<div><h3>Global summary</h3><img src=\"{html.escape(rel_global)}\"></div>")
        ego_path = gif_path or (step_paths[0] if step_paths else None)
        if ego_path:
            rel_ego = os.path.relpath(str(ego_path), start=str(html_path.parent))
            title = "Ego-centered GIF" if gif_path else "Ego-centered first step"
            lines.append(f"<div><h3>{title}</h3><img src=\"{html.escape(rel_ego)}\"></div>")
        lines.append("</div>")
        if step_paths:
            lines.append("<details><summary>Step PNGs</summary>")
            for step_path in step_paths:
                rel_step = os.path.relpath(str(step_path), start=str(html_path.parent))
                lines.append(f"<p><a href=\"{html.escape(rel_step)}\">{html.escape(Path(step_path).name)}</a></p>")
            lines.append("</details>")
        lines.append("</section>")
    lines.append("</body></html>")
    html_path.write_text("\n".join(lines))


def _iter_inputs(log_dir: Path, pkl_glob: Optional[str], max_scenes: Optional[int]) -> Iterable[Path]:
    if pkl_glob:
        paths = [Path(path) for path in sorted(glob.glob(pkl_glob))]
        if max_scenes is not None:
            paths = paths[:max_scenes]
        return paths
    return list(_iter_scene_pickles(log_dir, max_scenes if max_scenes is not None else 10**12))


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root)
    log_dir = data_root / "navsim_logs" / args.split
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_dir) if args.vis_dir is not None else output_dir / "visualizations"
    vis_every = max(1, args.vis_every)

    if not log_dir.exists():
        raise FileNotFoundError(f"Missing NAVSIM log directory: {log_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if args.execute_interval_length <= 0 or args.sim_interval_length <= 0:
        raise ValueError("Simulation and execution intervals must be positive.")
    execute_steps = int(round(args.execute_interval_length / args.sim_interval_length))
    if not math.isclose(execute_steps * args.sim_interval_length, args.execute_interval_length, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("--execute-interval-length must be an integer multiple of --sim-interval-length.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    plan_sampling = TrajectorySampling(
        num_poses=args.num_poses,
        interval_length=args.interval_length,
    )
    horizon_s = args.num_poses * args.interval_length
    sim_num_poses = int(round(horizon_s / args.sim_interval_length))
    sim_sampling = TrajectorySampling(
        num_poses=sim_num_poses,
        interval_length=args.sim_interval_length,
    )
    simulator = PDMSimulator(proposal_sampling=sim_sampling)

    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder

    config = RAPConfig(trajectory_sampling=plan_sampling)
    model = _load_model(checkpoint_path, device, config)
    feature_builder = RAPFeatureBuilder(config)
    sensor_config = _build_sensor_config()
    synthetic_sensor_root = data_root / "sensor_blobs_perturbed"
    original_data_root = _default_original_data_root(data_root)

    max_scenes = None if args.max_scenes <= 0 else args.max_scenes
    pkl_paths = list(_iter_inputs(log_dir, args.pkl_glob, max_scenes))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"recovery_rollout_{timestamp}.csv"
    summary_path = output_dir / f"recovery_rollout_{timestamp}.json"
    html_path = vis_dir / "index.html" if args.save_vis and args.make_html else None

    fieldnames = [
        "sample_token",
        "source_log_name",
        "pkl_path",
        "rollout_step",
        "observation_policy",
        "top_score",
        "current_x",
        "current_y",
        "current_heading",
        "current_timestamp",
        "step_ade",
        "step_fde",
        "step_ahe",
        "step_fhe",
        "step_cv_ade",
        "step_cv_fde",
        "executed_x",
        "executed_y",
        "executed_heading",
        "reference_x",
        "reference_y",
        "reference_heading",
        "executed_xy_error",
        "executed_heading_error",
        "executed_lateral_like_error",
        "pred_trajectory",
        "target_trajectory",
    ]

    rows: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []
    vis_rows: List[Dict[str, object]] = []
    vis_warnings: List[str] = []
    map_api_cache: Dict[str, object] = {}

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample_idx, pkl_path in enumerate(tqdm(pkl_paths, desc="Rolling out perturbed scenes")):
            try:
                frames = pickle.load(open(pkl_path, "rb"))
                if len(frames) < args.num_history_frames + 1:
                    raise ValueError("Not enough frames for history plus rollout.")

                start_idx = args.num_history_frames - 1
                history_frames = copy.deepcopy(frames[: args.num_history_frames])
                current_frame = history_frames[-1]
                current_ego_state = _ego_state_from_frame(current_frame)
                reference_global = _reference_global_poses(frames)
                sample_token = current_frame["token"]
                source_log_name = current_frame["log_name"]
                map_location = current_frame.get("map_location", "")
                should_visualize = (
                    args.save_vis
                    and sample_idx % vis_every == 0
                    and (args.vis_max_samples <= 0 or len(vis_rows) < args.vis_max_samples)
                )
                sample_vis_steps: List[Dict[str, object]] = []

                for rollout_step in range(args.rollout_steps):
                    current_pose = np.asarray(current_ego_state.rear_axle.serialize(), dtype=np.float64)
                    current_timestamp = int(current_ego_state.time_point.time_us)
                    ref_start = start_idx + int(round(rollout_step * args.execute_interval_length / args.interval_length))
                    target_local = _reference_segment_local(
                        reference_global,
                        current_pose,
                        ref_start,
                        args.num_poses,
                    )
                    if target_local is None:
                        break

                    pred_local, top_score = _run_rap(
                        model,
                        history_frames,
                        synthetic_sensor_root,
                        sensor_config,
                        feature_builder,
                        device,
                        args.use_rendered,
                        args.num_poses,
                    )
                    cv_local = np.zeros_like(target_local, dtype=np.float32)
                    current_velocity = np.asarray(history_frames[-1]["ego_dynamic_state"][:2], dtype=np.float32)
                    steps = np.arange(1, args.num_poses + 1, dtype=np.float32)[:, None]
                    cv_local[:, :2] = steps * args.interval_length * current_velocity[None]

                    proposal_states = _local_prediction_to_state_array(
                        pred_local,
                        current_ego_state,
                        plan_sampling,
                        sim_sampling,
                    )
                    simulated_states = simulator.simulate_proposals(
                        proposal_states,
                        current_ego_state,
                    )
                    executed_state_array = simulated_states[0, execute_steps]
                    executed_pose = _state_array_pose(executed_state_array)

                    ref_execute_index = ref_start + int(round(args.execute_interval_length / args.interval_length))
                    if ref_execute_index >= len(reference_global):
                        break
                    reference_pose = reference_global[ref_execute_index]
                    pose_errors = _pose_error(executed_pose, reference_pose)
                    metrics = _metrics(pred_local, target_local, cv_local)

                    row = {
                        "sample_token": sample_token,
                        "source_log_name": source_log_name,
                        "pkl_path": str(pkl_path),
                        "rollout_step": rollout_step,
                        "observation_policy": args.observation_policy,
                        "top_score": top_score,
                        "current_x": float(current_pose[0]),
                        "current_y": float(current_pose[1]),
                        "current_heading": float(current_pose[2]),
                        "current_timestamp": current_timestamp,
                        "step_ade": metrics["ade"],
                        "step_fde": metrics["fde"],
                        "step_ahe": metrics["ahe"],
                        "step_fhe": metrics["fhe"],
                        "step_cv_ade": metrics["cv_ade"],
                        "step_cv_fde": metrics["cv_fde"],
                        "executed_x": float(executed_pose[0]),
                        "executed_y": float(executed_pose[1]),
                        "executed_heading": float(executed_pose[2]),
                        "reference_x": float(reference_pose[0]),
                        "reference_y": float(reference_pose[1]),
                        "reference_heading": float(reference_pose[2]),
                        **pose_errors,
                        "pred_trajectory": _to_jsonable_trajectory(pred_local),
                        "target_trajectory": _to_jsonable_trajectory(target_local),
                    }
                    rows.append(row)
                    writer.writerow(row)

                    if should_visualize:
                        sample_vis_steps.append(
                            {
                                "sample_token": sample_token,
                                "source_log_name": source_log_name,
                                "pkl_path": str(pkl_path),
                                "map_location": map_location,
                                "rollout_step": rollout_step,
                                "current_pose": current_pose.copy(),
                                "current_timestamp": current_timestamp,
                                "executed_pose": executed_pose.astype(np.float64).copy(),
                                "reference_pose": reference_pose.astype(np.float64).copy(),
                                "pred_local": pred_local.astype(np.float32).copy(),
                                "target_local": target_local.astype(np.float32).copy(),
                                "top_score": top_score,
                                "executed_xy_error": pose_errors["executed_xy_error"],
                                "executed_heading_error": pose_errors["executed_heading_error"],
                            }
                        )

                    next_time = current_ego_state.time_point + TimeDuration.from_s(args.execute_interval_length)
                    next_ego_state = state_array_to_ego_state(
                        executed_state_array,
                        next_time,
                        current_ego_state.car_footprint.vehicle_parameters,
                    )
                    next_frame = _update_frame_from_ego_state(
                        history_frames[-1],
                        next_ego_state,
                        token_suffix=f"step{rollout_step + 1:02d}",
                    )
                    history_frames = history_frames[1:] + [next_frame]
                    current_ego_state = next_ego_state

                if should_visualize and sample_vis_steps:
                    try:
                        vis_rows.append(
                            _render_sample_visualizations(
                                sample_vis_steps,
                                vis_dir,
                                map_api_cache,
                                plan_sampling,
                                save_gif=args.save_gif,
                                gif_duration_ms=args.gif_duration_ms,
                            )
                        )
                    except Exception as exc:
                        warning = f"{pkl_path}: visualization failed: {repr(exc)}"
                        print(f"Warning: {warning}")
                        vis_warnings.append(warning)
            except Exception as exc:
                failures.append((str(pkl_path), repr(exc)))

    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        _write_html(vis_rows, html_path, args.observation_policy)

    summary = _summarize(rows, failures)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "data_root": str(data_root),
            "split": args.split,
            "num_requested": len(pkl_paths),
            "num_failed": len(failures),
            "use_rendered": args.use_rendered,
            "observation_policy": args.observation_policy,
            "sim_interval_length": args.sim_interval_length,
            "execute_interval_length": args.execute_interval_length,
            "rollout_steps": args.rollout_steps,
            "csv_path": str(csv_path),
            "original_data_root": str(original_data_root),
            "vis_dir": str(vis_dir) if args.save_vis else None,
            "num_visualized": len(vis_rows),
            "html_path": str(html_path) if html_path is not None else None,
            "visualization_warnings": vis_warnings[:20],
            "failures": failures[:20],
            "samples": rows,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
