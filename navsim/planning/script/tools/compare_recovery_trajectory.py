"""Compare validate and rollout recovery trajectory outputs.

Example:
python navsim/planning/script/tools/compare_recovery_trajectory.py \
  --validate-csv outputs/recovery_validation_vis/recovery_validation_20260604_151653.csv \
  --rollout-csv outputs/recovery_rollout_vis/recovery_rollout_20260604_173739.csv \
  --output-dir outputs/recovery_comparison_vis \
  --max-samples 0 \
  --make-html \
  --save-aggregate
"""

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.visualization.config import BEV_PLOT_CONFIG, ELLIS_5, NEW_TAB_10, TRAJECTORY_CONFIG


VALIDATE_REQUIRED_FIELDS = [
    "token",
    "log_name",
    "pkl_path",
    "ade",
    "fde",
    "ahe",
    "fhe",
    "cv_ade",
    "cv_fde",
    "lateral_error_initial",
    "lateral_error_final",
    "heading_error_initial",
    "heading_error_final",
    "time_to_recover",
    "top_score",
    "pred_trajectory",
    "target_trajectory",
]

ROLLOUT_REQUIRED_FIELDS = [
    "sample_token",
    "source_log_name",
    "pkl_path",
    "rollout_step",
    "current_x",
    "current_y",
    "current_heading",
    "executed_x",
    "executed_y",
    "executed_heading",
    "reference_x",
    "reference_y",
    "reference_heading",
    "step_ade",
    "step_fde",
    "step_ahe",
    "step_fhe",
    "executed_xy_error",
    "executed_heading_error",
    "executed_lateral_like_error",
    "top_score",
    "pred_trajectory",
    "target_trajectory",
]

ROLLOUT_CURRENT_FIELDS = ["current_x", "current_y", "current_heading"]

RECOVERY_LATERAL_THRESHOLD_M = 0.5
RECOVERY_HEADING_THRESHOLD_RAD = math.radians(5.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process validate_recovery_trajectory.py and "
            "rollout_recovery_trajectory.py CSV outputs for quantitative and visual comparison."
        )
    )
    parser.add_argument("--validate-csv", required=True, help="CSV produced by validate_recovery_trajectory.py")
    parser.add_argument("--rollout-csv", required=True, help="CSV produced by rollout_recovery_trajectory.py")
    parser.add_argument("--output-dir", default="outputs/recovery_comparison")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit selected matched samples. Use <=0 for all.")
    parser.add_argument("--make-html", action="store_true", help="Write output_dir/index.html")
    parser.add_argument("--save-aggregate", action="store_true", help="Save aggregate comparison plots.")
    parser.add_argument("--sample-token", default=None, help="Only compare this sample token.")
    return parser.parse_args()


def _safe_filename(value: str, fallback: str = "sample") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or fallback


def _read_csv(path: Path, required_fields: Sequence[str], csv_name: str) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {csv_name} CSV: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if csv_name == "rollout":
            missing_current = [field for field in ROLLOUT_CURRENT_FIELDS if field not in fieldnames]
            if missing_current:
                raise ValueError(
                    "Rollout CSV is missing current_x/current_y/current_heading. "
                    "Please first re-run or update rollout_recovery_trajectory.py to write these fields. "
                    f"Missing: {missing_current}"
                )
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise ValueError(f"{csv_name} CSV is missing required fields: {missing}")
        return list(reader)


def _as_float(value: object, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _as_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid rollout_step value: {value!r}") from exc


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def local_to_global(
    local_traj: np.ndarray,
    current_x: float,
    current_y: float,
    current_heading: float,
) -> np.ndarray:
    """Transform an ego-local [x, y, heading] trajectory into global/world coordinates."""
    local_traj = np.asarray(local_traj, dtype=np.float64)
    c = math.cos(float(current_heading))
    s = math.sin(float(current_heading))
    global_traj = local_traj.copy()
    x_local = local_traj[:, 0]
    y_local = local_traj[:, 1]
    global_traj[:, 0] = c * x_local - s * y_local + float(current_x)
    global_traj[:, 1] = s * x_local + c * y_local + float(current_y)
    global_traj[:, 2] = _wrap_angle(local_traj[:, 2] + float(current_heading))
    return global_traj


def _parse_trajectory(value: object, field_name: str, token: str, required: bool = True) -> Optional[np.ndarray]:
    if value is None or value == "":
        if required:
            raise ValueError(f"{token}: missing {field_name}")
        return None
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{token}: {field_name} is not valid JSON") from exc
    traj = np.asarray(parsed, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 3:
        raise ValueError(f"{token}: {field_name} must have shape [num_poses, 3], got {traj.shape}")
    return traj


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BEV_PLOT_CONFIG.get("background_color", "white"),
            "axes.facecolor": BEV_PLOT_CONFIG.get("background_color", "white"),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "font.size": 9,
            "legend.framealpha": 0.9,
        }
    )


def _plot_local_traj(
    ax: plt.Axes,
    traj: np.ndarray,
    color: str,
    label: str,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    display_x = traj[:, 1]
    display_y = traj[:, 0]
    ax.plot(
        display_x,
        display_y,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        marker=TRAJECTORY_CONFIG["agent"]["marker"],
        markersize=3.5,
        markeredgecolor="black",
        alpha=alpha,
        label=label,
    )
    ax.scatter(display_x[0], display_y[0], color=color, marker="o", s=28, zorder=4)
    ax.scatter(display_x[-1], display_y[-1], color=color, marker="X", s=48, zorder=4)


def _set_equal_limits(ax: plt.Axes, xy_arrays: Sequence[np.ndarray], margin: float = 3.0) -> None:
    points = [array[:, :2] for array in xy_arrays if array is not None and len(array)]
    if not points:
        return
    xy = np.concatenate(points, axis=0)
    x_min, y_min = np.nanmin(xy[:, 0]), np.nanmin(xy[:, 1])
    x_max, y_max = np.nanmax(xy[:, 0]), np.nanmax(xy[:, 1])
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0) / 2.0 + margin
    ax.set_xlim(center_x - span, center_x + span)
    ax.set_ylim(center_y - span, center_y + span)


def _save_step0_open_loop(
    token: str,
    validate_row: Dict[str, str],
    rollout_steps: List[Dict[str, str]],
    output_path: Path,
) -> Path:
    step0 = rollout_steps[0]
    validate_pred = _parse_trajectory(validate_row["pred_trajectory"], "validate pred_trajectory", token)
    validate_target = _parse_trajectory(validate_row["target_trajectory"], "validate target_trajectory", token)
    rollout_pred = _parse_trajectory(step0["pred_trajectory"], "rollout step0 pred_trajectory", token)
    rollout_target = _parse_trajectory(step0["target_trajectory"], "rollout step0 target_trajectory", token)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    _plot_local_traj(ax, validate_pred, ELLIS_5[0], "validate pred", "-")
    _plot_local_traj(ax, validate_target, NEW_TAB_10[4], "validate target", "-")
    _plot_local_traj(ax, rollout_pred, NEW_TAB_10[2], "rollout step0 pred", "--")
    _plot_local_traj(ax, rollout_target, NEW_TAB_10[0], "rollout step0 target", "--")
    ax.scatter([0.0], [0.0], color="black", marker="s", s=45, label="ego origin", zorder=5)
    ax.set_title(f"{token} step0 open-loop comparison")
    ax.set_xlabel("ego-local lateral y [m]")
    ax.set_ylabel("ego-local forward x [m]")
    ax.set_aspect("equal", adjustable="box")
    local_display_arrays = [
        np.column_stack([traj[:, 1], traj[:, 0], traj[:, 2]])
        for traj in (validate_pred, validate_target, rollout_pred, rollout_target)
    ]
    _set_equal_limits(ax, local_display_arrays)
    ax.invert_xaxis()
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_global_rollout(token: str, rollout_steps: List[Dict[str, str]], output_path: Path) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    current = np.asarray(
        [
            [
                _as_float(step["current_x"]),
                _as_float(step["current_y"]),
                _as_float(step["current_heading"]),
            ]
            for step in rollout_steps
        ],
        dtype=np.float64,
    )
    executed = np.asarray(
        [
            [
                _as_float(step["executed_x"]),
                _as_float(step["executed_y"]),
                _as_float(step["executed_heading"]),
            ]
            for step in rollout_steps
        ],
        dtype=np.float64,
    )
    reference = np.asarray(
        [
            [
                _as_float(step["reference_x"]),
                _as_float(step["reference_y"]),
                _as_float(step["reference_heading"]),
            ]
            for step in rollout_steps
            if np.isfinite(_as_float(step["reference_x"])) and np.isfinite(_as_float(step["reference_y"]))
        ],
        dtype=np.float64,
    )

    all_global_trajs: List[np.ndarray] = [current, executed]
    if len(reference):
        all_global_trajs.append(reference)

    for idx, step in enumerate(rollout_steps):
        step_id = _as_int(step["rollout_step"])
        current_pose = current[idx]
        pred_local = _parse_trajectory(step["pred_trajectory"], "rollout pred_trajectory", token)
        pred_global = local_to_global(pred_local, current_pose[0], current_pose[1], current_pose[2])
        all_global_trajs.append(pred_global)
        ax.plot(
            pred_global[:, 0],
            pred_global[:, 1],
            color=ELLIS_5[0],
            alpha=0.28,
            linewidth=1.2,
            label="step pred trajectories" if idx == 0 else None,
        )
        target_local = _parse_trajectory(step.get("target_trajectory"), "rollout target_trajectory", token, required=False)
        if target_local is not None:
            target_global = local_to_global(target_local, current_pose[0], current_pose[1], current_pose[2])
            all_global_trajs.append(target_global)
            ax.plot(
                target_global[:, 0],
                target_global[:, 1],
                color=NEW_TAB_10[4],
                alpha=0.28,
                linewidth=1.2,
                linestyle="--",
                label="step target trajectories" if idx == 0 else None,
            )
        ax.text(current_pose[0], current_pose[1], str(step_id), fontsize=7, color="#333333")

    ax.plot(current[:, 0], current[:, 1], color=NEW_TAB_10[0], marker="o", linewidth=2.1, label="current path")
    ax.plot(executed[:, 0], executed[:, 1], color=NEW_TAB_10[2], marker="o", linewidth=2.1, label="executed path")
    if len(reference):
        ax.plot(reference[:, 0], reference[:, 1], color=NEW_TAB_10[4], marker="o", linewidth=2.1, label="reference path")
    ax.scatter(current[0, 0], current[0, 1], color="#111111", marker="s", s=60, label="start", zorder=5)
    ax.scatter(executed[-1, 0], executed[-1, 1], color="#111111", marker="*", s=110, label="end", zorder=5)
    ax.set_title(f"{token} global rollout")
    ax.set_xlabel("global x [m]")
    ax.set_ylabel("global y [m]")
    ax.set_aspect("equal", adjustable="box")
    _set_equal_limits(ax, all_global_trajs, margin=5.0)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_metric_curves(
    token: str,
    validate_row: Dict[str, str],
    rollout_steps: List[Dict[str, str]],
    output_path: Path,
) -> Path:
    steps = np.asarray([_as_int(step["rollout_step"]) for step in rollout_steps], dtype=np.int64)
    series = [
        ("executed_xy_error", "#0057B8", "-", "o"),
        ("executed_heading_error", "#D62728", "-", "s"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for key, color, linestyle, marker in series:
        values = np.asarray([_as_float(step.get(key)) for step in rollout_steps], dtype=np.float64)
        ax.plot(
            steps,
            values,
            marker=marker,
            markersize=5.5,
            linewidth=2.6,
            color=color,
            linestyle=linestyle,
            label=key,
        )

    step0 = rollout_steps[0]
    final = rollout_steps[-1]
    text = "\n".join(
        [
            f"validate ADE/FDE/AHE/FHE: {_as_float(validate_row.get('ade')):.3f} / {_as_float(validate_row.get('fde')):.3f} / "
            f"{_as_float(validate_row.get('ahe')):.3f} / {_as_float(validate_row.get('fhe')):.3f}",
            f"rollout step0 ADE/FDE: {_as_float(step0.get('step_ade')):.3f} / {_as_float(step0.get('step_fde')):.3f}",
            f"final executed xy/head: {_as_float(final.get('executed_xy_error')):.3f} / "
            f"{_as_float(final.get('executed_heading_error')):.3f}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"},
    )
    ax.set_title(f"{token} rollout metric curves")
    ax.set_xlabel("rollout_step")
    ax.set_ylabel("error")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_summary_panel(token: str, image_paths: Sequence[Path], output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    titles = ["step0 open-loop", "global rollout", "metric curves"]
    for ax, image_path, title in zip(axes, image_paths, titles):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(token, fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_aggregate_error_curves(samples: Sequence[Dict[str, object]], output_path: Path) -> Optional[Path]:
    sample_steps: List[np.ndarray] = []
    sample_values: List[np.ndarray] = []
    for sample in samples:
        rollout_steps = sample["rollout_steps"]
        steps = np.asarray([_as_int(step["rollout_step"]) for step in rollout_steps], dtype=np.int64)
        values = np.asarray([_as_float(step.get("executed_xy_error")) for step in rollout_steps], dtype=np.float64)
        finite = np.isfinite(values)
        if finite.any():
            sample_steps.append(steps[finite])
            sample_values.append(values[finite])
    if not sample_values:
        return None

    all_steps = np.unique(np.concatenate(sample_steps))
    matrix = np.full((len(sample_values), len(all_steps)), np.nan, dtype=np.float64)
    for row_idx, (steps, values) in enumerate(zip(sample_steps, sample_values)):
        for step, value in zip(steps, values):
            col_idx = int(np.where(all_steps == step)[0][0])
            matrix[row_idx, col_idx] = value

    mean = np.nanmean(matrix, axis=0)
    median = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for steps, values in zip(sample_steps, sample_values):
        ax.plot(steps, values, color="0.75", linewidth=0.8, alpha=0.65)
    ax.fill_between(all_steps, q25, q75, color=NEW_TAB_10[0], alpha=0.18, label="25/75 percentile")
    ax.plot(all_steps, mean, color=NEW_TAB_10[0], linewidth=2.4, label="mean")
    ax.plot(all_steps, median, color=ELLIS_5[0], linewidth=2.4, linestyle="--", label="median")
    ax.set_title("aggregate executed_xy_error vs rollout_step")
    ax.set_xlabel("rollout_step")
    ax.set_ylabel("executed_xy_error [m]")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_aggregate_step0_scatter(samples: Sequence[Dict[str, object]], output_path: Path) -> Optional[Path]:
    validate_ade = np.asarray([sample["summary"]["validate_ade"] for sample in samples], dtype=np.float64)
    validate_fde = np.asarray([sample["summary"]["validate_fde"] for sample in samples], dtype=np.float64)
    rollout_ade = np.asarray([sample["summary"]["rollout_step0_ade"] for sample in samples], dtype=np.float64)
    rollout_fde = np.asarray([sample["summary"]["rollout_step0_fde"] for sample in samples], dtype=np.float64)
    if not (np.isfinite(validate_ade).any() or np.isfinite(validate_fde).any()):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    scatter_specs = [
        (axes[0], validate_ade, rollout_ade, "validate ADE", "rollout step0 step_ADE"),
        (axes[1], validate_fde, rollout_fde, "validate FDE", "rollout step0 step_FDE"),
    ]
    for ax, x_values, y_values, x_label, y_label in scatter_specs:
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        ax.scatter(x_values[finite], y_values[finite], color=NEW_TAB_10[0], alpha=0.8, edgecolor="black", linewidth=0.4)
        if finite.any():
            low = float(min(np.nanmin(x_values[finite]), np.nanmin(y_values[finite])))
            high = float(max(np.nanmax(x_values[finite]), np.nanmax(y_values[finite])))
            pad = max((high - low) * 0.05, 0.1)
            ax.plot([low - pad, high + pad], [low - pad, high + pad], color="0.35", linestyle="--", linewidth=1.0)
            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("validate open-loop vs rollout step0")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_aggregate_final_hist(samples: Sequence[Dict[str, object]], output_path: Path) -> Optional[Path]:
    final_xy = np.asarray([sample["summary"]["rollout_final_executed_xy_error"] for sample in samples], dtype=np.float64)
    final_heading = np.asarray(
        [sample["summary"]["rollout_final_executed_heading_error"] for sample in samples],
        dtype=np.float64,
    )
    if not (np.isfinite(final_xy).any() or np.isfinite(final_heading).any()):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(final_xy[np.isfinite(final_xy)], bins=20, color=NEW_TAB_10[2], alpha=0.85, edgecolor="white")
    axes[0].set_title("final executed_xy_error")
    axes[0].set_xlabel("meters")
    axes[0].set_ylabel("count")
    axes[1].hist(
        final_heading[np.isfinite(final_heading)],
        bins=20,
        color=NEW_TAB_10[6],
        alpha=0.85,
        edgecolor="white",
    )
    axes[1].set_title("final executed_heading_error")
    axes[1].set_xlabel("radians")
    axes[1].set_ylabel("count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _find_execute_interval_length(rollout_csv: Path, rollout_rows: Sequence[Dict[str, str]]) -> Tuple[float, str, Optional[str]]:
    if rollout_rows and "execute_interval_length" in rollout_rows[0]:
        value = _as_float(rollout_rows[0].get("execute_interval_length"))
        if np.isfinite(value) and value > 0:
            return value, "rollout_csv", None

    json_candidates = [rollout_csv.with_suffix(".json")]
    json_candidates.extend(sorted(rollout_csv.parent.glob("recovery_rollout_*.json"), reverse=True))
    seen = set()
    for json_path in json_candidates:
        if json_path in seen or not json_path.exists():
            continue
        seen.add(json_path)
        try:
            data = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        value = _as_float(data.get("execute_interval_length"))
        csv_path_in_json = data.get("csv_path")
        same_csv = not csv_path_in_json or Path(str(csv_path_in_json)).name == rollout_csv.name
        if same_csv and np.isfinite(value) and value > 0:
            return value, str(json_path), None

    assumption = "execute_interval_length not found in rollout CSV or sibling JSON; defaulted to 0.5 seconds"
    return 0.5, "default", assumption


def _time_to_recover(rollout_steps: List[Dict[str, str]], execute_interval_length: float) -> float:
    for step in rollout_steps:
        lateral = _as_float(step.get("executed_lateral_like_error"))
        heading = _as_float(step.get("executed_heading_error"))
        if lateral < RECOVERY_LATERAL_THRESHOLD_M and heading < RECOVERY_HEADING_THRESHOLD_RAD:
            return float((_as_int(step["rollout_step"]) + 1) * execute_interval_length)
    return float("nan")


def _build_sample_summary(
    token: str,
    validate_row: Dict[str, str],
    rollout_steps: List[Dict[str, str]],
    execute_interval_length: float,
) -> Dict[str, object]:
    step0 = rollout_steps[0]
    final = rollout_steps[-1]
    xy_errors = np.asarray([_as_float(step.get("executed_xy_error")) for step in rollout_steps], dtype=np.float64)
    finite_xy_errors = xy_errors[np.isfinite(xy_errors)]
    first_xy = float(finite_xy_errors[0]) if len(finite_xy_errors) else float("nan")
    final_xy = _as_float(final.get("executed_xy_error"))
    reduction = first_xy - final_xy if np.isfinite(first_xy) and np.isfinite(final_xy) else float("nan")
    reduction_ratio = reduction / first_xy if np.isfinite(reduction) and first_xy > 1e-6 else float("nan")

    validate_ade = _as_float(validate_row.get("ade"))
    validate_fde = _as_float(validate_row.get("fde"))
    rollout_step0_ade = _as_float(step0.get("step_ade"))
    rollout_step0_fde = _as_float(step0.get("step_fde"))

    return {
        "token": token,
        "validate_ade": validate_ade,
        "validate_fde": validate_fde,
        "rollout_step0_ade": rollout_step0_ade,
        "rollout_step0_fde": rollout_step0_fde,
        "rollout_final_executed_xy_error": final_xy,
        "rollout_final_executed_heading_error": _as_float(final.get("executed_heading_error")),
        "rollout_mean_executed_xy_error": float(np.nanmean(finite_xy_errors)) if len(finite_xy_errors) else float("nan"),
        "rollout_num_steps": len(rollout_steps),
        "step0_ade_delta": rollout_step0_ade - validate_ade,
        "step0_fde_delta": rollout_step0_fde - validate_fde,
        "rollout_xy_error_reduction": reduction,
        "rollout_xy_error_reduction_ratio": reduction_ratio,
        "rollout_time_to_recover": _time_to_recover(rollout_steps, execute_interval_length),
    }


def _json_sanitize(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(val) for val in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_summary_csv(summaries: Sequence[Dict[str, object]], output_path: Path) -> Path:
    fieldnames = [
        "token",
        "validate_ade",
        "validate_fde",
        "rollout_step0_ade",
        "rollout_step0_fde",
        "rollout_final_executed_xy_error",
        "rollout_final_executed_heading_error",
        "rollout_mean_executed_xy_error",
        "rollout_num_steps",
        "step0_ade_delta",
        "step0_fde_delta",
        "rollout_xy_error_reduction",
        "rollout_xy_error_reduction_ratio",
        "rollout_time_to_recover",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})
    return output_path


def _relative(path: Optional[Path], start: Path) -> Optional[str]:
    if path is None:
        return None
    return os.path.relpath(str(path), start=str(start))


def _write_html(
    samples: Sequence[Dict[str, object]],
    aggregate_paths: Dict[str, Optional[Path]],
    output_path: Path,
    metadata: Dict[str, object],
) -> Path:
    lines = [
        "<html><head><meta charset=\"utf-8\"><title>Recovery Comparison</title>",
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;color:#222;}"
        "section{border-bottom:1px solid #ddd;margin-bottom:28px;padding-bottom:24px;}"
        "img{max-width:100%;height:auto;border:1px solid #ddd;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:12px;line-height:1.35;}"
        ".note{background:#fff8e1;border-left:4px solid #edc948;padding:12px;margin:16px 0;}"
        "</style></head><body>",
        "<h1>Recovery Comparison</h1>",
        "<div class=\"note\">"
        "validate measures open-loop prediction quality. rollout measures closed-loop executed recovery behavior. "
        "Compare validate ADE/FDE with rollout step0 step_ADE/FDE only; do not treat rollout executed_*_error as the same metric. "
        "Current rollout observation_policy may be reuse_latest_rendered, so this is state closed-loop rather than full sensor closed-loop."
        "</div>",
        "<pre>"
        f"validate_csv={html.escape(str(metadata['validate_csv']))}\n"
        f"rollout_csv={html.escape(str(metadata['rollout_csv']))}\n"
        f"num_matched_selected={int(metadata['num_selected'])}\n"
        f"execute_interval_length={float(metadata['execute_interval_length']):.3f} "
        f"({html.escape(str(metadata['execute_interval_length_source']))})"
        "</pre>",
    ]
    if metadata.get("assumption"):
        lines.append(f"<p><strong>Assumption:</strong> {html.escape(str(metadata['assumption']))}</p>")

    available_aggregate = [(name, path) for name, path in aggregate_paths.items() if path is not None]
    if available_aggregate:
        lines.append("<h2>Aggregate</h2><div class=\"grid\">")
        for name, path in available_aggregate:
            rel = _relative(path, output_path.parent)
            lines.append(f"<div><h3>{html.escape(name)}</h3><img src=\"{html.escape(rel or '')}\"></div>")
        lines.append("</div>")

    for sample in samples:
        summary = sample["summary"]
        paths = sample["paths"]
        lines.append("<section>")
        lines.append(
            f"<h2>{html.escape(str(summary['token']))}</h2>"
            "<pre>"
            f"validate ADE/FDE: {float(summary['validate_ade']):.4f} / {float(summary['validate_fde']):.4f}\n"
            f"rollout step0 ADE/FDE: {float(summary['rollout_step0_ade']):.4f} / {float(summary['rollout_step0_fde']):.4f}\n"
            f"rollout final executed xy/heading: {float(summary['rollout_final_executed_xy_error']):.4f} / "
            f"{float(summary['rollout_final_executed_heading_error']):.4f}\n"
            f"rollout num steps: {int(summary['rollout_num_steps'])}"
            "</pre>"
        )
        lines.append("<div class=\"grid\">")
        for title, key in [
            ("Step0 open-loop", "step0_open_loop"),
            ("Global rollout", "global_rollout"),
            ("Metric curves", "metric_curves"),
            ("Summary", "summary"),
        ]:
            rel = _relative(paths.get(key), output_path.parent)
            if rel:
                lines.append(f"<div><h3>{title}</h3><img src=\"{html.escape(rel)}\"></div>")
        lines.append("</div></section>")
    lines.append("</body></html>")
    output_path.write_text("\n".join(lines))
    return output_path


def _group_rollout_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        token = row["sample_token"]
        grouped.setdefault(token, []).append(row)
    for token, token_rows in grouped.items():
        token_rows.sort(key=lambda row: _as_int(row["rollout_step"]))
    return grouped


def _select_tokens(
    validate_rows: Sequence[Dict[str, str]],
    rollout_by_token: Dict[str, List[Dict[str, str]]],
    sample_token: Optional[str],
    max_samples: Optional[int],
) -> List[str]:
    validate_order = [row["token"] for row in validate_rows]
    matched = [token for token in validate_order if token in rollout_by_token]
    if sample_token is not None:
        matched = [token for token in matched if token == sample_token]
    if max_samples is not None and max_samples > 0:
        matched = matched[:max_samples]
    return matched


def main() -> None:
    args = _parse_args()
    _plot_style()

    validate_csv = Path(args.validate_csv)
    rollout_csv = Path(args.rollout_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_rows = _read_csv(validate_csv, VALIDATE_REQUIRED_FIELDS, "validate")
    rollout_rows = _read_csv(rollout_csv, ROLLOUT_REQUIRED_FIELDS, "rollout")
    validate_by_token = {row["token"]: row for row in validate_rows}
    rollout_by_token = _group_rollout_rows(rollout_rows)
    execute_interval_length, execute_interval_source, assumption = _find_execute_interval_length(rollout_csv, rollout_rows)

    selected_tokens = _select_tokens(
        validate_rows,
        rollout_by_token,
        sample_token=args.sample_token,
        max_samples=args.max_samples,
    )
    if not selected_tokens:
        raise ValueError(
            "No matched samples found between validate.token and rollout.sample_token"
            + (f" for --sample-token {args.sample_token!r}" if args.sample_token else "")
        )

    samples: List[Dict[str, object]] = []
    sample_dir = output_dir / "samples"
    for token in selected_tokens:
        safe_token = _safe_filename(token)
        validate_row = validate_by_token[token]
        rollout_steps = rollout_by_token[token]
        summary = _build_sample_summary(token, validate_row, rollout_steps, execute_interval_length)
        step0_path = _save_step0_open_loop(
            token,
            validate_row,
            rollout_steps,
            sample_dir / f"{safe_token}_step0_open_loop.png",
        )
        global_path = _save_global_rollout(
            token,
            rollout_steps,
            sample_dir / f"{safe_token}_global_rollout.png",
        )
        metric_path = _save_metric_curves(
            token,
            validate_row,
            rollout_steps,
            sample_dir / f"{safe_token}_metric_curves.png",
        )
        summary_path = _save_summary_panel(
            token,
            [step0_path, global_path, metric_path],
            sample_dir / f"{safe_token}_summary.png",
        )
        samples.append(
            {
                "summary": summary,
                "rollout_steps": rollout_steps,
                "paths": {
                    "step0_open_loop": step0_path,
                    "global_rollout": global_path,
                    "metric_curves": metric_path,
                    "summary": summary_path,
                },
            }
        )

    summaries = [sample["summary"] for sample in samples]
    summary_csv_path = _write_summary_csv(summaries, output_dir / "comparison_summary.csv")

    aggregate_paths: Dict[str, Optional[Path]] = {
        "aggregate_error_curves": None,
        "aggregate_step0_scatter": None,
        "aggregate_final_hist": None,
    }
    if args.save_aggregate:
        aggregate_paths["aggregate_error_curves"] = _save_aggregate_error_curves(
            samples,
            output_dir / "aggregate_error_curves.png",
        )
        aggregate_paths["aggregate_step0_scatter"] = _save_aggregate_step0_scatter(
            samples,
            output_dir / "aggregate_step0_scatter.png",
        )
        aggregate_paths["aggregate_final_hist"] = _save_aggregate_final_hist(
            samples,
            output_dir / "aggregate_final_hist.png",
        )

    metadata = {
        "validate_csv": str(validate_csv),
        "rollout_csv": str(rollout_csv),
        "output_dir": str(output_dir),
        "num_validate_rows": len(validate_rows),
        "num_rollout_rows": len(rollout_rows),
        "num_matched": len([row for row in validate_rows if row["token"] in rollout_by_token]),
        "num_selected": len(samples),
        "sample_token": args.sample_token,
        "max_samples": args.max_samples,
        "execute_interval_length": execute_interval_length,
        "execute_interval_length_source": execute_interval_source,
        "assumption": assumption,
        "recovery_threshold": {
            "executed_lateral_like_error_m": RECOVERY_LATERAL_THRESHOLD_M,
            "executed_heading_error_rad": RECOVERY_HEADING_THRESHOLD_RAD,
        },
        "metric_interpretation": {
            "validate": "open-loop prediction quality",
            "rollout": "closed-loop executed recovery behavior",
            "step0_comparison": "validate ADE/FDE can be compared with rollout step0 step_ADE/FDE",
            "executed_error_warning": "rollout executed_*_error is not the same metric as validate ADE/FDE",
            "observation_policy_note": (
                "rollout observation_policy may be reuse_latest_rendered, so it is state closed-loop "
                "rather than full sensor closed-loop"
            ),
        },
        "summary_csv": str(summary_csv_path),
        "aggregate_paths": {key: str(path) if path is not None else None for key, path in aggregate_paths.items()},
        "samples": [
            {
                "summary": sample["summary"],
                "paths": {key: str(path) for key, path in sample["paths"].items()},
            }
            for sample in samples
        ],
    }
    summary_json_path = output_dir / "comparison_summary.json"
    summary_json_path.write_text(json.dumps(_json_sanitize(metadata), indent=2, sort_keys=True))

    html_path = None
    if args.make_html:
        html_path = _write_html(samples, aggregate_paths, output_dir / "index.html", metadata)

    print(f"Matched samples: {metadata['num_matched']}")
    print(f"Selected samples: {metadata['num_selected']}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    if html_path is not None:
        print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
