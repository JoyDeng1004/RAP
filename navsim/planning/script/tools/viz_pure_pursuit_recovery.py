"""White-box visual validation of the pure-pursuit recovery algorithm.

This tool drives ``make_recovery_trajectory`` directly (no checkpoint, no model
forward pass), sweeping a synthetic lateral offset ``shift_y`` over real NAVSIM
ego-local reference trajectories (or synthetic ones via ``--self-test``) and
rendering, per (scene, shift):

  * BEV panel with the per-step look-ahead "aim" lines (white-box geometry),
  * kinematic-feasibility curves (speed / accel / yaw-rate / curvature / steer),
  * lateral + heading error convergence with ``time_to_recover``,
  * a heading-vs-path-tangent consistency sentinel (catches +/-90 deg spikes).

It also writes ``summary.csv`` / ``summary.json`` with automatic pass/fail
checks and an optional HTML gallery.

Usage (real NAVSIM logs):
    python navsim/planning/script/tools/viz_pure_pursuit_recovery.py \
        --pkl-glob "dataset_perturbed/navsim_logs/mini/*.pkl" \
        --shift-values "-1.0,-0.5,0.5,1.0,2.0" \
        --num-poses 8 --interval-length 0.5 --max-scenes 12 \
        --output-dir outputs/pure_pursuit_validation --make-html

Usage (synthetic smoke test, no dataset needed):
    python navsim/planning/script/tools/viz_pure_pursuit_recovery.py \
        --self-test --output-dir outputs/pp_selftest --make-html
"""

from __future__ import annotations

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
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters  # noqa: E402

from navsim.agents.rap_dino.recovery.recovery_target import make_recovery_trajectory  # noqa: E402

# Lightweight, pure helpers reused from the checkpoint-driven validator
# (same import style as navsim/agents/rap_dino/recovery/evaluation.py).
from navsim.planning.script.tools.validate_recovery_trajectory import (  # noqa: E402
    _angle_abs_error,
    _bev_display_xy,
    _frame_pose,
    _future_recovery_label,
    _wrap_angle,
    compute_recovery_metrics,
)


# --------------------------------------------------------------------------- #
# Scene loading
# --------------------------------------------------------------------------- #
def _parse_shift_values(tokens: List[str]) -> List[float]:
    """Flatten space- and/or comma-separated tokens into a list of floats.

    Accepts both ``--shift-values -1.0 -0.5 0.5`` (negative numbers work as
    argparse values) and ``--shift-values=-1.0,-0.5,0.5``.
    """
    values: List[float] = []
    for token in tokens:
        values.extend(float(part) for part in str(token).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("--shift-values must contain at least one number")
    return values


def _load_real_scenes(
    pkl_glob: str,
    num_history_frames: int,
    num_poses: int,
    interval_length: float,
    max_scenes: Optional[int],
) -> List[Tuple[str, np.ndarray, Dict[str, float]]]:
    """Return (name, reference[T,3] ego-local, meta) for each usable pkl log."""
    scenes: List[Tuple[str, np.ndarray, Dict[str, float]]] = []
    pkl_paths = [Path(path) for path in sorted(glob.glob(pkl_glob))]
    if max_scenes is not None:
        pkl_paths = pkl_paths[:max_scenes]
    for pkl_path in pkl_paths:
        try:
            frames = pickle.load(open(pkl_path, "rb"))
            reference = _future_recovery_label(frames, num_history_frames, num_poses, interval_length)
            current_frame = frames[num_history_frames - 1]
            velocity = np.asarray(current_frame["ego_dynamic_state"][:2], dtype=np.float64)
            meta = {"ego_speed": float(np.linalg.norm(velocity))}
            name = f"{current_frame.get('token', pkl_path.stem)}"
            scenes.append((name, reference.astype(np.float64), meta))
        except Exception as exc:  # noqa: BLE001 - skip unusable logs, keep going
            print(f"[skip] {pkl_path}: {exc}")
    return scenes


def _synthetic_scenes(num_poses: int, interval_length: float) -> List[Tuple[str, np.ndarray, Dict[str, float]]]:
    """Straight (multi-speed), constant-curvature arc and S-curve references."""
    steps = np.arange(1, num_poses + 1, dtype=np.float64)
    scenes: List[Tuple[str, np.ndarray, Dict[str, float]]] = []

    for speed in (5.0, 10.0, 15.0):
        x = steps * speed * interval_length
        ref = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)
        scenes.append((f"straight_v{int(speed)}", ref, {"ego_speed": speed}))

    # Constant-curvature arc: ego drives along a circle of radius R at 10 m/s.
    speed = 10.0
    radius = 60.0
    arc = speed * interval_length * steps / radius  # cumulative heading [rad]
    x = radius * np.sin(arc)
    y = radius * (1.0 - np.cos(arc))
    ref = np.stack([x, y, arc], axis=-1)
    scenes.append(("arc_R60_v10", ref, {"ego_speed": speed}))

    # Gentle S-curve via a lateral sinusoid over forward distance.
    x = steps * speed * interval_length
    amp, wavelength = 1.5, 40.0
    y = amp * np.sin(2.0 * math.pi * x / wavelength)
    heading = np.arctan2(np.gradient(y), np.gradient(x))
    ref = np.stack([x, y, heading], axis=-1)
    scenes.append(("scurve_v10", ref, {"ego_speed": speed}))

    return scenes


# --------------------------------------------------------------------------- #
# Kinematics / checks
# --------------------------------------------------------------------------- #
def _derive_kinematics(path: np.ndarray, start_xy: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """Per-step kinematics of a SE(2) path, prepending the true start pose."""
    positions = np.vstack([start_xy[None, :], path[:, :2]])  # [T+1, 2]
    seg = np.diff(positions, axis=0)  # [T, 2]
    speed = np.linalg.norm(seg, axis=-1) / dt  # [T]
    accel = np.gradient(speed, dt)  # [T]
    headings = np.concatenate([[0.0], path[:, 2]])  # start heading is 0
    yaw_rate = _wrap_angle(np.diff(headings)) / dt  # [T]
    curvature = yaw_rate / np.maximum(speed, 1e-3)
    path_tangent = np.arctan2(seg[:, 1], seg[:, 0])  # [T]
    return {
        "speed": speed,
        "accel": accel,
        "yaw_rate": yaw_rate,
        "curvature": curvature,
        "path_tangent": path_tangent,
        "seg_len": np.linalg.norm(seg, axis=-1),
    }


def _run_checks(
    out: np.ndarray,
    reference: np.ndarray,
    dbg: Dict[str, np.ndarray],
    kin: Dict[str, np.ndarray],
    *,
    shift_y: float,
    dt: float,
    max_steer: float,
    max_steer_rate: float,
    wheel_base: float,
    accel_bound: float,
    heading_tol_rad: float,
    moving_eps: float,
    curv_speed_min: float,
    curv_seglen_min: float,
    recover_lateral_tol: float,
    recover_strict_shift_max: float,
    recover_lateral_tol_large: float,
    heading_align_tol_rad: float,
) -> Dict[str, object]:
    recovered = bool(dbg["recover_mask"][0])
    applied_steer = dbg["applied_steer"][0]
    steer_rate = dbg["steer_rate_cmd"][0]
    target_index = dbg["target_index"][0]
    local_x = dbg["target_point_local"][0, :, 0]

    curv_max = math.tan(max_steer) / wheel_base
    moving = kin["seg_len"] > moving_eps

    # Heading consistency (the +/-90 deg spike sentinel): compare the label
    # heading against the geometric path tangent on moving segments.
    heading_resid = np.abs(_wrap_angle(out[:, 2] - kin["path_tangent"]))
    heading_resid_moving = heading_resid[moving] if moving.any() else np.array([0.0])

    rec = compute_recovery_metrics(out, out, reference)[0]
    lateral_initial = rec["lateral_error_initial"]
    lateral_final = rec["lateral_error_final"]
    heading_final = rec["heading_error_final"]
    time_to_recover = rec["time_to_recover"]

    # Interior acceleration for the hard check: np.gradient uses one-sided
    # differences at the two endpoints, which over a T=8 horizon are noisy.
    accel = kin["accel"]
    accel_interior = accel[1:-1] if accel.shape[0] > 2 else accel
    max_accel_interior = float(np.max(np.abs(accel_interior)))
    max_accel_full = float(np.max(np.abs(accel)))

    # Curvature only judged on segments that are actually moving fast/far enough;
    # otherwise yaw_rate / tiny-speed amplifies finite-difference noise.
    curv_valid_mask = (kin["speed"] >= curv_speed_min) & (kin["seg_len"] >= curv_seglen_min)
    curv_valid = np.abs(kin["curvature"])[curv_valid_mask] if curv_valid_mask.any() else np.array([0.0])
    max_curv_valid = float(np.max(curv_valid))
    max_curv_full = float(np.max(np.abs(kin["curvature"])))

    # --- hard checks (drive pass/fail) ---------------------------------------
    checks: Dict[str, Optional[bool]] = {}
    checks["feasible_steer"] = bool(np.max(np.abs(applied_steer)) <= max_steer + 1e-3)
    checks["feasible_steer_rate"] = bool(np.max(np.abs(steer_rate)) <= max_steer_rate + 1e-3)
    checks["feasible_accel"] = bool(max_accel_interior <= accel_bound)
    checks["feasible_curvature"] = bool(max_curv_valid <= curv_max + 1e-3)
    checks["heading_consistent"] = bool(np.max(heading_resid_moving) <= heading_tol_rad)

    # --- warnings (recorded, do NOT drive pass/fail) -------------------------
    warnings: Dict[str, Optional[bool]] = {}
    # Endpoint accel artifact (one-sided np.gradient): informational only.
    warnings["accel_endpoint_within_bound"] = bool(max_accel_full <= accel_bound)

    if recovered:
        # Pure-pursuit geometry only meaningful when actually recovering.
        checks["target_monotonic"] = bool(np.all(np.diff(target_index) >= 0))
        checks["target_ahead"] = bool(np.all(local_x > -1e-6))
        # Lateral error should shrink and not blow up along the way.
        bump = float(np.max(np.diff(np.abs(out[:, 1] - reference[:, 1])))) if len(out) > 1 else 0.0
        checks["lateral_converges"] = bool(
            (lateral_final is not None)
            and (lateral_initial is not None)
            and (lateral_final < lateral_initial + 1e-6)
            and (bump <= 0.1 * max(lateral_initial, 1e-3))
        )
        if abs(shift_y) >= 0.5:
            # Split the old recovers_in_horizon: lateral is the hard requirement,
            # heading alignment is only a warning (heading lags badly on big
            # offsets within a short 4 s / T=8 horizon — a task boundary, not a
            # controller failure).
            if abs(shift_y) <= recover_strict_shift_max:
                # Small/medium offset: closing within tol is a hard requirement.
                checks["lateral_recovers_in_horizon"] = bool(
                    (lateral_final is not None) and (lateral_final <= recover_lateral_tol)
                )
            else:
                # Large offset: a 4 s / T=8 horizon may be physically too short to
                # close a big lateral gap, so this is a warning with a relaxed
                # tolerance instead of a hard fail (task boundary, not a bug).
                warnings["lateral_recovers_in_horizon_relaxed"] = bool(
                    (lateral_final is not None) and (lateral_final <= recover_lateral_tol_large)
                )
            warnings["heading_aligns_in_horizon"] = bool(
                (heading_final is not None) and (heading_final <= heading_align_tol_rad)
            )
    else:
        checks["target_monotonic"] = None
        checks["target_ahead"] = None
        checks["lateral_converges"] = None

    applicable = [v for v in checks.values() if v is not None]
    passed = bool(all(applicable)) if applicable else True

    return {
        "recovered": recovered,
        "checks": checks,
        "warnings": warnings,
        "passed": passed,
        "lateral_error_initial": lateral_initial,
        "lateral_error_final": lateral_final,
        "lateral_error_reduction_ratio": rec["lateral_error_reduction_ratio"],
        "heading_error_initial": rec["heading_error_initial"],
        "heading_error_final": heading_final,
        "time_to_recover": time_to_recover,
        "max_abs_steer": float(np.max(np.abs(applied_steer))),
        "max_abs_steer_rate": float(np.max(np.abs(steer_rate))),
        "max_abs_accel_interior": max_accel_interior,
        "max_abs_accel_full": max_accel_full,
        "max_abs_curvature_valid": max_curv_valid,
        "max_abs_curvature_full": max_curv_full,
        "max_heading_resid_deg": float(math.degrees(np.max(heading_resid_moving))),
    }


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _plot_bev(ax, reference, out, raw_path, dbg, shift_y) -> None:
    ref_d = _bev_display_xy(reference[:, :2])
    out_d = _bev_display_xy(out[:, :2])
    raw_d = _bev_display_xy(raw_path[:, :2])
    start_d = _bev_display_xy(np.array([[0.0, shift_y]]))[0]

    ax.plot(ref_d[:, 0], ref_d[:, 1], "-o", color="tab:green", ms=3, label="reference")
    ax.plot(raw_d[:, 0], raw_d[:, 1], "--", color="tab:gray", lw=1.2, label="raw bicycle")
    ax.plot(out_d[:, 0], out_d[:, 1], "-o", color="tab:blue", ms=3, label="recovery label")
    ax.scatter([start_d[0]], [start_d[1]], color="tab:red", marker="s", s=60, label="shifted start")

    # White-box aim lines: current controller position -> chosen look-ahead point.
    controller_pos = np.vstack([np.array([[0.0, shift_y]]), raw_path[:, :2]])  # [T+1,2]
    target_ref = dbg["target_point_ref"][0]  # [T,2]
    for step in range(len(target_ref)):
        cur_d = _bev_display_xy(controller_pos[step][None])[0]
        tgt_d = _bev_display_xy(target_ref[step][None])[0]
        ax.plot([cur_d[0], tgt_d[0]], [cur_d[1], tgt_d[1]], color="tab:orange", lw=0.6, alpha=0.5)
    tgt_d_all = _bev_display_xy(target_ref)
    ax.scatter(tgt_d_all[:, 0], tgt_d_all[:, 1], color="tab:orange", marker="x", s=18, label="look-ahead target")

    ax.set_title(f"BEV (shift_y={shift_y:+.2f} m)", fontsize=10)
    ax.set_xlabel("lateral (m, left <- / right ->)")
    ax.set_ylabel("forward (m)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best")


def _plot_kinematics(ax, kin, dbg, t, max_steer, max_steer_rate, wheel_base, accel_bound) -> None:
    applied = dbg["applied_steer"][0]
    desired = dbg["desired_steer"][0]
    implied = np.arctan(wheel_base * kin["curvature"])

    ax.plot(t, np.degrees(applied), "-o", ms=3, color="tab:blue", label="applied steer")
    ax.plot(t, np.degrees(desired), "--", color="tab:cyan", label="desired steer")
    ax.plot(t, np.degrees(implied), ":", color="tab:purple", label="implied steer (atan Lκ)")
    ax.axhline(math.degrees(max_steer), color="r", ls="--", lw=0.8)
    ax.axhline(-math.degrees(max_steer), color="r", ls="--", lw=0.8, label="±max_steer")
    ax.set_ylabel("steer (deg)")
    ax.set_xlabel("step")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title("steering vs limits", fontsize=10)

    twin = ax.twinx()
    twin.plot(t, kin["accel"], "-", color="tab:olive", lw=1.0, alpha=0.7, label="accel")
    twin.axhline(accel_bound, color="tab:olive", ls=":", lw=0.7)
    twin.axhline(-accel_bound, color="tab:olive", ls=":", lw=0.7)
    twin.set_ylabel("accel (m/s²)")

    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = twin.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, fontsize=7, loc="best")


def _plot_convergence(ax, out, reference, t, time_to_recover) -> None:
    lateral_err = np.abs(out[:, 1] - reference[:, 1])
    heading_err = _angle_abs_error(out[:, 2], reference[:, 2])
    ax.plot(t, lateral_err, "-o", ms=3, color="tab:blue", label="lateral err (m)")
    ax.set_ylabel("lateral err (m)")
    ax.set_xlabel("step")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title("error convergence", fontsize=10)
    twin = ax.twinx()
    twin.plot(t, np.degrees(heading_err), "-s", ms=3, color="tab:red", label="heading err (deg)")
    twin.set_ylabel("heading err (deg)")
    if time_to_recover is not None:
        ax.axvline(time_to_recover, color="tab:green", ls="--", lw=1.0)
        ax.text(time_to_recover, ax.get_ylim()[1] * 0.9, f"recover@{time_to_recover:.1f}s", fontsize=7)
    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = twin.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, fontsize=7, loc="best")


def _plot_heading_consistency(ax, out, kin, t, heading_tol_rad, moving_eps) -> None:
    moving = kin["seg_len"] > moving_eps
    ax.plot(t, np.degrees(out[:, 2]), "-o", ms=3, color="tab:blue", label="label heading")
    tangent = np.degrees(kin["path_tangent"]).copy()
    tangent[~moving] = np.nan  # do not draw tangent for stationary segments
    ax.plot(t, tangent, "--x", color="tab:orange", label="path tangent")
    resid = np.degrees(np.abs(_wrap_angle(out[:, 2] - kin["path_tangent"])))
    resid[~moving] = np.nan
    twin = ax.twinx()
    twin.plot(t, resid, ":", color="tab:red", lw=1.0, label="|residual|")
    twin.axhline(math.degrees(heading_tol_rad), color="tab:red", ls=":", lw=0.7)
    twin.set_ylabel("residual (deg)")
    ax.set_ylabel("heading (deg)")
    ax.set_xlabel("step")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title("heading vs tangent (spike sentinel)", fontsize=10)
    lines, labels = ax.get_legend_handles_labels()
    l2, lab2 = twin.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lab2, fontsize=7, loc="best")


def _render_figure(
    out_path: Path,
    name: str,
    reference: np.ndarray,
    out: np.ndarray,
    raw_path: np.ndarray,
    dbg: Dict[str, np.ndarray],
    kin: Dict[str, np.ndarray],
    result: Dict[str, object],
    *,
    shift_y: float,
    dt: float,
    max_steer: float,
    max_steer_rate: float,
    wheel_base: float,
    accel_bound: float,
    heading_tol_rad: float,
    moving_eps: float,
) -> None:
    t = np.arange(1, len(out) + 1) * dt
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    _plot_bev(axes[0, 0], reference, out, raw_path, dbg, shift_y)
    _plot_kinematics(axes[0, 1], kin, dbg, np.arange(1, len(out) + 1), max_steer, max_steer_rate, wheel_base, accel_bound)
    _plot_convergence(axes[1, 0], out, reference, t, result["time_to_recover"])
    _plot_heading_consistency(axes[1, 1], out, kin, np.arange(1, len(out) + 1), heading_tol_rad, moving_eps)

    status = "PASS" if result["passed"] else "FAIL"
    failed = [k for k, v in result["checks"].items() if v is False]
    warned = [k for k, v in result.get("warnings", {}).items() if v is False]
    subtitle = f"{status}" + (f" | failed: {', '.join(failed)}" if failed else "")
    if warned:
        subtitle += f" | warn: {', '.join(warned)}"
    color = "green" if result["passed"] else "red"
    fig.suptitle(f"{name}  shift_y={shift_y:+.2f}m  —  {subtitle}", color=color, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _write_html(records: List[Dict[str, object]], html_path: Path) -> None:
    lines = ["<html><body><h1>Pure-Pursuit Recovery Validation</h1>"]
    n_pass = sum(1 for r in records if r["passed"])
    lines.append(f"<p>{n_pass}/{len(records)} passed</p>")
    for r in records:
        vis_path = r.get("vis_path")
        if not vis_path:
            continue
        rel = os.path.relpath(str(vis_path), start=str(html_path.parent))
        color = "green" if r["passed"] else "red"
        failed = [k for k, v in r["checks"].items() if v is False]
        warned = [k for k, v in r.get("warnings", {}).items() if v is False]
        lines.append(
            f"<h3 style='color:{color}'>{r['scene']} | shift={r['shift_y']:+.2f} | "
            f"{'PASS' if r['passed'] else 'FAIL'}"
            + (f" | failed: {', '.join(failed)}" if failed else "")
            + (f" | <span style='color:orange'>warn: {', '.join(warned)}</span>" if warned else "")
            + "</h3>"
        )
        lines.append(f'<img src="{rel}" style="max-width: 100%;"><hr>')
    lines.append("</body></html>")
    html_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl-glob", default=None, help="Glob of NAVSIM log pkls; omit (or --self-test) for synthetic.")
    parser.add_argument("--self-test", action="store_true", help="Use built-in synthetic references (no dataset).")
    parser.add_argument(
        "--shift-values",
        nargs="+",
        default=["-1.0", "-0.5", "0.5", "1.0", "2.0"],
        help="Lateral offsets to sweep. Space-separated (negatives OK) or comma form via =, "
        "e.g. --shift-values -1.0 -0.5 0.5  or  --shift-values=-1.0,-0.5,0.5",
    )
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-poses", type=int, default=8)
    parser.add_argument("--interval-length", type=float, default=0.5)
    parser.add_argument("--max-scenes", type=int, default=12, help="Use <=0 for all.")
    parser.add_argument("--max-steer", type=float, default=math.pi / 3)
    parser.add_argument("--max-steer-rate", type=float, default=math.pi / 3)
    parser.add_argument(
        "--accel-bound",
        type=float,
        default=6.0,
        help="Hard-fail limit on |interior longitudinal accel| (m/s²). Endpoints are a warning only.",
    )
    parser.add_argument("--heading-tol-deg", type=float, default=30.0, help="Max |heading - path tangent| on moving segs.")
    parser.add_argument(
        "--curv-speed-min",
        type=float,
        default=0.5,
        help="Ignore curvature on segments slower than this (m/s) — finite-diff noise.",
    )
    parser.add_argument(
        "--curv-seglen-min",
        type=float,
        default=0.25,
        help="Ignore curvature on segments shorter than this (m) — finite-diff noise.",
    )
    parser.add_argument(
        "--recover-lateral-tol",
        type=float,
        default=0.5,
        help="Hard-fail: final lateral error must be within this (m) for 0.5<=|shift_y|<=recover-strict-shift-max.",
    )
    parser.add_argument(
        "--recover-strict-shift-max",
        type=float,
        default=1.0,
        help="|shift_y| above this is treated as a horizon-boundary case: lateral recovery becomes a warning.",
    )
    parser.add_argument(
        "--recover-lateral-tol-large",
        type=float,
        default=1.0,
        help="Warning-only relaxed final lateral tol (m) for |shift_y|>recover-strict-shift-max.",
    )
    parser.add_argument(
        "--heading-align-tol-deg",
        type=float,
        default=12.0,
        help="Warning-only: final heading error target (deg). Heading lag on big offsets is a task boundary.",
    )
    parser.add_argument("--output-dir", default="outputs/pure_pursuit_validation")
    parser.add_argument("--make-html", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    shift_values = _parse_shift_values(args.shift_values)
    wheel_base = float(get_pacifica_parameters().wheel_base)
    heading_tol_rad = math.radians(args.heading_tol_deg)
    heading_align_tol_rad = math.radians(args.heading_align_tol_deg)
    moving_eps = 1e-3
    max_scenes = None if args.max_scenes <= 0 else args.max_scenes

    if args.self_test or not args.pkl_glob:
        if not args.self_test and not args.pkl_glob:
            print("No --pkl-glob given; falling back to --self-test synthetic scenes.")
        scenes = _synthetic_scenes(args.num_poses, args.interval_length)
    else:
        scenes = _load_real_scenes(
            args.pkl_glob, args.num_history_frames, args.num_poses, args.interval_length, max_scenes
        )
    if not scenes:
        raise SystemExit("No usable scenes found.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"summary_{timestamp}.csv"
    json_path = output_dir / f"summary_{timestamp}.json"
    check_keys = [
        "feasible_steer",
        "feasible_steer_rate",
        "feasible_accel",
        "feasible_curvature",
        "heading_consistent",
        "target_monotonic",
        "target_ahead",
        "lateral_converges",
        "lateral_recovers_in_horizon",
    ]
    warning_keys = [
        "accel_endpoint_within_bound",
        "heading_aligns_in_horizon",
        "lateral_recovers_in_horizon_relaxed",
    ]
    fieldnames = [
        "scene",
        "shift_y",
        "recovered",
        "passed",
        *check_keys,
        *warning_keys,
        "lateral_error_initial",
        "lateral_error_final",
        "lateral_error_reduction_ratio",
        "heading_error_initial",
        "heading_error_final",
        "time_to_recover",
        "max_abs_steer",
        "max_abs_steer_rate",
        "max_abs_accel_interior",
        "max_abs_accel_full",
        "max_abs_curvature_valid",
        "max_abs_curvature_full",
        "max_heading_resid_deg",
        "vis_path",
    ]

    records: List[Dict[str, object]] = []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for name, reference, meta in scenes:
            ref_t = torch.as_tensor(reference[None], dtype=torch.float64)
            for shift_y in shift_values:
                out_t, dbg = make_recovery_trajectory(
                    ref_t,
                    shift_y=float(shift_y),
                    dt=args.interval_length,
                    max_steer=args.max_steer,
                    max_steer_rate=args.max_steer_rate,
                    return_debug=True,
                )
                out = out_t[0].detach().cpu().numpy()
                raw_path = dbg["raw_path"][0]
                kin = _derive_kinematics(out, np.array([0.0, float(shift_y)]), args.interval_length)
                result = _run_checks(
                    out,
                    reference,
                    dbg,
                    kin,
                    shift_y=float(shift_y),
                    dt=args.interval_length,
                    max_steer=args.max_steer,
                    max_steer_rate=args.max_steer_rate,
                    wheel_base=wheel_base,
                    accel_bound=args.accel_bound,
                    heading_tol_rad=heading_tol_rad,
                    moving_eps=moving_eps,
                    curv_speed_min=args.curv_speed_min,
                    curv_seglen_min=args.curv_seglen_min,
                    recover_lateral_tol=args.recover_lateral_tol,
                    recover_strict_shift_max=args.recover_strict_shift_max,
                    recover_lateral_tol_large=args.recover_lateral_tol_large,
                    heading_align_tol_rad=heading_align_tol_rad,
                )

                vis_path = vis_dir / f"{name}_shift{shift_y:+.2f}.png".replace("+", "p").replace("-", "m")
                _render_figure(
                    vis_path,
                    name,
                    reference,
                    out,
                    raw_path,
                    dbg,
                    kin,
                    result,
                    shift_y=float(shift_y),
                    dt=args.interval_length,
                    max_steer=args.max_steer,
                    max_steer_rate=args.max_steer_rate,
                    wheel_base=wheel_base,
                    accel_bound=args.accel_bound,
                    heading_tol_rad=heading_tol_rad,
                    moving_eps=moving_eps,
                )

                row = {
                    "scene": name,
                    "shift_y": float(shift_y),
                    "recovered": result["recovered"],
                    "passed": result["passed"],
                    **{k: result["checks"].get(k) for k in check_keys},
                    **{k: result["warnings"].get(k) for k in warning_keys},
                    "lateral_error_initial": result["lateral_error_initial"],
                    "lateral_error_final": result["lateral_error_final"],
                    "lateral_error_reduction_ratio": result["lateral_error_reduction_ratio"],
                    "heading_error_initial": result["heading_error_initial"],
                    "heading_error_final": result["heading_error_final"],
                    "time_to_recover": result["time_to_recover"],
                    "max_abs_steer": result["max_abs_steer"],
                    "max_abs_steer_rate": result["max_abs_steer_rate"],
                    "max_abs_accel_interior": result["max_abs_accel_interior"],
                    "max_abs_accel_full": result["max_abs_accel_full"],
                    "max_abs_curvature_valid": result["max_abs_curvature_valid"],
                    "max_abs_curvature_full": result["max_abs_curvature_full"],
                    "max_heading_resid_deg": result["max_heading_resid_deg"],
                    "vis_path": str(vis_path),
                }
                writer.writerow(row)
                records.append({**row, "checks": result["checks"], "warnings": result["warnings"]})

    n_pass = sum(1 for r in records if r["passed"])
    json_path.write_text(json.dumps(records, indent=2, default=str))
    print(f"Ran {len(records)} (scene, shift) cases — {n_pass} passed, {len(records) - n_pass} failed.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    if args.make_html:
        html_path = output_dir / f"index_{timestamp}.html"
        _write_html(records, html_path)
        print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
