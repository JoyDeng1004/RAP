from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, Tuple, Union

import numpy as np
import torch
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from navsim.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import (
    BatchKinematicBicycleModel,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    DynamicStateIndex,
    StateIndex,
)


def _wrap_heading(heading: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(heading), torch.cos(heading))


def _empty_debug(
    trajectory: torch.Tensor,
    shift: torch.Tensor,
    speed: torch.Tensor,
    recover: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Debug payload for the degenerate no-recovery early return.

    All rows fall back to the reference, so per-step internals are zero-filled
    and the look-ahead target is the reference pose itself at each step.
    """
    batch_size, steps, _ = trajectory.shape
    traj_np = trajectory.detach().cpu().double().numpy()
    indices = np.tile(np.arange(steps, dtype=np.int64), (batch_size, 1))
    return {
        "raw_path": traj_np.copy(),
        "target_index": indices,
        "target_point_ref": traj_np[:, :, :2].copy(),
        "target_point_local": np.zeros((batch_size, steps, 2), dtype=np.float64),
        "desired_steer": np.zeros((batch_size, steps), dtype=np.float64),
        "applied_steer": np.zeros((batch_size, steps), dtype=np.float64),
        "steer_rate_cmd": np.zeros((batch_size, steps), dtype=np.float64),
        "speed": speed.detach().cpu().double().numpy(),
        "lookahead": np.zeros(batch_size, dtype=np.float64),
        "shift": shift.detach().cpu().double().numpy(),
        "recover_mask": recover.detach().cpu().numpy(),
    }


def _infer_initial_speed(trajectory: torch.Tensor, dt: float = 0.5) -> torch.Tensor:
    if trajectory.ndim != 3 or trajectory.shape[-1] < 2:
        raise ValueError(f"trajectory must have shape [B, T, 3], got {tuple(trajectory.shape)}")
    if trajectory.shape[1] == 0:
        raise ValueError("trajectory must contain at least one future pose")

    first_delta = trajectory[:, 0, :2]
    speed = torch.linalg.norm(first_delta, dim=-1) / dt
    if trajectory.shape[1] > 1:
        next_delta = trajectory[:, 1, :2] - trajectory[:, 0, :2]
        fallback = torch.linalg.norm(next_delta, dim=-1) / dt
        speed = torch.where(speed > 1e-4, speed, fallback)
    return speed


def _adaptive_lookahead(
    speed: torch.Tensor,
    *,
    k: float = 0.4,
    min_lookahead: float = 2.0,
    lo: float = 2.5,
    hi: float = 8.0,
) -> torch.Tensor:
    return torch.clamp(k * speed + min_lookahead, min=lo, max=hi)


def _as_shift_tensor(
    shift_y: Union[torch.Tensor, float],
    trajectory: torch.Tensor,
) -> torch.Tensor:
    if isinstance(shift_y, torch.Tensor):
        shift = shift_y.to(device=trajectory.device, dtype=trajectory.dtype)
    else:
        shift = torch.full((trajectory.shape[0],), float(shift_y), device=trajectory.device, dtype=trajectory.dtype)
    if shift.ndim == 0:
        shift = shift.repeat(trajectory.shape[0])
    if shift.shape != (trajectory.shape[0],):
        raise ValueError(f"shift_y must be scalar or shape [B], got {tuple(shift.shape)}")
    return shift


def _recompute_heading(
    xy: torch.Tensor,
    init_xy: torch.Tensor,
    fallback_heading: torch.Tensor,
    *,
    min_step: float = 1e-3,
) -> torch.Tensor:
    """Derive heading from the (possibly post-processed) path.

    Two robustness measures compared to a naive finite-difference + ``atan2``:

    * The predecessor of the first step is the **true** initial pose
      ``init_xy = (0, shift)`` rather than the origin, so the first heading is
      not biased by the whole lateral offset.
    * Heading is forward-filled across (near-)stationary segments. Clamping
      passes such as ``cummax`` on X can create zero-length / purely-lateral
      segments; without forward-fill ``atan2`` would snap those to +/-90 deg
      and produce visible heading spikes ("画龙").
    """
    prev_xy = torch.cat([init_xy[:, None, :], xy[:, :-1, :]], dim=1)
    delta = xy - prev_xy
    seg_len = torch.linalg.norm(delta, dim=-1)
    raw_heading = torch.atan2(delta[..., 1], delta[..., 0])
    moving = seg_len > min_step
    heading = torch.where(moving, raw_heading, fallback_heading)
    # Forward-fill heading over stationary segments so flats inherit the last
    # valid heading instead of snapping to a spurious +/-90 deg.
    for step in range(1, heading.shape[1]):
        carry = ~moving[:, step]
        heading[:, step] = torch.where(carry, heading[:, step - 1], heading[:, step])
    return heading


def _limit_label_curvature(
    out: torch.Tensor,
    init_xy: torch.Tensor,
    max_curvature: float,
) -> torch.Tensor:
    """Cap the label curvature to a kinematically feasible value, then rebuild.

    Splitting the post-processing into independent X and Y edits can bend the
    label into a corner sharper than the vehicle can physically drive (turn
    radius below ``wheel_base / tan(max_steer)``). This pass keeps each step's
    **arc length** but **rate-limits the heading change** to
    ``|Δθ| <= max_curvature * segment_length`` (the definition of a curvature
    bound), then re-integrates the path from those headings. It also recomputes
    the heading consistently with the rebuilt path.

    For an already-feasible path every heading change is within the bound, so
    ``new_theta == theta`` and the rebuild reproduces the input exactly — i.e.
    this is an identity for good trajectories and only files down the corners
    that exceed the limit. Near-stationary segments (``L ~ 0``) get
    ``Δθ_max ~ 0`` and therefore inherit the previous heading, which also
    forward-fills heading without the spurious +/-90 deg snap.
    """
    if out.shape[1] == 0:
        return out

    batch_size, steps, _ = out.shape
    xy = out[..., :2]
    prev_xy = torch.cat([init_xy[:, None, :], xy[:, :-1, :]], dim=1)
    seg = xy - prev_xy
    seg_len = torch.linalg.norm(seg, dim=-1)              # [B, T]
    theta = torch.atan2(seg[..., 1], seg[..., 0])         # [B, T]

    new_theta = torch.empty_like(theta)
    prev_theta = torch.zeros(batch_size, device=out.device, dtype=out.dtype)  # ego starts at heading 0
    for step in range(steps):
        dtheta_max = max_curvature * seg_len[:, step]
        d = theta[:, step] - prev_theta
        d = torch.atan2(torch.sin(d), torch.cos(d))       # wrap to [-pi, pi]
        d = torch.clamp(d, -dtheta_max, dtheta_max)
        prev_theta = prev_theta + d
        new_theta[:, step] = prev_theta

    cos_t = torch.cos(new_theta)
    sin_t = torch.sin(new_theta)
    new_xy = torch.empty_like(xy)
    point = init_xy
    for step in range(steps):
        point = point + seg_len[:, step : step + 1] * torch.stack([cos_t[:, step], sin_t[:, step]], dim=-1)
        new_xy[:, step] = point

    result = out.clone()
    result[..., :2] = new_xy
    result[..., 2] = new_theta
    return result


def _enforce_monotonic_lateral_recovery(
    output: torch.Tensor,
    reference: torch.Tensor,
    shift: torch.Tensor,
    max_descent_slope: float = 0.4,
) -> torch.Tensor:
    """Keep the finite-horizon recovery target from overshooting laterally.

    A monotonic non-increasing envelope is applied to the *lateral error*
    (distance to the reference), starting from ``|shift|``. This still blocks
    overshoot and oscillation: the error can only shrink, and once it touches
    the reference ``prev_abs`` locks near zero so it never crosses to the far
    side.

    Unlike before, the error is **not** forced linearly down to exactly zero at
    the last step. That terminal yank compressed the tail and produced large
    spurious acceleration / curvature on big offsets; any residual the
    kinematics cannot remove within the horizon is now kept as-is.

    The lateral error is additionally **rate-limited**: per step it may not drop
    by more than ``max_descent_slope`` times the forward progress of that step.
    A fast lateral collapse over a short forward segment is exactly what bent the
    label into a sharp, high-curvature corner; capping the descent slope keeps
    the lateral approach angle (and therefore the corner) bounded.

    Only the lateral (Y) channel is modified here; heading is recomputed once,
    consistently, after all geometric post-processing.
    """
    if output.shape[1] == 0:
        return output

    out = output.clone()
    steps = out.shape[1]
    sign = torch.sign(shift).view(-1, 1)
    lateral = out[..., 1] - reference[..., 1]
    forward_x = out[..., 0]
    prev_abs = torch.abs(shift)
    prev_x = torch.zeros_like(shift)  # the recovery starts at X = 0
    bounded_lateral = torch.empty_like(lateral)
    for step in range(steps):
        forward_step = torch.clamp(forward_x[:, step] - prev_x, min=0.0)
        # Lowest the error may reach this step: a rate-limited descent so the
        # lateral path angle stays below ~atan(max_descent_slope).
        lower = torch.clamp(prev_abs - max_descent_slope * forward_step, min=0.0)
        current = torch.clamp(torch.minimum(torch.abs(lateral[:, step]), prev_abs), min=lower)
        bounded_lateral[:, step] = sign[:, 0] * current
        prev_abs = current
        prev_x = forward_x[:, step]

    out[..., 1] = reference[..., 1] + bounded_lateral
    return out


def _enforce_forward_progress(
    output: torch.Tensor,
    reference: torch.Tensor,
    min_progress_ratio: float = 0.25,
) -> torch.Tensor:
    """Avoid recovery labels that move backward when the reference moves forward.

    This **trusts the raw bicycle path**: a genuine forward step is kept exactly
    as integrated, even if it is slow. Only steps that actually move backward or
    stall (``x[t] <= kept x[t-1]``) are corrected, and then by a small *positive*
    increment rather than a ``cummax`` plateau, so the label never stalls into a
    zero-length segment. Re-spacing every step (as a blanket min-increment did)
    distorted otherwise-feasible low-speed geometry into sharp, high-curvature
    corners; only touching the misbehaving steps preserves the raw geometry.

    Only the longitudinal (X) channel is modified here; heading is recomputed
    once, consistently, after all geometric post-processing.
    """
    if output.shape[1] <= 1:
        return output

    ref_forward = torch.all(torch.diff(reference[..., 0], dim=1) >= -1e-4, dim=1)
    if not torch.any(ref_forward):
        return output

    out = output.clone()
    x = out[ref_forward, :, 0]                       # [n, T]
    ref_x = reference[ref_forward, :, 0]             # [n, T]

    # Minimum positive increment used only to step over a genuine backward/stall,
    # derived from the reference's median forward step and floored away from zero.
    ref_dx = torch.diff(ref_x, dim=1).clamp(min=0.0)
    if ref_dx.shape[1] > 0:
        median_dx = ref_dx.median(dim=1).values
    else:
        median_dx = torch.zeros_like(x[:, 0])
    min_dx = torch.clamp(min_progress_ratio * median_dx, min=1e-3)

    fixed = x.clone()
    for step in range(1, x.shape[1]):
        # Keep the raw forward step; only intervene when it would go backward.
        backward = x[:, step] <= fixed[:, step - 1]
        fixed[:, step] = torch.where(backward, fixed[:, step - 1] + min_dx, x[:, step])
    out[ref_forward, :, 0] = fixed
    return out


def make_recovery_trajectory(
    trajectory: torch.Tensor,
    shift_y: Union[torch.Tensor, float],
    dt: float = 0.5,
    lookahead_m: float | None = None,
    wheel_base: float = get_pacifica_parameters().wheel_base,
    max_steer: float = np.pi / 3,
    max_steer_rate: float = np.pi / 3,
    speed_eps: float = 0.1,
    lookahead_lead: int = 2,
    lookahead_k: float = 0.4,
    min_lookahead: float = 2.0,
    lookahead_lo: float = 2.5,
    lookahead_hi: float = 8.0,
    return_debug: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, np.ndarray]]]:
    """Build a pure-pursuit recovery target in the original ego-local frame.

    ``shift_y > 0`` means the ego is offset to the left of the original ego
    frame. The generated target starts from that shifted state and steers back
    toward the unshifted reference trajectory.

    Coordinate-frame contract (NAVSIM/PDM convention): ``trajectory`` is a
    ``[B, T, 3]`` sequence of future poses expressed **relative to the current
    ego pose (0, 0, 0)**; ``trajectory[:, 0]`` is the pose one ``dt`` ahead and
    the origin itself is not included. Passing a global-frame trajectory (one
    that does not start near the origin) breaks the displacement / speed
    assumptions below, which is why an entry guard is enforced.

    :param max_steer_rate: [rad/s] hard limit on the commanded front-wheel
        angle rate, applied on top of the model's first-order steering filter.
    :param speed_eps: [m/s] below this inferred speed a recovery is ill-posed
        (the kinematic model barely moves); those rows return the reference.
    :param lookahead_lead: time-indexed look-ahead lead in steps. At control
        step ``t`` the controller aims at ``reference[min(t + lookahead_lead,
        T - 1)]`` instead of selecting a target by geometric distance. With the
        constant model speed this is a speed-proportional look-ahead and keeps
        the target well-defined and monotonic across the whole horizon. The
        ``lookahead_*`` distance below is only used as a steering-denominator
        floor and as the straighten magnitude, no longer for target selection.
    :param return_debug: when ``True`` also return a dict of per-step internal
        pure-pursuit state (white-box validation). The default ``False`` keeps
        the original single-tensor return and behaviour unchanged. Keys:

        * ``raw_path`` ``[B, T, 3]``: bicycle integration output *before* the
          geometric post-processing (lateral clamp / forward progress / heading
          recompute), for comparing raw vs post-processed.
        * ``target_index`` ``[B, T]`` int: reference index the controller aimed
          at each step (the look-ahead target).
        * ``target_point_ref`` ``[B, T, 2]``: that reference point in the
          original ego-local frame.
        * ``target_point_local`` ``[B, T, 2]``: that reference point expressed in
          the car body frame at the step ``(local_x, local_y)``.
        * ``desired_steer`` ``[B, T]``: pure-pursuit steering angle command
          (after clipping to ``max_steer``), before the rate limit.
        * ``applied_steer`` ``[B, T]``: steering angle actually realised by the
          kinematic model after propagation.
        * ``steer_rate_cmd`` ``[B, T]``: commanded steering rate (after the
          ``max_steer_rate`` limit).
        * ``speed`` / ``lookahead`` / ``shift`` ``[B]``: inferred initial speed,
          look-ahead distance and lateral offset per row.
        * ``recover_mask`` ``[B]`` bool: rows that were actually recovered (vs
          returned as the reference because of no shift / too low speed).
    """

    shift = _as_shift_tensor(shift_y, trajectory)

    # --- Coordinate-frame guards (C3) ----------------------------------------
    if not torch.isfinite(trajectory).all():
        raise ValueError("trajectory contains non-finite values")
    first_step_reach = torch.linalg.norm(trajectory[:, 0, :2], dim=-1)
    if torch.any(first_step_reach > 50.0):
        raise ValueError(
            "trajectory[:, 0] is too far from the origin; make_recovery_trajectory "
            "expects an ego-local trajectory whose first pose is one dt ahead of (0, 0, 0)."
        )

    speed = _infer_initial_speed(trajectory, dt=dt)

    # Rows that should actually be recovered: meaningfully shifted *and* moving
    # fast enough for the kinematic model to make progress (L3).
    recover = (torch.abs(shift) > 1e-8) & (speed > speed_eps)
    if not torch.any(recover):
        out = trajectory.clone()
        if return_debug:
            return out, _empty_debug(trajectory, shift, speed, recover)
        return out

    if lookahead_m is None:
        lookahead = _adaptive_lookahead(
            speed,
            k=lookahead_k,
            min_lookahead=min_lookahead,
            lo=lookahead_lo,
            hi=lookahead_hi,
        )
    else:
        lookahead = torch.full_like(speed, float(lookahead_m))

    device = trajectory.device
    dtype = trajectory.dtype
    traj_np = trajectory.detach().cpu().double().numpy()
    shift_np = shift.detach().cpu().double().numpy()
    speed_np = speed.detach().cpu().double().numpy()
    lookahead_np = lookahead.detach().cpu().double().numpy()

    batch_size, steps, _ = trajectory.shape
    states = np.zeros((batch_size, StateIndex.size()), dtype=np.float64)
    states[:, StateIndex.Y] = shift_np
    states[:, StateIndex.VELOCITY_X] = speed_np

    vehicle = get_pacifica_parameters()
    if abs(float(wheel_base) - float(vehicle.wheel_base)) > 1e-9:
        if hasattr(vehicle, "_replace"):
            vehicle = vehicle._replace(wheel_base=wheel_base)
        else:
            vehicle = replace(vehicle, wheel_base=wheel_base)
    model = BatchKinematicBicycleModel(vehicle=vehicle, max_steering_angle=max_steer)
    sampling_time = TimePoint(int(dt * 1e6))

    output = np.zeros((batch_size, steps, 3), dtype=np.float64)
    rows = np.arange(batch_size)
    steer_rate_limit = float(max_steer_rate) * dt
    lead = max(int(lookahead_lead), 0)

    if return_debug:
        dbg_target_index = np.zeros((batch_size, steps), dtype=np.int64)
        dbg_target_point_ref = np.zeros((batch_size, steps, 2), dtype=np.float64)
        dbg_target_point_local = np.zeros((batch_size, steps, 2), dtype=np.float64)
        dbg_desired_steer = np.zeros((batch_size, steps), dtype=np.float64)
        dbg_applied_steer = np.zeros((batch_size, steps), dtype=np.float64)
        dbg_steer_rate_cmd = np.zeros((batch_size, steps), dtype=np.float64)

    for step in range(steps):
        commands = np.zeros((batch_size, len(DynamicStateIndex)), dtype=np.float64)

        # --- Time-indexed look-ahead target selection over the batch --------
        # Aim at the reference pose ``lead`` steps ahead of the current control
        # step (same index for every row, it only depends on ``step``). With a
        # constant model speed this is a speed-proportional look-ahead that is
        # always defined and monotonic, so the distance-based candidate scan and
        # its degenerate branches (no point ahead / behind target) are gone.
        xy = states[:, StateIndex.POINT]                       # [B, 2]
        heading = states[:, StateIndex.HEADING]                # [B]
        delta = traj_np[:, :, :2] - xy[:, None, :]             # [B, T, 2]
        cos_h = np.cos(heading)[:, None]
        sin_h = np.sin(heading)[:, None]
        local_x_all = cos_h * delta[..., 0] + sin_h * delta[..., 1]   # [B, T]
        local_y_all = -sin_h * delta[..., 0] + cos_h * delta[..., 1]  # [B, T]

        target_idx = np.full(batch_size, min(step + lead, steps - 1), dtype=np.int64)
        local_x = local_x_all[rows, target_idx]
        local_y = local_y_all[rows, target_idx]

        # Safety net: if the time-indexed target landed behind the rear axle
        # (heavy steering / longitudinal lag pushed the reference point back),
        # aim at a virtual point straight ahead so alpha=0 and the wheel simply
        # straightens (rate-limited) instead of chasing a behind point.
        behind = local_x <= 1e-6
        if np.any(behind):
            local_x[behind] = lookahead_np[behind]
            local_y[behind] = 0.0

        # Pure-pursuit denominator floored at the lookahead distance so a close
        # fall-back point cannot produce an over-aggressive steer command (L2).
        denom = np.maximum(np.maximum(np.hypot(local_x, local_y), lookahead_np), 1e-6)
        alpha = np.arctan2(local_y, np.maximum(local_x, 1e-6))
        desired_steer = np.arctan2(2.0 * wheel_base * np.sin(alpha), denom)
        desired_steer = np.clip(desired_steer, -max_steer, max_steer)

        # Limit the front-wheel angle rate on top of the model filter (L4).
        current_steer = states[:, StateIndex.STEERING_ANGLE]
        steer_delta = np.clip(desired_steer - current_steer, -steer_rate_limit, steer_rate_limit)
        commands[:, DynamicStateIndex.STEERING_RATE] = steer_delta / dt

        states = model.propagate_state(states, commands, sampling_time)
        output[:, step, 0] = states[:, StateIndex.X]
        output[:, step, 1] = states[:, StateIndex.Y]
        output[:, step, 2] = states[:, StateIndex.HEADING]

        if return_debug:
            tp_ref = traj_np[rows, target_idx, :2].copy()
            if np.any(behind):
                # Record the virtual straight-ahead target so the aim overlay and
                # the local_x>0 check reflect the straighten behaviour.
                tp_ref[behind] = xy[behind] + lookahead_np[behind, None] * np.stack(
                    [np.cos(heading[behind]), np.sin(heading[behind])], axis=1
                )
            dbg_target_index[:, step] = target_idx
            dbg_target_point_ref[:, step] = tp_ref
            dbg_target_point_local[:, step, 0] = local_x
            dbg_target_point_local[:, step, 1] = local_y
            dbg_desired_steer[:, step] = desired_steer
            dbg_applied_steer[:, step] = states[:, StateIndex.STEERING_ANGLE]
            dbg_steer_rate_cmd[:, step] = commands[:, DynamicStateIndex.STEERING_RATE]

    raw_path = output.copy()
    out = torch.as_tensor(output, device=device, dtype=dtype)

    # --- Geometric post-processing -------------------------------------------
    # The true predecessor of the first output pose is the shifted start pose
    # (0, shift), not the origin (C1).
    init_xy = torch.stack([torch.zeros_like(shift), shift], dim=-1)  # [B, 2]
    out = _enforce_monotonic_lateral_recovery(out, trajectory, shift)
    out = _enforce_forward_progress(out, trajectory)
    # Final feasibility pass: cap the label curvature to the vehicle limit and
    # rebuild XY + heading from it. This files down the sharp corners that the
    # independent X/Y edits above can introduce on low-speed / sharp geometry,
    # and is an identity for already-feasible trajectories. It also recomputes
    # heading consistently with the rebuilt path (subsumes _recompute_heading).
    curv_max = math.tan(max_steer) / float(wheel_base)
    out = _limit_label_curvature(out, init_xy, curv_max)
    out[..., 2] = _wrap_heading(out[..., 2])

    # Non-recoverable rows (no shift / too slow) fall back to the reference.
    out = torch.where(recover.view(-1, 1, 1), out, trajectory)

    if return_debug:
        debug = {
            "raw_path": raw_path,
            "target_index": dbg_target_index,
            "target_point_ref": dbg_target_point_ref,
            "target_point_local": dbg_target_point_local,
            "desired_steer": dbg_desired_steer,
            "applied_steer": dbg_applied_steer,
            "steer_rate_cmd": dbg_steer_rate_cmd,
            "speed": speed_np,
            "lookahead": lookahead_np,
            "shift": shift_np,
            "recover_mask": recover.detach().cpu().numpy(),
        }
        return out, debug
    return out
