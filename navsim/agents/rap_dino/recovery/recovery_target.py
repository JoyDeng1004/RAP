from __future__ import annotations

import math
from dataclasses import replace
from typing import Union

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


def _enforce_monotonic_lateral_recovery(
    output: torch.Tensor,
    reference: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Keep the finite-horizon recovery target from overshooting laterally."""
    if output.shape[1] == 0:
        return output

    out = output.clone()
    steps = out.shape[1]
    sign = torch.sign(shift).view(-1, 1)
    lateral = out[..., 1] - reference[..., 1]
    decay = torch.linspace(
        1.0,
        0.0,
        steps,
        device=out.device,
        dtype=out.dtype,
    )
    prev_abs = torch.abs(shift)
    bounded_lateral = torch.empty_like(lateral)
    for step in range(steps):
        allowed = torch.minimum(prev_abs, torch.abs(shift) * decay[step])
        current = torch.minimum(torch.abs(lateral[:, step]), allowed)
        bounded_lateral[:, step] = sign[:, 0] * current
        prev_abs = current

    out[..., 1] = reference[..., 1] + bounded_lateral

    prev_xy = torch.cat([torch.zeros_like(out[:, :1, :2]), out[:, :-1, :2]], dim=1)
    delta = out[..., :2] - prev_xy
    heading = torch.atan2(delta[..., 1], torch.clamp(delta[..., 0], min=1e-6))
    moving = torch.linalg.norm(delta, dim=-1) > 1e-6
    out[..., 2] = torch.where(moving, heading, out[..., 2])
    return out


def _enforce_forward_progress(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Avoid recovery labels that move backward when the reference moves forward."""
    if output.shape[1] <= 1:
        return output

    ref_forward = torch.all(torch.diff(reference[..., 0], dim=1) >= -1e-4, dim=1)
    if not torch.any(ref_forward):
        return output

    out = output.clone()
    forward_x = torch.cummax(out[ref_forward, :, 0], dim=1).values
    out[ref_forward, :, 0] = forward_x

    prev_xy = torch.cat([torch.zeros_like(out[:, :1, :2]), out[:, :-1, :2]], dim=1)
    delta = out[..., :2] - prev_xy
    heading = torch.atan2(delta[..., 1], torch.clamp(delta[..., 0], min=1e-6))
    moving = torch.linalg.norm(delta, dim=-1) > 1e-6
    out[..., 2] = torch.where(moving, heading, out[..., 2])
    return out


def make_recovery_trajectory(
    trajectory: torch.Tensor,
    shift_y: Union[torch.Tensor, float],
    dt: float = 0.5,
    lookahead_m: float | None = None,
    wheel_base: float = get_pacifica_parameters().wheel_base,
    max_steer: float = np.pi / 3,
    lookahead_k: float = 0.4,
    min_lookahead: float = 2.0,
    lookahead_lo: float = 2.5,
    lookahead_hi: float = 8.0,
) -> torch.Tensor:
    """Build a pure-pursuit recovery target in the original ego-local frame.

    ``shift_y > 0`` means the ego is offset to the left of the original ego
    frame. The generated target starts from that shifted state and steers back
    toward the unshifted reference trajectory.
    """

    shift = _as_shift_tensor(shift_y, trajectory)
    if torch.all(torch.abs(shift) <= 1e-8):
        return trajectory.clone()

    speed = _infer_initial_speed(trajectory, dt=dt)
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
    target_indices = np.zeros(batch_size, dtype=np.int64)
    for step in range(steps):
        commands = np.zeros((batch_size, len(DynamicStateIndex)), dtype=np.float64)
        for batch_idx in range(batch_size):
            xy = states[batch_idx, StateIndex.POINT]
            future = traj_np[batch_idx, :, :2]
            heading = states[batch_idx, StateIndex.HEADING]

            delta = future - xy[None]
            local_x_all = math.cos(heading) * delta[:, 0] + math.sin(heading) * delta[:, 1]
            local_y_all = -math.sin(heading) * delta[:, 0] + math.cos(heading) * delta[:, 1]
            distances = np.linalg.norm(delta, axis=-1)

            indices = np.arange(steps)
            ahead = (indices >= target_indices[batch_idx]) & (local_x_all > 1e-6)
            candidates = np.flatnonzero(ahead & (distances >= lookahead_np[batch_idx]))
            if len(candidates):
                target_idx = int(candidates[0])
            else:
                ahead_candidates = np.flatnonzero(ahead)
                if len(ahead_candidates):
                    target_idx = int(ahead_candidates[-1])
                else:
                    commands[batch_idx, DynamicStateIndex.STEERING_RATE] = (
                        -states[batch_idx, StateIndex.STEERING_ANGLE]
                    ) / dt
                    continue
            target_indices[batch_idx] = max(target_indices[batch_idx], target_idx)

            local_x = local_x_all[target_idx]
            local_y = local_y_all[target_idx]
            distance = max(math.hypot(local_x, local_y), 1e-6)
            alpha = math.atan2(local_y, max(local_x, 1e-6))
            desired_steer = math.atan2(2.0 * wheel_base * math.sin(alpha), distance)
            desired_steer = float(np.clip(desired_steer, -max_steer, max_steer))
            commands[batch_idx, DynamicStateIndex.STEERING_RATE] = (
                desired_steer - states[batch_idx, StateIndex.STEERING_ANGLE]
            ) / dt

        states = model.propagate_state(states, commands, sampling_time)
        output[:, step, 0] = states[:, StateIndex.X]
        output[:, step, 1] = states[:, StateIndex.Y]
        output[:, step, 2] = states[:, StateIndex.HEADING]

    out = torch.as_tensor(output, device=device, dtype=dtype)
    out = _enforce_monotonic_lateral_recovery(out, trajectory, shift)
    out = _enforce_forward_progress(out, trajectory)
    out[..., 2] = _wrap_heading(out[..., 2])
    return out
