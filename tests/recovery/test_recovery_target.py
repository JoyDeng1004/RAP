import math

import torch

from navsim.agents.rap_dino.navsim_config import RAPConfig
from navsim.agents.rap_dino.recovery.recovery_target import (
    _adaptive_lookahead,
    _infer_initial_speed,
    make_recovery_trajectory,
)


def _straight_trajectory(batch_size=1, steps=8, speed=4.0, dt=0.5, dtype=torch.float32):
    x = torch.arange(1, steps + 1, dtype=dtype) * speed * dt
    y = torch.zeros_like(x)
    heading = torch.zeros_like(x)
    return torch.stack([x, y, heading], dim=-1).repeat(batch_size, 1, 1)


def test_rap_config_recovery_defaults_preserve_old_behavior():
    config = RAPConfig()

    assert config.ref2d_observation_aug is False
    assert config.ref2d_aug_scope == "sca"
    assert config.ref2d_aug_y_range == (-1.0, 1.0)
    assert config.ref2d_aug_prob == 1.0
    assert config.recovery_target_enabled is False


def test_zero_shift_returns_input_exactly():
    traj = _straight_trajectory(batch_size=2)

    out = make_recovery_trajectory(traj, shift_y=torch.zeros(2))

    torch.testing.assert_close(out, traj, atol=1e-5, rtol=0.0)


def test_positive_shift_converges_lateral_error_monotonically():
    traj = _straight_trajectory(batch_size=1, speed=5.0)

    out = make_recovery_trajectory(traj, shift_y=torch.tensor([1.0]))

    err = torch.abs(out[0, :, 1] - traj[0, :, 1])
    assert torch.all(err[1:] <= err[:-1] + 1e-5)
    assert err[-1] < err[0]


def _assert_forward_coordinate_does_not_fold_back(output):
    dx = torch.diff(output[..., 0], dim=-1)
    assert torch.all(dx >= -1e-4), output[..., 0]


def test_positive_shift_straight_recovery_does_not_fold_back():
    traj = _straight_trajectory(batch_size=1, steps=10, speed=5.0)

    out = make_recovery_trajectory(traj, shift_y=torch.tensor([1.0]))

    _assert_forward_coordinate_does_not_fold_back(out)


def test_negative_shift_straight_recovery_does_not_fold_back():
    traj = _straight_trajectory(batch_size=1, steps=10, speed=5.0)

    out = make_recovery_trajectory(traj, shift_y=torch.tensor([-1.0]))

    _assert_forward_coordinate_does_not_fold_back(out)


def test_high_speed_straight_recovery_does_not_fold_back():
    traj = _straight_trajectory(batch_size=2, steps=10, speed=15.0)

    out = make_recovery_trajectory(traj, shift_y=torch.tensor([1.0, -1.0]))

    _assert_forward_coordinate_does_not_fold_back(out)


def test_recovery_shape_dtype_and_heading_range():
    traj = _straight_trajectory(batch_size=2, dtype=torch.float64)

    out = make_recovery_trajectory(traj, shift_y=torch.tensor([1.0, -0.5], dtype=torch.float64))

    assert out.shape == traj.shape
    assert out.dtype == traj.dtype
    assert torch.all(out[..., 2] <= math.pi)
    assert torch.all(out[..., 2] >= -math.pi)


def test_initial_speed_is_inferred_from_uniform_straight_trajectory():
    speed = 6.0
    traj = _straight_trajectory(batch_size=1, speed=speed)

    inferred = _infer_initial_speed(traj, dt=0.5)

    assert torch.abs(inferred[0] - speed) / speed < 0.05


def test_lookahead_is_speed_adaptive_and_clipped():
    low = _adaptive_lookahead(torch.tensor([0.5]), k=0.4, min_lookahead=2.0, lo=2.5, hi=8.0)
    high = _adaptive_lookahead(torch.tensor([20.0]), k=0.4, min_lookahead=2.0, lo=2.5, hi=8.0)

    assert low.item() != high.item()
    assert 2.5 <= low.item() <= 8.0
    assert 2.5 <= high.item() <= 8.0
