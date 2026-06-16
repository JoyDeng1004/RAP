import torch

from navsim.agents.rap_dino.recovery.recovery_target import make_recovery_trajectory
from tests.bevformer.test_sca_only_shift import _run_encoder


def test_positive_shift_observation_and_recovery_target_use_same_left_positive_convention():
    shifted, _, _, _ = _run_encoder(shift_y=1.0)
    baseline, _, _, _ = _run_encoder(shift_y=0.0)
    obs_delta_u = (
        shifted["ref2d_debug"]["reference_points_cam"][..., 0]
        - baseline["ref2d_debug"]["reference_points_cam"][..., 0]
    ).mean()
    obs_dir = torch.sign(obs_delta_u)

    # Convention: shift_y > 0 means the ego is left of the original ego frame.
    # Therefore the SCA metric y coordinate moves left-positive, and the
    # recovery target initially remains left-positive while curving back to GT.
    traj = torch.stack(
        [
            torch.arange(1, 9, dtype=torch.float32) * 2.0,
            torch.zeros(8),
            torch.zeros(8),
        ],
        dim=-1,
    )[None]
    recovery = make_recovery_trajectory(traj, shift_y=torch.tensor([1.0]))
    tgt_dir = torch.sign(recovery[0, 0, 1] - traj[0, 0, 1])

    assert obs_dir == tgt_dir
