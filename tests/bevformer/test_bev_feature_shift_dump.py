import torch

from tests.bevformer.test_sca_only_shift import _run_encoder


def test_zero_shift_records_identical_bev_feature():
    pre, _, _, _ = _run_encoder(shift_y=0.0)
    post, _, _, _ = _run_encoder(shift_y=0.0)

    torch.testing.assert_close(
        pre["ref2d_debug"]["bev_feature"],
        post["ref2d_debug"]["bev_feature"],
        rtol=0.0,
        atol=0.0,
    )
