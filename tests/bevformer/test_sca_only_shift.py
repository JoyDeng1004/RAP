import types

import torch

from navsim.agents.rap_dino.bevformer.encoder import BEVFormerEncoder
from navsim.agents.rap_dino.rap_model import RAPModel


def _minimal_encoder():
    encoder = BEVFormerEncoder.__new__(BEVFormerEncoder)
    encoder.return_intermediate = False
    encoder.num_points_in_pillar = 1
    encoder.pc_range = [-32, -32, -2.0, 32, 32, 6.0]
    encoder.half_width = 1.0
    encoder.half_length = 2.0
    encoder.rear_axle_to_center = 0.0
    encoder.lidar_height = 0.0
    encoder.layers = []

    def fake_point_sampling(self, reference_points, img_metas):
        metric_points = reference_points.permute(1, 2, 0, 3)[None, ..., :2]
        projected = torch.stack([metric_points[..., 1], metric_points[..., 1]], dim=-1)
        bev_mask = projected[..., 1] > 0.75
        return projected, bev_mask

    encoder.point_sampling = types.MethodType(fake_point_sampling, encoder)
    return encoder


def _run_encoder(shift_y):
    encoder = _minimal_encoder()
    batch_size, queries, channels = 1, 4, 2
    bev_query = torch.zeros(queries, batch_size, channels)
    bev_pos = torch.zeros_like(bev_query)
    ref_2d = torch.tensor(
        [[[2.0, 0.0, 0.0], [4.0, 0.25, 0.0], [6.0, 0.5, 0.0], [8.0, 0.75, 0.0]]]
    )
    lidar2img = torch.eye(4).reshape(1, 1, 4, 4)
    features = {
        "lidar2img": lidar2img,
        "img_shape": [[(256, 1024, 3)]],
        "ref2d_aug_shift_y": torch.tensor([shift_y]),
        "ref2d_debug": {},
    }

    encoder.forward(
        bev_query,
        None,
        None,
        bev_h=1,
        bev_w=queries,
        bev_pos=bev_pos,
        spatial_shapes=None,
        level_start_index=None,
        ref_2d=ref_2d,
        img_metas=features,
        features=features,
    )
    return features, ref_2d, lidar2img.clone(), id(lidar2img)


def test_sca_shift_keeps_tsa_ref_pos_unchanged_and_changes_sca_projection():
    shifted, _, _, _ = _run_encoder(shift_y=1.0)
    baseline, _, _, _ = _run_encoder(shift_y=0.0)

    torch.testing.assert_close(shifted["ref2d_debug"]["ref_pos"], baseline["ref2d_debug"]["ref_pos"])
    torch.testing.assert_close(
        shifted["ref2d_debug"]["hybird_ref_2d"],
        baseline["ref2d_debug"]["hybird_ref_2d"],
    )
    assert (
        shifted["ref2d_debug"]["reference_points_cam"]
        - baseline["ref2d_debug"]["reference_points_cam"]
    ).abs().max() > 1e-4


def test_sca_shift_can_change_bev_mask_and_does_not_mutate_lidar2img():
    shifted, _, lidar2img_before, lidar2img_id = _run_encoder(shift_y=1.0)
    baseline, _, _, _ = _run_encoder(shift_y=0.0)

    assert shifted["ref2d_debug"]["bev_mask"].sum() != baseline["ref2d_debug"]["bev_mask"].sum()
    assert id(shifted["lidar2img"]) == lidar2img_id
    torch.testing.assert_close(shifted["lidar2img"], lidar2img_before)


def test_rap_model_forward_passes_features_to_refiner_for_shift_plumbing():
    observed = {}

    class FakeBackbone:
        def __call__(self, camera_feature, img_metas):
            return (torch.zeros(1, 1, 1, 2), None, None, {"img_metas": img_metas})

    class FakeLinear:
        def __call__(self, ego_status):
            return torch.zeros(ego_status.shape[0], 2)

    class FakeEmbedding:
        weight = torch.zeros(4, 2)

    class FakeRefiner:
        def __call__(self, bev_feature, proposal_list, image_feature, features):
            observed["shift"] = features["ref2d_aug_shift_y"].clone()
            proposals = torch.zeros(features["ego_status"].shape[0], 2, 2, 3)
            proposal_list.append(proposals)
            return bev_feature, proposal_list

    class FakeScorer:
        def __call__(self, proposals, bev_feature):
            batch_size, proposal_num = proposals.shape[:2]
            scores = torch.zeros(batch_size, proposal_num, 1)
            return scores, None, None, None, None, None, None

    model = RAPModel.__new__(RAPModel)
    model.b2d = False
    model._backbone = FakeBackbone()
    model.hist_encoding = FakeLinear()
    model.init_feature = FakeEmbedding()
    model._trajectory_head = [FakeRefiner()]
    model.scorer = FakeScorer()
    model.lambda_scheduler = lambda progress: 0.0
    model.domain_classifier = lambda feat, lambd: torch.zeros(feat.shape[2])
    model.progress = 0.0
    model.batch_size = 1

    features = {
        "ego_status": torch.zeros(1, 1, 11),
        "camera_feature": torch.zeros(1, 1, 3, 4, 4),
        "ref2d_aug_shift_y": torch.tensor([0.75]),
    }

    model.forward(features, targets={})

    torch.testing.assert_close(observed["shift"], torch.tensor([0.75]))
