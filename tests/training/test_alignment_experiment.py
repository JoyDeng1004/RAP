import pytest
import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace

from navsim.planning.training.alignment import (
    build_task_mask,
    compose_alignment_loss,
    domain_alignment_loss,
    mask_batch_mapping,
    spatial_alignment_loss,
    validate_alignment_config,
)


def test_r100_task_mask_selects_only_valid_real_samples():
    valid = torch.tensor([True, False, True])

    task_mask, task_is_real = build_task_mask(valid, task_real_ratio=1.0)

    assert task_mask.tolist() == [False, False, False, True, True]
    assert task_is_real.tolist() == [True, True]


def test_s100_task_mask_selects_only_paired_raster_samples():
    valid = torch.tensor([True, False, True])

    task_mask, task_is_real = build_task_mask(valid, task_real_ratio=0.0)

    assert task_mask.tolist() == [True, False, True, False, False]
    assert task_is_real.tolist() == [False, False]


def test_mixed_task_mask_selects_exact_ratio_and_one_modality_per_pair():
    valid = torch.tensor([True, True, False, True, True])
    generator = torch.Generator().manual_seed(7)

    task_mask, task_is_real = build_task_mask(valid, 0.5, generator=generator)

    assert int(task_mask.sum()) == 4
    assert int(task_is_real.sum()) == 2
    paired_raster = task_mask[: len(valid)][valid]
    selected_real = task_mask[len(valid) :]
    assert torch.logical_xor(paired_raster, selected_real).all()


def test_mask_batch_mapping_handles_predictions_and_token_lists():
    mask = torch.tensor([False, True, False, True])
    mapping = {
        "trajectory": torch.arange(4),
        "proposal_list": [torch.arange(4), torch.arange(4) + 10],
        "token": ["a", "b", "c", "d"],
        "metadata": "unchanged",
    }

    masked = mask_batch_mapping(mapping, mask)

    assert masked["trajectory"].tolist() == [1, 3]
    assert [value.tolist() for value in masked["proposal_list"]] == [[1, 3], [11, 13]]
    assert masked["token"] == ["b", "d"]
    assert masked["metadata"] == "unchanged"


def test_spatial_alignment_detaches_raster_branch_only():
    raster = torch.tensor([[1.0], [9.0], [3.0]], requires_grad=True)
    real = torch.tensor([[2.0], [5.0]], requires_grad=True)
    features = torch.cat([raster, real])

    loss = spatial_alignment_loss(features, 3, torch.tensor([True, False, True]))
    loss.backward()

    assert torch.count_nonzero(raster.grad) == 0
    assert real.grad is not None
    assert real.grad.tolist() == [[1.0], [2.0]]


def test_domain_loss_uses_only_valid_pairs():
    logits = torch.tensor([0.0, 100.0, 0.0, 0.0, 0.0])

    loss = domain_alignment_loss(logits, 3, torch.tensor([True, False, True]))

    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_global_alignment_grl_reverses_encoder_gradient():
    from navsim.agents.rap_dino.rap_model import grad_reverse

    feature = torch.tensor([1.0, -2.0], requires_grad=True)
    grad_reverse(feature, lambd=0.25).sum().backward()

    assert feature.grad.tolist() == pytest.approx([-0.25, -0.25])


def test_rap_grl_schedule_has_paper_scale_and_endpoints():
    from navsim.agents.rap_dino.rap_model import LambdaScheduler

    schedule = LambdaScheduler(gamma=10.0, scale=0.1)

    assert schedule(0.0) == pytest.approx(0.0)
    assert schedule(1.0) == pytest.approx(0.1, abs=1e-5)
    assert schedule(-2.0) == schedule(0.0)
    assert schedule(3.0) == schedule(1.0)


def test_hash_sampler_is_order_independent_and_covers_logs_first():
    from navsim.planning.script.build_alignment_small_data import select_hash_round_robin

    records = [
        {"log_name": "log_a", "token": "a2", "selection_hash": "20"},
        {"log_name": "log_b", "token": "b1", "selection_hash": "10"},
        {"log_name": "log_a", "token": "a1", "selection_hash": "01"},
    ]
    expected = [("log_a", "a1"), ("log_b", "b1")]

    first = select_hash_round_robin(records, 2)
    second = select_hash_round_robin(list(reversed(records)), 2)

    assert [(item["log_name"], item["token"]) for item in first] == expected
    assert [(item["log_name"], item["token"]) for item in second] == expected


def test_camera_loader_uses_explicit_independent_raster_root(tmp_path):
    from navsim.common.dataclasses import Cameras

    real_root = tmp_path / "sensor_blobs" / "trainval"
    raster_root = tmp_path / "render_output"
    relative = "log/CAM_F0/image.jpg"
    (real_root / relative).parent.mkdir(parents=True)
    (raster_root / relative).parent.mkdir(parents=True)
    Image.fromarray(np.full((1080, 1920, 3), 17, dtype=np.uint8)).save(real_root / relative)
    Image.fromarray(np.full((1120, 1920, 3), 29, dtype=np.uint8)).save(raster_root / relative)
    identity = np.eye(3)
    camera_dict = {
        "CAM_F0": {
            "data_path": relative,
            "sensor2lidar_rotation": identity,
            "sensor2lidar_translation": np.zeros(3),
            "cam_intrinsic": identity,
            "distortion": np.zeros(5),
        }
    }

    cameras = Cameras.from_camera_dict(
        real_root, camera_dict, ["cam_f0"],
        rendered_sensor_blobs_path=raster_root,
        strict_camera_loading=True,
    )

    assert cameras.cam_f0.real_valid is True
    assert cameras.cam_f0.rendered_valid is True
    assert cameras.cam_f0.image.shape == cameras.cam_f0.rendered_image.shape == (1080, 1920, 3)
    assert int(cameras.cam_f0.image[0, 0, 0]) == 17
    assert int(cameras.cam_f0.rendered_image[0, 0, 0]) == 29


def test_strict_camera_loader_does_not_substitute_missing_raster(tmp_path):
    from navsim.common.dataclasses import Cameras

    real_root = tmp_path / "real"
    relative = "log/CAM_F0/image.jpg"
    (real_root / relative).parent.mkdir(parents=True)
    Image.fromarray(np.ones((1080, 1920, 3), dtype=np.uint8)).save(real_root / relative)
    identity = np.eye(3)
    camera_dict = {
        "CAM_F0": {
            "data_path": relative,
            "sensor2lidar_rotation": identity,
            "sensor2lidar_translation": np.zeros(3),
            "cam_intrinsic": identity,
            "distortion": np.zeros(5),
        }
    }

    with pytest.raises(FileNotFoundError):
        Cameras.from_camera_dict(
            real_root, camera_dict, ["cam_f0"],
            rendered_sensor_blobs_path=tmp_path / "missing-raster-root",
            strict_camera_loading=True,
        )


@pytest.mark.parametrize(
    ("use_spatial", "use_global", "expected"),
    [
        (False, False, 10.0),
        (True, False, 10.4),
        (False, True, 10.6),
        (True, True, 11.0),
    ],
)
def test_four_alignment_switch_combinations(use_spatial, use_global, expected):
    total = compose_alignment_loss(
        task_loss=torch.tensor(10.0),
        spatial_loss=torch.tensor(2.0),
        domain_loss=torch.tensor(3.0),
        use_spatial_align=use_spatial,
        use_global_align=use_global,
        spatial_weight=0.2,
        domain_weight=0.2,
    )

    assert total.item() == pytest.approx(expected)


def test_alignment_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="task_real_ratio"):
        validate_alignment_config(1.1, "real")
    with pytest.raises(ValueError, match="eval_input_modality"):
        validate_alignment_config(1.0, "both")


def test_small_subset_deterministically_keeps_only_valid_real_pairs():
    from navsim.planning.script.run_training import _limit_dataset, _dataset_tokens

    class DummyDataset(torch.utils.data.Dataset):
        tokens = ["a", "b", "c", "d", "e"]

        def __len__(self):
            return len(self.tokens)

        def __getitem__(self, index):
            valid = index in (1, 3, 4)
            return {"camera_valid": torch.tensor(valid)}, {}

    subset = _limit_dataset(
        DummyDataset(), max_samples=2, require_camera_valid=True
    )

    assert list(subset.indices) == [1, 3]
    assert _dataset_tokens(subset) == ["b", "d"]

    whitelisted = _limit_dataset(
        DummyDataset(),
        max_samples=2,
        require_camera_valid=True,
        allowed_tokens={"b", "e"},
    )
    assert list(whitelisted.indices) == [1, 4]
    assert _dataset_tokens(whitelisted) == ["b", "e"]


class _DummyRAPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.progress = 0.0
        self.batch_size = 0


class _DummyAgent(torch.nn.Module):
    def __init__(self, use_spatial_align, use_global_align, eval_input_modality="real"):
        super().__init__()
        self._config = SimpleNamespace(
            distill_feature=True,
            distill_feature_weight=0.2,
            task_real_ratio=1.0,
            use_spatial_align=use_spatial_align,
            use_global_align=use_global_align,
            domain_align_weight=0.2,
            eval_input_modality=eval_input_modality,
        )
        self._rap_model = _DummyRAPModel()
        self.last_task_camera = None
        self.last_task_tokens = None
        self.last_forward_camera = None

    def forward(self, features, targets=None, return_score=False):
        camera = features["camera_feature"].float()
        self.last_forward_camera = camera.detach().clone()
        scalar = camera.reshape(camera.shape[0], -1).mean(dim=1)
        trajectory = scalar[:, None, None].expand(-1, 1, 3)
        return {
            "trajectory": trajectory,
            "bev_feature": scalar[:, None],
            "domain_logits": scalar,
        }

    def compute_loss(self, features, targets, prediction):
        self.last_task_camera = features["camera_feature"].detach().clone()
        self.last_task_tokens = list(targets["token"])
        return {"loss": prediction["trajectory"].mean()}


def _paired_dummy_batch():
    features = {
        "camera_feature": torch.tensor([[10.0], [20.0], [30.0]]),
        "rendered_camera_feature": torch.tensor([[1.0], [2.0], [3.0]]),
        "camera_valid": torch.tensor([True, False, True]),
        "ego_status": torch.zeros(3, 1),
    }
    targets = {
        "trajectory": torch.zeros(3, 1, 3),
        "score_mask": torch.ones(3, dtype=torch.bool),
        "token": ["a", "b", "c"],
    }
    return features, targets


def test_noalign_and_fullalign_use_identical_r100_task_supervision():
    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    outputs = []
    for use_alignment in (False, True):
        agent = _DummyAgent(use_alignment, use_alignment)
        module = AgentLightningModule(agent)
        module.log = lambda *args, **kwargs: None
        module.train()
        loss = module._step_distill(_paired_dummy_batch(), "train")
        outputs.append((agent, loss))

    noalign_agent, noalign_loss = outputs[0]
    fullalign_agent, fullalign_loss = outputs[1]
    assert noalign_agent.last_task_camera.flatten().tolist() == [10.0, 30.0]
    assert fullalign_agent.last_task_camera.flatten().tolist() == [10.0, 30.0]
    assert noalign_agent.last_task_tokens == fullalign_agent.last_task_tokens == ["a", "c"]
    assert fullalign_loss > noalign_loss


def test_predict_step_uses_real_input_and_filters_frame_names():
    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    agent = _DummyAgent(False, False, eval_input_modality="real")
    module = AgentLightningModule(agent)
    features, targets = _paired_dummy_batch()
    features["frame_name"] = ["fa", "fb", "fc"]

    prediction = module.predict_step((features, targets), 0)

    assert agent.last_forward_camera.flatten().tolist() == [10.0, 30.0]
    assert prediction["frame_name"] == ["fa", "fc"]
