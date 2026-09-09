import pytest
import torch
import numpy as np
import json
from PIL import Image
from pathlib import Path
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


def test_navsim_scenario_reconstructs_timepoints_for_placeholder_history():
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    current_timestamp = 10_000_000
    frames = [None, None, None] + [
        SimpleNamespace(
            timestamp=current_timestamp + index * 500_000,
            token=f"token-{index}",
            roadblock_ids=["roadblock"],
        )
        for index in range(11)
    ]
    scene = SimpleNamespace(
        scene_metadata=SimpleNamespace(
            num_history_frames=4,
            map_name="us-ma-boston",
            log_name="log",
        ),
        frames=frames,
    )

    scenario = NavSimScenario(scene, map_root="/unused", map_version="nuplan-maps-v1.0")

    assert [point.time_us for point in scenario._time_points[:4]] == [
        8_500_000,
        9_000_000,
        9_500_000,
        10_000_000,
    ]
    assert scenario.token == "token-0"
    assert scenario.get_time_point(0).time_us == current_timestamp


def test_target_only_cache_needs_no_sensor_or_raster_root(tmp_path):
    from navsim.planning.script.run_dataset_caching import TargetOnlyAgent
    from navsim.planning.training.dataset import Dataset, load_feature_target_from_pickle

    token = "target-only-token"

    class FakeScene:
        scene_metadata = SimpleNamespace(log_name="log", initial_token=token)

        def get_agent_input(self):
            raise AssertionError("target-only caching must not load sensors")

        def get_future_trajectory(self, num_trajectory_frames):
            return SimpleNamespace(poses=np.zeros((num_trajectory_frames, 3), dtype=np.float32))

    class FakeSceneLoader:
        tokens = [token]

        def get_scene_from_token(self, requested_token):
            assert requested_token == token
            return FakeScene()

    config = SimpleNamespace(trajectory_sampling=SimpleNamespace(num_poses=10))
    provider = TargetOnlyAgent(config)
    Dataset(
        scene_loader=FakeSceneLoader(),
        feature_builders=provider.get_feature_builders(),
        target_builders=provider.get_target_builders(),
        cache_path=str(tmp_path),
        force_cache_computation=True,
    )

    target_path = tmp_path / "log" / token / "rap_target.gz"
    assert target_path.is_file()
    assert not list(tmp_path.rglob("rap_feature.gz"))
    target = load_feature_target_from_pickle(target_path)
    assert target["trajectory"].shape == (10, 3)
    assert torch.isfinite(target["trajectory"]).all()


def test_paired_trainval_config_resolves_stride_one():
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).parents[2] / "navsim/planning/script/config/training"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        cfg = compose(
            config_name="default_training",
            overrides=[
                "agent=rap_agent",
                "dataset=navsim_dataset",
                "train_test_split=paired_trainval",
            ],
        )

    assert cfg.target_only is False
    assert cfg.train_test_split.scene_filter.frame_interval == 1
    assert len(cfg.train_test_split.scene_filter.log_names) == 54


def test_hash_sampler_is_proportional_exact_and_order_independent():
    from navsim.planning.script.build_alignment_small_data import select_hash_proportional

    records = []
    for log_name, size in (("log_a", 6), ("log_b", 3), ("log_c", 1)):
        records.extend(
            {"log_name": log_name, "token": f"{log_name}-{index}", "selection_hash": f"{index:02d}"}
            for index in range(size)
        )

    first = select_hash_proportional(records, 5)
    second = select_hash_proportional(list(reversed(records)), 5)
    counts = {log: sum(item["log_name"] == log for item in first) for log in ("log_a", "log_b", "log_c")}

    assert counts == {"log_a": 3, "log_b": 1, "log_c": 1}
    assert len(first) == 5
    assert [item["token"] for item in first] == [item["token"] for item in second]


def test_topup_replaces_failed_token_in_hash_order(tmp_path):
    from navsim.planning.script.build_alignment_small_data import deterministic_topup

    candidates = [
        {"split": "train", "log_name": "a", "token": "a0", "selection_hash": "00"},
        {"split": "train", "log_name": "a", "token": "a1", "selection_hash": "01"},
        {"split": "train", "log_name": "b", "token": "b0", "selection_hash": "00"},
    ]
    selected = [candidates[0], candidates[2]]

    def attempt(record):
        if record["token"] == "a0":
            raise RuntimeError("render failed")
        return record["token"]

    final, results, replacements = deterministic_topup(
        candidates, selected, "train", attempt, tmp_path / "topup.jsonl", max_replacement_rate=1.0
    )

    assert [item["token"] for item in final] == ["a1", "b0"]
    assert results == ["a1", "b0"]
    assert replacements[0]["fallback_cross_log"] is False


@pytest.mark.parametrize("size", [2498, 464, 11489])
def test_exact_distributed_sampler_has_no_padding_or_drop(size):
    from navsim.planning.training.samplers import ExactDistributedSampler

    tokens = [f"token-{index}" for index in range(size)]
    samplers = [
        ExactDistributedSampler(range(size), 4, rank, seed=17, shuffle=True, tokens=tokens)
        for rank in range(4)
    ]
    shards = [set(iter(sampler)) for sampler in samplers]
    assert set.union(*shards) == set(range(size))
    assert sum(len(shard) for shard in shards) == size
    assert all(not shards[left] & shards[right] for left in range(4) for right in range(left + 1, 4))
    assert len({(len(sampler) + 31) // 32 for sampler in samplers}) == 1
    first = samplers[0].global_token_order(0)
    assert first == ExactDistributedSampler(range(size), 4, 0, 17, True, tokens).global_token_order(0)
    assert first != samplers[0].global_token_order(1)


def test_epoch_token_checksum_callback_records_train_and_val_for_20_epochs(tmp_path):
    from navsim.planning.script.run_training import EpochTokenChecksumCallback
    from navsim.planning.training.samplers import ExactDistributedSampler

    train = ExactDistributedSampler(range(11), 1, 0, seed=3, tokens=[f"t{i}" for i in range(11)])
    val = ExactDistributedSampler(range(5), 1, 0, seed=3, shuffle=False, tokens=[f"v{i}" for i in range(5)])
    callback = EpochTokenChecksumCallback(tmp_path, train, val)
    trainer = SimpleNamespace(current_epoch=0, is_global_zero=True)
    for epoch in range(20):
        trainer.current_epoch = epoch
        callback.on_train_epoch_start(trainer, None)
    rows = [json.loads(line) for line in (tmp_path / "epoch_token_checksums.jsonl").read_text().splitlines()]

    assert len(rows) == 40
    assert sum(row["stage"] == "train" for row in rows) == 20
    assert sum(row["stage"] == "val" for row in rows) == 20
    assert {row["num_tokens"] for row in rows if row["stage"] == "train"} == {11}
    before = len(rows)
    trainer.is_global_zero = False
    callback.on_train_epoch_start(trainer, None)
    assert len((tmp_path / "epoch_token_checksums.jsonl").read_text().splitlines()) == before


def test_stage_a_callback_path_replaces_agent_checkpoints(tmp_path):
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint
    from navsim.planning.script.run_training import (
        EpochTokenChecksumCallback,
        LossComponentCallback,
        _build_training_callbacks,
    )
    from navsim.planning.training.samplers import ExactDistributedSampler

    old_checkpoint = ModelCheckpoint(dirpath=tmp_path / "agent-checkpoints", save_top_k=1)
    marker = object()
    agent = SimpleNamespace(get_training_callbacks=lambda: [old_checkpoint, marker])
    cfg = OmegaConf.create(
        {"output_dir": str(tmp_path / "run"), "final_step_checkpoint_only": True}
    )
    train = ExactDistributedSampler(range(2), 1, 0, tokens=["t0", "t1"])
    val = ExactDistributedSampler(range(1), 1, 0, shuffle=False, tokens=["v0"])

    callbacks = _build_training_callbacks(cfg, agent, train, val)
    checkpoints = [item for item in callbacks if isinstance(item, ModelCheckpoint)]

    assert marker in callbacks
    assert len(checkpoints) == 1
    assert checkpoints[0] is not old_checkpoint
    assert checkpoints[0].dirpath == str(tmp_path / "run" / "checkpoints")
    assert checkpoints[0].save_last is True
    assert checkpoints[0].save_top_k == 0
    assert sum(isinstance(item, EpochTokenChecksumCallback) for item in callbacks) == 1
    assert sum(isinstance(item, LossComponentCallback) for item in callbacks) == 1


@pytest.mark.parametrize("progress", [0.0, 0.5, 1.0])
def test_loss_component_callback_writes_live_grl_and_reconstructs_total(tmp_path, progress):
    from navsim.agents.rap_dino.rap_model import LambdaScheduler
    from navsim.planning.script.run_training import LossComponentCallback

    grl = LambdaScheduler(gamma=10.0, scale=0.1)(progress)
    task, spatial, global_loss = 2.0, 3.0, 4.0
    spatial_weight, global_weight = 0.002, 0.1
    module = SimpleNamespace(
        _last_loss_components={
            "progress_p": progress,
            "task_loss": torch.tensor(task),
            "spatial_loss_raw": torch.tensor(spatial),
            "global_loss_raw": torch.tensor(global_loss),
            "spatial_weight_effective": spatial_weight,
            "global_weight_effective": global_weight,
            "grl_lambda": torch.tensor(grl),
            "total_loss": torch.tensor(task + spatial_weight * spatial + global_weight * global_loss),
        }
    )
    trainer = SimpleNamespace(current_epoch=2, global_step=9, is_global_zero=True)
    callback = LossComponentCallback(tmp_path)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    row = json.loads((tmp_path / "loss_components.jsonl").read_text())

    assert row["grl_lambda"] == pytest.approx(grl)
    assert row["total_loss"] == pytest.approx(
        row["task_loss"]
        + row["spatial_weight_effective"] * row["spatial_loss_raw"]
        + row["global_weight_effective"] * row["global_loss_raw"],
        abs=1e-6,
    )
    assert row["grl_lambda"] == pytest.approx(
        {0.0: 0.0, 0.5: 0.0986614298, 1.0: 0.0999909204}[progress], abs=1e-7
    )


def test_loss_component_callback_is_rank_safe(tmp_path):
    from navsim.planning.script.run_training import LossComponentCallback

    module = SimpleNamespace(
        _last_loss_components={
            "progress_p": 0.0,
            "task_loss": 1.0,
            "spatial_loss_raw": 2.0,
            "global_loss_raw": 3.0,
            "spatial_weight_effective": 0.0,
            "global_weight_effective": 0.0,
            "grl_lambda": 0.0,
            "total_loss": 1.0,
        }
    )
    trainer = SimpleNamespace(current_epoch=0, global_step=1, is_global_zero=False)
    callback = LossComponentCallback(tmp_path)

    callback.on_train_batch_end(trainer, module, None, None, 0)

    assert not (tmp_path / "loss_components.jsonl").exists()


def test_topup_aborts_above_one_percent(tmp_path):
    from navsim.planning.script.build_alignment_small_data import deterministic_topup

    candidates = [
        {"split": "train", "log_name": "a", "token": f"t{index:03d}", "selection_hash": f"{index:03d}"}
        for index in range(102)
    ]

    def attempt(record):
        if record["token"] in {"t000", "t001"}:
            raise RuntimeError("render failed")
        return record["token"]

    with pytest.raises(RuntimeError, match="replacement rate"):
        deterministic_topup(
            candidates, candidates[:100], "train", attempt, tmp_path / "topup.jsonl"
        )


def test_atomic_image_interrupt_cleanup(tmp_path):
    from navsim.planning.script.build_alignment_small_data import atomic_write_validated_image

    final = tmp_path / "CAM_B0" / "frame.jpg"
    image = np.full((16, 16, 3), 127, dtype=np.uint8)

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_write_validated_image(final, image, lambda _: (_ for _ in ()).throw(RuntimeError("interrupted")))

    assert not final.exists()
    assert not list(final.parent.glob("*.tmp-*"))


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


def _placeholder_agent_input(rendered_present):
    from navsim.common.dataclasses import Camera, Cameras

    identity = np.eye(3, dtype=np.float32)
    cameras = []
    for index, present in enumerate(rendered_present):
        cameras.append(Camera(
            image=np.full((20, 30, 3), 10 + index, dtype=np.uint8),
            rendered_image=np.full((20, 30, 3), 20 + index, dtype=np.uint8) if present else None,
            sensor2lidar_rotation=identity,
            sensor2lidar_translation=np.zeros(3, dtype=np.float32),
            intrinsics=identity,
            distortion=np.zeros(5, dtype=np.float32),
            real_valid=True,
            rendered_valid=present,
        ))
    return SimpleNamespace(cameras=[Cameras(
        cam_b0=cameras[0], cam_f0=cameras[1], cam_l0=cameras[2], cam_r0=cameras[3],
        cam_l1=Camera(), cam_l2=Camera(), cam_r1=Camera(), cam_r2=Camera(),
    )])


def test_rendered_placeholder_defaults_off_and_all_missing_still_raises():
    from navsim.agents.rap_dino.bevformer.bev_feature_build import _get_bev_feature

    with pytest.raises(ValueError, match="Expected four camera views, got 0"):
        _get_bev_feature(_placeholder_agent_input([False] * 4))


def test_rendered_placeholder_rejects_partial_missing_even_when_enabled():
    from navsim.agents.rap_dino.bevformer.bev_feature_build import _get_bev_feature

    with pytest.raises(ValueError, match="Expected four camera views, got 3"):
        _get_bev_feature(
            _placeholder_agent_input([True, True, True, False]),
            allow_missing_rendered_placeholder=True,
        )


def test_rendered_placeholder_is_zero_keeps_real_validity_and_logs_once(tmp_path, monkeypatch):
    from navsim.agents.rap_dino.bevformer import bev_feature_build

    warnings = []
    monkeypatch.setattr(bev_feature_build.logger, "warning", lambda *args: warnings.append(args))
    log_path = tmp_path / "worker.json"
    agent_input = _placeholder_agent_input([False] * 4)
    first = bev_feature_build._get_bev_feature(
        agent_input,
        allow_missing_rendered_placeholder=True,
        rendered_placeholder_log_path=str(log_path),
    )
    second = bev_feature_build._get_bev_feature(
        agent_input,
        allow_missing_rendered_placeholder=True,
        rendered_placeholder_log_path=str(log_path),
    )

    assert torch.count_nonzero(first["rendered_camera_feature"]) == 0
    assert first["rendered_camera_feature"].shape == first["camera_feature"].shape
    assert bool(first["camera_valid"])
    assert torch.equal(first["camera_valid"], second["camera_valid"])
    assert log_path.is_file()
    assert len(warnings) == 1


def test_rendered_placeholder_switch_is_enabled_only_by_pdm_scoring_config(tmp_path):
    from hydra import compose, initialize_config_dir
    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from navsim.planning.script.run_pdm_score import _configure_rendered_placeholder

    assert RAPConfig().allow_missing_rendered_placeholder is False
    config_dir = Path(__file__).parents[2] / "navsim/planning/script/config/pdm_scoring"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=["agent=rap_agent"])
    assert cfg.allow_missing_rendered_placeholder is True
    assert cfg.agent.config.allow_missing_rendered_placeholder is False

    agent = SimpleNamespace(_config=RAPConfig())
    cfg.output_dir = str(tmp_path)
    _configure_rendered_placeholder(agent, cfg, thread_id="thread", node_id=3)
    assert agent._config.allow_missing_rendered_placeholder is True
    assert agent._config.rendered_placeholder_log_path == str(
        tmp_path / "rendered_placeholder_workers/worker-3-thread.json"
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
