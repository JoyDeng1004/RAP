"""Integrity gates over the frozen Stage-A manifests and paired cache."""

from __future__ import annotations

import gzip
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml


REPO = Path(__file__).resolve().parents[2]
STAGE = Path(os.environ.get("STAGE_A_OUT", REPO / "outputs/alignment_stage_a"))
INPUT = STAGE / "input_data"
PAIRED = STAGE / "paired_cache"
CAMERA_ORDER = ["CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0"]


def _json(name: str):
    path = INPUT / name
    assert path.is_file(), f"missing Stage-A artifact: {path}"
    return json.loads(path.read_text())


def _load_gzip(path: Path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def test_frozen_tokens_logs_and_official_splits_are_isolated():
    manifest = _json("stage_a_token_manifest.json")
    split = yaml.safe_load(
        (REPO / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml").read_text()
    )
    train_tokens, val_tokens = set(manifest["train_tokens"]), set(manifest["val_tokens"])
    train_logs, val_logs = set(manifest["train_logs"]), set(manifest["val_logs"])
    test_logs = set(split["test_logs"])

    assert len(train_tokens) == len(manifest["train_tokens"])
    assert len(val_tokens) == len(manifest["val_tokens"])
    assert not train_tokens & val_tokens
    assert not train_logs & val_logs
    assert train_logs <= set(split["train_logs"])
    assert val_logs <= set(split["val_logs"])
    assert not (train_logs | val_logs) & test_logs
    assert manifest["camera_order"] == CAMERA_ORDER


def test_dataset_manifest_exactly_aligns_with_paired_cache():
    token_manifest = _json("stage_a_token_manifest.json")
    dataset = _json("dataset_manifest.json")
    samples = dataset["samples"]
    wanted = set(token_manifest["train_tokens"]) | set(token_manifest["val_tokens"])
    actual = [sample["token"] for sample in samples]

    assert dataset["camera_order"] == CAMERA_ORDER
    assert len(actual) == len(wanted) == dataset["train_count"] + dataset["val_count"]
    assert len(actual) == len(set(actual))
    assert set(actual) == wanted
    cache_dirs = {path.name for log in PAIRED.iterdir() if log.is_dir() for path in log.iterdir() if path.is_dir()}
    assert cache_dirs == wanted

    for sample in samples:
        token_dir = PAIRED / sample["log_name"] / sample["token"]
        assert Path(sample["feature_cache_path"]) == token_dir / "rap_feature.gz"
        assert Path(sample["target_cache_path"]) == token_dir / "rap_target.gz"
        assert all(path.is_file() for path in (token_dir / "rap_feature.gz", token_dir / "rap_target.gz"))


def test_every_selected_sample_has_four_real_and_raster_cameras():
    samples = _json("dataset_manifest.json")["samples"]
    for sample in samples:
        assert sample["camera_order"] == CAMERA_ORDER
        assert [camera["camera"] for camera in sample["cameras"]] == CAMERA_ORDER
        for camera in sample["cameras"]:
            relative = Path(camera["relative_path"])
            assert relative.parts[:2] == (sample["log_name"], camera["camera"])
            assert Path(camera["real_path"]).is_file()
            assert Path(camera["raster_path"]).is_file()


def test_targets_are_hardlinked_to_frozen_source_and_numerically_valid():
    samples = _json("dataset_manifest.json")["samples"]
    for sample in samples:
        source = Path(sample["target_path"])
        paired = Path(sample["target_cache_path"])
        assert source.stat().st_ino == paired.stat().st_ino
        assert sample["target_sha256"] == sample["target_cache_sha256"]

    # Loading representative records catches serialization, shape, dtype and value errors
    # without inflating the gate by decoding every multi-megabyte feature archive.
    by_split = {split: [sample for sample in samples if sample["split"] == split] for split in ("train", "val")}
    selected = [records[index] for records in by_split.values() for index in (0, len(records) // 2, -1)]
    for sample in selected:
        target = _load_gzip(Path(sample["target_cache_path"]))
        trajectory = torch.as_tensor(target["trajectory"])
        assert target["token"] == sample["token"]
        assert trajectory.shape == (10, 3)
        assert trajectory.dtype in (torch.float32, torch.float64)
        assert torch.isfinite(trajectory).all()


def test_raw_camera_to_cached_tensor_numeric_regression():
    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder
    from navsim.common.dataclasses import AgentInput
    from navsim.planning.script.build_alignment_small_data import _sensor_config

    sample = _json("dataset_manifest.json")["samples"][0]
    log_root = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs/trainval"
    real_root = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "sensor_blobs/trainval"
    raster_root = Path(os.environ["RASTER_4CAM_ROOT"])
    frames = pickle.load(open(log_root / f"{sample['log_name']}.pkl", "rb"))
    indices = [index for index, frame in enumerate(frames) if frame["token"] == sample["token"]]
    assert len(indices) == 1 and indices[0] >= 3
    history = frames[indices[0] - 3 : indices[0] + 1]
    agent_input = AgentInput.from_scene_dict_list(
        history,
        real_root,
        4,
        _sensor_config(),
        rendered_sensor_blobs_path=raster_root,
        strict_camera_loading=True,
    )
    rebuilt = RAPFeatureBuilder(RAPConfig()).compute_features(agent_input)
    cached = _load_gzip(Path(sample["feature_cache_path"]))

    assert bool(cached["camera_valid"])
    for name in ("camera_feature", "rendered_camera_feature", "img_shape", "lidar2img", "ego_status"):
        assert cached[name].shape == rebuilt[name].shape
        assert torch.isfinite(cached[name]).all()
        assert torch.equal(cached[name], rebuilt[name]), name
    assert cached["camera_feature"].shape == (4, 3, 448, 768)
    assert cached["rendered_camera_feature"].shape == (4, 3, 448, 768)


def test_rendered_placeholder_switch_is_bitwise_inert_for_real_train_token():
    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder
    from navsim.common.dataclasses import AgentInput
    from navsim.planning.script.build_alignment_small_data import _sensor_config

    sample = _json("dataset_manifest.json")["samples"][0]
    log_root = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs/trainval"
    real_root = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "sensor_blobs/trainval"
    raster_root = Path(os.environ["RASTER_4CAM_ROOT"])
    frames = pickle.load(open(log_root / f"{sample['log_name']}.pkl", "rb"))
    frame_index = next(index for index, frame in enumerate(frames) if frame["token"] == sample["token"])
    agent_input = AgentInput.from_scene_dict_list(
        frames[frame_index - 3 : frame_index + 1],
        real_root,
        4,
        _sensor_config(),
        rendered_sensor_blobs_path=raster_root,
        strict_camera_loading=True,
    )

    normal = RAPFeatureBuilder(RAPConfig()).compute_features(agent_input)
    enabled = RAPFeatureBuilder(
        RAPConfig(allow_missing_rendered_placeholder=True)
    ).compute_features(agent_input)

    for name in ("camera_feature", "camera_valid", "img_shape", "lidar2img"):
        assert torch.equal(normal[name], enabled[name]), name


def test_b0_manifest_and_input_audit_are_complete():
    dataset = _json("dataset_manifest.json")
    audit = _json("input_audit.json")
    rows = [json.loads(line) for line in (INPUT / "b0_generation_manifest.jsonl").read_text().splitlines()]
    tokens = [row["frame_token"] for row in rows]

    assert audit["status"] == "passed", audit
    assert len(rows) == dataset["train_count"] + dataset["val_count"]
    assert len(tokens) == len(set(tokens))
    assert all(row["camera"] == "CAM_B0" for row in rows)
    assert all(row["f0_mae_0_to_255"] <= 1.0 for row in rows)
    assert all(row["finite_fraction"] == 1.0 and row["nonzero_fraction"] > 0 for row in rows)
