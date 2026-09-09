from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from navsim.planning.script.validate_stage_a_seed import ValidationError, validate


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path):
    stage = tmp_path / "alignment_stage_a"
    noalign = stage / "training" / "r100-noalign" / "seed_0"
    fullalign = stage / "training" / "r100-fullalign" / "seed_0"
    train_tokens = [f"t{index}" for index in range(256)]
    val_tokens = ["v0", "v1"]
    _write_json(stage / "input_data" / "stage_a_token_manifest.json", {"train_tokens": train_tokens, "val_tokens": val_tokens})

    init = stage / "init_checkpoints" / "seed_0.ckpt"
    init.parent.mkdir(parents=True)
    init.write_bytes(b"frozen initialization")
    init_sha = hashlib.sha256(init.read_bytes()).hexdigest()
    _write_json(stage / "init_checkpoints" / "sha256.json", {"seed_0": {"file_sha256": init_sha}})

    checksum_rows = [
        {"epoch": epoch, "stage": split, "num_tokens": len(train_tokens if split == "train" else val_tokens), "sequence_sha256": hashlib.sha256(f"{epoch}/{split}".encode()).hexdigest()}
        for epoch in range(2)
        for split in ("train", "val")
    ]
    steps = 4
    for run_dir, condition, enabled in (
        (noalign, "r100-noalign", False),
        (fullalign, "r100-fullalign", True),
    ):
        config = {
            "alignment_condition_name": condition,
            "experiment_name": f"stage-a-{condition}",
            "experiment_uid": "seed_0",
            "output_dir": str(run_dir),
            "seed": 0,
            "max_train_samples": None,
            "max_val_samples": None,
            "shuffle_train": True,
            "shuffle_val": False,
            "include_auxiliary_datasets": False,
            "agent": {
                "checkpoint_path": str(init),
                "lr": 1e-4,
                "config": {
                    "use_spatial_align": enabled,
                    "use_global_align": enabled,
                    "dino_init_from_pretrained": False,
                    "distill_feature_weight": 0.002,
                    "domain_align_weight": 0.1,
                },
            },
            "dataloader": {"params": {"batch_size": 64}},
            "trainer": {"params": {
                "default_root_dir": str(run_dir),
                "max_epochs": 2,
                "precision": "16-mixed",
                "gradient_clip_val": 0.0,
                "limit_train_batches": 1.0,
                "limit_val_batches": 1.0,
                "devices": 2,
                "num_nodes": 1,
                "accumulate_grad_batches": 1,
            }},
        }
        _write_json(run_dir / "reproducibility_manifest.json", {
            "seed": 0,
            "initialization_checkpoint": str(init),
            "initialization_checkpoint_sha256": init_sha,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "hydra_config": config,
        })
        _write_jsonl(run_dir / "epoch_token_checksums.jsonl", checksum_rows)
        rows = []
        for index in range(steps):
            progress = index / (steps - 1)
            task = 2.0 if index == 0 else 2.0 + index + (0.5 if enabled else 0.0)
            spatial_weight = 0.002 if enabled else 0.0
            global_weight = 0.1 if enabled else 0.0
            spatial_raw, global_raw = 3.0, 4.0
            rows.append({
                "epoch": index // 2,
                "global_step": index + 1,
                "progress_p": progress,
                "task_loss": task,
                "spatial_loss_raw": spatial_raw,
                "global_loss_raw": global_raw,
                "spatial_weight_effective": spatial_weight,
                "global_weight_effective": global_weight,
                "grl_lambda": 0.1 * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0),
                "total_loss": task + spatial_weight * spatial_raw + global_weight * global_raw,
            })
        _write_jsonl(run_dir / "loss_components.jsonl", rows)
        checkpoint = run_dir / "checkpoints" / "last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
    return noalign, fullalign


def test_validate_stage_a_seed_accepts_controlled_pair(tmp_path):
    noalign, fullalign = _fixture(tmp_path)
    result = validate(noalign, fullalign, train_count=256, epochs=2)
    assert result["status"] == "passed"
    assert result["optimizer_steps"] == 4


def test_validate_stage_a_seed_rejects_checksum_divergence(tmp_path):
    noalign, fullalign = _fixture(tmp_path)
    path = fullalign / "epoch_token_checksums.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["sequence_sha256"] = "0" * 64
    _write_jsonl(path, rows)
    with pytest.raises(ValidationError, match="epoch token checksums differ"):
        validate(noalign, fullalign, train_count=256, epochs=2)
