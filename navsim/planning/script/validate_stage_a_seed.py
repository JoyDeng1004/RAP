"""Validate one completed Stage-A noalign/fullalign seed pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TOLERANCE = 1e-6
EXPECTED_CONFIG_DIFFERENCES = {
    "agent.config.use_global_align",
    "agent.config.use_spatial_align",
    "alignment_condition_name",
    "experiment_name",
    "output_dir",
    "trainer.params.default_root_dir",
}
LOSS_KEYS = {
    "epoch",
    "global_step",
    "progress_p",
    "task_loss",
    "spatial_loss_raw",
    "global_loss_raw",
    "spatial_weight_effective",
    "global_weight_effective",
    "grl_lambda",
    "total_loss",
}


class ValidationError(RuntimeError):
    """Raised when a frozen Stage-A invariant does not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing artifact: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(line.strip()), f"blank JSONL line at {path}:{line_number}")
        row = json.loads(line)
        _require(isinstance(row, dict), f"expected object at {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_differences(left: Any, right: Any, path: tuple[str, ...] = ()) -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in set(left) | set(right):
            child = path + (str(key),)
            if key not in left or key not in right:
                differences.add(".".join(child))
            else:
                differences.update(_config_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {".".join(path)}
        differences = set()
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.update(
                _config_differences(left_item, right_item, path + (str(index),))
            )
        return differences
    return set() if left == right else {".".join(path)}


def _validate_config_pair(
    noalign: Path,
    fullalign: Path,
    no_manifest: dict[str, Any],
    full_manifest: dict[str, Any],
    epochs: int,
) -> tuple[int, int]:
    no_cfg = no_manifest["hydra_config"]
    full_cfg = full_manifest["hydra_config"]
    _require(isinstance(no_cfg, dict) and isinstance(full_cfg, dict), "invalid hydra_config")

    differences = _config_differences(no_cfg, full_cfg)
    _require(
        differences == EXPECTED_CONFIG_DIFFERENCES,
        "uncontrolled resolved-config differences: "
        f"got={sorted(differences)} expected={sorted(EXPECTED_CONFIG_DIFFERENCES)}",
    )

    expected = (
        (no_cfg, noalign, "r100-noalign", False),
        (full_cfg, fullalign, "r100-fullalign", True),
    )
    for cfg, run_dir, condition, enabled in expected:
        _require(cfg["alignment_condition_name"] == condition, f"wrong condition: {run_dir}")
        _require(cfg["experiment_name"] == f"stage-a-{condition}", f"wrong experiment name: {run_dir}")
        _require(cfg["experiment_uid"] == f"seed_{cfg['seed']}", f"unfrozen experiment_uid: {run_dir}")
        _require(Path(cfg["output_dir"]).resolve() == run_dir, f"wrong output_dir: {run_dir}")
        _require(
            Path(cfg["trainer"]["params"]["default_root_dir"]).resolve() == run_dir,
            f"wrong default_root_dir: {run_dir}",
        )
        agent_cfg = cfg["agent"]["config"]
        _require(agent_cfg["use_spatial_align"] is enabled, f"wrong spatial switch: {run_dir}")
        _require(agent_cfg["use_global_align"] is enabled, f"wrong global switch: {run_dir}")
        _require(agent_cfg["dino_init_from_pretrained"] is False, f"runtime DINO init enabled: {run_dir}")
        _require(float(agent_cfg["distill_feature_weight"]) == 0.002, f"wrong spatial weight: {run_dir}")
        _require(float(agent_cfg["domain_align_weight"]) == 0.1, f"wrong global weight: {run_dir}")
        trainer = cfg["trainer"]["params"]
        _require(int(trainer["max_epochs"]) == epochs, f"wrong epoch budget: {run_dir}")
        _require(trainer["precision"] == "16-mixed", f"wrong precision: {run_dir}")
        _require(float(trainer["gradient_clip_val"]) == 0.0, f"wrong gradient clipping: {run_dir}")
        _require(float(trainer["limit_train_batches"]) == 1.0, f"train batches limited: {run_dir}")
        _require(float(trainer["limit_val_batches"]) == 1.0, f"val batches limited: {run_dir}")
        _require(cfg["max_train_samples"] is None and cfg["max_val_samples"] is None, f"samples limited: {run_dir}")
        _require(cfg["shuffle_train"] is True and cfg["shuffle_val"] is False, f"wrong shuffle: {run_dir}")
        _require(cfg["include_auxiliary_datasets"] is False, f"auxiliary data enabled: {run_dir}")
        _require(float(cfg["agent"]["lr"]) == 1e-4, f"wrong lr: {run_dir}")
        _require(int(cfg["seed"]) == int(no_manifest["seed"]), f"seed mismatch: {run_dir}")

    trainer = no_cfg["trainer"]["params"]
    devices = int(trainer["devices"]) * int(trainer["num_nodes"])
    batch_size = int(no_cfg["dataloader"]["params"]["batch_size"])
    accumulation = int(trainer["accumulate_grad_batches"])
    effective_batch = devices * batch_size * accumulation
    _require(effective_batch == 128, f"effective global batch is {effective_batch}, expected 128")
    return effective_batch, int(no_cfg["seed"])


def _validate_initialization(
    stage_root: Path,
    no_manifest: dict[str, Any],
    full_manifest: dict[str, Any],
    seed: int,
) -> str:
    expected_path = (stage_root / "init_checkpoints" / f"seed_{seed}.ckpt").resolve()
    paths = [Path(manifest["initialization_checkpoint"]).resolve() for manifest in (no_manifest, full_manifest)]
    _require(paths == [expected_path, expected_path], f"wrong initialization paths: {paths}")
    _require("RAP_DINO_navsimv" not in str(expected_path), f"forbidden initialization: {expected_path}")
    _require(expected_path.is_file(), f"missing initialization checkpoint: {expected_path}")

    recorded = [manifest["initialization_checkpoint_sha256"] for manifest in (no_manifest, full_manifest)]
    expected_metadata = _load_json(stage_root / "init_checkpoints" / "sha256.json")
    expected_sha = expected_metadata[f"seed_{seed}"]["file_sha256"]
    _require(recorded == [expected_sha, expected_sha], f"manifest initialization SHA mismatch: {recorded}")
    actual_sha = _sha256(expected_path)
    _require(actual_sha == expected_sha, f"checkpoint SHA mismatch: {actual_sha} != {expected_sha}")
    return actual_sha


def _validate_tokens_and_epoch_checksums(
    stage_root: Path,
    noalign: Path,
    fullalign: Path,
    no_manifest: dict[str, Any],
    full_manifest: dict[str, Any],
    train_count: int,
    epochs: int,
) -> int:
    frozen = _load_json(stage_root / "input_data" / "stage_a_token_manifest.json")
    frozen_train = frozen["train_tokens"]
    frozen_val = frozen["val_tokens"]
    _require(len(frozen_train) == train_count, f"frozen train count is {len(frozen_train)}, expected {train_count}")

    for split, frozen_tokens in (("train", frozen_train), ("val", frozen_val)):
        no_tokens = no_manifest[f"{split}_tokens"]
        full_tokens = full_manifest[f"{split}_tokens"]
        _require(len(no_tokens) == len(set(no_tokens)), f"duplicate {split} token in noalign")
        _require(len(full_tokens) == len(set(full_tokens)), f"duplicate {split} token in fullalign")
        _require(set(no_tokens) == set(frozen_tokens), f"noalign {split} tokens differ from frozen manifest")
        _require(set(full_tokens) == set(frozen_tokens), f"fullalign {split} tokens differ from frozen manifest")
        _require(no_tokens == full_tokens, f"condition {split} token order differs")

    no_rows = _load_jsonl(noalign / "epoch_token_checksums.jsonl")
    full_rows = _load_jsonl(fullalign / "epoch_token_checksums.jsonl")
    _require(no_rows == full_rows, "condition epoch token checksums differ")
    _require(len(no_rows) == 2 * epochs, f"expected {2 * epochs} epoch checksum rows, got {len(no_rows)}")
    expected_pairs = {(epoch, stage) for epoch in range(epochs) for stage in ("train", "val")}
    actual_pairs = {(int(row["epoch"]), row["stage"]) for row in no_rows}
    _require(actual_pairs == expected_pairs, "epoch/stage checksum coverage is incomplete")
    for row in no_rows:
        expected_count = train_count if row["stage"] == "train" else len(frozen_val)
        _require(int(row["num_tokens"]) == expected_count, f"wrong token count in checksum row: {row}")
        digest = row["sequence_sha256"]
        _require(isinstance(digest, str) and len(digest) == 64, f"invalid checksum: {row}")
    return len(frozen_val)


def _validate_loss_rows(
    noalign: Path,
    fullalign: Path,
    train_count: int,
    epochs: int,
    effective_batch: int,
) -> int:
    no_rows = _load_jsonl(noalign / "loss_components.jsonl")
    full_rows = _load_jsonl(fullalign / "loss_components.jsonl")
    expected_steps = math.ceil(train_count / effective_batch) * epochs
    _require(len(no_rows) == len(full_rows) == expected_steps, f"loss row count mismatch: no={len(no_rows)} full={len(full_rows)} expected={expected_steps}")

    expected_global_steps = list(range(1, expected_steps + 1))
    for label, rows, spatial_weight, global_weight in (
        ("noalign", no_rows, 0.0, 0.0),
        ("fullalign", full_rows, 0.002, 0.1),
    ):
        _require([int(row["global_step"]) for row in rows] == expected_global_steps, f"{label} global steps are incomplete")
        progress_values = []
        grl_values = []
        for row in rows:
            _require(LOSS_KEYS <= set(row), f"{label} loss row missing keys: {row}")
            numeric = [float(row[key]) for key in LOSS_KEYS - {"epoch", "global_step"}]
            _require(all(math.isfinite(value) for value in numeric), f"{label} has non-finite loss row: {row}")
            _require(abs(float(row["spatial_weight_effective"]) - spatial_weight) <= TOLERANCE, f"{label} spatial weight mismatch")
            _require(abs(float(row["global_weight_effective"]) - global_weight) <= TOLERANCE, f"{label} global weight mismatch")
            reconstructed = (
                float(row["task_loss"])
                + float(row["spatial_weight_effective"]) * float(row["spatial_loss_raw"])
                + float(row["global_weight_effective"]) * float(row["global_loss_raw"])
            )
            _require(abs(float(row["total_loss"]) - reconstructed) <= TOLERANCE, f"{label} total loss reconstruction failed")
            if label == "noalign":
                _require(abs(float(row["total_loss"]) - float(row["task_loss"])) <= TOLERANCE, "NoAlign total_loss != task_loss")
            progress = float(row["progress_p"])
            expected_grl = 0.1 * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            _require(abs(float(row["grl_lambda"]) - expected_grl) <= TOLERANCE, f"{label} GRL schedule mismatch")
            progress_values.append(progress)
            grl_values.append(float(row["grl_lambda"]))
        _require(all(left <= right for left, right in zip(progress_values, progress_values[1:])), f"{label} progress is not monotonic")
        _require(all(left <= right + TOLERANCE for left, right in zip(grl_values, grl_values[1:])), f"{label} GRL is not monotonic")
        _require(abs(progress_values[0]) <= TOLERANCE, f"{label} first progress is not zero")
        _require(abs(progress_values[-1] - 1.0) <= TOLERANCE, f"{label} final progress is not one")

    _require(abs(float(no_rows[0]["task_loss"]) - float(full_rows[0]["task_loss"])) <= TOLERANCE, "first optimizer-step task_loss differs between conditions")
    return expected_steps


def _validate_checkpoints(run_dir: Path) -> None:
    expected = (run_dir / "checkpoints" / "last.ckpt").resolve()
    checkpoints = sorted(path.resolve() for path in run_dir.rglob("*.ckpt"))
    _require(checkpoints == [expected], f"expected only {expected}, got {checkpoints}")


def validate(noalign: Path, fullalign: Path, train_count: int, epochs: int) -> dict[str, Any]:
    noalign = noalign.resolve()
    fullalign = fullalign.resolve()
    _require(noalign.is_dir() and fullalign.is_dir(), "both run directories must exist")
    stage_root = noalign.parents[2]
    _require(fullalign.parents[2] == stage_root, "condition runs are under different Stage-A roots")

    no_manifest = _load_json(noalign / "reproducibility_manifest.json")
    full_manifest = _load_json(fullalign / "reproducibility_manifest.json")
    _require(int(no_manifest["seed"]) == int(full_manifest["seed"]), "condition seeds differ")
    effective_batch, seed = _validate_config_pair(noalign, fullalign, no_manifest, full_manifest, epochs)
    init_sha = _validate_initialization(stage_root, no_manifest, full_manifest, seed)
    val_count = _validate_tokens_and_epoch_checksums(stage_root, noalign, fullalign, no_manifest, full_manifest, train_count, epochs)
    optimizer_steps = _validate_loss_rows(noalign, fullalign, train_count, epochs, effective_batch)
    _validate_checkpoints(noalign)
    _validate_checkpoints(fullalign)
    return {
        "effective_global_batch": effective_batch,
        "epochs": epochs,
        "initialization_checkpoint_sha256": init_sha,
        "optimizer_steps": optimizer_steps,
        "seed": seed,
        "status": "passed",
        "train_tokens": train_count,
        "val_tokens": val_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noalign", required=True, type=Path)
    parser.add_argument("--fullalign", required=True, type=Path)
    parser.add_argument("--train-count", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(validate(args.noalign, args.fullalign, args.train_count, args.epochs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
