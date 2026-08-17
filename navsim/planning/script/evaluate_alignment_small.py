"""Evaluate a trained small alignment run on the deterministic real subset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import navsim  # noqa: F401 - applies the transformers / PyTorch compatibility shim
from navsim.planning.script.run_training import _limit_dataset, _tokens_from_manifest
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset


def _latest_last_checkpoint(output_dir: Path) -> Path:
    candidates = list(output_dir.glob("csv_logs/version_*/checkpoints/last.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No last.ckpt found under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _scalar(loss_dict, key):
    value = loss_dict.get(key)
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


@hydra.main(config_path="config/training", config_name="alignment_small", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    checkpoint_path = _latest_last_checkpoint(output_dir)

    # The trained Lightning checkpoint is the sole model initialization here.
    # Avoid loading the original 4 GB initialization checkpoint first.
    cfg.agent.checkpoint_path = None
    agent = instantiate(cfg.agent)
    module = AgentLightningModule(agent)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint

    dataset = CacheOnlyDataset(
        cache_path=cfg.cache_path,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.val_logs,
        split="val",
    )
    allowed_tokens = _tokens_from_manifest(cfg.subset_token_manifest_path, "val")
    dataset = _limit_dataset(
        dataset,
        cfg.max_val_samples,
        require_camera_valid=True,
        allowed_tokens=allowed_tokens,
    )
    base_dataset = dataset.dataset
    token_to_log = {
        token: path.parent.name
        for token, path in base_dataset._valid_cache_paths.items()
    }
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
    )

    device = torch.device("cuda")
    module.to(device).eval()
    module.agent._rap_model.progress = 1.0
    rows = []
    with torch.inference_mode():
        for features, targets in dataloader:
            features = _move_to_device(features, device)
            targets = _move_to_device(targets, device)
            valid = features["camera_valid"].bool()
            if not bool(valid.all()):
                raise RuntimeError("The paired evaluation subset contains an invalid camera")
            module.agent._rap_model.batch_size = int(valid.sum())
            real_features = {
                key: value[valid]
                for key, value in features.items()
                if key not in {"camera_valid", "rendered_camera_feature"}
            }
            prediction = module.agent.forward(real_features, targets)
            loss_dict = module.agent.compute_loss(real_features, targets, prediction)
            predicted = prediction["trajectory"][:, :, :2]
            target = targets["trajectory"][:, :, :2]
            point_errors = torch.linalg.norm(predicted - target, dim=-1)
            token = targets["token"][0]
            rows.append(
                {
                    "condition": cfg.alignment_condition_name,
                    "seed": int(cfg.seed),
                    "token": token,
                    "log_name": token_to_log[token],
                    "ade_real": float(point_errors.mean().cpu()),
                    "fde_real": float(point_errors[:, -1].mean().cpu()),
                    "score": _scalar(loss_dict, "score"),
                    "best_score": _scalar(loss_dict, "best_score"),
                    "loss": _scalar(loss_dict, "loss"),
                    "trajectory_loss": _scalar(loss_dict, "trajectory_loss"),
                    "sub_score_loss": _scalar(loss_dict, "sub_score_loss"),
                    "final_score_loss": _scalar(loss_dict, "final_score_loss"),
                    "prediction_xy": json.dumps(predicted[0].float().cpu().tolist()),
                    "target_xy": json.dumps(target[0].float().cpu().tolist()),
                }
            )

    csv_path = output_dir / "per_token_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} paired real samples to {csv_path}")


if __name__ == "__main__":
    main()
