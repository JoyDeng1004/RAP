from typing import Tuple
from pathlib import Path
import logging
import json
import subprocess
import hashlib
import math
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import navsim  # applies the PyTorch 2.1 / transformers pytree compatibility shim
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, WaymoCacheOnlyDataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from omegaconf import OmegaConf
from navsim.planning.training.samplers import ExactDistributedSampler

import random
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _json_number(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


class EpochTokenChecksumCallback(Callback):
    """Record the frozen global train and validation token orders per epoch."""

    def __init__(self, output_dir, train_sampler, val_sampler):
        self.path = Path(output_dir) / "epoch_token_checksums.jsonl"
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        self.train_sampler.set_epoch(epoch)
        self.val_sampler.set_epoch(epoch)
        if not trainer.is_global_zero:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for stage, sampler in (("train", self.train_sampler), ("val", self.val_sampler)):
                tokens = sampler.global_token_order(epoch)
                digest = hashlib.sha256()
                for token in tokens:
                    encoded = token.encode("utf-8")
                    digest.update(len(encoded).to_bytes(4, "big"))
                    digest.update(encoded)
                row = {
                    "epoch": epoch,
                    "stage": stage,
                    "num_tokens": len(tokens),
                    "sequence_sha256": digest.hexdigest(),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")


class LossComponentCallback(Callback):
    """Persist each optimizer step's live loss weights and GRL coefficient."""

    def __init__(self, output_dir):
        self.path = Path(output_dir) / "loss_components.jsonl"

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero or not hasattr(pl_module, "_last_loss_components"):
            return
        values = pl_module._last_loss_components
        row = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            **{key: _json_number(value) for key, value in values.items()},
        }
        reconstructed = (
            row["task_loss"]
            + row["spatial_weight_effective"] * row["spatial_loss_raw"]
            + row["global_weight_effective"] * row["global_loss_raw"]
        )
        if abs(row["total_loss"] - reconstructed) > 1e-6:
            raise RuntimeError(
                f"loss component reconstruction failed: {row['total_loss']} != {reconstructed}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _distributed_topology(cfg):
    devices = int(cfg.trainer.params.devices)
    nodes = int(cfg.trainer.params.num_nodes)
    world_size = devices * nodes
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    return world_size, node_rank * devices + local_rank


def _assert_equal_batch_counts(dataset_size, world_size, batch_size, label):
    shard_sizes = [max(math.ceil((dataset_size - rank) / world_size), 0) for rank in range(world_size)]
    batch_counts = [math.ceil(size / batch_size) for size in shard_sizes]
    if len(set(batch_counts)) != 1:
        raise RuntimeError(
            f"{label} exact shards have unequal batch counts: sizes={shard_sizes}, batches={batch_counts}"
        )


def _build_training_callbacks(cfg, agent, train_sampler=None, val_sampler=None):
    """Build Stage-A callbacks from the run-level output directory."""
    callbacks = list(agent.get_training_callbacks())
    if bool(getattr(cfg, "final_step_checkpoint_only", False)):
        callbacks = [callback for callback in callbacks if not isinstance(callback, ModelCheckpoint)]
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(cfg.output_dir) / "checkpoints",
                save_last=True,
                save_top_k=0,
            )
        )
    if train_sampler is not None:
        if val_sampler is None:
            raise ValueError("val_sampler is required when train_sampler is provided")
        callbacks.extend(
            [
                EpochTokenChecksumCallback(cfg.output_dir, train_sampler, val_sampler),
                LossComponentCallback(cfg.output_dir),
            ]
        )
    return callbacks


def _limit_dataset(
    dataset, max_samples, require_camera_valid=False, allowed_tokens=None
):
    """Take a deterministic subset from a token-sorted dataset.

    Alignment experiments require paired real/raster samples. When requested,
    skip cache entries whose real camera is unavailable instead of silently
    building a subset that cannot produce real-domain metrics.
    """
    if max_samples is None and not require_camera_valid and allowed_tokens is None:
        return dataset

    dataset_tokens = _dataset_tokens(dataset)
    selected_indices = []
    for index in range(len(dataset)):
        if allowed_tokens is not None and dataset_tokens[index] not in allowed_tokens:
            continue
        if require_camera_valid:
            features, _ = dataset[index]
            if not bool(features["camera_valid"]):
                continue
        selected_indices.append(index)
        if max_samples is not None and len(selected_indices) >= int(max_samples):
            break

    if (require_camera_valid or allowed_tokens is not None) and not selected_indices:
        raise RuntimeError(
            "No samples matched the token whitelist and camera_valid requirements"
        )
    return Subset(dataset, selected_indices)


def _tokens_from_cache(cache_path):
    """Read a token whitelist from an existing RAP feature cache tree."""
    if cache_path is None:
        return None
    root = Path(cache_path)
    tokens = {path.parent.name for path in root.glob("*/*/rap_feature.gz")}
    if not tokens:
        raise RuntimeError(f"No RAP feature tokens found under {root}")
    return tokens


def _tokens_from_manifest(path, split):
    if path is None:
        return None
    with Path(path).open(encoding="utf-8") as file:
        manifest = json.load(file)
    key = f"{split}_tokens"
    if key not in manifest:
        raise KeyError(f"Missing {key!r} in token manifest {path}")
    return set(manifest[key])


def _require_passed_input_audit(path):
    if path is None:
        return
    with Path(path).open(encoding="utf-8") as file:
        audit = json.load(file)
    if audit.get("status") != "passed":
        raise RuntimeError(f"Input audit did not pass: {path}")


def _sha256(path):
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_tokens(dataset):
    """Resolve token identifiers through Subset/ConcatDataset wrappers."""
    if isinstance(dataset, Subset):
        parent_tokens = _dataset_tokens(dataset.dataset)
        return [parent_tokens[index] for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [token for child in dataset.datasets for token in _dataset_tokens(child)]
    return [str(token) for token in getattr(dataset, "tokens", [])]


def _write_reproducibility_manifest(cfg, train_data, val_data) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    manifest = {
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": int(cfg.seed),
        "initialization_checkpoint": str(cfg.agent.checkpoint_path),
        "initialization_checkpoint_sha256": _sha256(cfg.agent.checkpoint_path),
        "train_tokens": _dataset_tokens(train_data),
        "val_tokens": _dataset_tokens(val_data),
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
    }
    with (output_dir / "reproducibility_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        rendered_sensor_blobs_path=(
            Path(cfg.rendered_sensor_blobs_path)
            if getattr(cfg, "rendered_sensor_blobs_path", None) is not None else None
        ),
        strict_camera_loading=bool(getattr(cfg, "strict_camera_loading", False)),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        rendered_sensor_blobs_path=(
            Path(cfg.rendered_sensor_blobs_path)
            if getattr(cfg, "rendered_sensor_blobs_path", None) is not None else None
        ),
        strict_camera_loading=bool(getattr(cfg, "strict_camera_loading", False)),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    _require_passed_input_audit(getattr(cfg, "input_audit_path", None))

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"

        cached_logs = [log_name.name.replace(".pkl", "") for log_name in Path(cfg.cache_path).iterdir()]
        train_logs = [log_name for log_name in cached_logs if log_name not in cfg.val_logs]
        val_logs = [log_name for log_name in cached_logs if log_name in cfg.val_logs]

        if 'waymo' in cfg.dataset['_target_']:
            train_data = WaymoCacheOnlyDataset(
                cache_path=cfg.cache_path,
                split='training'
            )
            val_data = WaymoCacheOnlyDataset(
                cache_path=cfg.cache_path,
                split='val',
            )
            # # split val_data by 80/20
            # import random
            # from torch.utils.data import ConcatDataset, Subset
            # N = len(val_data)
            # indices = random.sample(range(N), int(0.8*N))
            # the_rest = [i for i in range(N) if i not in indices]
            # train_data = Subset(val_data, indices)
            # val_data = Subset(val_data, the_rest)
        else:
            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
            log_names=train_logs,
        )
            val_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=val_logs,
                split='val'
            )

            if cfg.include_auxiliary_datasets:
                train_data_perturbed = CacheOnlyDataset(
                    cache_path=cfg.cache_path_perturbed,
                    feature_builders=agent.get_feature_builders(),
                    target_builders=agent.get_target_builders())
                N = len(train_data_perturbed)
                indices = random.sample(range(N), int(0.1*N))
                print(f'len(perturbed): {len(indices)}')
                train_data_perturbed = Subset(train_data_perturbed, indices)

                train_data_others = CacheOnlyDataset(
                    cache_path=cfg.cache_path_others,
                    feature_builders=agent.get_feature_builders(),
                    target_builders=agent.get_target_builders())

                train_data_others.score_mask=False
                N = len(train_data_others)
                indices = random.sample(range(N), int(0.05*N))
                print(f'len(others): {len(indices)}')
                train_data_others = Subset(train_data_others, indices)

                train_data = ConcatDataset([train_data, train_data_perturbed, train_data_others])

    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    require_camera_valid = bool(getattr(cfg, "require_camera_valid_subset", False))
    cache_tokens = _tokens_from_cache(getattr(cfg, "subset_token_cache_path", None))
    train_manifest_tokens = _tokens_from_manifest(
        getattr(cfg, "subset_token_manifest_path", None), "train"
    )
    val_manifest_tokens = _tokens_from_manifest(
        getattr(cfg, "subset_token_manifest_path", None), "val"
    )
    train_allowed_tokens = train_manifest_tokens or cache_tokens
    val_allowed_tokens = val_manifest_tokens or cache_tokens
    train_data = _limit_dataset(
        train_data,
        cfg.max_train_samples,
        require_camera_valid=require_camera_valid,
        allowed_tokens=train_allowed_tokens,
    )
    val_data = _limit_dataset(
        val_data,
        cfg.max_val_samples,
        require_camera_valid=require_camera_valid,
        allowed_tokens=val_allowed_tokens,
    )
    _write_reproducibility_manifest(cfg, train_data, val_data)

    logger.info("Building Datasets")
    train_sampler = val_sampler = None
    if bool(getattr(cfg, "exact_distributed_coverage", False)):
        world_size, rank = _distributed_topology(cfg)
        batch_size = int(cfg.dataloader.params.batch_size)
        _assert_equal_batch_counts(len(train_data), world_size, batch_size, "train")
        _assert_equal_batch_counts(len(val_data), world_size, batch_size, "val")
        train_sampler = ExactDistributedSampler(
            train_data, world_size, rank, seed=int(cfg.seed), shuffle=bool(cfg.shuffle_train),
            tokens=_dataset_tokens(train_data),
        )
        val_sampler = ExactDistributedSampler(
            val_data, world_size, rank, seed=int(cfg.seed), shuffle=bool(cfg.shuffle_val),
            tokens=_dataset_tokens(val_data),
        )
    train_dataloader = DataLoader(
        train_data, **cfg.dataloader.params,
        shuffle=False if train_sampler is not None else bool(cfg.shuffle_train),
        sampler=train_sampler,
    )
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(
        val_data, **cfg.dataloader.params, shuffle=False, sampler=val_sampler
    )
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    experiment_loggers = [CSVLogger(save_dir=cfg.output_dir, name="csv_logs")]
    if cfg.use_wandb_logger:
        experiment_loggers.append(
            WandbLogger(project="rap", name=cfg.experiment_name, id=cfg.experiment_name)
        )
    callbacks = _build_training_callbacks(cfg, agent, train_sampler, val_sampler)
    trainer = pl.Trainer(
        **cfg.trainer.params,
        callbacks=callbacks,
        logger=experiment_loggers,
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.resume_training_checkpoint,
    )


if __name__ == "__main__":
    main()
