from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, WaymoCacheOnlyDataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from pytorch_lightning.loggers import WandbLogger

import random
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


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
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
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

    # Check explicit resume paths before loading datasets and model weights.
    resume_ckpt_path = cfg.get("resume_ckpt_path")
    if resume_ckpt_path not in (None, "last") and not Path(resume_ckpt_path).is_file():
        raise FileNotFoundError(f"resume_ckpt_path does not exist: {resume_ckpt_path}")
    logger.info("Resume checkpoint: %s", resume_ckpt_path or "none (fresh start)")

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

        val_log_set = set(cfg.val_logs)
        cached_logs = [log_name.name.replace(".pkl", "") for log_name in Path(cfg.cache_path).iterdir()]
        train_logs = [log_name for log_name in cached_logs if log_name not in val_log_set]
        if cfg.get("restrict_train_logs", False):
            allowed = set(cfg.train_logs)
            train_logs = [log_name for log_name in train_logs if log_name in allowed]

        # A separate validation cache keeps val/score comparable across datasets.
        val_cache_path = cfg.get("val_cache_path") or cfg.cache_path
        val_cached_logs = [log_name.name.replace(".pkl", "") for log_name in Path(val_cache_path).iterdir()]
        val_logs = [log_name for log_name in val_cached_logs if log_name in val_log_set]

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
            if not train_logs:
                raise ValueError(
                    f"No training logs in {cfg.cache_path}. With restrict_train_logs=true the "
                    "cache must contain logs listed in train_logs (set it false to train on a "
                    "cache from another dataset, e.g. nuScenes)."
                )
            if not val_logs:
                raise ValueError(
                    f"No validation logs from val_logs found in {val_cache_path}. "
                    "Point val_cache_path at the cache that holds them."
                )

            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=train_logs,
            )
            # False for sources without a metric cache (e.g. nuScenes): rap_loss then
            # skips their scorer losses and down-weights their trajectory loss.
            train_data.score_mask = cfg.get("train_score_mask", True)
            val_data = CacheOnlyDataset(
                cache_path=val_cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=val_logs,
                split='val'
            )
            val_data.score_mask = cfg.get("val_score_mask", True)

            train_parts = [("main", train_data)]
            if cfg.get("cache_path_perturbed"):
                train_data_perturbed = CacheOnlyDataset(
                    cache_path=cfg.cache_path_perturbed,
                    feature_builders=agent.get_feature_builders(),
                    target_builders=agent.get_target_builders())
                N = len(train_data_perturbed)
                indices = random.sample(range(N), int(0.1*N))
                train_parts.append(("perturbed", Subset(train_data_perturbed, indices)))

            if cfg.get("cache_path_others"):
                train_data_others = CacheOnlyDataset(
                    cache_path=cfg.cache_path_others,
                    feature_builders=agent.get_feature_builders(),
                    target_builders=agent.get_target_builders())
                train_data_others.score_mask=False
                N = len(train_data_others)
                indices = random.sample(range(N), int(0.05*N))
                train_parts.append(("others", Subset(train_data_others, indices)))

            # Each extra source supplies cache_path and optional sampling metadata.
            for i, src in enumerate(cfg.get("extra_train_sources") or []):
                extra_logs = [p.name for p in Path(src.cache_path).iterdir() if p.name not in val_log_set]
                extra = CacheOnlyDataset(
                    cache_path=src.cache_path,
                    feature_builders=agent.get_feature_builders(),
                    target_builders=agent.get_target_builders(),
                    log_names=extra_logs,
                )
                extra.score_mask = src.get("score_mask", False)
                extra.rig_id = src.get("rig_id", 0)
                extra.use_real = src.get("use_real", True)
                train_parts.append((src.get("name", f"extra{i}"), extra))

            logger.info("Training sources: %s", ", ".join(f"{name}={len(ds)}" for name, ds in train_parts))
            logger.info("Validation: %d samples from %s", len(val_data), val_cache_path)
            train_data = ConcatDataset([ds for _, ds in train_parts])

    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks(), logger=WandbLogger(project="rap", name=cfg.experiment_name, id=cfg.experiment_name),
            )

    if cfg.get("validate_only", False):
        # The agent loads validation weights from agent.checkpoint_path.
        logger.info("Validate only: %d samples", len(val_data))
        trainer.validate(model=lightning_module, dataloaders=val_dataloader)
        return

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
