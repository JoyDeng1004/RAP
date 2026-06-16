import copy

import torch
import torch.nn as nn

from navsim.agents.rap_dino.navsim_config import RAPConfig
from navsim.agents.rap_dino.rap_agent import RAPAgent
from navsim.planning.training.agent_lightning_module import AgentLightningModule


def _recovery_config():
    config = RAPConfig(
        pdm_scorer=False,
        recovery_target_enabled=True,
        ref2d_observation_aug=True,
        sub_score_weight=0,
        final_score_weight=0,
        pred_ce_weight=0,
        pred_l1_weight=0,
        pred_area_weight=0,
        agent_class_weight=0,
        agent_box_weight=0,
        bev_semantic_weight=0,
    )
    return config


def _straight_targets(batch_size=1, steps=8):
    x = torch.arange(1, steps + 1, dtype=torch.float32)[None, :, None] * 2.0
    y = torch.zeros(batch_size, steps, 1)
    h = torch.zeros(batch_size, steps, 1)
    return {
        "trajectory": torch.cat([x.repeat(batch_size, 1, 1), y, h], dim=-1),
        "score_mask": torch.zeros(batch_size, dtype=torch.bool),
    }


def test_prepare_recovery_batch_sets_shift_score_mask_and_changes_target():
    module = AgentLightningModule.__new__(AgentLightningModule)
    module.agent = type("Agent", (), {"_config": _recovery_config()})()
    features = {"camera_valid": torch.ones(1, dtype=torch.bool)}
    targets = _straight_targets()
    original = targets["trajectory"].clone()

    out_features, out_targets = module._prepare_recovery_batch(features, targets)

    assert "ref2d_aug_shift_y" in out_features
    assert torch.all(out_targets["score_mask"])
    assert not torch.allclose(out_targets["trajectory"], original)


def test_recovery_rap_loss_is_trajectory_loss_without_auxiliary_targets():
    config = _recovery_config()
    agent = RAPAgent.__new__(RAPAgent)
    agent._config = config
    agent.bce_logit_loss = nn.BCEWithLogitsLoss()

    targets = _straight_targets()
    targets["score_mask"] = torch.ones(1, dtype=torch.bool)
    proposals = targets["trajectory"][:, None].repeat(1, 2, 1, 1)
    proposals[:, 1, :, 1] = 1.0
    pred = {
        "proposals": proposals,
        "proposal_list": [proposals],
        "pred_logit": torch.zeros(1, 2, 1),
        "pred_logit2": None,
        "agent_states": torch.zeros(1, 1),
        "bev_semantic_map": torch.zeros(1, 2, 4, 4),
    }

    loss_dict = agent.compute_loss({}, targets, pred)

    torch.testing.assert_close(loss_dict["loss"], loss_dict["trajectory_loss"])
    torch.testing.assert_close(loss_dict["final_score_loss"], torch.zeros(()))
    torch.testing.assert_close(loss_dict["agent_class_loss"] if "agent_class_loss" in loss_dict else torch.zeros(()), torch.zeros(()))


def test_score_mask_false_downweights_trajectory_loss_by_ten():
    config = _recovery_config()
    agent = RAPAgent.__new__(RAPAgent)
    agent._config = config
    agent.bce_logit_loss = nn.BCEWithLogitsLoss()

    targets_true = _straight_targets()
    targets_true["score_mask"] = torch.ones(1, dtype=torch.bool)
    targets_false = copy.deepcopy(targets_true)
    targets_false["score_mask"] = torch.zeros(1, dtype=torch.bool)
    proposals = torch.zeros(1, 2, 8, 3)
    pred = {
        "proposals": proposals,
        "proposal_list": [proposals],
        "pred_logit": torch.zeros(1, 2, 1),
        "pred_logit2": None,
        "agent_states": None,
        "bev_semantic_map": None,
    }

    full = agent.compute_loss({}, targets_true, pred)["trajectory_loss"]
    downweighted = agent.compute_loss({}, targets_false, pred)["trajectory_loss"]

    torch.testing.assert_close(full, downweighted * 10.0)


def test_baseline_and_offset_recovery_targets_differ():
    module = AgentLightningModule.__new__(AgentLightningModule)
    module.agent = type("Agent", (), {"_config": _recovery_config()})()
    targets = _straight_targets()

    _, recovery_targets = module._prepare_recovery_batch({}, targets)

    assert not torch.allclose(recovery_targets["trajectory"], targets["trajectory"])
