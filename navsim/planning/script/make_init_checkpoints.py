#!/usr/bin/env python3
"""Create the three provenance-frozen Stage-A initialization checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import navsim  # noqa: F401 - install the PyTorch/transformers pytree shim first
import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.rap_dino.navsim_config import RAPConfig
from navsim.agents.rap_dino.rap_agent import RAPAgent


BACKBONE_PREFIX = "_rap_model._backbone.img_backbone."


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_sha256(state_dict: dict[str, torch.Tensor], backbone: bool) -> str:
    digest = hashlib.sha256()
    selected = ((name, value) for name, value in state_dict.items() if name.startswith(BACKBONE_PREFIX) == backbone)
    for name, tensor in sorted(selected):
        tensor = tensor.detach().cpu().contiguous()
        for value in (name, str(tensor.dtype), str(tuple(tensor.shape))):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _assert_architecture(config: RAPConfig, reference: dict[str, Any], backbone: torch.nn.Module) -> None:
    actual = {
        "dino_model_name": config.dino_model_name,
        "hidden_size": backbone.config.hidden_size,
        "intermediate_size": backbone.config.intermediate_size,
        "num_hidden_layers": backbone.config.num_hidden_layers,
        "num_attention_heads": backbone.config.num_attention_heads,
        "patch_size": backbone.config.patch_size,
        "num_register_tokens": backbone.config.num_register_tokens,
        "use_gated_mlp": backbone.config.use_gated_mlp,
        "tf_d_model": config.tf_d_model,
        "backbone_param_count": len(backbone.state_dict()),
    }
    for field, value in actual.items():
        if reference[field] != value:
            raise RuntimeError(f"DINO architecture mismatch for {field}: {value!r} != {reference[field]!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"duplicate seeds: {args.seeds}")

    stage = Path(os.environ["STAGE_A_OUT"])
    reference = json.loads((stage / "dino_reference_config.json").read_text())
    pretrained = Path(os.environ["DINO_PRETRAINED_CKPT"])
    if pretrained.resolve() == Path(os.environ["REF_DINO_CKPT"]).resolve():
        raise RuntimeError("the architecture-reference RAP checkpoint is forbidden as initialization")
    if _file_sha256(pretrained) != reference["pretrained_checkpoint_sha256"]:
        raise RuntimeError("Meta DINOv3 checkpoint checksum changed since Phase -1")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError(f"initialization output is non-empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    provenance: dict[str, Any] = {}

    for seed in args.seeds:
        pl.seed_everything(seed, workers=True)
        config = RAPConfig(
            dino_init_from_pretrained=True,
            train_metric_cache_path=str(Path(os.environ["NAVSIM_EXP_ROOT"]) / "train_metric_cache"),
            trajectory_sampling=TrajectorySampling(time_horizon=5, interval_length=0.5),
        )
        agent = RAPAgent(config=config, lr=1e-4, checkpoint_path=None)
        backbone = agent._rap_model._backbone.img_backbone
        _assert_architecture(config, reference, backbone)
        state_dict = agent.state_dict()
        backbone_keys = [name for name in state_dict if name.startswith(BACKBONE_PREFIX)]
        if len(backbone_keys) != reference["backbone_param_count"]:
            raise RuntimeError(f"backbone state has {len(backbone_keys)} keys, expected {reference['backbone_param_count']}")

        output = args.out_dir / f"seed_{seed}.ckpt"
        temporary = output.with_suffix(".ckpt.tmp")
        torch.save({"state_dict": state_dict}, temporary)
        os.replace(temporary, output)
        provenance[f"seed_{seed}"] = {
            "file_sha256": _file_sha256(output),
            "backbone_sha256": _group_sha256(state_dict, backbone=True),
            "head_sha256": _group_sha256(state_dict, backbone=False),
            "source": (
                f"meta_official:{pretrained.name}@sha256:{reference['pretrained_checkpoint_sha256']}; "
                f"architecture_id=hf:{reference['dino_model_name']}; random_head_seed={seed}"
            ),
            "training_data": "none — freshly initialized",
            "training_objective": "none",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
        }
        del state_dict, backbone, agent
        gc.collect()

    (args.out_dir / "sha256.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
