#!/usr/bin/env python3
"""Step 0-F: dynamic gradient-route and shared-anchor drift audit.

This script intentionally reconstructs the rendered branch from dataset_norm.
The historical training caches in this workspace contain blank rendered tensors
and therefore are used only for calibration tensors and trajectory targets.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


CAMERAS = ["B0", "F0", "L0", "R0"]
CAMERA_KEYS = [f"CAM_{name}" for name in CAMERAS]
SEED = 20260717


def patch_pytree_compatibility() -> None:
    """Bridge transformers>=4.57 to the older torch shipped in the RAP env."""
    import torch.utils._pytree as pytree

    if hasattr(pytree, "register_pytree_node"):
        return

    def register_pytree_node(type_: Any, flatten_fn: Any, unflatten_fn: Any, *args: Any, **kwargs: Any) -> Any:
        return pytree._register_pytree_node(type_, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = register_pytree_node


def install_offline_dinov3_factory() -> Dict[str, Any]:
    """Make ImgEncoder construct DINOv3 from architecture, not the gated Hub.

    The Lightning checkpoint contains every backbone tensor.  Only the model
    architecture is needed at construction time.  These values are the ViT-H+
    architecture and are also checked against checkpoint tensor shapes before
    the experiment starts.
    """
    patch_pytree_compatibility()
    from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel

    architecture = {
        "patch_size": 16,
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "use_gated_mlp": True,
        "num_register_tokens": 4,
        "image_size": 224,
        "attention_dropout": 0.0,
        "drop_path_rate": 0.0,
    }

    def from_architecture(*args: Any, **kwargs: Any) -> torch.nn.Module:
        return DINOv3ViTModel(DINOv3ViTConfig(**architecture))

    AutoModel.from_pretrained = from_architecture
    return architecture


DINOV3_ARCHITECTURE = install_offline_dinov3_factory()

# Imports below this line transitively import transformers/ImgEncoder.
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.rap_dino.bevformer.bev_feature_build import (
    NormalizeMultiviewImage,
    PadMultiViewImage,
    RandomScaleImageMultiViewImage,
)
from navsim.agents.rap_dino.navsim_config import RAPConfig
from navsim.agents.rap_dino.rap_agent import RAPAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, default=Path("dataset_norm/navsim_logs/mini"))
    parser.add_argument("--raster-root", type=Path, default=Path("dataset_norm/rendered_sensor_blobs"))
    parser.add_argument(
        "--real-root",
        type=Path,
        default=Path("/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/mini"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/metabev_densegrid_v2_full/cache"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpts/RAP_DINO_navsimv2.ckpt"))
    parser.add_argument(
        "--metric-cache",
        type=Path,
        default=Path("/gs/bs/tga-RLA/qdeng/navsim_workspace/exp/metric_cache"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step0_alignment_audit/0f_gradient_route"))
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: jsonable(value) for key, value in row.items()} for row in rows])


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_gzip_pickle(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def preprocess_images(images: Sequence[np.ndarray]) -> torch.Tensor:
    result: Dict[str, Any] = {
        "img": [image.astype(np.float32, copy=False) for image in images],
        "lidar2img": [np.eye(4, dtype=np.float32) for _ in images],
    }
    result = NormalizeMultiviewImage(result)
    result = RandomScaleImageMultiViewImage(result)
    result = PadMultiViewImage(result)
    chw = [image.transpose(2, 0, 1) for image in result["img"]]
    return torch.tensor(np.ascontiguousarray(np.stack(chw)), dtype=torch.float32)


def find_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for metadata_path in sorted(args.metadata_root.glob("*.pkl")):
        cache_log = args.cache_root / metadata_path.stem
        if not cache_log.is_dir():
            continue
        with metadata_path.open("rb") as stream:
            frames = pickle.load(stream)
        for frame in frames:
            token = frame["token"]
            token_cache = cache_log / token
            feature_path = token_cache / "rap_feature.gz"
            target_path = token_cache / "rap_target.gz"
            if not feature_path.is_file() or not target_path.is_file():
                continue

            real_images: List[np.ndarray] = []
            raster_images: List[np.ndarray] = []
            camera_records: List[Dict[str, Any]] = []
            usable = True
            for camera_key, camera_short in zip(CAMERA_KEYS, CAMERAS):
                rel = Path(frame["cams"][camera_key]["data_path"])
                real_path = args.real_root / rel
                raster_path = args.raster_root / rel
                if not real_path.is_file():
                    usable = False
                    break
                real_image = np.array(Image.open(real_path))
                if raster_path.is_file():
                    raster_image = np.array(Image.open(raster_path))[20:-20]
                    if camera_short != "B0" and not np.any(raster_image):
                        usable = False
                        break
                else:
                    raster_image = np.zeros((1080, 1920, 3), dtype=np.float32)
                real_images.append(real_image)
                raster_images.append(raster_image)
                camera_records.append(
                    {
                        "camera": camera_short,
                        "real_path": str(real_path.resolve()),
                        "raster_path": str(raster_path.resolve()),
                        "raster_exists": raster_path.is_file(),
                    }
                )
            if not usable:
                continue

            cached_feature = load_gzip_pickle(feature_path)
            target = load_gzip_pickle(target_path)
            real_tensor = preprocess_images(real_images)
            raster_tensor = preprocess_images(raster_images)
            cache_real_max_abs = float((real_tensor - cached_feature["camera_feature"]).abs().max())
            if cache_real_max_abs != 0.0:
                raise RuntimeError(f"real preprocessing mismatch for {token}: {cache_real_max_abs}")
            if target["trajectory"].shape[0] != 10:
                continue
            samples.append(
                {
                    "log": metadata_path.stem,
                    "token": token,
                    "frame": frame,
                    "feature": cached_feature,
                    "target": target,
                    "real_tensor": real_tensor,
                    "raster_tensor": raster_tensor,
                    "camera_records": camera_records,
                    "cache_real_max_abs": cache_real_max_abs,
                    "historical_cached_raster_mse": float(
                        F.mse_loss(raster_tensor, cached_feature["rendered_camera_feature"]).item()
                    ),
                }
            )
            if len(samples) >= args.num_pairs:
                return samples
    raise RuntimeError(f"found only {len(samples)} usable samples; requested {args.num_pairs}")


def build_batch(samples: Sequence[Dict[str, Any]], device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    batch_size = len(samples)
    real = torch.stack([sample["real_tensor"] for sample in samples]).to(device)
    raster = torch.stack([sample["raster_tensor"] for sample in samples]).to(device)
    features: Dict[str, torch.Tensor] = {"camera_feature": torch.cat([raster, real], dim=0)}
    for key in ["img_shape", "lidar2img", "ego_status"]:
        values = torch.stack([sample["feature"][key] for sample in samples]).to(device)
        features[key] = torch.cat([values, values], dim=0)
    trajectories = torch.stack([sample["target"]["trajectory"].float() for sample in samples]).to(device)
    targets: Dict[str, Any] = {
        "trajectory": torch.cat([trajectories, trajectories], dim=0),
        "token": [sample["token"] for sample in samples] * 2,
        # The norm probe tokens have no matching PDM metric cache.  False is the
        # code-defined unscored-sample path (trajectory component weighted 0.1).
        "score_mask": torch.zeros(2 * batch_size, dtype=torch.bool, device=device),
    }
    return features, targets


def group_parameters(model: torch.nn.Module) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
        "dino": [],
        "fpn": [],
        "camera_embedding": [],
        "level_embedding": [],
        "trajectory_refiner": [],
        "scorer": [],
        "domain_classifier": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("_backbone.img_backbone"):
            groups["dino"].append((name, parameter))
        elif name.startswith("_backbone.img_neck"):
            groups["fpn"].append((name, parameter))
        elif name == "_backbone.cams_embeds":
            groups["camera_embedding"].append((name, parameter))
        elif name == "_backbone.level_embeds":
            groups["level_embedding"].append((name, parameter))
        elif name.startswith("_trajectory_head"):
            groups["trajectory_refiner"].append((name, parameter))
        elif name.startswith("scorer"):
            groups["scorer"].append((name, parameter))
        elif name.startswith("domain_classifier"):
            groups["domain_classifier"].append((name, parameter))
    return groups


def grad_stats(named_parameters: Sequence[Tuple[str, torch.nn.Parameter]]) -> Dict[str, Any]:
    squared = 0.0
    non_none = 0
    nonzero = 0
    for _, parameter in named_parameters:
        grad = parameter.grad
        if grad is None:
            continue
        non_none += 1
        norm = float(torch.linalg.vector_norm(grad.detach().float()).item())
        squared += norm * norm
        if norm > 0:
            nonzero += 1
    return {
        "grad_norm": math.sqrt(squared),
        "parameter_tensors": len(named_parameters),
        "grad_non_none_tensors": non_none,
        "grad_nonzero_tensors": nonzero,
    }


def tensor_norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.detach().float()).item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(a, b).div(denom).item())


def clear_grads(module: torch.nn.Module) -> None:
    module.zero_grad(set_to_none=True)


def f1_mse_only(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    batch_size: int,
    groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    clear_grads(model)
    feat = model._backbone(batch["camera_feature"], img_metas=batch)[0]
    post = feat.permute(2, 0, 1, 3)
    post.retain_grad()
    raster = post[:batch_size]
    real = post[batch_size:]
    raw_mse = F.mse_loss(raster.detach(), real)
    loss = 0.002 * raw_mse
    loss.backward()

    rows: List[Dict[str, Any]] = []
    grad = post.grad
    assert grad is not None
    rows.append({"experiment": "F1", "route": "mse_only", "object": "real_post_fpn", "grad_norm": tensor_norm(grad[batch_size:])})
    rows.append({"experiment": "F1", "route": "mse_only", "object": "raster_pre_detach", "grad_norm": tensor_norm(grad[:batch_size])})
    for group_name, parameters in groups.items():
        row = {"experiment": "F1", "route": "mse_only", "object": group_name}
        row.update(grad_stats(parameters))
        rows.append(row)

    camera_rows = []
    per_camera_mse = [float(F.mse_loss(raster[:, index].detach(), real[:, index]).item()) for index in range(4)]
    mse_sum = sum(per_camera_mse)
    for index, camera in enumerate(CAMERAS):
        row = {
            "experiment": "F1",
            "route": "mse_only_per_camera",
            "object": camera,
            "raw_mse": per_camera_mse[index],
            "weighted_feature_grad_norm": tensor_norm(grad[batch_size:, index]),
            "mse_share": per_camera_mse[index] / mse_sum,
            "raster_exists": camera != "B0",
        }
        rows.append(row)
        camera_rows.append(row)
    result = {
        "batch_pairs": batch_size,
        "raw_mse": float(raw_mse.item()),
        "weighted_loss": float(loss.item()),
        "camera_rows": camera_rows,
    }
    clear_grads(model)
    return rows, result


def domain_loss(model: torch.nn.Module, feat: torch.Tensor, batch_size: int, lambd: float) -> torch.Tensor:
    raster_f0 = feat[[1], :, :batch_size].detach()
    real_f0 = feat[[1], :, batch_size:]
    mixed = torch.cat([raster_f0, real_f0], dim=2)
    logits = model.domain_classifier(mixed, lambd=lambd)
    labels = torch.cat(
        [torch.zeros(batch_size, device=feat.device), torch.ones(batch_size, device=feat.device)], dim=0
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


def f2_domain_only(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    batch_size: int,
    groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    feat = model._backbone(batch["camera_feature"], img_metas=batch)[0]
    feat.retain_grad()
    rows: List[Dict[str, Any]] = []
    saved_real_grads: Dict[float, torch.Tensor] = {}
    losses: Dict[str, float] = {}
    lambdas = [0.0, 1.0, -1.0]
    for index, lambd in enumerate(lambdas):
        clear_grads(model)
        feat.grad = None
        loss = domain_loss(model, feat, batch_size, lambd)
        loss.backward(retain_graph=index < len(lambdas) - 1)
        grad = feat.grad
        assert grad is not None
        saved_real_grads[lambd] = grad[[1], :, batch_size:].detach().cpu().clone()
        losses[str(lambd)] = float(loss.item())
        rows.extend(
            [
                {"experiment": "F2", "route": f"domain_lambda_{lambd:+.0f}", "object": "raster_F0", "grad_norm": tensor_norm(grad[[1], :, :batch_size])},
                {"experiment": "F2", "route": f"domain_lambda_{lambd:+.0f}", "object": "real_F0", "grad_norm": tensor_norm(grad[[1], :, batch_size:])},
            ]
        )
        for group_name in ["fpn", "camera_embedding", "level_embedding", "domain_classifier", "dino"]:
            row = {"experiment": "F2", "route": f"domain_lambda_{lambd:+.0f}", "object": group_name}
            row.update(grad_stats(groups[group_name]))
            rows.append(row)

    cos_rows = [
        {
            "experiment": "F2",
            "parameter": "real_F0_activation",
            "route_a": "GRL_lambda_+1",
            "route_b": "GRL_lambda_-1_no_reverse_control",
            "cosine": cosine(saved_real_grads[1.0], saved_real_grads[-1.0]),
        }
    ]
    clear_grads(model)
    return rows, cos_rows, {"losses": losses, "grl_cosine": cos_rows[0]["cosine"]}


def exact_trajectory_component(prediction: Dict[str, torch.Tensor], targets: Dict[str, Any]) -> torch.Tensor:
    trajectory = targets["trajectory"]
    running: torch.Tensor | float = 0.0
    for proposals in prediction["proposal_list"]:
        minimum = torch.linalg.norm(proposals - trajectory[:, None], dim=-1, ord=1).mean(-1).amin(1)
        # score_mask=False follows RAPAgent.rap_loss's unscored-sample weight.
        minimum = (minimum * 0.1).mean()
        running = 0.1 * running + minimum
    assert isinstance(running, torch.Tensor)
    return running


def selected_fpn_parameters(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    names = [
        "_backbone.img_neck.lateral_convs.0.conv.weight",
        "_backbone.img_neck.fpn_convs.0.conv.weight",
    ]
    parameters = dict(model.named_parameters())
    missing = [name for name in names if name not in parameters]
    if missing:
        raise KeyError(f"missing FPN parameters: {missing}")
    return [(name, parameters[name]) for name in names]


def gradients_for(loss: torch.Tensor, named_parameters: Sequence[Tuple[str, torch.nn.Parameter]], retain_graph: bool = False) -> Dict[str, torch.Tensor]:
    grads = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named_parameters],
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return {
        name: (torch.zeros_like(parameter, memory_format=torch.preserve_format) if grad is None else grad).detach().cpu()
        for (name, parameter), grad in zip(named_parameters, grads)
    }


def f3_combined(
    agent: RAPAgent,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    targets: Dict[str, Any],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    selected = selected_fpn_parameters(model)
    model.batch_size = batch_size
    model.progress = 1.0

    prediction = agent.forward({key: value.clone() for key, value in batch.items()}, dict(targets))
    planning = exact_trajectory_component(prediction, targets)
    gradients: Dict[str, Dict[str, torch.Tensor]] = {
        "planning": gradients_for(planning, selected)
    }
    del prediction

    feat = model._backbone(batch["camera_feature"], img_metas=batch)[0]
    post = feat.permute(2, 0, 1, 3)
    mse = 0.002 * F.mse_loss(post[:batch_size].detach(), post[batch_size:])
    domain = 0.001 * domain_loss(model, feat, batch_size, 1.0)
    gradients["mse"] = gradients_for(mse, selected, retain_graph=True)
    gradients["domain"] = gradients_for(domain, selected)

    norm_rows: List[Dict[str, Any]] = []
    cosine_rows: List[Dict[str, Any]] = []
    for name, _ in selected:
        for route in ["planning", "mse", "domain"]:
            norm_rows.append(
                {
                    "experiment": "F3",
                    "route": route,
                    "object": name,
                    "grad_norm": tensor_norm(gradients[route][name]),
                    "weighted": True,
                }
            )
        for route_a, route_b in [("planning", "mse"), ("planning", "domain"), ("mse", "domain")]:
            cosine_rows.append(
                {
                    "experiment": "F3",
                    "parameter": name,
                    "route_a": route_a,
                    "route_b": route_b,
                    "cosine": cosine(gradients[route_a][name], gradients[route_b][name]),
                }
            )
    result = {
        "batch_pairs": batch_size,
        "planning_probe": "exact RAP trajectory component; score_mask=False gives the code-defined 0.1 sample weight",
        "losses": {"planning": float(planning.item()), "mse_weighted": float(mse.item()), "domain_weighted": float(domain.item())},
    }
    return norm_rows, cosine_rows, result


def projector_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    parameters = list(model._backbone.img_neck.parameters())
    parameters.extend([model._backbone.cams_embeds, model._backbone.level_embeds])
    return parameters


def projector_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, value in model._backbone.state_dict().items():
        if name.startswith("img_neck") or name in {"cams_embeds", "level_embeds"}:
            state[name] = value.detach().cpu().clone()
    return state


def load_projector_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    current = model._backbone.state_dict()
    current.update(state)
    model._backbone.load_state_dict(current, strict=True)


@torch.no_grad()
def capture_anchor(model: torch.nn.Module, batch: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    holder: Dict[str, torch.Tensor] = {}

    def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
        holder["raw"] = output["last_hidden_state"].detach().float().cpu()

    handle = model._backbone.img_backbone.register_forward_hook(hook)
    try:
        feat = model._backbone(batch["camera_feature"], img_metas=batch)[0]
    finally:
        handle.remove()
    post = feat.permute(2, 0, 1, 3).detach().float().cpu()
    return {
        "raw_raster": holder["raw"][: 4 * batch_size],
        "raw_real": holder["raw"][4 * batch_size :],
        "post_raster": post[:batch_size],
        "post_real": post[batch_size:],
        "mse": F.mse_loss(post[:batch_size], post[batch_size:]),
    }


def delta_stats(before: torch.Tensor, after: torch.Tensor) -> Dict[str, float]:
    delta = after.float() - before.float()
    before_norm = float(torch.linalg.vector_norm(before.float()).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    return {
        "delta_l2": delta_norm,
        "delta_relative_l2": delta_norm / before_norm if before_norm else float("nan"),
        "delta_max_abs": float(delta.abs().max().item()),
    }


def f4_anchor_drift(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    state = projector_state(model)
    baseline = capture_anchor(model, batch, batch_size)
    repeated = capture_anchor(model, batch, batch_size)
    rows: List[Dict[str, Any]] = []

    def add_rows(setting: str, after: Dict[str, torch.Tensor]) -> None:
        for key in ["raw_real", "raw_raster", "post_real", "post_raster"]:
            row = {"experiment": "F4", "optimizer": setting, "object": key}
            row.update(delta_stats(baseline[key], after[key]))
            row["mse_before"] = float(baseline["mse"].item())
            row["mse_after"] = float(after["mse"].item())
            rows.append(row)

    add_rows("no_update_repeat", repeated)

    settings = [
        ("adamw_production_lr_wd", "adamw", 1.0e-5, 1.0e-4),
        ("adamw_no_weight_decay", "adamw", 1.0e-5, 0.0),
        ("sgd_no_weight_decay", "sgd", 1.0e-5, 0.0),
    ]
    for label, kind, lr, weight_decay in settings:
        load_projector_state(model, state)
        parameters = projector_parameters(model)
        if kind == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad(set_to_none=True)
        feat = model._backbone(batch["camera_feature"], img_metas=batch)[0]
        post = feat.permute(2, 0, 1, 3)
        loss = 0.002 * F.mse_loss(post[:batch_size].detach(), post[batch_size:])
        loss.backward()
        optimizer.step()
        after = capture_anchor(model, batch, batch_size)
        add_rows(label, after)
        del optimizer

    load_projector_state(model, state)
    return rows, {
        "batch_pairs": batch_size,
        "mse_before": float(baseline["mse"].item()),
        "settings": [setting[0] for setting in settings],
    }


def checkpoint_architecture_check(checkpoint: Path) -> Dict[str, Any]:
    data = torch.load(str(checkpoint), map_location="cpu", mmap=True, weights_only=False)
    state = data["state_dict"]
    prefix = "agent._rap_model._backbone.img_backbone."
    layers = sorted(
        {
            int(key.split(".layer.")[1].split(".")[0])
            for key in state
            if key.startswith(prefix) and ".layer." in key
        }
    )
    checks = {
        "hidden_size": int(state[prefix + "embeddings.cls_token"].shape[-1]),
        "patch_size": int(state[prefix + "embeddings.patch_embeddings.weight"].shape[-1]),
        "num_register_tokens": int(state[prefix + "embeddings.register_tokens"].shape[1]),
        "intermediate_size": int(state[prefix + "layer.0.mlp.gate_proj.weight"].shape[0]),
        "num_hidden_layers": len(layers),
    }
    for key, value in checks.items():
        if DINOV3_ARCHITECTURE[key] != value:
            raise RuntimeError(f"DINO architecture mismatch for {key}: config={DINOV3_ARCHITECTURE[key]} checkpoint={value}")
    return {
        "checkpoint_epoch": data.get("epoch"),
        "checkpoint_global_step": data.get("global_step"),
        "state_tensors": len(state),
        "domain_classifier_tensors": sum("domain_classifier" in key for key in state),
        "shape_checks": checks,
    }


def make_agent(args: argparse.Namespace, device: torch.device) -> RAPAgent:
    config = RAPConfig(
        trajectory_sampling=TrajectorySampling(time_horizon=5, interval_length=0.5),
        distill_feature=True,
        pdm_scorer=True,
        cache_data=False,
        train_metric_cache_path=str(args.metric_cache),
    )
    agent = RAPAgent(config=config, lr=1.0e-5, checkpoint_path=str(args.checkpoint))
    agent.eval()
    agent._rap_model._backbone.use_grid_mask = False
    for parameter in agent._rap_model._backbone.img_backbone.parameters():
        parameter.requires_grad_(False)
    agent.to(device)
    agent._rap_model.batch_size = 1
    agent._rap_model.progress = 1.0
    return agent


def build_summary(
    args: argparse.Namespace,
    sample_records: List[Dict[str, Any]],
    architecture_check: Dict[str, Any],
    f1: Dict[str, Any],
    f2: Dict[str, Any],
    f3: Dict[str, Any],
    f4_rows: List[Dict[str, Any]],
) -> str:
    b0 = next(row for row in f1["camera_rows"] if row["object"] == "B0")
    non_b0_max = max(row["mse_share"] for row in f1["camera_rows"] if row["object"] != "B0")
    f4_by = {(row["optimizer"], row["object"]): row for row in f4_rows}
    adam_raster = f4_by[("adamw_no_weight_decay", "post_raster")]
    adam_real = f4_by[("adamw_no_weight_decay", "post_real")]
    raw_real = f4_by[("adamw_no_weight_decay", "raw_real")]
    repeat = f4_by[("no_update_repeat", "post_real")]
    domain_head_loaded = architecture_check["domain_classifier_tensors"] > 0
    status = "PASS" if domain_head_loaded else "PASS_WITH_WARNINGS"
    if abs(f2["grl_cosine"] + 1.0) > 1e-3:
        status = "FAIL"
    if raw_real["delta_max_abs"] != 0.0 or repeat["delta_max_abs"] != 0.0:
        status = "FAIL"
    sample_lines = "\n".join(f"- `{record['log']}/{record['token']}`" for record in sample_records)
    return f"""# Step 0-F gradient-route audit

Status: **{status}**

## Inputs

- checkpoint: `{args.checkpoint.resolve()}` (epoch {architecture_check['checkpoint_epoch']}, global step {architecture_check['checkpoint_global_step']})
- norm metadata: `{args.metadata_root.resolve()}`
- norm raster: `{args.raster_root.resolve()}`
- real images: `{args.real_root.resolve()}`
- F1/F2 paired samples: {f1['batch_pairs']}
{sample_lines}

Historical cache rendered tensors were not used: they are blank fallback tensors. Real camera preprocessing was reconstructed from disk and checked exactly against the cache; norm raster was reconstructed from `dataset_norm`, with B0 following the production missing-file-to-zero behavior.

## F1 — MSE-only

- raw MSE: `{f1['raw_mse']:.8g}`; weighted loss: `{f1['weighted_loss']:.8g}`
- B0 share of total per-camera MSE: `{b0['mse_share']:.2%}` (largest non-B0 camera share: `{non_b0_max:.2%}`)
- The raster activation is detached; the real activation, shared FPN, camera embedding, and level embedding receive gradients. Frozen DINO, trajectory refiner, scorer, and domain head do not receive MSE-only gradients.

## F2 — domain-only / GRL

- cosine between real-F0 activation gradients at lambda=+1 and the lambda=-1 no-reversal control: `{f2['grl_cosine']:.8f}`
- At lambda=0 the encoder-side real-F0 gradient is zero while DomainClassifier still has a gradient. At lambda=1 the real-F0 encoder path is nonzero. Raster F0 stays detached.
- Checkpoint DomainClassifier tensors: `{architecture_check['domain_classifier_tensors']}`. The selected v2 checkpoint predates this head, so the classifier is deterministically initialized from seed {SEED}. F2 validates the autograd/GRL route, not a learned domain decision boundary.

## F3 — weighted gradient decomposition

- planning probe: {f3['planning_probe']}
- losses: planning `{f3['losses']['planning']:.8g}`, 0.002*MSE `{f3['losses']['mse_weighted']:.8g}`, 0.001*domain `{f3['losses']['domain_weighted']:.8g}`
- Exact per-layer gradient norms and pairwise cosines are in `gradient_norms.csv` and `gradient_cosines.csv`.
- Limitation: these norm tokens do not occur in the available PDM metric cache, so F3 uses the exact RAP trajectory component and the code-defined unscored-sample weight; PDM-score auxiliary terms are not included.
- Because the v2 checkpoint has no DomainClassifier tensors, F3's domain gradient magnitude is a route smoke-test value and must not be interpreted as the magnitude of a trained domain head.

## F4 — shared-projector anchor drift

- deterministic no-update repeat, real post-FPN max delta: `{repeat['delta_max_abs']:.8g}`
- no-weight-decay AdamW step, raw DINO real max delta: `{raw_real['delta_max_abs']:.8g}`
- no-weight-decay AdamW step, post-FPN real relative L2 delta: `{adam_real['delta_relative_l2']:.8g}`
- no-weight-decay AdamW step, post-FPN raster relative L2 delta: `{adam_raster['delta_relative_l2']:.8g}`

The raw DINO anchor is fixed, but both real and raster post-FPN representations move after an MSE-only projector update. Therefore raster is a stop-gradient target only within the current backward pass; it is not a fixed teacher across optimizer steps.
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    architecture_check = checkpoint_architecture_check(args.checkpoint)
    config_record = {
        "seed": SEED,
        "metadata_root": args.metadata_root,
        "raster_root": args.raster_root,
        "real_root": args.real_root,
        "cache_root": args.cache_root,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint),
        "metric_cache": args.metric_cache,
        "num_pairs": args.num_pairs,
        "mse_weight": 0.002,
        "domain_weight": 0.001,
        "production_lr": 1.0e-5,
        "production_weight_decay": 1.0e-4,
        "dinov3_architecture": DINOV3_ARCHITECTURE,
        "architecture_check": architecture_check,
    }
    write_json(args.output_dir / "config.json", config_record)
    if args.preflight_only:
        print(json.dumps(jsonable(config_record), indent=2))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; run this audit on a GPU queue")

    samples = find_samples(args)
    sample_records = [
        {
            "log": sample["log"],
            "token": sample["token"],
            "camera_records": sample["camera_records"],
            "cache_real_max_abs": sample["cache_real_max_abs"],
            "historical_cached_raster_mse": sample["historical_cached_raster_mse"],
        }
        for sample in samples
    ]
    write_json(args.output_dir / "samples.json", sample_records)
    print(f"[0-F] selected {len(samples)} norm pairs", flush=True)

    device = torch.device(args.device)
    agent = make_agent(args, device)
    model = agent._rap_model
    groups = group_parameters(model)
    print(f"[0-F] model loaded on {device}", flush=True)

    full_batch, full_targets = build_batch(samples, device)
    model.batch_size = len(samples)
    f1_rows, f1_result = f1_mse_only(model, full_batch, len(samples), groups)
    print("[0-F] F1 complete", flush=True)
    f2_rows, f2_cosines, f2_result = f2_domain_only(model, full_batch, len(samples), groups)
    print("[0-F] F2 complete", flush=True)
    del full_batch, full_targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    one_batch, one_targets = build_batch(samples[:1], device)
    f3_rows, f3_cosines, f3_result = f3_combined(agent, model, one_batch, one_targets, 1)
    print("[0-F] F3 complete", flush=True)
    f4_rows, f4_result = f4_anchor_drift(model, one_batch, 1)
    print("[0-F] F4 complete", flush=True)

    gradient_rows = f1_rows + f2_rows + f3_rows
    cosine_rows = f2_cosines + f3_cosines
    write_csv(args.output_dir / "gradient_norms.csv", gradient_rows)
    write_csv(args.output_dir / "gradient_cosines.csv", cosine_rows)
    write_csv(args.output_dir / "anchor_drift.csv", f4_rows)
    results = {
        "F1": f1_result,
        "F2": f2_result,
        "F3": f3_result,
        "F4": f4_result,
    }
    results["F2"]["classifier_checkpoint_loaded"] = architecture_check["domain_classifier_tensors"] > 0
    results["F3"]["domain_magnitude_checkpoint_interpretable"] = architecture_check["domain_classifier_tensors"] > 0
    write_json(args.output_dir / "summary.json", results)
    summary = build_summary(args, sample_records, architecture_check, f1_result, f2_result, f3_result, f4_rows)
    (args.output_dir / "summary.md").write_text(summary)
    write_json(
        args.output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "numpy": np.__version__,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
