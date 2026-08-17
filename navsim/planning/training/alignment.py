"""Pure helpers for controlled Raster-to-Real alignment experiments."""

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def validate_alignment_config(task_real_ratio: float, eval_input_modality: str) -> None:
    """Fail early when an experiment configuration is not meaningful."""
    if not 0.0 <= task_real_ratio <= 1.0:
        raise ValueError(f"task_real_ratio must be in [0, 1], got {task_real_ratio}")
    if eval_input_modality not in {"real", "raster"}:
        raise ValueError(
            "eval_input_modality must be either 'real' or 'raster', "
            f"got {eval_input_modality!r}"
        )


def build_task_mask(
    real_valid_mask: Tensor,
    task_real_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Select exactly one planning-supervision modality for every valid pair.

    The combined forward order is ``[all raster, valid real]``.  The returned
    mask has that combined length.  The second return value marks which paired
    samples selected the real modality and is useful for logging/tests.
    """
    validate_alignment_config(task_real_ratio, "real")
    real_valid_mask = real_valid_mask.bool()
    device = real_valid_mask.device
    raster_batch_size = real_valid_mask.numel()
    num_pairs = int(real_valid_mask.sum().item())
    task_mask = torch.zeros(raster_batch_size + num_pairs, dtype=torch.bool, device=device)
    choose_real = torch.zeros(num_pairs, dtype=torch.bool, device=device)

    if num_pairs == 0:
        return task_mask, choose_real

    num_real = int(round(num_pairs * task_real_ratio))
    if num_real == num_pairs:
        choose_real.fill_(True)
    elif num_real > 0:
        # Generate on CPU so a caller can pass a seeded CPU generator while the
        # batch mask itself lives on a GPU.
        selected = torch.randperm(num_pairs, generator=generator)[:num_real].to(device)
        choose_real[selected] = True

    paired_raster_indices = torch.nonzero(real_valid_mask, as_tuple=False).flatten()
    task_mask[paired_raster_indices[~choose_real]] = True
    task_mask[raster_batch_size + torch.nonzero(choose_real, as_tuple=False).flatten()] = True
    return task_mask, choose_real


def mask_batch_mapping(mapping: Mapping[str, Any], mask: Tensor) -> Dict[str, Any]:
    """Apply a batch mask to RAP feature, target, or prediction mappings."""
    batch_size = mask.numel()
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None:
            result[key] = None
        elif key == "proposal_list" and isinstance(value, (list, tuple)):
            result[key] = [item[mask] for item in value]
        elif isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            result[key] = value[mask]
        elif isinstance(value, list) and len(value) == batch_size:
            result[key] = [item for item, keep in zip(value, mask.tolist()) if keep]
        else:
            result[key] = value
    return result


def spatial_alignment_loss(
    bev_feature: Tensor,
    raster_batch_size: int,
    real_valid_mask: Tensor,
) -> Tensor:
    """Pointwise MSE from real features to detached paired raster features."""
    raster = bev_feature[:raster_batch_size][real_valid_mask].detach()
    real = bev_feature[raster_batch_size:]
    if raster.shape != real.shape:
        raise ValueError(
            f"Paired raster/real feature shapes differ: {tuple(raster.shape)} vs {tuple(real.shape)}"
        )
    return F.mse_loss(real, raster)


def domain_alignment_loss(
    domain_logits: Tensor,
    raster_batch_size: int,
    real_valid_mask: Tensor,
) -> Tensor:
    """Balanced BCE over paired raster/real domain predictions only."""
    raster_logits = domain_logits[:raster_batch_size][real_valid_mask]
    real_logits = domain_logits[raster_batch_size:]
    if raster_logits.shape != real_logits.shape:
        raise ValueError(
            f"Paired raster/real logit shapes differ: {tuple(raster_logits.shape)} vs {tuple(real_logits.shape)}"
        )
    logits = torch.cat([raster_logits, real_logits], dim=0)
    labels = torch.cat([torch.zeros_like(raster_logits), torch.ones_like(real_logits)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def compose_alignment_loss(
    task_loss: Tensor,
    spatial_loss: Tensor,
    domain_loss: Tensor,
    use_spatial_align: bool,
    use_global_align: bool,
    spatial_weight: float,
    domain_weight: float,
) -> Tensor:
    """Compose independently gated alignment terms with the task loss."""
    total = task_loss
    if use_spatial_align:
        total = total + spatial_weight * spatial_loss
    if use_global_align:
        total = total + domain_weight * domain_loss
    return total
