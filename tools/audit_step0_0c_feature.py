#!/usr/bin/env python3
"""Step 0-C component feature/MSE/gradient ablation on norm F0/L0/R0.

The script reuses the checkpoint construction and exact preprocessing checks
from Step 0-F.  B0 remains in the four-camera tensor only as the production
blank fallback; every reported loss is explicitly sliced to F0/L0/R0.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import platform
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools import audit_step0_0f as base
from tools.audit_step0_0a import jpeg_roundtrip_rgb, reconstruct_scenario
from tools.audit_step0_0c_geometry import render_variant

CAMERAS = ("F0", "L0", "R0")
CAMERA_INDEX = {"F0": 1, "L0": 2, "R0": 3}
VARIANTS = (
    "full", "blank", "agent_only", "map_only", "traffic_light_only",
    "full_without_agents", "full_without_map", "full_without_traffic_lights",
)
COMPONENT_FOR_LEAVEOUT = {
    "full_without_agents": "agent",
    "full_without_map": "map",
    "full_without_traffic_lights": "traffic_light",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=Path("dataset_norm/navsim_logs/mini"))
    parser.add_argument("--raster-root", type=Path, default=Path("dataset_norm/rendered_sensor_blobs"))
    parser.add_argument("--real-root", type=Path, default=Path("/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/mini"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/metabev_densegrid_v2_full/cache"))
    parser.add_argument("--checkpoint", type=Path, default=Path("ckpts/RAP_DINO_navsimv2.ckpt"))
    parser.add_argument("--metric-cache", type=Path, default=Path("/gs/bs/tga-RLA/qdeng/navsim_workspace/exp/metric_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step0_alignment_audit/0c_raster_components"))
    parser.add_argument("--num-pairs", type=int, default=32)
    parser.add_argument("--gradient-pairs", type=int, default=32)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=base.SEED)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: base.jsonable(row.get(key, "")) for key in fields} for row in rows])


def find_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = args.sample_offset + args.num_pairs
    rng = random.Random(args.seed)
    per_log: list[tuple[Path, list[dict[str, Any]]]] = []
    for metadata_path in sorted(args.metadata_root.glob("*.pkl")):
        with metadata_path.open("rb") as stream:
            frames = pickle.load(stream)
        candidates = []
        for frame in frames:
            token_cache = args.cache_root / metadata_path.stem / frame["token"]
            if (token_cache / "rap_feature.gz").is_file() and (token_cache / "rap_target.gz").is_file():
                candidates.append(frame)
        rng.shuffle(candidates)
        if candidates:
            per_log.append((metadata_path, candidates))
    candidate_target = min(sum(len(candidates) for _, candidates in per_log), requested + max(64, requested))
    selected: list[tuple[Path, dict[str, Any]]] = []
    depth = 0
    while len(selected) < candidate_target:
        added = False
        for metadata_path, candidates in per_log:
            if depth < len(candidates):
                selected.append((metadata_path, candidates[depth])); added = True
                if len(selected) >= candidate_target:
                    break
        if not added:
            break
        depth += 1
    samples: list[dict[str, Any]] = []
    for metadata_path, frame in selected:
        token = frame["token"]
        token_cache = args.cache_root / metadata_path.stem / token
        real_images: list[np.ndarray] = []
        raster_images: list[np.ndarray] = []
        camera_records: list[dict[str, Any]] = []
        usable = True
        for camera_key, camera_short in zip(base.CAMERA_KEYS, base.CAMERAS):
            rel = Path(frame["cams"][camera_key]["data_path"])
            real_path, raster_path = args.real_root / rel, args.raster_root / rel
            if not real_path.is_file():
                usable = False; break
            real_image = np.asarray(Image.open(real_path).convert("RGB"))
            raster_image = np.asarray(Image.open(raster_path).convert("RGB"))[20:-20] if raster_path.is_file() else np.zeros((1080, 1920, 3), np.uint8)
            if camera_short != "B0" and not np.any(raster_image):
                usable = False; break
            real_images.append(real_image); raster_images.append(raster_image)
            camera_records.append({"camera": camera_short, "real_path": str(real_path.resolve()), "raster_path": str(raster_path.resolve()), "raster_exists": raster_path.is_file()})
        if not usable:
            continue
        feature = base.load_gzip_pickle(token_cache / "rap_feature.gz")
        target = base.load_gzip_pickle(token_cache / "rap_target.gz")
        real_tensor, raster_tensor = base.preprocess_images(real_images), base.preprocess_images(raster_images)
        cache_delta = float((real_tensor - feature["camera_feature"]).abs().max())
        if cache_delta != 0 or target["trajectory"].shape[0] != 10:
            continue
        samples.append({"log": metadata_path.stem, "token": token, "frame": frame, "feature": feature, "target": target, "real_tensor": real_tensor,
            "raster_tensor": raster_tensor, "camera_records": camera_records, "cache_real_max_abs": cache_delta,
            "historical_cached_raster_mse": float(F.mse_loss(raster_tensor, feature["rendered_camera_feature"]).item())})
        if len(samples) >= requested:
            break
    if len(samples) < requested:
        raise RuntimeError(f"found only {len(samples)} stratified samples; requested {requested}")
    return samples[args.sample_offset:requested]


def reconstruct_for_sample(sample: Mapping[str, Any], map_cache: dict[str, Any]) -> dict[str, Any]:
    scenario = reconstruct_scenario(sample["frame"], map_cache)
    anns = sample["frame"]["anns"]
    scenario["anns"]["instance_tokens"] = anns.get("instance_tokens", [])
    scenario["anns"]["track_tokens"] = anns.get("track_tokens", [])
    return scenario


def variant_tensor(sample: Mapping[str, Any], scenario: Mapping[str, Any], variant: str) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    components = {
        "full": {"traffic_light", "map", "agent"}, "blank": set(), "agent_only": {"agent"},
        "map_only": {"map"}, "traffic_light_only": {"traffic_light"},
        "full_without_agents": {"traffic_light", "map"}, "full_without_map": {"traffic_light", "agent"},
        "full_without_traffic_lights": {"map", "agent"},
    }[variant]
    images: list[np.ndarray] = [np.zeros((1080, 1920, 3), np.uint8)]
    coverage: dict[str, dict[str, float]] = {}
    for camera in CAMERAS:
        raw = render_variant(scenario, f"CAM_{camera}", components)
        encoded = jpeg_roundtrip_rgb(raw)[20:-20]
        if variant == "full":
            stored = np.asarray(Image.open(Path(sample["camera_records"][CAMERA_INDEX[camera]]["raster_path"])).convert("RGB"))[20:-20]
            if not np.array_equal(encoded, stored):
                raise RuntimeError(f"full variant mismatch for {sample['token']} {camera}")
            encoded = stored
        images.append(encoded)
        raw_fg = float(np.any(raw != 0, axis=-1).mean())
        resized = cv2.resize(encoded, (768, 432), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, 0, 16, 0, 0, cv2.BORDER_CONSTANT, value=0)
        coverage[camera] = {"foreground_fraction_render": raw_fg, "foreground_fraction_model": float(np.any(padded != 0, axis=-1).mean())}
    return base.preprocess_images(images).unsqueeze(0), coverage


def forward_capture(model: torch.nn.Module, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    holder: dict[str, torch.Tensor] = {}
    def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
        holder["raw"] = output["last_hidden_state"]
    handle = model._backbone.img_backbone.register_forward_hook(hook)
    try:
        feature = model._backbone(tensor, img_metas={})[0].permute(2, 0, 1, 3)
    finally:
        handle.remove()
    raw = holder["raw"].reshape(tensor.shape[0], tensor.shape[1], holder["raw"].shape[-2], holder["raw"].shape[-1])
    return raw, feature


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.detach().float() - b.detach().float()) ** 2)).item())


def norm_dict(values: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(torch.sum(value.detach().float() ** 2).item()) for value in values.values()))


def cosine_dict(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    dot = sum(float(torch.sum(a[key].detach().float() * b[key].detach().float()).item()) for key in a)
    na, nb = norm_dict(a), norm_dict(b)
    return dot / (na * nb) if na and nb else float("nan")


def subtract_dict(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: a[key] - b[key] for key in a}


def gradients_for_scopes(model: torch.nn.Module, raster: torch.Tensor, real: torch.Tensor, selected: Sequence[tuple[str, torch.nn.Parameter]]) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    paired = torch.cat([raster, real], dim=0)
    _, post = forward_capture(model, paired)
    raster_post, real_post = post[:1].detach(), post[1:]
    losses = {"joint_3cam": 0.002 * F.mse_loss(raster_post[:, 1:4], real_post[:, 1:4])}
    for camera, index in CAMERA_INDEX.items():
        losses[f"single_{camera}"] = 0.002 * F.mse_loss(raster_post[:, index], real_post[:, index])
    gradients: dict[str, dict[str, torch.Tensor]] = {}
    for count, (scope, loss) in enumerate(losses.items()):
        gradients[scope] = base.gradients_for(loss, selected, retain_graph=count < len(losses) - 1)
    return gradients, {scope: float(loss.detach().item()) for scope, loss in losses.items()}


def planning_grad(agent: Any, sample: Mapping[str, Any], device: torch.device, selected: Sequence[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    batch, targets = base.build_batch([sample], device)
    agent._rap_model.batch_size = 1
    prediction = agent.forward({key: value.clone() for key, value in batch.items()}, dict(targets))
    loss = base.exact_trajectory_component(prediction, targets)
    return base.gradients_for(loss, selected)


def heatmap(array: torch.Tensor) -> np.ndarray:
    values = torch.linalg.vector_norm(array.detach().float(), dim=-1).cpu().numpy()
    return values.reshape(28, 48).astype(np.float32)


def save_heatmap_overlay(real_path: Path, array_path: Path, png_path: Path, scale: float) -> None:
    values = np.load(array_path)
    normalized = np.clip(values / max(scale, 1e-12), 0, 1)
    colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    colored = cv2.resize(colored, (768, 448), interpolation=cv2.INTER_NEAREST)
    real = np.asarray(Image.open(real_path).convert("RGB").resize((768, 448)))
    overlay = np.uint8(0.55 * real + 0.45 * colored)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(png_path)


def bootstrap_groups(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], metrics: Sequence[str], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for group, members in sorted(groups.items()):
        record: dict[str, Any] = dict(zip(keys, group)); record["n"] = len(members); record["logs"] = len({str(row["log"]) for row in members})
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in members if math.isfinite(float(row[metric]))], dtype=float)
            if not len(values):
                record[f"{metric}_mean"] = float("nan"); record[f"{metric}_ci_low"] = float("nan"); record[f"{metric}_ci_high"] = float("nan"); continue
            estimates = [float(np.mean(rng.choices(values.tolist(), k=len(values)))) for _ in range(2000)]
            record[f"{metric}_mean"] = float(np.mean(values)); record[f"{metric}_ci_low"] = float(np.percentile(estimates, 2.5)); record[f"{metric}_ci_high"] = float(np.percentile(estimates, 97.5))
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    architecture = base.checkpoint_architecture_check(args.checkpoint)
    samples = find_samples(args)
    device = torch.device(args.device)
    agent = base.make_agent(args, device)
    model = agent._rap_model
    selected = base.selected_fpn_parameters(model)
    map_cache: dict[str, Any] = {}
    ablation_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    heatmap_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        print(f"[0-C] {sample_index + 1}/{len(samples)} {sample['log']}/{sample['token']}", flush=True)
        scenario = reconstruct_for_sample(sample, map_cache)
        tensors: dict[str, torch.Tensor] = {}
        coverage: dict[str, dict[str, dict[str, float]]] = {}
        for variant in VARIANTS:
            tensor, cov = variant_tensor(sample, scenario, variant)
            tensors[variant] = tensor.to(device)
            coverage[variant] = cov
        real = sample["real_tensor"].unsqueeze(0).to(device)
        with torch.no_grad():
            raw_real, post_real = forward_capture(model, real)
            raw_full, post_full = forward_capture(model, tensors["full"])
        full_mse = {camera: float(F.mse_loss(post_real[:, index], post_full[:, index]).item()) for camera, index in CAMERA_INDEX.items()}
        full_mse_joint = float(F.mse_loss(post_real[:, 1:4], post_full[:, 1:4]).item())
        heat_root = args.output_dir / "ablations" / "token_heatmaps" / sample["token"]
        for camera, index in CAMERA_INDEX.items():
            real_full = heatmap(post_real[:, index] - post_full[:, index])
            path = heat_root / f"{camera}_real_full.npy"; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, real_full)
            grad_map = heatmap((2 * 0.002 / post_real[:, 1:4].numel()) * (post_real[:, index] - post_full[:, index]))
            grad_path = heat_root / f"{camera}_grad.npy"; np.save(grad_path, grad_map)
            real_path = Path(sample["camera_records"][index]["real_path"])
            heatmap_rows += [
                {"log": sample["log"], "frame_token": sample["token"], "camera": camera, "heatmap_type": "real_full", "component": "all", "npy_path": str(path), "real_path": str(real_path)},
                {"log": sample["log"], "frame_token": sample["token"], "camera": camera, "heatmap_type": "gradient", "component": "all", "npy_path": str(grad_path), "real_path": str(real_path)},
            ]
        variant_features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {"full": (raw_full, post_full)}
        for variant in VARIANTS:
            if variant == "full":
                raw_variant, post_variant = raw_full, post_full
            else:
                with torch.no_grad():
                    raw_variant, post_variant = forward_capture(model, tensors[variant])
                variant_features[variant] = (raw_variant, post_variant)
            for camera, index in CAMERA_INDEX.items():
                mse = float(F.mse_loss(post_real[:, index], post_variant[:, index]).item())
                ablation_rows.append({"log": sample["log"], "frame_token": sample["token"], "camera": camera, "scope": "single_camera", "variant": variant,
                    **coverage[variant][camera], "raw_dino_distance_to_full": rms(raw_variant[:, index], raw_full[:, index]), "post_fpn_distance_to_full": rms(post_variant[:, index], post_full[:, index]),
                    "raw_mse_real_to_variant": mse, "weighted_mse_real_to_variant": 0.002 * mse, "delta_loss_vs_full": mse - full_mse[camera]})
            mse_joint = float(F.mse_loss(post_real[:, 1:4], post_variant[:, 1:4]).item())
            ablation_rows.append({"log": sample["log"], "frame_token": sample["token"], "camera": "ALL_3CAM", "scope": "joint_3cam", "variant": variant,
                "foreground_fraction_render": float(np.mean([coverage[variant][camera]["foreground_fraction_render"] for camera in CAMERAS])),
                "foreground_fraction_model": float(np.mean([coverage[variant][camera]["foreground_fraction_model"] for camera in CAMERAS])),
                "raw_dino_distance_to_full": rms(raw_variant[:, 1:4], raw_full[:, 1:4]), "post_fpn_distance_to_full": rms(post_variant[:, 1:4], post_full[:, 1:4]),
                "raw_mse_real_to_variant": mse_joint, "weighted_mse_real_to_variant": 0.002 * mse_joint,
                "delta_loss_vs_full": mse_joint - full_mse_joint})
            if variant in COMPONENT_FOR_LEAVEOUT:
                component = COMPONENT_FOR_LEAVEOUT[variant]
                for camera, index in CAMERA_INDEX.items():
                    remove = heatmap(post_full[:, index] - post_variant[:, index])
                    path = heat_root / f"{camera}_remove_{component}.npy"; np.save(path, remove)
                    heatmap_rows.append({"log": sample["log"], "frame_token": sample["token"], "camera": camera, "heatmap_type": "remove_component", "component": component, "npy_path": str(path), "real_path": sample["camera_records"][index]["real_path"]})
        if sample_index < args.gradient_pairs:
            planning = planning_grad(agent, sample, device, selected)
            full_grads, full_losses = gradients_for_scopes(model, tensors["full"], real, selected)
            for variant in VARIANTS:
                grads, losses = (full_grads, full_losses) if variant == "full" else gradients_for_scopes(model, tensors[variant], real, selected)
                for scope, values in grads.items():
                    delta = subtract_dict(values, full_grads[scope])
                    camera = "ALL_3CAM" if scope == "joint_3cam" else scope.removeprefix("single_")
                    gradient_rows.append({"log": sample["log"], "frame_token": sample["token"], "camera": camera, "scope": "joint_3cam" if scope == "joint_3cam" else "single_camera", "variant": variant,
                        "weighted_mse": losses[scope], "lateral_conv_grad_norm": base.tensor_norm(values[selected[0][0]]), "fpn_conv_grad_norm": base.tensor_norm(values[selected[1][0]]),
                        "combined_grad_norm": norm_dict(values), "grad_cosine_vs_full_mse": cosine_dict(values, full_grads[scope]), "grad_cosine_vs_planning": cosine_dict(values, planning),
                        "delta_grad_norm_vs_full": norm_dict(delta), "delta_grad_cosine_vs_full_mse": cosine_dict(delta, full_grads[scope]), "delta_grad_cosine_vs_planning": cosine_dict(delta, planning),
                        "planning_probe": "exact RAP trajectory component; score_mask=False (0.1 sample weight); excludes PDM-score auxiliaries"})
        sample_rows.append({"sample_index": sample_index, "log": sample["log"], "frame_token": sample["token"], "cache_real_max_abs": sample["cache_real_max_abs"]})
        del tensors, variant_features, real
        torch.cuda.empty_cache()
    finite_arrays = [np.load(row["npy_path"]) for row in heatmap_rows]
    global_scale = float(np.percentile(np.concatenate([array.ravel() for array in finite_arrays]), 99)) if finite_arrays else 1.0
    visualization_rows: list[dict[str, Any]] = []
    for row in heatmap_rows:
        png = Path(row["npy_path"]).with_suffix(".png")
        save_heatmap_overlay(Path(row["real_path"]), Path(row["npy_path"]), png, global_scale)
        visualization_rows.append({"split": "mini", "log": row["log"], "frame_token": row["frame_token"], "camera": row["camera"], "panel_type": "token_heatmap", "element_type": row["component"], "element_id": "",
            "selection_rule": "fixed_checkpoint_subset", "rank": "", "real_path": row["real_path"], "raster_path": "", "panel_path": str(png), "heatmap_npy_path": row["npy_path"], "heatmap_type": row["heatmap_type"], "color_scale_global_p99": global_scale})
    write_csv(args.output_dir / "component_ablation.csv", ablation_rows)
    write_csv(args.output_dir / "component_gradient.csv", gradient_rows)
    ablation_summary = bootstrap_groups(ablation_rows, ("variant", "camera", "scope"), ("raw_dino_distance_to_full", "post_fpn_distance_to_full", "raw_mse_real_to_variant", "delta_loss_vs_full"), args.seed)
    gradient_summary = bootstrap_groups(gradient_rows, ("variant", "camera", "scope"), ("combined_grad_norm", "grad_cosine_vs_full_mse", "grad_cosine_vs_planning", "delta_grad_norm_vs_full"), args.seed)
    write_csv(args.output_dir / "component_ablation_summary.csv", ablation_summary)
    write_csv(args.output_dir / "component_gradient_summary.csv", gradient_summary)
    write_csv(args.output_dir / "feature_visualization_index.csv", visualization_rows)
    write_csv(args.output_dir / "feature_samples.csv", sample_rows)
    summary = {"status": "PASS", "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": base.sha256(args.checkpoint), "architecture": architecture,
        "samples": len(samples), "gradient_samples": min(args.gradient_pairs, len(samples)), "cameras": list(CAMERAS), "b0_excluded_from_losses": True,
        "variants": list(VARIANTS), "ablation_rows": len(ablation_rows), "gradient_rows": len(gradient_rows), "ablation_summary_rows": len(ablation_summary), "gradient_summary_rows": len(gradient_summary), "heatmap_arrays": len(heatmap_rows), "heatmap_global_p99": global_scale,
        "planning_probe_limitation": "exact RAP trajectory component with score_mask=False; PDM-score auxiliary losses excluded",
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device)}}
    (args.output_dir / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "feature_summary.md").write_text(
        "# Step 0-C feature/gradient ablation\n\n"
        f"- 状态：**PASS**\n- checkpoint：`{args.checkpoint}`（SHA256 `{summary['checkpoint_sha256']}`）\n"
        f"- forward 子集：{len(samples)} 帧 × F0/L0/R0；gradient 子集：{min(args.gradient_pairs, len(samples))} 帧\n"
        f"- 输出：{len(ablation_rows)} ablation rows，{len(gradient_rows)} gradient rows，{len(heatmap_rows)} float heatmaps\n"
        "- 主损失严格排除 B0。GridMask 已关闭。所有 component variant 使用相同虚拟相机、JPEG round-trip、crop/resize/pad/normalize。\n"
        "- planning cosine 只使用 0-F 同一简化 trajectory component，不包含 PDM-score auxiliary losses。结果不能解释为 planner 性能因果。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
