#!/usr/bin/env python3
"""Merge Step 0-C geometry/feature deliverables and write the final summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def select(rows: Sequence[Mapping[str, str]], variant: str, camera: str = "ALL_3CAM") -> Mapping[str, str]:
    return next(row for row in rows if row.get("variant") == variant and row.get("camera") == camera and row.get("scope") == "joint_3cam")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step0_alignment_audit/0c_raster_components"))
    args = parser.parse_args(); root = args.output_dir
    geometry = json.loads((root / "geometry_summary.json").read_text())
    feature = json.loads((root / "feature_summary.json").read_text())
    ablation = read_csv(root / "component_ablation_summary.csv")
    gradient = read_csv(root / "component_gradient_summary.csv")
    visual_candidates = read_csv(root / "visualization_index.csv") + read_csv(root / "overlay_visualization_index.csv") + read_csv(root / "feature_visualization_index.csv")
    visual_by_key = {}
    for row in visual_candidates:
        key = (row.get("panel_path", ""), row.get("heatmap_npy_path", ""), row.get("panel_type", ""))
        visual_by_key[key] = row
    visual = list(visual_by_key.values())
    write_csv(root / "visualization_index.csv", visual)
    figures = []
    for row in visual:
        figures.append("<figure><img loading='lazy' src='{}'><figcaption>{} {} · {} · {} · {}</figcaption></figure>".format(
            row.get("panel_path", ""), row.get("frame_token", ""), row.get("camera", ""), row.get("panel_type", ""), row.get("element_type", ""), row.get("selection_rule", "")))
    html = """<!doctype html><meta charset='utf-8'><title>Step 0-C panels</title>
<style>body{{background:#111;color:#eee;font-family:sans-serif}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px}}figure{{margin:0}}img{{width:100%}}figcaption{{padding:4px}}</style>
<h1>Step 0-C visualization index</h1><main>{}</main>""".format("\n".join(figures))
    (root / "panel_index.html").write_text(html, encoding="utf-8")
    required = [
        "agent_raster_visibility.csv", "map_element_visibility.csv", "traffic_light_visibility.csv", "component_coverage.csv",
        "component_ablation.csv", "component_gradient.csv", "visualization_index.csv", "geometry_summary.md", "feature_summary.md",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    leaveouts = {}
    for variant, component in (("full_without_agents", "agent"), ("full_without_map", "map"), ("full_without_traffic_lights", "traffic_light")):
        a, g = select(ablation, variant), select(gradient, variant)
        leaveouts[component] = {
            "delta_mse_mean": float(a["delta_loss_vs_full_mean"]), "delta_mse_ci": [float(a["delta_loss_vs_full_ci_low"]), float(a["delta_loss_vs_full_ci_high"])],
            "delta_grad_norm_mean": float(g["delta_grad_norm_vs_full_mean"]), "grad_cosine_full_mean": float(g["grad_cosine_vs_full_mse_mean"]),
            "grad_cosine_planning_mean": float(g["grad_cosine_vs_planning_mean"]),
        }
    status = "PASS" if geometry["status"] == feature["status"] == "PASS" and not missing else "FAIL"
    result = {"status": status, "geometry": geometry, "feature": feature, "joint_3cam_leave_one_out": leaveouts, "missing_deliverables": missing,
        "interpretation_limit": "Raster-target content and alignment-gradient attribution only; no planner-performance causal claim."}
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    coverage = geometry["coverage_log_bootstrap_95ci"]
    lines = ["# Step 0-C raster components / ownership / ablation", "", f"Status: **{status}**", "", "## 执行范围", "",
        f"- 几何主扫描：{geometry['frames']} 帧 × F0/L0/R0；production RGB exact reproduction {geometry['rgb_reproduction']['exact_equal']}/{geometry['rgb_reproduction']['rows']}。",
        f"- feature forward：{feature['samples']} 帧；gradient：{feature['gradient_samples']} 帧；checkpoint SHA256 `{feature['checkpoint_sha256']}`。",
        "- B0 完全排除于 0-C loss/gradient 主统计；GridMask 关闭。", "", "## Final ownership coverage", "", "| scale | light | map | agent | blank |", "|---|---:|---:|---:|---:|"]
    for scale in ("render", "model"):
        vals = [coverage[f"{scale}.{name}"]["mean"] for name in ("traffic_light_coverage", "map_coverage", "agent_coverage", "blank_fraction")]
        lines.append("| {} | {} | {} | {} | {} |".format(scale, *[f"{value:.4%}" for value in vals]))
    lines += ["", "## Joint-3cam leave-one-out", "", "| removed component | Δ raw MSE mean (95% CI) | ||Δg|| mean | cos(g_variant,g_full) | cos(g_variant,g_planning probe) |", "|---|---:|---:|---:|---:|"]
    for component, value in leaveouts.items():
        lines.append(f"| {component} | {value['delta_mse_mean']:.6g} [{value['delta_mse_ci'][0]:.6g}, {value['delta_mse_ci'][1]:.6g}] | {value['delta_grad_norm_mean']:.6g} | {value['grad_cosine_full_mean']:.4f} | {value['grad_cosine_planning_mean']:.4f} |")
    lines += ["", "## 边界", "", "这些结果只描述 raster target 的内容、覆盖关系和 alignment gradient 归因。正负 ΔMSE 不代表 planner 性能改善或下降。planning cosine 使用与 0-F 相同的简化 trajectory component，不包含 PDM-score auxiliaries。", ""]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if missing:
        raise RuntimeError(f"missing deliverables: {missing}")


if __name__ == "__main__":
    main()
