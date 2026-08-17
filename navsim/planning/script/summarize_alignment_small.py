"""Summarize the paired 32/16 alignment smoke run without overstating it."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "outputs/alignment_small"
CONDITIONS = {
    "r100-noalign": OUTPUT_ROOT / "alignment-small-r100-noalign/seed_0",
    "r100-fullalign": OUTPUT_ROOT / "alignment-small-r100-fullalign/seed_0",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    manifests = {
        name: json.load((path / "reproducibility_manifest.json").open(encoding="utf-8"))
        for name, path in CONDITIONS.items()
    }
    no_manifest, full_manifest = manifests.values()
    controlled_equalities = {
        key: no_manifest[key] == full_manifest[key]
        for key in (
            "git_commit", "seed", "initialization_checkpoint",
            "initialization_checkpoint_sha256", "train_tokens", "val_tokens",
        )
    }
    if not all(controlled_equalities.values()):
        raise RuntimeError(f"Controlled inputs differ: {controlled_equalities}")

    condition_rows = {
        name: _read_csv(path / "per_token_metrics.csv")
        for name, path in CONDITIONS.items()
    }
    indexed = {
        name: {row["token"]: row for row in rows}
        for name, rows in condition_rows.items()
    }
    no_tokens, full_tokens = set(indexed["r100-noalign"]), set(indexed["r100-fullalign"])
    if no_tokens != full_tokens or no_tokens != set(no_manifest["val_tokens"]):
        raise RuntimeError("Evaluation tokens do not match the locked validation manifest")

    paired_rows = []
    for token in sorted(no_tokens):
        no_row, full_row = indexed["r100-noalign"][token], indexed["r100-fullalign"][token]
        row = {"token": token, "log_name": no_row["log_name"]}
        for metric in ("ade_real", "fde_real", "score", "best_score", "loss", "trajectory_loss"):
            no_value, full_value = float(no_row[metric]), float(full_row[metric])
            row[f"noalign_{metric}"] = no_value
            row[f"fullalign_{metric}"] = full_value
            row[f"delta_{metric}"] = full_value - no_value
        paired_rows.append(row)

    paired_path = OUTPUT_ROOT / "paired_validation_metrics.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    metric_summary = {}
    for metric in ("ade_real", "fde_real", "score", "best_score", "loss", "trajectory_loss"):
        no_values = [row[f"noalign_{metric}"] for row in paired_rows]
        full_values = [row[f"fullalign_{metric}"] for row in paired_rows]
        deltas = [row[f"delta_{metric}"] for row in paired_rows]
        metric_summary[metric] = {
            "noalign": _summary(no_values),
            "fullalign": _summary(full_values),
            "paired_delta_fullalign_minus_noalign": _summary(deltas),
            "delta_positive_count": sum(value > 0 for value in deltas),
            "delta_negative_count": sum(value < 0 for value in deltas),
            "delta_zero_count": sum(value == 0 for value in deltas),
        }

    data_audit = json.load(
        (OUTPUT_ROOT / "input_data/input_audit.json").open(encoding="utf-8")
    )
    token_manifest = json.load(
        (OUTPUT_ROOT / "input_data/token_manifest.json").open(encoding="utf-8")
    )
    generation_mae = [
        item["front_reproduction_mae_0_to_255"]
        for item in token_manifest["back_camera_generation"]
    ]
    summary = {
        "experiment_class": "continued-alignment 32/16 smoke diagnostic",
        "research_conclusion_allowed": False,
        "reason": (
            "single seed, 16 validation tokens, 16 optimizer steps, unknown checkpoint training provenance, "
            "and CAM_B0 was generated only after three-camera subset eligibility"
        ),
        "controlled_equalities": controlled_equalities,
        "initialization_checkpoint_sha256": no_manifest["initialization_checkpoint_sha256"],
        "data_audit": data_audit,
        "generated_back_camera_front_reproduction_mae": _summary(generation_mae),
        "num_paired_validation_tokens": len(paired_rows),
        "metrics": metric_summary,
        "not_measured": [
            "NAVSIM v2 EPDMS", "NAVSIM v2 Stage 1/Stage 2 metrics",
            "NAVSIM v2 per-rollout sub-score transitions", "multi-seed uncertainty",
        ],
    }
    with (OUTPUT_ROOT / "alignment_small_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    ade = metric_summary["ade_real"]
    fde = metric_summary["fde_real"]
    markdown = f"""# RAP alignment small smoke 结果

## 当前结论

这次运行只证明修正后的四相机 raw-image → paired cache → loss → checkpoint → 16-token real validation 链路能够跑通。**它不能回答 alignment 是否导致 planning regression。**

限制是单 seed、16 个 validation token、仅 16 个 optimizer step、初始化 checkpoint 的完整训练来源未知，而且原 raster root 没有 `CAM_B0`；本次只对 hash 入选 token 补渲染了 `CAM_B0`。

## 已确认的控制变量

- NoAlign/FullAlign checkpoint SHA256 相同：`{no_manifest['initialization_checkpoint_sha256']}`。
- 加载日志显示该 checkpoint 不含 `domain_classifier` 的 6 个参数；两组按同一 `seed=0` 随机初始化这部分。checkpoint 是否含 spatial alignment 训练历史仍未知。
- train/val token 完全相同：`32 / 16`；两组均只用 real planning supervision。
- real/raster/target root 见 `input_data/input_audit.json`，target 逐字节一致。
- `CAM_B0` 生成时对已有 `CAM_F0` 的复现 MAE：mean `{statistics.mean(generation_mae):.4f}`，max `{max(generation_mae):.4f}`（像素范围 0–255）。

## 16-token paired real validation

| Metric | NoAlign mean | FullAlign mean | Paired delta mean | delta > 0 / < 0 / = 0 |
|---|---:|---:|---:|---:|
| ADE (lower is better) | {ade['noalign']['mean']:.6f} | {ade['fullalign']['mean']:.6f} | {ade['paired_delta_fullalign_minus_noalign']['mean']:+.6f} | {ade['delta_positive_count']} / {ade['delta_negative_count']} / {ade['delta_zero_count']} |
| FDE (lower is better) | {fde['noalign']['mean']:.6f} | {fde['fullalign']['mean']:.6f} | {fde['paired_delta_fullalign_minus_noalign']['mean']:+.6f} | {fde['delta_positive_count']} / {fde['delta_negative_count']} / {fde['delta_zero_count']} |

这里的 ADE/FDE delta 只能描述这次 smoke 的方向，不能解释成总体 regression effect。

## 没有测量

- NAVSIM v2 EPDMS、Stage 1/Stage 2 和 sub-score transition；
- 多 seed 方差或置信区间；
- 完整合格池 small pilot；
- 论文规模 confirmatory experiment。
"""
    (ROOT / "docs/rap-alignment-small-results.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
