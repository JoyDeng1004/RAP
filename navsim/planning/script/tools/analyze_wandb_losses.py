#!/usr/bin/env python3
"""Analyze local ref2d wandb loss export CSVs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_EXPORT_DIR = Path("outputs/wandb_loss_export")
DEFAULT_FIG_DIR = Path("docs/figs/wandb_loss")
CELL_ORDER = [
    "baseline",
    "shift_only_log",
    "recovery_only_log",
    "recovery_aux_only_log_l03",
    "offset_recovery_log",
    "offset_recovery_aux_log_l03",
]
LOSS_WEIGHTS = {
    "trajectory_loss": 1.0,
    "inter_loss": 0.0,
    "sub_score_loss": 0.0,
    "final_score_loss": 1.0,
    "pred_ce_loss": 1.0,
    "pred_l1_loss": 0.1,
    "pred_area_loss": 2.0,
    "recovery_aux_trajectory_loss": 0.3,
}
CORE_METRICS = [
    "loss",
    "trajectory_loss",
    "final_score_loss",
    "pred_ce_loss",
    "pred_l1_loss",
    "pred_area_loss",
    "recovery_aux_trajectory_loss",
    "score",
    "best_score",
    "min_loss",
    "inter_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def load_long(export_dir: Path) -> pd.DataFrame:
    path = export_dir / "all_runs_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run export_wandb_losses.py first")
    df = pd.read_csv(path)
    df = df[df["aggregation"].eq("epoch")].copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["epoch", "value"])
    df["epoch"] = df["epoch"].astype(int)
    return df


def tail_slope(group: pd.DataFrame) -> float:
    group = group.sort_values("epoch")
    if len(group) < 2:
        return float("nan")
    tail_n = max(2, math.ceil(len(group) * 0.2))
    tail = group.tail(tail_n)
    x = tail["epoch"].astype(float)
    y = tail["value"].astype(float)
    if x.nunique() < 2:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def summarize_health(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, cell, split, metric), group in df.groupby(["run", "cell", "split", "metric"]):
        group = group.sort_values("epoch")
        values = group["value"]
        tail_n = max(2, math.ceil(len(group) * 0.2))
        tail = values.tail(tail_n)
        min_idx = values.idxmin()
        max_ratio = values.abs().max() / max(values.abs().median(), 1e-12)
        rows.append(
            {
                "run": run,
                "cell": cell,
                "split": split,
                "metric": metric,
                "n_epoch": len(group),
                "first": values.iloc[0],
                "last": values.iloc[-1],
                "min": values.loc[min_idx],
                "min_epoch": int(group.loc[min_idx, "epoch"]),
                "tail_mean": tail.mean(),
                "tail_std": tail.std(ddof=0),
                "tail_cv": tail.std(ddof=0) / max(abs(tail.mean()), 1e-12),
                "tail_slope": tail_slope(group),
                "nan_count": values.isna().sum(),
                "inf_count": (~values.apply(math.isfinite)).sum(),
                "max_abs_over_median_abs": max_ratio,
                "explosion_flag": max_ratio > 10,
            }
        )
    return pd.DataFrame(rows)


def summarize_gap(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["run", "cell", "metric", "epoch"],
        columns="split",
        values="value",
        aggfunc="last",
    ).reset_index()
    if "train" not in pivot or "val" not in pivot:
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["train", "val"])
    pivot["gap"] = pivot["val"] - pivot["train"]
    rows = []
    for (run, cell, metric), group in pivot.groupby(["run", "cell", "metric"]):
        group = group.sort_values("epoch")
        best_idx = group["val"].idxmin()
        rows.append(
            {
                "run": run,
                "cell": cell,
                "metric": metric,
                "n_epoch": len(group),
                "first_gap": group["gap"].iloc[0],
                "last_gap": group["gap"].iloc[-1],
                "mean_gap": group["gap"].mean(),
                "tail_gap": group["gap"].tail(max(2, math.ceil(len(group) * 0.2))).mean(),
                "gap_slope": tail_slope(group.rename(columns={"gap": "value"})),
                "best_val_epoch": int(group.loc[best_idx, "epoch"]),
                "best_val": group.loc[best_idx, "val"],
                "last_train": group["train"].iloc[-1],
                "last_val": group["val"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def summarize_cells(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, cell, split, metric), group in df.groupby(["run", "cell", "split", "metric"]):
        group = group.sort_values("epoch")
        min_idx = group["value"].idxmin()
        rows.append(
            {
                "run": run,
                "cell": cell,
                "split": split,
                "metric": metric,
                "first": group["value"].iloc[0],
                "last": group["value"].iloc[-1],
                "min": group.loc[min_idx, "value"],
                "min_epoch": int(group.loc[min_idx, "epoch"]),
                "mean": group["value"].mean(),
                "tail_mean": group["value"].tail(max(2, math.ceil(len(group) * 0.2))).mean(),
            }
        )
    out = pd.DataFrame(rows)
    baseline = out[out["cell"].eq("baseline")][["split", "metric", "last", "min", "tail_mean"]]
    baseline = baseline.rename(columns={"last": "baseline_last", "min": "baseline_min", "tail_mean": "baseline_tail_mean"})
    out = out.merge(baseline, on=["split", "metric"], how="left")
    out["delta_last_vs_baseline"] = out["last"] - out["baseline_last"]
    out["delta_min_vs_baseline"] = out["min"] - out["baseline_min"]
    out["delta_tail_mean_vs_baseline"] = out["tail_mean"] - out["baseline_tail_mean"]
    return out


def summarize_contrib(df: pd.DataFrame) -> pd.DataFrame:
    epoch_df = df[df["metric"].isin(set(LOSS_WEIGHTS) | {"loss"})]
    pivot = epoch_df.pivot_table(index=["run", "cell", "split", "epoch"], columns="metric", values="value", aggfunc="last").reset_index()
    rows = []
    for _, row in pivot.iterrows():
        loss = row.get("loss")
        if pd.isna(loss) or abs(loss) < 1e-12:
            continue
        for metric, weight in LOSS_WEIGHTS.items():
            if metric not in row or pd.isna(row[metric]):
                continue
            contribution = weight * row[metric]
            rows.append(
                {
                    "run": row["run"],
                    "cell": row["cell"],
                    "split": row["split"],
                    "epoch": int(row["epoch"]),
                    "metric": metric,
                    "raw_value": row[metric],
                    "weight": weight,
                    "weighted_contribution": contribution,
                    "contribution_share_of_loss": contribution / loss,
                }
            )
    contrib = pd.DataFrame(rows)
    if contrib.empty:
        return contrib
    return (
        contrib.sort_values("epoch")
        .groupby(["run", "cell", "split", "metric"], as_index=False)
        .tail(1)
        .sort_values(["cell", "split", "metric"])
    )


def plot_metric(df: pd.DataFrame, metric: str, split: str, fig_dir: Path) -> Path | None:
    data = df[(df["metric"].eq(metric)) & (df["split"].eq(split))]
    if data.empty:
        return None
    plt.figure(figsize=(9, 5))
    for cell in CELL_ORDER:
        group = data[data["cell"].eq(cell)].sort_values("epoch")
        if group.empty:
            continue
        plt.plot(group["epoch"], group["value"], marker="o", linewidth=1.8, label=cell)
    plt.xlabel("epoch")
    plt.ylabel(f"{split}/{metric}")
    plt.title(f"{split}/{metric}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{split}_{metric}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_train_val(df: pd.DataFrame, metric: str, fig_dir: Path) -> Path | None:
    data = df[df["metric"].eq(metric)]
    if data.empty:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    plotted = False
    for ax, cell in zip(axes, CELL_ORDER):
        cell_data = data[data["cell"].eq(cell)]
        for split, style in (("train", "-"), ("val", "--")):
            group = cell_data[cell_data["split"].eq(split)].sort_values("epoch")
            if not group.empty:
                plotted = True
                ax.plot(group["epoch"], group["value"], style, marker="o", label=split)
        ax.set_title(cell, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    if not plotted:
        plt.close(fig)
        return None
    fig.suptitle(f"train vs val: {metric}")
    fig.supxlabel("epoch")
    fig.supylabel(metric)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"train_val_{metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def write_outputs(df: pd.DataFrame, export_dir: Path, fig_dir: Path) -> None:
    health = summarize_health(df)
    gap = summarize_gap(df)
    cells = summarize_cells(df)
    contrib = summarize_contrib(df)
    health.to_csv(export_dir / "summary_health.csv", index=False)
    gap.to_csv(export_dir / "summary_gap.csv", index=False)
    cells.to_csv(export_dir / "summary_cells.csv", index=False)
    contrib.to_csv(export_dir / "summary_contributions.csv", index=False)

    for metric in CORE_METRICS:
        plot_metric(df, metric, "val", fig_dir)
        if metric in {"loss", "trajectory_loss", "score", "best_score", "recovery_aux_trajectory_loss"}:
            plot_train_val(df, metric, fig_dir)


def main() -> None:
    args = parse_args()
    df = load_long(args.export_dir)
    write_outputs(df, args.export_dir, args.fig_dir)
    print(f"Wrote summaries to {args.export_dir}")
    print(f"Wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
