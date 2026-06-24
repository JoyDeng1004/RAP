#!/usr/bin/env python3
"""Export ref2d loss histories to local CSV files.

The primary offline path parses Lightning progress-bar metrics from local
training logs. The optional online path uses the wandb API when requested.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_OUT_DIR = Path("outputs/wandb_loss_export")
DEFAULT_LOG_DIR = Path("outputs/ref2d_matrix_e10/logs")
DEFAULT_PROJECT = "rap"
DEFAULT_PREFIX = "ref2d_"
CELL_ALIASES = {
    "shift_only": "shift_only_log",
    "recovery_only": "recovery_only_log",
    "offset_recovery": "offset_recovery_log",
}
RUN_TO_CELL_PREFIXES = ("ref2d_e10_", "ref2d_")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_RE = re.compile(r"Epoch\s+(\d+):")
METRIC_RE = re.compile(
    r"((?:train|val)/[A-Za-z0-9_]+_(?:step|epoch))="
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", help="Explicit wandb run ids to export.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Run id prefix. Default: {DEFAULT_PREFIX}")
    parser.add_argument("--exclude-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--source",
        choices=("logs", "wandb"),
        default="logs",
        help="Use local training logs or the wandb API. Default: logs",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def run_to_cell(run: str) -> str:
    cell = run
    for prefix in RUN_TO_CELL_PREFIXES:
        if cell.startswith(prefix):
            cell = cell[len(prefix) :]
            break
    return CELL_ALIASES.get(cell, cell)


def log_name_to_run(log_path: Path) -> str:
    return f"ref2d_e10_{CELL_ALIASES.get(log_path.stem, log_path.stem)}"


def split_metric_key(key: str) -> tuple[str, str, str]:
    split, rest = key.split("/", 1)
    if rest.endswith("_epoch"):
        return split, rest[: -len("_epoch")], "epoch"
    if rest.endswith("_step"):
        return split, rest[: -len("_step")], "step"
    return split, rest, "raw"


def clean_log_text(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_log(log_path: Path, run: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    last_by_epoch: dict[int, dict[str, float | int | str]] = {}
    event_index = 0

    text = clean_log_text(log_path.read_text(errors="replace"))
    for segment in re.split(r"[\r\n]+", text):
        metric_pairs = METRIC_RE.findall(segment)
        if not metric_pairs:
            continue
        epoch_match = EPOCH_RE.search(segment)
        epoch = int(epoch_match.group(1)) if epoch_match else None
        row: dict[str, float | int | str] = {"step": event_index}
        if epoch is not None:
            row["epoch"] = epoch
        for key, raw_value in metric_pairs:
            row[key] = float(raw_value)
        rows.append(row)
        event_index += 1

        if epoch is not None and any(key.endswith("_epoch") for key, _ in metric_pairs):
            last_by_epoch[epoch] = row

    if last_by_epoch:
        rows = [last_by_epoch[epoch] for epoch in sorted(last_by_epoch)]

    for index, row in enumerate(rows):
        row.setdefault("step", index)
        if "epoch" not in row:
            row["epoch"] = index
    return rows


def export_logs(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_paths = sorted(args.log_dir.glob("*.log"))
    if args.runs:
        wanted = set(args.runs)
        log_paths = [p for p in log_paths if log_name_to_run(p) in wanted]

    exported: list[tuple[str, str, Path]] = []
    long_rows: list[dict[str, float | int | str]] = []
    for log_path in log_paths:
        run = log_name_to_run(log_path)
        if not run.startswith(args.prefix):
            continue
        if args.exclude_smoke and run.endswith("_smoke"):
            continue
        rows = parse_log(log_path, run)
        if not rows:
            print(f"warning: no metrics parsed from {log_path}", file=sys.stderr)
            continue

        fields = ["step", "epoch"] + sorted({key for row in rows for key in row if key not in {"step", "epoch"}})
        out_csv = args.out_dir / f"{run}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        cell = run_to_cell(run)
        for row in rows:
            for key, value in row.items():
                if key in {"step", "epoch"} or not isinstance(value, float):
                    continue
                split, metric, aggregation = split_metric_key(key)
                long_rows.append(
                    {
                        "run": run,
                        "cell": cell,
                        "step": row["step"],
                        "epoch": row["epoch"],
                        "metric": metric,
                        "split": split,
                        "aggregation": aggregation,
                        "value": value,
                    }
                )
        exported.append((run, cell, out_csv))

    long_csv = args.out_dir / "all_runs_long.csv"
    with long_csv.open("w", newline="") as f:
        fields = ["run", "cell", "step", "epoch", "metric", "split", "aggregation", "value"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)
    return exported


def export_wandb(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    import wandb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    if args.runs:
        runs = [api.run(f"{project_path}/{run}") for run in args.runs]
    else:
        runs = [run for run in api.runs(project_path) if run.id.startswith(args.prefix)]

    exported: list[tuple[str, str, Path]] = []
    long_rows: list[dict[str, float | int | str]] = []
    for run_obj in runs:
        run = run_obj.id
        if args.exclude_smoke and run.endswith("_smoke"):
            continue
        try:
            rows = list(run_obj.scan_history())
        except Exception as exc:
            print(f"warning: failed to export {run}: {exc}", file=sys.stderr)
            continue
        if not rows:
            print(f"warning: no history for {run}", file=sys.stderr)
            continue

        normalized = []
        for idx, row in enumerate(rows):
            normalized_row = {"step": row.get("_step", idx), "epoch": row.get("epoch", "")}
            for key, value in row.items():
                if "/" in key and isinstance(value, (int, float)):
                    normalized_row[key] = value
            normalized.append(normalized_row)

        fields = ["step", "epoch"] + sorted({key for row in normalized for key in row if key not in {"step", "epoch"}})
        out_csv = args.out_dir / f"{run}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(normalized)

        cell = run_to_cell(run)
        for row in normalized:
            for key, value in row.items():
                if key in {"step", "epoch"} or not isinstance(value, (int, float)):
                    continue
                split, metric, aggregation = split_metric_key(key)
                long_rows.append(
                    {
                        "run": run,
                        "cell": cell,
                        "step": row["step"],
                        "epoch": row["epoch"],
                        "metric": metric,
                        "split": split,
                        "aggregation": aggregation,
                        "value": value,
                    }
                )
        exported.append((run, cell, out_csv))

    with (args.out_dir / "all_runs_long.csv").open("w", newline="") as f:
        fields = ["run", "cell", "step", "epoch", "metric", "split", "aggregation", "value"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)
    return exported


def main() -> None:
    args = parse_args()
    if args.source == "logs":
        exported = export_logs(args)
    else:
        exported = export_wandb(args)
    print(f"Exported {len(exported)} runs to {args.out_dir}")
    for run, cell, path in exported:
        print(f"{run}\t{cell}\t{path}")


if __name__ == "__main__":
    main()
