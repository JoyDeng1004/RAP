#!/usr/bin/env python3
"""Record-only distribution audit for the frozen Stage-A subset."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Callable


ROUTE_NAMES = ("TURN_LEFT", "GO_STRAIGHT", "TURN_RIGHT", "UNKNOWN")


def _route_command(record: dict[str, Any]) -> str:
    command = record["driving_command"]
    if not isinstance(command, list) or len(command) != 4:
        raise ValueError(f"invalid driving_command for {record.get('token')}: {command!r}")
    active = [index for index, value in enumerate(command) if value == 1]
    if len(active) != 1 or any(value not in (0, 1) for value in command):
        raise ValueError(f"driving_command is not one-hot for {record.get('token')}: {command!r}")
    return ROUTE_NAMES[active[0]]


def _actor_bin(record: dict[str, Any]) -> str:
    value = int(record["actor_count_30m"])
    if value <= 5:
        return "0-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    return ">20"


def _continuous_bin(field: str) -> Callable[[dict[str, Any]], str]:
    def classify(record: dict[str, Any]) -> str:
        value = float(record[field])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {field} for {record.get('token')}: {value}")
        if value < -2:
            return "<-2"
        if value < -0.5:
            return "-2~-0.5"
        if value <= 0.5:
            return "-0.5~0.5"
        if value <= 2:
            return "0.5~2"
        return ">2"

    return classify


def _proportions(records: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, float]:
    if not records:
        raise ValueError("cannot audit an empty record set")
    counts = Counter(key(record) for record in records)
    return {name: counts[name] / len(records) for name in sorted(counts)}


def _distribution(
    full: list[dict[str, Any]], selected: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    full_p = _proportions(full, key)
    selected_p = _proportions(selected, key)
    categories = sorted(set(full_p) | set(selected_p))
    table = {
        category: {"full": full_p.get(category, 0.0), "subset": selected_p.get(category, 0.0)}
        for category in categories
    }
    tvd = 0.5 * sum(abs(row["full"] - row["subset"]) for row in table.values())
    return {"full_count": len(full), "subset_count": len(selected), "proportions": table, "tvd": tvd}


def _sampling_rates(full: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    full_counts = Counter(record["log_name"] for record in full)
    selected_counts = Counter(record["log_name"] for record in selected)
    if set(selected_counts) != set(full_counts):
        missing = sorted(set(full_counts) - set(selected_counts))
        extra = sorted(set(selected_counts) - set(full_counts))
        raise ValueError(f"subset log coverage mismatch: missing={missing}, extra={extra}")
    rates = {name: selected_counts[name] / count for name, count in sorted(full_counts.items())}
    values = list(rates.values())
    return {
        "per_log": rates,
        "min": min(values),
        "max": max(values),
        "median": median(values),
        "max_min_ratio": max(values) / min(values),
    }


def _threshold(thresholds: dict[str, Any], key: str) -> float:
    aliases = {
        "categorical_tvd": ("categorical_tvd", "categorical_tvd_max"),
        "continuous_tvd": ("continuous_tvd", "binned_continuous_tvd_max"),
        "sampling_ratio": ("per_log_sampling_rate_max_min", "per_log_sampling_rate_max_min_ratio"),
    }
    for alias in aliases[key]:
        if alias in thresholds:
            return float(thresholds[alias])
    raise KeyError(f"threshold file lacks one of {aliases[key]}")


def audit(token_manifest: Path, dataset_manifest: Path, thresholds_path: Path) -> dict[str, Any]:
    token_data = json.loads(token_manifest.read_text())
    dataset_data = json.loads(dataset_manifest.read_text())
    thresholds = json.loads(thresholds_path.read_text())
    qualified = token_data["qualified_candidates"]
    selected = dataset_data["samples"]

    wanted = set(token_data["train_tokens"]) | set(token_data["val_tokens"])
    selected_tokens = [record["token"] for record in selected]
    if len(selected_tokens) != len(set(selected_tokens)) or set(selected_tokens) != wanted:
        raise ValueError("dataset_manifest samples do not exactly match the frozen token manifest")

    metrics: dict[str, dict[str, Any]] = {}
    sampling: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []
    specs = {
        "route_command": (_route_command, "categorical_tvd"),
        "map_location": (lambda record: str(record["map_location"]), "categorical_tvd"),
        "actor_density_bin": (_actor_bin, "continuous_tvd"),
        "endpoint_dx_bin": (_continuous_bin("endpoint_dx"), "continuous_tvd"),
        "endpoint_dy_bin": (_continuous_bin("endpoint_dy"), "continuous_tvd"),
        "endpoint_dyaw_bin": (_continuous_bin("endpoint_dyaw"), "continuous_tvd"),
    }
    for split in ("train", "val"):
        full_split = [record for record in qualified if record["split"] == split]
        selected_split = [record for record in selected if record["split"] == split]
        metrics[split] = {}
        for name, (classifier, threshold_key) in specs.items():
            result = _distribution(full_split, selected_split, classifier)
            result["threshold"] = _threshold(thresholds, threshold_key)
            result["passed"] = result["tvd"] <= result["threshold"]
            metrics[split][name] = result
            if not result["passed"]:
                violations.append(
                    {"split": split, "metric": name, "value": result["tvd"], "threshold": result["threshold"]}
                )
        sampling[split] = _sampling_rates(full_split, selected_split)
        sampling[split]["threshold"] = _threshold(thresholds, "sampling_ratio")
        sampling[split]["passed"] = sampling[split]["max_min_ratio"] <= sampling[split]["threshold"]
        if not sampling[split]["passed"]:
            violations.append(
                {
                    "split": split,
                    "metric": "per_log_sampling_rate_max_min_ratio",
                    "value": sampling[split]["max_min_ratio"],
                    "threshold": sampling[split]["threshold"],
                }
            )

    summary = {
        split: {
            "full_count": sum(record["split"] == split for record in qualified),
            "subset_count": sum(record["split"] == split for record in selected),
            "tvds": {name: result["tvd"] for name, result in metrics[split].items()},
            "per_log_sampling_rate_max_min_ratio": sampling[split]["max_min_ratio"],
        }
        for split in ("train", "val")
    }
    return {
        "status": "passed" if not violations else "failed",
        "record_only": True,
        "thresholds": thresholds,
        "metrics": metrics,
        "per_log_sampling_rate": sampling,
        "summary": summary,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.token_manifest, args.dataset_manifest, args.thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(f"distribution audit failed: {result['violations']}")


if __name__ == "__main__":
    main()
