from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

import numpy as np

from navsim.planning.script.tools.validate_recovery_trajectory import (
    _metrics,
    compute_recovery_metrics,
)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def selected_and_best_proposal_metrics(
    prediction: Dict,
    target: np.ndarray,
    cv_baseline: np.ndarray,
    original_reference: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute deployment selected metrics and best-proposal upper-bound metrics."""

    selected = _to_numpy(prediction["trajectory"])
    target = _to_numpy(target)
    cv_baseline = _to_numpy(cv_baseline)
    if selected.ndim == 3:
        selected = selected[0]
    if target.ndim == 3:
        target = target[0]
    if cv_baseline.ndim == 3:
        cv_baseline = cv_baseline[0]

    result = {f"selected_{key}": value for key, value in _metrics(selected, target, cv_baseline).items()}

    proposals = prediction.get("proposals")
    if proposals is None:
        proposals = prediction.get("trajectory") if "score" in prediction else None
    if proposals is not None:
        proposals_np = _to_numpy(proposals)
        if proposals_np.ndim == 4:
            proposals_np = proposals_np[0]
        proposal_errors = np.linalg.norm(proposals_np[:, :, :2] - target[None, :, :2], axis=-1).mean(axis=-1)
        best_idx = int(np.argmin(proposal_errors))
        best = proposals_np[best_idx]
        best_metrics = _metrics(best, target, cv_baseline)
        result.update({f"best_proposal_{key}": value for key, value in best_metrics.items()})
        result["proposal_selection_ade_gap"] = result["selected_ade"] - result["best_proposal_ade"]
        result["proposal_selection_fde_gap"] = result["selected_fde"] - result["best_proposal_fde"]

    recovery_metrics, error = compute_recovery_metrics(selected, target, original_reference)
    result.update({f"recovery_{key}": value for key, value in recovery_metrics.items()})
    if error is not None:
        result["recovery_metrics_error"] = error
    return result


def _jsonable(metrics: Dict) -> Dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _load_prediction_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    prediction = {"trajectory": data["trajectory"]}
    if "proposals" in data:
        prediction["proposals"] = data["proposals"]
    if "score" in data:
        prediction["score"] = data["score"]
    return {
        "prediction": prediction,
        "target": data["target"],
        "cv_baseline": data["cv_baseline"],
        "original_reference": data["original_reference"] if "original_reference" in data else None,
    }


def _parse_named_npz(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --prediction-npz")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("NAME cannot be empty")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute selected/best-proposal and recovery metrics from npz dumps.")
    parser.add_argument(
        "--prediction-npz",
        action="append",
        type=_parse_named_npz,
        required=True,
        help="NAME=PATH npz with trajectory, target, cv_baseline, optional proposals/original_reference.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    table = {}
    for name, path in args.prediction_npz:
        sample = _load_prediction_npz(path)
        table[name] = _jsonable(
            selected_and_best_proposal_metrics(
                sample["prediction"],
                sample["target"],
                sample["cv_baseline"],
                sample["original_reference"],
            )
        )

    text = json.dumps(table, indent=2, sort_keys=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    print(text)


if __name__ == "__main__":
    main()
