"""Train offline MLP probes for RAP drift representations.
Usage:
python navsim/planning/script/tools/train_mlp_drift_probe.py \
    --input-dir outputs/drift_probe_features_perturbed \
    --output-dir outputs/drift_probe_features_perturbed/mlp_probe \
    --split scene \
    --device cuda
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# transformers>=4.56 may be imported indirectly by torch dynamo/onnx in this
# environment; torch 2.1 still exposes the same pytree helper under a private
# name.
if (
    not hasattr(torch.utils._pytree, "register_pytree_node")
    and hasattr(torch.utils._pytree, "_register_pytree_node")
):
    def _register_pytree_node_compat(typ, flatten_fn, unflatten_fn, **kwargs):
        kwargs.pop("serialized_type_name", None)
        return torch.utils._pytree._register_pytree_node(
            typ, flatten_fn, unflatten_fn, **kwargs
        )

    torch.utils._pytree.register_pytree_node = _register_pytree_node_compat


class DriftProbeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLP drift probes from collect_drift_probe_features.py output."
    )
    parser.add_argument("--input-dir", default="outputs/drift_probe_features")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="random", choices=["random", "scene", "heldout_drift"])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout-delta-lat", default="-0.5,0.5")
    parser.add_argument("--heldout-delta-yaw-deg", default="")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--linear-alpha", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=0, help="Use >0 to debug on first N feature groups.")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_info_from_name(path: Path, summary: Dict) -> Dict[str, object]:
    stem = path.stem
    if "features" in summary and stem in summary["features"]:
        info = dict(summary["features"][stem])
        info["feature_key"] = stem
        return info
    parts = stem.split("__call")
    hook_name = parts[0].replace("__", ".")
    call_rest = parts[1] if len(parts) > 1 else "0__unknown"
    call_idx_str, pooling_name = call_rest.split("__", 1)
    return {
        "feature_key": stem,
        "hook_name": hook_name,
        "call_idx": int(call_idx_str),
        "pooling_name": pooling_name,
    }


def _split_indices(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    n = len(df)
    all_indices = np.arange(n)

    if args.split == "random":
        shuffled = rng.permutation(all_indices)
        n_test = max(1, int(round(n * args.test_fraction)))
        test_idx = np.sort(shuffled[:n_test])
        train_idx = np.sort(shuffled[n_test:])
    elif args.split == "scene":
        scene_values = df["scene_token"].fillna(df["token"]).astype(str).to_numpy()
        scenes = np.unique(scene_values)
        shuffled_scenes = rng.permutation(scenes)
        n_test = max(1, int(round(len(scenes) * args.test_fraction)))
        test_scenes = set(shuffled_scenes[:n_test])
        test_mask = np.array([scene in test_scenes for scene in scene_values])
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]
    else:
        heldout_lat = np.asarray(_parse_float_list(args.heldout_delta_lat), dtype=np.float64)
        heldout_yaw_deg = np.asarray(_parse_float_list(args.heldout_delta_yaw_deg), dtype=np.float64)
        test_mask = np.zeros(n, dtype=bool)
        if len(heldout_lat):
            lat = df["delta_lat"].to_numpy(dtype=np.float64)
            test_mask |= np.isclose(lat[:, None], heldout_lat[None], atol=1e-6).any(axis=1)
        if len(heldout_yaw_deg):
            yaw_deg = df["delta_yaw_deg"].to_numpy(dtype=np.float64)
            test_mask |= np.isclose(yaw_deg[:, None], heldout_yaw_deg[None], atol=1e-6).any(axis=1)
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Invalid split {args.split}: train={len(train_idx)} test={len(test_idx)}")
    return train_idx, test_idx


def _standardize(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x[train_idx])
    x_test = x_scaler.transform(x[test_idx])
    y_train = y_scaler.fit_transform(y[train_idx])
    y_test = y_scaler.transform(y[test_idx])
    return x_train, x_test, y_train, y_test, x_scaler, y_scaler


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    model = DriftProbeMLP(x_train.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model.train()
    for _epoch in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred[:, 0], yb[:, 0]) + loss_fn(pred[:, 1], yb[:, 1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_test), args.batch_size):
            xb = torch.from_numpy(x_test[start : start + args.batch_size].astype(np.float32)).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE_delta_lat": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "MAE_delta_yaw": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "RMSE_delta_lat": float(math.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "RMSE_delta_yaw": float(math.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "R2_delta_lat": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "R2_delta_yaw": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "sign_accuracy_delta_lat": _sign_accuracy(y_true[:, 0], y_pred[:, 0]),
        "sign_accuracy_delta_yaw": _sign_accuracy(y_true[:, 1], y_pred[:, 1]),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _behavior_correlations(
    behavior_df: pd.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    b = behavior_df.iloc[test_idx].reset_index(drop=True)
    recovery_ratio = b["recovery_ratio"].to_numpy(dtype=np.float64)
    correction_sign = b["correction_direction_sign"].to_numpy(dtype=np.float64)
    return {
        "corr_delta_lat_hat_recovery_ratio": _corr(y_pred[:, 0], recovery_ratio),
        "corr_abs_delta_lat_hat_recovery_ratio": _corr(np.abs(y_pred[:, 0]), recovery_ratio),
        "corr_delta_lat_hat_correction_direction_sign": _corr(y_pred[:, 0], correction_sign),
        "corr_delta_lat_true_recovery_ratio": _corr(y_true[:, 0], recovery_ratio),
        "corr_delta_lat_true_correction_direction_sign": _corr(y_true[:, 0], correction_sign),
    }


def _fit_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha > 0:
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _save_json(path: Path, rows: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2))


def _scatter_plot(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(5.5, 5))
    plt.scatter(x, y, s=10, alpha=0.6)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        lo = float(min(x[finite].min(), y[finite].min()))
        hi = float(max(x[finite].max(), y[finite].max()))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _bar_plot(path: Path, labels: List[str], values: List[float], ylabel: str, title: str) -> None:
    plt.figure(figsize=(max(7, 0.5 * len(labels)), 4.5))
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _select_best_row(results: pd.DataFrame) -> Optional[pd.Series]:
    if len(results) == 0:
        return None
    rank_value = results["R2_delta_lat"].fillna(-np.inf) + results["R2_delta_yaw"].fillna(-np.inf)
    return results.iloc[int(rank_value.to_numpy().argmax())]


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "mlp_probe"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    metadata = pd.read_csv(input_dir / "metadata.csv")
    behavior = pd.read_csv(input_dir / "behavior_metrics.csv")
    with open(input_dir / "hook_summary.json", "r") as f:
        summary = json.load(f)

    y = metadata[["delta_lat", "delta_yaw"]].to_numpy(dtype=np.float64)
    train_idx, test_idx = _split_indices(metadata, args)

    feature_paths = sorted((input_dir / "features").glob("*.npy"))
    if args.max_features > 0:
        feature_paths = feature_paths[: args.max_features]
    if not feature_paths:
        raise FileNotFoundError(f"No .npy feature matrices found under {input_dir / 'features'}")

    result_rows: List[Dict[str, object]] = []
    controls_rows: List[Dict[str, object]] = []
    corr_rows: List[Dict[str, object]] = []
    prediction_cache: Dict[str, Dict[str, np.ndarray]] = {}

    for feature_path in tqdm(feature_paths, desc="Training drift probes"):
        info = _feature_info_from_name(feature_path, summary)
        x = np.load(feature_path)
        if len(x) != len(metadata):
            raise ValueError(f"{feature_path} has {len(x)} rows but metadata has {len(metadata)}")
        x = x.astype(np.float64)

        x_train, x_test, y_train_scaled, _y_test_scaled, _x_scaler, y_scaler = _standardize(
            x,
            y,
            train_idx,
            test_idx,
        )
        y_true_test = y[test_idx]

        pred_scaled = _train_mlp(x_train, y_train_scaled, x_test, args, device)
        pred = y_scaler.inverse_transform(pred_scaled)
        metrics = _metrics(y_true_test, pred)
        base = {
            "feature_key": info["feature_key"],
            "hook_name": info["hook_name"],
            "call_idx": int(info["call_idx"]),
            "pooling_name": info["pooling_name"],
            "model_type": "mlp",
            "num_train": int(len(train_idx)),
            "num_test": int(len(test_idx)),
            **metrics,
        }
        result_rows.append(base)
        corr_rows.append({**base, **_behavior_correlations(behavior, test_idx, y_true_test, pred)})
        prediction_cache[str(info["feature_key"])] = {
            "y_true": y_true_test,
            "mlp_pred": pred,
            "test_idx": test_idx,
        }

        linear_scaled = _fit_linear(x_train, y_train_scaled, x_test, args.linear_alpha)
        linear_pred = y_scaler.inverse_transform(linear_scaled)
        controls_rows.append(
            {
                **base,
                "model_type": "linear",
                **_metrics(y_true_test, linear_pred),
            }
        )

        shuffled_train = y_train_scaled.copy()
        np.random.default_rng(args.seed).shuffle(shuffled_train)
        shuffled_scaled = _train_mlp(x_train, shuffled_train, x_test, args, device)
        shuffled_pred = y_scaler.inverse_transform(shuffled_scaled)
        controls_rows.append(
            {
                **base,
                "model_type": "mlp_shuffled_labels",
                **_metrics(y_true_test, shuffled_pred),
            }
        )

    results_df = pd.DataFrame(result_rows)
    controls_df = pd.DataFrame(controls_rows)
    corr_df = pd.DataFrame(corr_rows)

    results_df.to_csv(output_dir / "mlp_probe_results.csv", index=False)
    _save_json(output_dir / "mlp_probe_results.json", result_rows)
    controls_df.to_csv(output_dir / "controls_results.csv", index=False)
    corr_df.to_csv(output_dir / "behavior_correlation.csv", index=False)

    ranking = results_df.copy()
    ranking["rank_score"] = ranking["R2_delta_lat"].fillna(-999.0) + ranking["R2_delta_yaw"].fillna(-999.0)
    ranking = ranking.sort_values("rank_score", ascending=False)
    ranking.to_csv(output_dir / "layer_callidx_pooling_ranking.csv", index=False)

    best = _select_best_row(results_df)
    if best is not None:
        key = str(best["feature_key"])
        cached = prediction_cache[key]
        y_true = cached["y_true"]
        y_pred = cached["mlp_pred"]
        _scatter_plot(
            plots_dir / "true_vs_pred_delta_lat.png",
            y_true[:, 0],
            y_pred[:, 0],
            "true delta_lat",
            "predicted delta_lat",
            f"Best MLP delta_lat: {best['hook_name']} call {best['call_idx']}",
        )
        _scatter_plot(
            plots_dir / "true_vs_pred_delta_yaw.png",
            y_true[:, 1],
            y_pred[:, 1],
            "true delta_yaw",
            "predicted delta_yaw",
            f"Best MLP delta_yaw: {best['hook_name']} call {best['call_idx']}",
        )
        test_behavior = behavior.iloc[cached["test_idx"]].reset_index(drop=True)
        _scatter_plot(
            plots_dir / "pred_delta_lat_vs_recovery_ratio.png",
            y_pred[:, 0],
            test_behavior["recovery_ratio"].to_numpy(dtype=np.float64),
            "predicted delta_lat",
            "recovery_ratio",
            "Probe prediction vs recovery behavior",
        )

    labels = [
        f"{row.hook_name}\ncall{int(row.call_idx)}"
        for row in results_df.itertuples()
    ]
    _bar_plot(
        plots_dir / "r2_by_hook_and_callidx.png",
        labels,
        results_df["R2_delta_lat"].to_numpy(dtype=np.float64).tolist(),
        "R2 delta_lat",
        "MLP drift decodability by hook/call_idx",
    )
    _bar_plot(
        plots_dir / "r2_delta_yaw_by_hook_and_callidx.png",
        labels,
        results_df["R2_delta_yaw"].to_numpy(dtype=np.float64).tolist(),
        "R2 delta_yaw",
        "MLP yaw drift decodability by hook/call_idx",
    )

    merged_control = results_df[["feature_key", "R2_delta_lat"]].rename(columns={"R2_delta_lat": "mlp_r2"})
    linear_control = controls_df[controls_df["model_type"] == "linear"][["feature_key", "R2_delta_lat"]].rename(
        columns={"R2_delta_lat": "linear_r2"}
    )
    shuffled_control = controls_df[controls_df["model_type"] == "mlp_shuffled_labels"][
        ["feature_key", "R2_delta_lat"]
    ].rename(columns={"R2_delta_lat": "shuffled_r2"})
    merged_control = merged_control.merge(linear_control, on="feature_key").merge(shuffled_control, on="feature_key")
    _bar_plot(
        plots_dir / "mlp_vs_linear_by_hook.png",
        merged_control["feature_key"].tolist(),
        (merged_control["mlp_r2"] - merged_control["linear_r2"]).to_numpy(dtype=np.float64).tolist(),
        "MLP R2 - linear R2 for delta_lat",
        "Nonlinear probe gain over linear control",
    )
    _bar_plot(
        plots_dir / "real_vs_shuffled_control.png",
        merged_control["feature_key"].tolist(),
        (merged_control["mlp_r2"] - merged_control["shuffled_r2"]).to_numpy(dtype=np.float64).tolist(),
        "real-label R2 - shuffled-label R2 for delta_lat",
        "Real probe vs shuffled-label control",
    )

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "num_features": len(feature_paths),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "note": (
            "Probe performance shows nonlinear decodability of drift labels from frozen RAP features; "
            "it does not establish causal use by RAP for recovery behavior."
        ),
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"Wrote probe results to {output_dir}")


if __name__ == "__main__":
    main()
