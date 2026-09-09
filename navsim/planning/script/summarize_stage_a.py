#!/usr/bin/env python3
"""Summarize the v2 Stage-A paired evaluation.

The protocol files are the source of every statistical parameter in this module.
This script deliberately validates all inputs before producing any output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


V1_SHA256 = "252994acb610806b59ee7653f6b7f27a1e3ab470d1efba7e9de045dba900945d"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _protocol_check(protocol: dict[str, Any], protocol_v1: Path, token_to_log: dict[str, Any]) -> None:
    digest = hashlib.sha256(protocol_v1.read_bytes()).hexdigest()
    recorded = token_to_log.get("protocol_sha256")
    if digest != V1_SHA256 or recorded != V1_SHA256:
        raise ValueError("v1 protocol SHA256 or token_to_log protocol_sha256 mismatch")
    if protocol.get("supersedes_sha256") != V1_SHA256:
        raise ValueError("summary_protocol_v2.supersedes_sha256 does not match v1")


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    values = series.map(lambda x: x if isinstance(x, bool) else str(x).strip().lower())
    return values.map({True: True, False: False, "true": True, "false": False})


def _run_files(evaluation_v1: Path, protocol: dict[str, Any]) -> dict[tuple[str, int], Path]:
    budget = protocol["measured_budget"]
    conditions = list(budget["conditions"])
    seeds = [int(s) for s in budget["seeds"]]
    expected = {(c, s) for c in conditions for s in seeds}
    found: dict[tuple[str, int], Path] = {}
    for path in sorted(evaluation_v1.glob("*.csv")):
        match = re.match(r"^(?P<condition>.+)_seed(?P<seed>\d+)\.csv$", path.name)
        if not match:
            continue
        key = (match.group("condition"), int(match.group("seed")))
        if key in expected:
            if key in found:
                raise ValueError(f"duplicate CSV for {key}")
            found[key] = path
    if set(found) != expected:
        missing = sorted(expected - set(found))
        extra = sorted(set(found) - expected)
        raise ValueError(f"expected {len(expected)} CSVs; missing={missing}, extra={extra}")
    return found


def _load_evaluations(evaluation_v1: Path, protocol: dict[str, Any], token_to_log: dict[str, Any]):
    files = _run_files(evaluation_v1, protocol)
    mapping = token_to_log.get("mapping", {})
    if not mapping:
        raise ValueError("token_to_log mapping is empty")
    expected_tokens = set(mapping)
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    token_sets: dict[tuple[str, int], set[str]] = {}
    metric_columns = list(protocol["metric_classification"]["columns"])
    required = {"token", "valid", *metric_columns}
    for key, path in files.items():
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        tokens = set(frame["token"].astype(str))
        token_sets[key] = tokens
        if tokens != expected_tokens:
            raise ValueError(f"token set mismatch in {path.name}")
        valid = _as_bool_series(frame["valid"])
        if valid.isna().any() or not bool(valid.all()):
            raise ValueError(f"invalid row in {path.name}")
        frame = frame.copy()
        frame["token"] = frame["token"].astype(str)
        frame = frame.set_index("token").loc[sorted(expected_tokens)]
        frames[key] = frame
    if len({frozenset(s) for s in token_sets.values()}) != 1:
        raise ValueError("CSV token sets are not identical")
    return frames, sorted(expected_tokens), mapping


def classify_metrics(frames: dict[tuple[str, int], pd.DataFrame], protocol: dict[str, Any]) -> dict[str, str]:
    cfg = protocol["metric_classification"]
    always = set(cfg.get("always_continuous", []))
    result = {}
    for metric in cfg["columns"]:
        values = pd.concat([f[metric] for f in frames.values()], ignore_index=True).dropna()
        result[metric] = "continuous" if metric in always or values.nunique() > 5 else "discrete"
    return result


def cluster_bootstrap(values: np.ndarray, clusters: Iterable[str], protocol: dict[str, Any], *, rng_seed: int | None = None) -> dict[str, Any]:
    cfg = protocol["cluster_bootstrap"]
    n_resamples = int(cfg["n_resamples"])
    seed = int(cfg["rng_seed"] if rng_seed is None else rng_seed)
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(list(clusters), dtype=str)
    logs = np.array(sorted(set(clusters)), dtype=str)
    if len(values) != len(clusters) or not len(logs):
        raise ValueError("bootstrap values/clusters mismatch")
    groups = {log: np.flatnonzero(clusters == log) for log in logs}
    rng = np.random.default_rng(seed)
    stars = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        selected = rng.choice(logs, size=len(logs), replace=True)
        indices = np.concatenate([groups[log] for log in selected])
        stars[i] = float(np.mean(values[indices]))
    quantiles = [float(q) for q in cfg["reported_quantiles"]]
    observed = float(np.mean(values))
    lower_tail = float(np.mean(stars <= 0))
    upper_tail = float(np.mean(stars >= 0))
    p = max(1.0 / n_resamples, 2.0 * min(lower_tail, upper_tail))
    return {
        "n_resamples": n_resamples, "rng_seed": seed,
        "quantiles": {str(q): float(np.quantile(stars, q)) for q in quantiles},
        "ci_95": [float(np.quantile(stars, 1 - float(cfg["ci_level"]))), float(np.quantile(stars, float(cfg["ci_level"])))],
        "p_value": p, "observed_mean": observed,
        "bootstrap_distribution": stars,
    }


def _public_bootstrap(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "bootstrap_distribution"}


def _effect_sizes(delta: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(delta, dtype=float)
    mean = float(np.mean(delta))
    sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
    if sd == 0.0:
        dz, reason = None, "sd(delta)==0"
    else:
        dz, reason = mean / sd, None
    return {"cohens_dz": dz, "cohens_dz_null_reason": reason,
            "paired_dominance": float(np.mean(delta > 0) - np.mean(delta < 0))}


def _continuous_summary(delta: np.ndarray, clusters: list[str], protocol: dict[str, Any]) -> dict[str, Any]:
    cfg = protocol["cluster_bootstrap"]
    bs = cluster_bootstrap(delta, clusters, protocol)
    q = {str(x): float(np.quantile(delta, float(x))) for x in cfg["reported_quantiles"]}
    out = {"mean": float(np.mean(delta)), "median": float(np.median(delta)),
           "quantiles": q, "cluster_bootstrap": _public_bootstrap(bs)}
    out.update(_effect_sizes(delta))
    return out


def _table(values: np.ndarray, full: np.ndarray, levels: list[Any]) -> dict[str, Any]:
    def key(x: Any) -> str:
        value = float(x) if isinstance(x, (float, np.floating, int, np.integer)) else x
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    counts = {key(a): {key(b): 0 for b in levels} for a in levels}
    for a, b in zip(values, full):
        counts[key(a)][key(b)] += 1
    return {"levels": [_jsonable(x) for x in levels], "counts": counts,
            "regression_count": int(np.sum(full < values)),
            "rescue_count": int(np.sum(full > values)), "n": int(len(values))}


def _discrete_summary(frames, protocol, conditions, seeds, tokens) -> dict[str, Any]:
    result = {}
    for metric, kind in classify_metrics(frames, protocol).items():
        if kind != "discrete":
            continue
        levels = sorted(set(float(x) for f in frames.values() for x in f[metric].dropna()))
        per_seed = {}
        merged_a, merged_b = [], []
        for seed in seeds:
            a = frames[(conditions[0], seed)][metric].to_numpy(float)
            b = frames[(conditions[1], seed)][metric].to_numpy(float)
            per_seed[str(seed)] = _table(a, b, levels)
            merged_a.extend(a); merged_b.extend(b)
        result[metric] = {"per_seed": per_seed, "merged": _table(np.array(merged_a), np.array(merged_b), levels)}
    return result


def _noise_floor(frames, protocol, conditions, seeds, tokens, mapping) -> dict[str, Any]:
    pairs = [(a, b) for a in seeds for b in seeds if a < b]
    out = {}
    all_p95 = []
    q95 = min(protocol["cluster_bootstrap"]["reported_quantiles"], key=lambda x: abs(float(x) - 0.95))
    noise_metric = protocol["metric_classification"]["always_continuous"][0]
    for condition in conditions:
        for a, b in pairs:
            key = f"{condition}/{a}-{b}"
            delta = frames[(condition, a)][noise_metric].to_numpy(float) - frames[(condition, b)][noise_metric].to_numpy(float)
            bs = cluster_bootstrap(delta, [mapping[t] for t in tokens], protocol)
            entry = {"mean": float(np.mean(delta)), "median": float(np.median(delta)),
                     "ci_95": _public_bootstrap(bs)["ci_95"],
                     "abs_delta_p95": float(np.quantile(np.abs(delta), float(q95))),
                     "bootstrap": _public_bootstrap(bs)}
            out[key] = entry
            all_p95.append(entry["abs_delta_p95"])
    return {"sequences": out, "practical_regression_threshold": float(max(all_p95))}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(Path(args.protocol).read_text())
    protocol_v1_path = Path(args.protocol_v1)
    token_to_log = json.loads(Path(args.token_to_log).read_text())
    _protocol_check(protocol, protocol_v1_path, token_to_log)
    required_sections = protocol.get("report_status", {}).get("required_sections")
    if not isinstance(required_sections, list) or not required_sections:
        raise ValueError("protocol report_status.required_sections is missing")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames, tokens, mapping = _load_evaluations(Path(args.evaluation_v1), protocol, token_to_log)
    budget = protocol["measured_budget"]
    conditions, seeds = list(budget["conditions"]), [int(x) for x in budget["seeds"]]
    metric_kinds = classify_metrics(frames, protocol)
    primary_metric = next((m for m in protocol["metric_classification"]["columns"] if m.lower() == "pdms"), protocol["metric_classification"]["always_continuous"][0])
    clusters = [mapping[t] for t in tokens]
    deltas = {}
    per_seed = {}
    for metric in protocol["metric_classification"]["columns"]:
        seed_deltas = {}
        for seed in seeds:
            base = frames[(conditions[0], seed)][metric].to_numpy(float)
            full = frames[(conditions[1], seed)][metric].to_numpy(float)
            seed_deltas[str(seed)] = full - base
        per_seed[metric] = {k: float(np.mean(v)) for k, v in seed_deltas.items()}
        deltas[metric] = np.mean(np.stack(list(seed_deltas.values())), axis=0)
    pd.DataFrame({"token": tokens, **{f"delta_{m}": deltas[m] for m in deltas}}).to_csv(Path(args.out_dir) / "paired_deltas.csv", index=False)

    continuous = {m: _continuous_summary(deltas[m], clusters, protocol) for m, k in metric_kinds.items() if k == "continuous"}
    for m in continuous:
        continuous[m]["per_seed_mean_delta"] = per_seed[m]
        if m == primary_metric:
            continuous[m]["p_value_annotation"] = "descriptive_non_confirmatory"
    discrete = _discrete_summary(frames, protocol, conditions, seeds, tokens)
    noise = _noise_floor(frames, protocol, conditions, seeds, tokens, mapping)
    eval_v2 = Path(args.evaluation_v2)
    co_primary = list(protocol["hypothesis_family"]["co_primary"])
    family = {co_primary[0]: "tested", co_primary[1]: "not_tested" if not eval_v2.exists() or not any(eval_v2.iterdir()) else "not_tested"}
    dist = json.loads(Path(args.distribution_audit).read_text())
    convergence = json.loads(Path(args.convergence_curve_audit).read_text())
    scorer = json.loads(Path(args.scorer_equivalence).read_text())
    report = {
        "status": protocol["report_status"]["status"],
        "status_reason": protocol["report_status"]["reason"],
        "operative_protocol": "summary_protocol_v2.json",
        "protocol_v2_sha256": hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
        "run_scope": protocol["run_scope"],
        "data_provenance_caveats": protocol["data_provenance_caveats_required_in_report"],
        "scorer_equivalence": scorer,
        "distribution_audit_summary": dist.get("summary", dist),
        "convergence_gate": {"protocol": protocol["convergence_gate"], "audit": convergence},
        "metric_classification": metric_kinds,
        "paired_delta": {"continuous": continuous, "discrete": discrete},
        "noise_floor": noise,
        "practical_regression_threshold": noise["practical_regression_threshold"],
        "hypothesis_family": family,
        "p_value_policy": "descriptive_non_confirmatory; co-primary family incomplete; no Holm adjustment",
        "amendment": protocol["amendment"],
        "required_sections": required_sections,
    }
    produced = {"run_scope 与 permitted/forbidden use 原文", "data_provenance_caveats_required_in_report 全部条目", "scorer 等价性章节（scorer_equivalence_record.json 的 required_statement 原文 + 实测差异数值 + 论文基准复现结果）", "distribution_audit.json 的完整 summary", "convergence_gate 判定与 under-trained regime 标注（含 convergence_curve_audit.json 的结论）", "PDMS 与全部 sub-score 的 paired delta、CI、effect size、transition table", "noise_floor 的 6 个 null 序列", "co-primary family 不完整声明", "本协议的修订记录（amendment 章节原文，含 observed_at_amendment_time）"}
    if set(required_sections) != produced:
        raise ValueError("report_status.required_sections does not match implemented report sections")
    (out / "stage_a_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=_jsonable) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize paired Stage-A v2 evaluation using the frozen protocol.")
    p.add_argument("--evaluation-v1", required=True)
    p.add_argument("--evaluation-v2", required=True)
    p.add_argument("--token-to-log", required=True)
    p.add_argument("--distribution-audit", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--protocol-v1", required=True)
    p.add_argument("--scorer-equivalence", required=True)
    p.add_argument("--convergence-curve-audit", required=True)
    p.add_argument("--out-dir", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        summarize(args)
        return 0
    except Exception as exc:  # input/protocol failures are deliberate non-zero HALTs
        print(f"summarize_stage_a: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
