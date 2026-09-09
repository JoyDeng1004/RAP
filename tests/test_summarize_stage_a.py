"""Synthetic-only contract tests for summarize_stage_a.py.

No file in outputs/alignment_stage_a/evaluation_v1 is read by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "navsim/planning/script/summarize_stage_a.py"
V1 = ROOT / "outputs/alignment_stage_a/summary/summary_protocol.json"
V2 = ROOT / "outputs/alignment_stage_a/summary/summary_protocol_v2.json"
V1_SHA = "252994acb610806b59ee7653f6b7f27a1e3ab470d1efba7e9de045dba900945d"


def _write_fixture(tmp_path: Path, *, delta_by_seed=None, n_logs=10, tokens_per_log=20):
    root = tmp_path / "fixture"
    root.mkdir(parents=True)
    ev1 = root / "evaluation_v1"; ev1.mkdir()
    ev2 = root / "evaluation_v2"; ev2.mkdir()
    out = root / "summary_out"
    protocol = json.loads(V2.read_text())
    protocol["measured_budget"] = {"seeds": [0, 1, 2], "conditions": ["noalign", "fullalign"], "runs": 6}
    protocol["cluster_bootstrap"]["n_resamples"] = 10000
    protocol_path = root / "v2.json"; protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    v1_path = root / "v1.json"; v1_path.write_bytes(V1.read_bytes())
    tokens = [f"tok{i:04d}" for i in range(n_logs * tokens_per_log)]
    mapping = {t: f"log{int(i / tokens_per_log):02d}" for i, t in enumerate(tokens)}
    token_path = root / "token_to_log.json"
    token_path.write_text(json.dumps({"mapping": mapping, "protocol_sha256": V1_SHA, "n_tokens": len(tokens), "n_logs": n_logs}) + "\n")
    cols = protocol["metric_classification"]["columns"]
    delta_by_seed = delta_by_seed or {0: 0.05, 1: 0.05, 2: 0.05}
    for condition in protocol["measured_budget"]["conditions"]:
        for seed in protocol["measured_budget"]["seeds"]:
            rows = []
            for i, tok in enumerate(tokens):
                row = {"token": tok, "valid": True, "NC": i % 2, "DAC": i % 3 / 2, "EP": (i % 5) / 4, "TTC": i % 7 / 6, "C": i % 2, "DDC": i % 3 / 2, "PDMS": 0.4 + i * 0.0001}
                if condition == "fullalign":
                    row.update({"NC": (i + 1) % 2, "PDMS": row["PDMS"] + delta_by_seed[seed]})
                rows.append(row)
            pd.DataFrame(rows, columns=["token", "valid", *cols]).to_csv(ev1 / f"{condition}_seed{seed}.csv", index=False)
    for name, data in [("distribution.json", {"summary": {"synthetic": True}}), ("scorer.json", {"reporting_rules": {"required_statement": "synthetic"}}), ("convergence.json", {"status": "complete", "runs": {}})]:
        (root / name).write_text(json.dumps(data) + "\n")
    args = [sys.executable, str(SCRIPT), "--evaluation-v1", str(ev1), "--evaluation-v2", str(ev2), "--token-to-log", str(token_path), "--distribution-audit", str(root / "distribution.json"), "--protocol", str(protocol_path), "--protocol-v1", str(v1_path), "--scorer-equivalence", str(root / "scorer.json"), "--convergence-curve-audit", str(root / "convergence.json"), "--out-dir", str(out)]
    return root, args, protocol_path, token_path, ev1, ev2, out


def _run(args):
    return subprocess.run(args, text=True, capture_output=True)


def test_synthetic_contracts_all_groups(tmp_path):
    root, args, protocol_path, token_path, ev1, ev2, out = _write_fixture(tmp_path)
    good = _run(args)
    assert good.returncode == 0, good.stderr
    report = json.loads((out / "stage_a_report.json").read_text())
    deltas = json.loads((out / "stage_a_report.json").read_text())["paired_delta"]["continuous"]
    assert deltas["PDMS"]["mean"] == pytest.approx(0.05, abs=1e-12)  # C1
    assert deltas["PDMS"]["median"] == pytest.approx(0.05, abs=1e-12)
    assert report["status"] == "incomplete" and report["hypothesis_family"]["EPDMS_v2_navhard_two_stage"] == "not_tested"  # H1
    assert report["paired_delta"]["continuous"]["PDMS"]["p_value_annotation"] == "descriptive_non_confirmatory"  # H3
    assert "holm_adjusted_p" not in json.dumps(report)  # H3
    assert report["metric_classification"]["NC"] == "discrete" and report["metric_classification"]["PDMS"] == "continuous"  # E1/E5
    table = report["paired_delta"]["discrete"]["NC"]
    assert set(table) == {"per_seed", "merged"} and set(table["per_seed"]) == {"0", "1", "2"}  # F5
    expected_scenes = len(json.loads(token_path.read_text())["mapping"])
    for seed_table in table["per_seed"].values():
        assert sum(map(sum, (row.values() for row in seed_table["counts"].values()))) == expected_scenes  # F2
    for src, row in table["merged"]["counts"].items():
        for dst, count in row.items():
            assert count == sum(seed_table["counts"][src][dst] for seed_table in table["per_seed"].values())  # F5
    ddc_counts = report["paired_delta"]["discrete"]["DDC"]["per_seed"]["0"]["counts"]
    assert set(ddc_counts) == {"0", "0.5", "1"}  # F4
    assert all(set(row) == {"0", "0.5", "1"} for row in ddc_counts.values())
    assert sum(len(row) for row in ddc_counts.values()) == 9 and sum(count == 0 for row in ddc_counts.values() for count in row.values()) > 0
    assert len(report["noise_floor"]["sequences"]) == 6  # G1
    # I1: PDMS is non-constant in this fixture; direct helper covers the zero-sd branch below.

    # A1/A2/A3: each integrity mutation must fail before analysis.
    bad_v1 = root / "bad_v1.json"; bad_v1.write_bytes(V1.read_bytes() + b"x")
    bad = args.copy(); bad[bad.index("--protocol-v1") + 1] = str(bad_v1)
    assert _run(bad).returncode != 0
    bad_v2 = json.loads(protocol_path.read_text()); bad_v2["supersedes_sha256"] = "0" * 64; protocol_path.write_text(json.dumps(bad_v2))
    assert _run(args).returncode != 0
    protocol_path.write_text(json.dumps(json.loads(V2.read_text()) | {"measured_budget": {"seeds": [0, 1, 2], "conditions": ["noalign", "fullalign"], "runs": 6}}))
    token = json.loads(token_path.read_text()); token["protocol_sha256"] = "0" * 64; token_path.write_text(json.dumps(token))
    assert _run(args).returncode != 0


def test_input_failures_and_pairing_edges(tmp_path):
    root, args, protocol_path, token_path, ev1, ev2, out = _write_fixture(tmp_path)
    # B1
    (ev1 / "fullalign_seed2.csv").unlink(); assert _run(args).returncode != 0
    root, args, protocol_path, token_path, ev1, ev2, out = _write_fixture(tmp_path / "b2")
    x = pd.read_csv(ev1 / "fullalign_seed2.csv"); x.loc[0, "token"] = "wrong"; x.to_csv(ev1 / "fullalign_seed2.csv", index=False); assert _run(args).returncode != 0
    root, args, protocol_path, token_path, ev1, ev2, out = _write_fixture(tmp_path / "b3")
    x = pd.read_csv(ev1 / "noalign_seed0.csv"); x.loc[0, "valid"] = False; x.to_csv(ev1 / "noalign_seed0.csv", index=False); assert _run(args).returncode != 0
    root, args, protocol_path, token_path, ev1, ev2, out = _write_fixture(tmp_path / "b4")
    token = json.loads(token_path.read_text()); token["mapping"].pop(next(iter(token["mapping"]))); token_path.write_text(json.dumps(token)); assert _run(args).returncode != 0
    # C2/C3
    root, args, *_ = _write_fixture(tmp_path / "c2", delta_by_seed={0: .03, 1: .06, 2: .09}); assert _run(args).returncode == 0
    d = json.loads((root / "summary_out/stage_a_report.json").read_text())["paired_delta"]["continuous"]["PDMS"]
    assert d["mean"] == pytest.approx(.06, abs=1e-12)
    assert d["per_seed_mean_delta"]["0"] == pytest.approx(.03, abs=1e-12)
    assert d["per_seed_mean_delta"]["1"] == pytest.approx(.06, abs=1e-12)
    assert d["per_seed_mean_delta"]["2"] == pytest.approx(.09, abs=1e-12)
    root, args, *_ = _write_fixture(tmp_path / "c3", delta_by_seed={0: 0, 1: 0, 2: 0}); assert _run(args).returncode == 0
    d = json.loads((root / "summary_out/stage_a_report.json").read_text())["paired_delta"]["continuous"]["PDMS"]
    assert d["mean"] == 0 and d["paired_dominance"] == 0 and d["cohens_dz"] is None and d["cohens_dz_null_reason"] == "sd(delta)==0"


def test_bootstrap_and_effect_helpers(tmp_path):
    sys.path.insert(0, str(ROOT))
    from navsim.planning.script.summarize_stage_a import _effect_sizes, cluster_bootstrap
    p = json.loads(V2.read_text()); p["cluster_bootstrap"]["n_resamples"] = 10000
    vals = np.array([1.0] * 20 + [0.0] * 180); clusters = ["hot"] * 20 + [f"l{i}" for i in range(9) for _ in range(20)]
    a = cluster_bootstrap(vals, clusters, p); b = cluster_bootstrap(vals, clusters, p)
    changed_seed = copy.deepcopy(p); changed_seed["cluster_bootstrap"]["rng_seed"] = 7
    c = cluster_bootstrap(vals, clusters, changed_seed)
    assert np.array_equal(a["bootstrap_distribution"], b["bootstrap_distribution"])  # D1
    assert not np.array_equal(a["bootstrap_distribution"], c["bootstrap_distribution"])  # D2
    assert a["n_resamples"] == 10000 and len(a["bootstrap_distribution"]) == 10000  # D4
    shortened = copy.deepcopy(p); shortened["cluster_bootstrap"]["n_resamples"] = 37
    assert len(cluster_bootstrap(vals, clusters, shortened)["bootstrap_distribution"]) == 37
    token_rng = np.random.default_rng(1); token_stars = np.array([np.mean(vals[token_rng.integers(0, len(vals), len(vals))]) for _ in range(10000)])
    assert (a["ci_95"][1] - a["ci_95"][0]) > 2 * (np.quantile(token_stars, .975) - np.quantile(token_stars, .025))  # D3
    assert _effect_sizes(np.array([1., 2., 3.]))["cohens_dz"] == pytest.approx(2 / 1, abs=1e-12)
    assert _effect_sizes(np.array([1., 1.]))["cohens_dz"] is None
    assert _effect_sizes(np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, 0]))["paired_dominance"] == pytest.approx(.3, abs=1e-12)  # I2/I3


def test_protocol_required_sections_and_eval_v2_absent(tmp_path):
    root, args, protocol_path, token_path, ev1, ev2, out = _write_fixture(tmp_path)
    assert _run(args).returncode == 0  # H1: empty evaluation_v2
    shutil.rmtree(root / "evaluation_v2")
    shutil.rmtree(out)
    assert _run(args).returncode == 0  # H2: missing evaluation_v2
    missing_v2_report = json.loads((out / "stage_a_report.json").read_text())
    assert missing_v2_report["status"] == "incomplete"
    assert missing_v2_report["hypothesis_family"]["EPDMS_v2_navhard_two_stage"] == "not_tested"
    p = json.loads(protocol_path.read_text()); p["report_status"]["required_sections"] = p["report_status"]["required_sections"][:-1]; protocol_path.write_text(json.dumps(p))
    assert _run(args).returncode != 0  # H4


def test_metric_classification_boundaries(tmp_path):
    root, args, *_ = _write_fixture(tmp_path)
    # Boundary classification is tested through the public helper with synthetic frames.
    sys.path.insert(0, str(ROOT))
    from navsim.planning.script.summarize_stage_a import classify_metrics
    proto = json.loads(V2.read_text()); cols = proto["metric_classification"]["columns"]
    frame = pd.DataFrame({c: [0, 1, 2, 3, 4, 5] for c in cols})
    frame["PDMS"] = [0, 0, 0, 0, 0, 0]
    assert classify_metrics({("x", 0): frame}, proto)["PDMS"] == "continuous"
    frame["C"] = [0, 1, 2, 3, 4, 5]
    assert classify_metrics({("x", 0): frame}, proto)["C"] == "continuous"  # E4
    frame["C"] = [0, 1, 2, 3, 4, 0]
    assert classify_metrics({("x", 0): frame}, proto)["C"] == "discrete"  # E3
    wide = pd.DataFrame({c: list(range(100)) for c in cols})
    assert classify_metrics({("x", 0): wide}, proto)["C"] == "continuous"  # E2


def test_transition_table_and_cluster_null_contract(tmp_path):
    sys.path.insert(0, str(ROOT))
    from navsim.planning.script.summarize_stage_a import _table
    a = np.array([0, 0, 0, 0, 1, 1]); b = np.array([0, 1, 1, 1, 0, 1])
    table = _table(a, b, [0, 1])
    assert table["counts"] == {"0": {"0": 1, "1": 3}, "1": {"0": 1, "1": 1}}  # F1
    assert sum(map(sum, (v.values() for v in table["counts"].values()))) == 6
    assert table["regression_count"] == 1 and table["rescue_count"] == 3  # F3: n10 != n01
    a = np.array([0, .5, 1]); b = np.array([1, .5, 0])
    table = _table(a, b, [0, .5, 1])
    assert set(table["counts"]) == {"0", "0.5", "1"}  # F4
    assert all(set(row) == {"0", "0.5", "1"} for row in table["counts"].values())
    assert sum(len(row) for row in table["counts"].values()) == 9
    assert sum(map(sum, (v.values() for v in table["counts"].values()))) == 3
    assert sum(count == 0 for row in table["counts"].values() for count in row.values()) == 6


def test_noise_floor_known_p95_and_seed_structure(tmp_path):
    root, args, protocol_path, token_path, ev1, ev2, out = _write_fixture(
        tmp_path, delta_by_seed={0: 0.0, 1: 0.125, 2: 0.375}
    )
    noalign = [pd.read_csv(ev1 / f"noalign_seed{seed}.csv") for seed in (0, 1, 2)]
    assert noalign[0].equals(noalign[1]) and noalign[0].equals(noalign[2])
    result = _run(args)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "stage_a_report.json").read_text())
    sequences = report["noise_floor"]["sequences"]
    expected_p95 = {
        "noalign/0-1": 0.0,
        "noalign/0-2": 0.0,
        "noalign/1-2": 0.0,
        "fullalign/0-1": 0.125,
        "fullalign/0-2": 0.375,
        "fullalign/1-2": 0.25,
    }
    assert set(sequences) == set(expected_p95)  # G1
    for key in ("noalign/0-1", "noalign/0-2", "noalign/1-2"):
        assert sequences[key]["mean"] == 0.0 and sequences[key]["median"] == 0.0  # G2
    for key in ("fullalign/0-1", "fullalign/0-2", "fullalign/1-2"):
        assert sequences[key]["mean"] != 0.0 and sequences[key]["median"] != 0.0
    for key, expected in expected_p95.items():
        assert sequences[key]["abs_delta_p95"] == pytest.approx(expected, abs=1e-12)  # G3
    expected_threshold = max(expected_p95.values())
    assert report["practical_regression_threshold"] == pytest.approx(expected_threshold, abs=1e-12)
    assert report["noise_floor"]["practical_regression_threshold"] == pytest.approx(expected_threshold, abs=1e-12)
