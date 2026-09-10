#!/usr/bin/env python3
"""A1 follow-up: quantify the clean, usable subset of a RAP dataset directory.

Context
-------
`docs/audits/2026-09-09_asset_audit/A1_split_membership.md` judged `dataset_norm`
CONTAMINATED (10 logs / 1,365 tokens intersect `navtest`).  That judgement is a
hard gate (Sprint Plan Sec 3.1 / CP-2b) but it does not answer the follow-up
question:

    after removing everything that touches the evaluation side, how much
    legally usable material is actually left, and how much of it has a
    complete C_d raster set on disk?

This script answers exactly that and nothing else.

Design decisions worth knowing before reading the numbers
--------------------------------------------------------
1. The clean pool is defined ADDITIVELY, not subtractively.

   `A1_split_membership` recorded 44,389 dataset tokens that appear in no
   official split token list, and explicitly warned that they must not be
   auto-interpreted as legal training tokens (the official YAMLs only list the
   scene-filtered token subset).  Therefore:

       clean = (dataset tokens INTERSECT approved-training-pool tokens)
               MINUS evaluation-side tokens

   and never "everything minus the known-bad".

2. `SD-12` (which pool counts as "the legal NAVSIM training set") is still
   undecided, so this script evaluates ALL THREE candidates side by side and
   reports them as decision material.  It does not pick one.

3. Exclusion is reported at BOTH token level and log level, because Sprint Plan
   Sec 3.1 states the hard gate as "intersection must be 0" without specifying a
   granularity, and the two give very different answers here.  Choosing between
   them is a Human decision, not this script's.

4. Raster completeness is counted against the C_d camera set (4 cameras, see
   Sprint Plan Sec 7.2), because a token missing any C_d camera cannot be used by
   F0/F1.  Each raster root is counted separately; roots are never merged, per
   `A1_rap_datasets.md` ("`rendered_sensor_blobs_4cam_v1` is an independent root
   and must not auto-complete the primary root without a manifest/pair audit").

Compliance
----------
Read-only, per Sprint Plan Sec 3.0.  The only writes are the two report files
under ``--out-dir``; a guard refuses to run if that directory resolves inside a
dataset root.  Emits both ``.md`` (human) and ``.jsonl`` (machine) with the
standard header fields.  Directory scanning uses a single ``os.scandir`` walk and
set membership rather than one ``os.path.exists`` per expected path, so this does
not repeat the per-file ``stat`` storm that forced A1 to abort two scans.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import pickle
import socket
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

JST = timezone(timedelta(hours=9))

# Sprint Plan Sec 7.2: RAP consumes 4 of the 8 NAVSIM cameras.
C_D_CAMERAS = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0")

SCENE_FILTER_DIR = Path(
    "navsim/planning/script/config/common/train_test_split/scene_filter"
)

# SD-12 candidates.  kind is how the pool file must be parsed.
POOL_CANDIDATES = {
    "B1_navtrain": {
        "kind": "scene_filter",
        "rel_path": SCENE_FILTER_DIR / "navtrain.yaml",
        "note": "official NAVSIM training split; smallest, most conservative",
    },
    "B2_navall": {
        "kind": "scene_filter",
        "rel_path": SCENE_FILTER_DIR / "navall.yaml",
        "note": "full trainval pool; largest, needs its own navtest separation proof",
    },
    "B3_split_config": {
        "kind": "log_split",
        "rel_path": Path("process_data/default_train_val_test_log_split.yaml"),
        "key": "train_logs",
        "note": "the pool build_alignment_small_data.py actually consumes; LOG-LEVEL ONLY",
    },
}

# Evaluation-side splits that must not leak into training.
EVAL_SPLITS = ("navtest", "navhard_two_stage", "warmup_test_e2e", "navmini")


# --------------------------------------------------------------------------
# Minimal YAML readers
#
# The scene_filter files are large (navall.yaml is 415k lines) but structurally
# trivial: a bare `key:` line followed by `  - 'value'` entries.  A line parser is
# both faster and dependency-free, which matters for an audit script that must run
# identically on a login node and on a laptop.
# --------------------------------------------------------------------------


def parse_simple_yaml_lists(path: Path, wanted: Sequence[str]) -> Dict[str, List[str]]:
    """Extract top-level ``key:`` -> list-of-scalars sections from a simple YAML file."""
    wanted_set = set(wanted)
    out: Dict[str, List[str]] = {k: [] for k in wanted}
    current: Optional[str] = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            # rstrip() not rstrip("\n"): navtrain.yaml and navtest.yaml write
            # `tokens:` with trailing spaces, which would defeat the endswith(":")
            # section test below and silently yield an empty token list.
            line = raw.rstrip()
            if not line.strip():
                continue

            stripped = line.lstrip()
            indented = len(line) - len(stripped)

            if indented == 0 and stripped.endswith(":") and " " not in stripped[:-1]:
                key = stripped[:-1]
                current = key if key in wanted_set else None
                continue

            if indented == 0 and ":" in stripped and not stripped.startswith("-"):
                # A scalar key/value line such as `max_scenes: null` ends any section.
                current = None
                continue

            if current is not None and stripped.startswith("-"):
                value = stripped[1:].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                if value and value.lower() not in ("null", "~"):
                    out[current].append(value)

    return out


def load_scene_filter(path: Path) -> Tuple[Set[str], Set[str]]:
    """Return (log_names, tokens) for a scene_filter YAML."""
    parsed = parse_simple_yaml_lists(path, ("log_names", "tokens"))
    return set(parsed["log_names"]), set(parsed["tokens"])


def load_log_split(path: Path, key: str) -> Set[str]:
    """Return the log-name set under ``key`` of a train/val/test log-split YAML."""
    parsed = parse_simple_yaml_lists(path, (key,))
    return set(parsed[key])


# --------------------------------------------------------------------------
# Dataset metadata
# --------------------------------------------------------------------------


class Record:
    __slots__ = ("token", "log_name", "cam_paths")

    def __init__(self, token: str, log_name: str, cam_paths: Dict[str, str]):
        self.token = token
        self.log_name = log_name
        self.cam_paths = cam_paths


def normalise_rel(path_value: object) -> Optional[str]:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text:
        return None
    return os.path.normpath(text).lstrip("/")


def read_dataset_metadata(
    metadata_dir: Path, cameras: Sequence[str], max_logs: Optional[int]
) -> Tuple[List[Record], Dict[str, int]]:
    """Load every per-log pickle and flatten it into Record objects."""
    pkl_files = sorted(metadata_dir.glob("*.pkl"))
    if max_logs is not None:
        pkl_files = pkl_files[:max_logs]

    records: List[Record] = []
    stats = {
        "pkl_files": len(pkl_files),
        "unreadable_pkl": 0,
        "records": 0,
        "records_missing_token": 0,
        "records_missing_log": 0,
        "records_missing_cams_key": 0,
        "cam_entries_absent": 0,
    }

    for pkl in pkl_files:
        try:
            with pkl.open("rb") as fh:
                frame_infos = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - an unreadable pickle is a finding
            stats["unreadable_pkl"] += 1
            print(f"  [warn] unreadable pickle {pkl.name}: {exc}", file=sys.stderr)
            continue

        for info in frame_infos:
            stats["records"] += 1
            token = info.get("token")
            log_name = info.get("log_name")
            if not token:
                stats["records_missing_token"] += 1
                continue
            if not log_name:
                stats["records_missing_log"] += 1
                log_name = pkl.stem

            cams = info.get("cams")
            if not isinstance(cams, dict):
                stats["records_missing_cams_key"] += 1
                cams = {}

            cam_paths: Dict[str, str] = {}
            for cam_id in cameras:
                entry = cams.get(cam_id)
                rel = None
                if isinstance(entry, dict):
                    rel = normalise_rel(entry.get("data_path"))
                elif entry is not None:
                    rel = normalise_rel(entry)
                if rel is None:
                    stats["cam_entries_absent"] += 1
                else:
                    cam_paths[cam_id] = rel

            records.append(Record(str(token), str(log_name), cam_paths))

    return records, stats


def scan_raster_root(root: Path) -> Set[str]:
    """Collect every file under ``root`` as a path relative to ``root``.

    One iterative scandir walk, no per-file stat.  Returns an empty set if the
    root does not exist, so a missing root is reported rather than raised.
    """
    found: Set[str] = set()
    if not root.is_dir():
        return found

    root_str = str(root)
    stack = [root_str]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            found.add(os.path.relpath(entry.path, root_str))
                    except OSError:
                        continue
        except OSError as exc:  # noqa: PERF203 - unreadable dirs are a finding
            print(f"  [warn] cannot scan {current}: {exc}", file=sys.stderr)
    return found


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------


def git_head(rap_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(rap_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return out.stdout.strip() or "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def compute(args: argparse.Namespace) -> Dict[str, object]:
    rap_root = args.rap_root.resolve()
    dataset_root = (rap_root / args.dataset).resolve()
    metadata_dir = dataset_root / args.metadata_subdir
    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())

    if not metadata_dir.is_dir():
        raise SystemExit(f"metadata dir not found: {metadata_dir}")

    print(f"[1/4] reading metadata from {metadata_dir}", file=sys.stderr)
    records, meta_stats = read_dataset_metadata(metadata_dir, cameras, args.max_logs)
    ds_tokens = {r.token for r in records}
    ds_logs = {r.log_name for r in records}
    by_token = {r.token: r for r in records}
    print(
        f"      {len(records)} records / {len(ds_tokens)} tokens / {len(ds_logs)} logs",
        file=sys.stderr,
    )

    # ---- evaluation-side splits -------------------------------------------
    print("[2/4] loading evaluation-side splits", file=sys.stderr)
    eval_tokens: Set[str] = set()
    eval_logs: Set[str] = set()
    eval_detail = []
    for name in EVAL_SPLITS:
        path = rap_root / SCENE_FILTER_DIR / f"{name}.yaml"
        if not path.is_file():
            eval_detail.append(
                {"split": name, "status": "MISSING_SPLIT_DEFINITION",
                 "log_intersection": None, "token_intersection": None}
            )
            print(f"      [MISSING] {name}.yaml", file=sys.stderr)
            continue
        logs, tokens = load_scene_filter(path)
        eval_tokens |= tokens
        eval_logs |= logs
        eval_detail.append(
            {
                "split": name,
                "status": "PRESENT",
                "official_logs": len(logs),
                "official_tokens": len(tokens),
                "log_intersection": len(ds_logs & logs),
                "token_intersection": len(ds_tokens & tokens),
            }
        )
        print(
            f"      {name}: {len(ds_logs & logs)} logs / "
            f"{len(ds_tokens & tokens)} tokens intersect the dataset",
            file=sys.stderr,
        )

    missing_eval_defs = [d["split"] for d in eval_detail if d["status"] != "PRESENT"]

    # ---- raster roots -----------------------------------------------------
    print("[3/4] scanning raster roots", file=sys.stderr)
    raster_roots = [r.strip() for r in args.raster_roots.split(",") if r.strip()]
    raster_index: Dict[str, Set[str]] = {}
    for rel_root in raster_roots:
        root = dataset_root / rel_root
        files = scan_raster_root(root)
        raster_index[rel_root] = files
        print(f"      {rel_root}: {len(files)} files", file=sys.stderr)

    def raster_report(token_set: Set[str]) -> Dict[str, Dict[str, object]]:
        """Per raster root: how many of these tokens have a complete camera set."""
        out: Dict[str, Dict[str, object]] = {}
        for rel_root, files in raster_index.items():
            per_camera = {cam: 0 for cam in cameras}
            complete = 0
            for token in token_set:
                rec = by_token.get(token)
                if rec is None:
                    continue
                hits = 0
                for cam in cameras:
                    rel = rec.cam_paths.get(cam)
                    if rel is not None and rel in files:
                        per_camera[cam] += 1
                        hits += 1
                if hits == len(cameras):
                    complete += 1
            out[rel_root] = {
                "root_exists": (dataset_root / rel_root).is_dir(),
                "files_in_root": len(files),
                "tokens_with_camera_present": per_camera,
                "tokens_with_complete_camera_set": complete,
            }
        return out

    # ---- per-pool clean subset --------------------------------------------
    print("[4/4] computing clean subsets per SD-12 candidate", file=sys.stderr)
    pools = []
    for pool_id, spec in POOL_CANDIDATES.items():
        path = rap_root / spec["rel_path"]
        entry: Dict[str, object] = {
            "pool_id": pool_id,
            "pool_file": str(spec["rel_path"]),
            "kind": spec["kind"],
            "note": spec["note"],
        }
        if not path.is_file():
            entry["status"] = "MISSING_POOL_FILE"
            pools.append(entry)
            print(f"      [MISSING] {pool_id}", file=sys.stderr)
            continue

        if spec["kind"] == "scene_filter":
            pool_logs, pool_tokens = load_scene_filter(path)
            token_level_available = True
        else:
            pool_logs = load_log_split(path, spec["key"])
            pool_tokens = set()
            token_level_available = False

        entry["status"] = "PRESENT"
        entry["pool_logs"] = len(pool_logs)
        entry["pool_tokens"] = len(pool_tokens) if token_level_available else None
        entry["token_level_available"] = token_level_available

        # Additive intersection with the dataset.
        in_pool_logs = ds_logs & pool_logs
        if token_level_available:
            in_pool_tokens = ds_tokens & pool_tokens
        else:
            # B3 lists logs only.  Falling back to "all tokens of in-pool logs" is a
            # strictly weaker guarantee and is flagged as such in the report.
            in_pool_tokens = {r.token for r in records if r.log_name in in_pool_logs}

        # Variant A: token-level exclusion (keeps logs, drops offending tokens).
        clean_tokens_tok = in_pool_tokens - eval_tokens
        clean_logs_tok = {by_token[t].log_name for t in clean_tokens_tok}

        # Variant B: log-level exclusion (drops any log that touches the eval side).
        clean_logs_log = in_pool_logs - eval_logs
        clean_tokens_log = {
            t for t in in_pool_tokens if by_token[t].log_name in clean_logs_log
        } - eval_tokens

        entry["intersection"] = {
            "logs": len(in_pool_logs),
            "tokens": len(in_pool_tokens),
            "tokens_derived_from_logs": not token_level_available,
        }
        entry["clean_token_level_exclusion"] = {
            "logs": len(clean_logs_tok),
            "tokens": len(clean_tokens_tok),
            "raster": raster_report(clean_tokens_tok),
        }
        entry["clean_log_level_exclusion"] = {
            "logs": len(clean_logs_log),
            "tokens": len(clean_tokens_log),
            "raster": raster_report(clean_tokens_log),
        }
        entry["sd3_ten_percent_of_pool_logs"] = round(len(pool_logs) * 0.10)
        pools.append(entry)
        print(
            f"      {pool_id}: token-level {len(clean_tokens_tok)} tokens / "
            f"{len(clean_logs_tok)} logs | log-level {len(clean_tokens_log)} tokens / "
            f"{len(clean_logs_log)} logs",
            file=sys.stderr,
        )

    return {
        "header": {
            "audit_time_jst": datetime.now(JST).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "git_head": git_head(rap_root),
            "rap_root": str(rap_root),
            "executor": args.executor or getpass.getuser(),
            "script": "scripts/audit/a1_clean_subset.py",
            "read_only": True,
        },
        "inputs": {
            "dataset": args.dataset,
            "dataset_root": str(dataset_root),
            "metadata_dir": str(metadata_dir),
            "cameras": list(cameras),
            "raster_roots": raster_roots,
            "max_logs": args.max_logs,
        },
        "dataset_stats": {
            **meta_stats,
            "unique_tokens": len(ds_tokens),
            "unique_logs": len(ds_logs),
        },
        "eval_splits": eval_detail,
        "missing_eval_split_definitions": missing_eval_defs,
        "pools": pools,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def render_markdown(result: Dict[str, object]) -> str:
    h = result["header"]
    inp = result["inputs"]
    ds = result["dataset_stats"]
    lines: List[str] = []
    add = lines.append

    add("# A1 追加｜污染剔除后的干净可用池")
    add("")
    add(f"- 审计时间：{h['audit_time_jst']}")
    add(f"- 执行主机：`{h['host']}`")
    add(f"- Git HEAD：`{h['git_head']}`")
    add(f"- RAP_ROOT：`{h['rap_root']}`")
    add(f"- 执行者：{h['executor']}")
    add(f"- 脚本：`{h['script']}`（只读）")
    add(f"- 相机集合（`C_d`，Sprint Plan §7.2）：{', '.join(inp['cameras'])}")
    add("")
    add("## 方法")
    add("")
    add("干净池按**加法**定义：")
    add("")
    add("```")
    add("clean = (dataset tokens ∩ 已批准训练池 tokens) − 评测侧 tokens")
    add("```")
    add("")
    add("不使用减法（全部 token 减去已知污染），因为 `A1_split_membership` 记录了 "
        "44,389 个不在任何官方 split token 列表内的 token，并明确警告不得把它们"
        "自动视为合法训练 token。")
    add("")
    add("`SD-12` 未裁决，因此三个候选池全部计算，本报告仅作决策材料，不代替裁决。")
    add("")
    add("## 数据集实测")
    add("")
    add("| 项 | 值 |")
    add("|---|---:|")
    add(f"| metadata pickle | {ds['pkl_files']} |")
    add(f"| 不可读 pickle | {ds['unreadable_pkl']} |")
    add(f"| records | {ds['records']} |")
    add(f"| unique tokens | {ds['unique_tokens']} |")
    add(f"| unique logs | {ds['unique_logs']} |")
    add(f"| 缺 token 字段 | {ds['records_missing_token']} |")
    add(f"| 缺 cams 字段 | {ds['records_missing_cams_key']} |")
    add(f"| `C_d` 相机条目缺失 | {ds['cam_entries_absent']} |")
    add("")
    add("## 评测侧 split")
    add("")
    add("| split | 状态 | 官方 logs | 官方 tokens | ∩ logs | ∩ tokens |")
    add("|---|---|---:|---:|---:|---:|")
    for d in result["eval_splits"]:
        if d["status"] != "PRESENT":
            add(f"| `{d['split']}` | **{d['status']}** | — | — | — | — |")
        else:
            add(
                f"| `{d['split']}` | PRESENT | {d['official_logs']} | "
                f"{d['official_tokens']} | {d['log_intersection']} | "
                f"{d['token_intersection']} |"
            )
    add("")
    if result["missing_eval_split_definitions"]:
        add(
            "> 🔴 以下评测侧 split 定义缺失，其交集**无法核验**，"
            "本报告的干净池对它们不提供任何保证："
            + "、".join(f"`{s}`" for s in result["missing_eval_split_definitions"])
        )
        add("")

    add("## 每个 `SD-12` 候选池的干净子集")
    add("")
    for pool in result["pools"]:
        add(f"### `{pool['pool_id']}` — `{pool['pool_file']}`")
        add("")
        add(f"{pool['note']}")
        add("")
        if pool["status"] != "PRESENT":
            add(f"**{pool['status']}** — 无法计算。")
            add("")
            continue
        add(
            f"池规模：{pool['pool_logs']} logs"
            + (
                f" / {pool['pool_tokens']} tokens"
                if pool["token_level_available"]
                else "（**仅 log 级，无 token 列表**）"
            )
            + f"；`SD-3` 的 10% ≈ {pool['sd3_ten_percent_of_pool_logs']} logs"
        )
        add("")
        inter = pool["intersection"]
        add(f"与 dataset 的交集：{inter['logs']} logs / {inter['tokens']} tokens"
            + ("（token 由 log 反推，保证更弱）" if inter["tokens_derived_from_logs"] else ""))
        add("")
        add("| 剔除粒度 | 剩余 logs | 剩余 tokens | raster root | 该 root 内文件数 | `C_d` 四相机齐全的 token |")
        add("|---|---:|---:|---|---:|---:|")
        for label, key in (
            ("token 级", "clean_token_level_exclusion"),
            ("log 级", "clean_log_level_exclusion"),
        ):
            block = pool[key]
            first = True
            for rel_root, rep in block["raster"].items():
                prefix = (
                    f"| **{label}** | {block['logs']} | {block['tokens']} "
                    if first
                    else "| | | "
                )
                marker = "" if rep["root_exists"] else " ⚠️不存在"
                add(
                    prefix
                    + f"| `{rel_root}`{marker} | {rep['files_in_root']} | "
                    f"**{rep['tokens_with_complete_camera_set']}** |"
                )
                first = False
            if first:  # no raster roots configured
                add(f"| **{label}** | {block['logs']} | {block['tokens']} | — | — | — |")
        add("")
        add("每路相机单独命中数：")
        add("")
        for label, key in (
            ("token 级", "clean_token_level_exclusion"),
            ("log 级", "clean_log_level_exclusion"),
        ):
            for rel_root, rep in pool[key]["raster"].items():
                per = rep["tokens_with_camera_present"]
                detail = "、".join(f"{cam} {n}" for cam, n in per.items())
                add(f"- {label} / `{rel_root}`：{detail}")
        add("")

    add("## 如何读这张表")
    add("")
    add("1. **最后一列才是能否跑 F0 的答案。** token 数再多，若 `C_d` 四相机不齐，"
        "该 token 对 F0/F1 不可用。")
    add("2. **两个 raster root 不合并。** 依 `A1_rap_datasets.md`，"
        "`rendered_sensor_blobs_4cam_v1` 是独立根，未经 manifest/pair 审计不得用于"
        "补齐主根。")
    add("3. **token 级与 log 级剔除的差距本身就是一个待裁决问题。** "
        "Sprint Plan §3.1 只写了「交集必须为 0」，未指定粒度。若同一 driving log "
        "的 token 分处训练与评测两侧是否构成泄漏，需 Human 以 `SD-*` 裁决。")
    add("4. 本报告不改变 `A1` 的 CONTAMINATED 判定，也不解除 `CP-2b`。")
    add("")
    return "\n".join(lines) + "\n"


def render_jsonl(result: Dict[str, object]) -> str:
    rows: List[Dict[str, object]] = [
        {"record": "header", **result["header"]},
        {"record": "inputs", **result["inputs"]},
        {"record": "dataset_stats", **result["dataset_stats"]},
    ]
    for d in result["eval_splits"]:
        rows.append({"record": "eval_split", **d})
    for p in result["pools"]:
        rows.append({"record": "pool", **p})
    return "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=False) for r in rows) + "\n"


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantify the clean, usable subset of a RAP dataset directory (read-only).",
    )
    parser.add_argument(
        "--rap-root",
        type=Path,
        default=Path(os.environ.get("RAP_ROOT", "/gs/bs/tga-RLA/qdeng/RAP")),
    )
    parser.add_argument("--dataset", default="dataset_norm")
    parser.add_argument("--metadata-subdir", default="navsim_logs/mini")
    parser.add_argument("--cameras", default=",".join(C_D_CAMERAS))
    parser.add_argument(
        "--raster-roots",
        default="rendered_sensor_blobs,rendered_sensor_blobs_4cam_v1",
        help="comma-separated, relative to the dataset root; counted separately, never merged",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <rap-root>/docs/audits/2026-09-09_asset_audit",
    )
    parser.add_argument("--out-name", default="A1_clean_subset")
    parser.add_argument("--executor", default=None)
    parser.add_argument(
        "--max-logs",
        type=int,
        default=None,
        help="smoke-test knob: only read the first N metadata pickles",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="print the markdown report and write nothing",
    )
    args = parser.parse_args()

    rap_root = args.rap_root.resolve()
    out_dir = (args.out_dir or (rap_root / "docs/audits/2026-09-09_asset_audit")).resolve()

    # Read-only guard (Sprint Plan Sec 3.0): never write inside an audited dataset.
    dataset_root = (rap_root / args.dataset).resolve()
    for forbidden in (dataset_root, rap_root / "dataset_norm", rap_root / "dataset_aug",
                      rap_root / "dataset_perturbed", rap_root / "ckpts"):
        try:
            out_dir.relative_to(forbidden.resolve())
        except ValueError:
            continue
        raise SystemExit(f"refusing to write inside an audited directory: {out_dir}")

    result = compute(args)
    markdown = render_markdown(result)

    if args.stdout_only:
        print(markdown)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.out_name}.md"
    jsonl_path = out_dir / "raw" / f"{args.out_name}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(markdown, encoding="utf-8")
    jsonl_path.write_text(render_jsonl(result), encoding="utf-8")

    print(f"\nwrote {md_path}", file=sys.stderr)
    print(f"wrote {jsonl_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
