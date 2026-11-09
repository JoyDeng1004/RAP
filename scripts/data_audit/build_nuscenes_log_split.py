#!/usr/bin/env python3
"""从 nuScenes manifest 生成 RAP 的 train/val/test log 划分。

只读脚本：只读 _index/manifest.json，只往 --out 写一个 yaml。

=============================================================================
为什么必须有这一步（直接按 pkl 名划分是错的）
=============================================================================
create_nuscenes_metadata.py 是【每个 scene 一个 pkl】，且把 info["log_name"]
写成了 scene_name（见该文件 line 295，因为 dataloader 是拿 pkl 文件名当 log 名的）。
而 run_training.py:36-46 的划分机制是：

    train_scene_filter.log_names = [n for n in scene_filter.log_names
                                    if n in cfg.train_logs]

也就是在 pkl 名的粒度上切。于是"log 划分"实际退化成了"scene 划分"。

对 nuPlan 这没问题（一个 log 一个 pkl）。对 nuScenes 是数据泄漏：
trainval 的 850 个 scene 只来自 68 个真实 log，单个 log 最多出 34 个 scene，
同一 log 的相邻 scene 在时间上只差几十秒、走的是同一条街。把它们劈到
train 和 val 两边，val 会被系统性地高估。

所以这里的做法是：**按真实 log_name 分组，整组进同一个 split，再展开成
scene 名列表**写进 train_logs / val_logs / test_logs。对 RAP 来说它读到的
仍然是一串 pkl 名，接口不变；但分组边界是对的。

=============================================================================
划分策略
=============================================================================
- 分层：按 location 分别划分。nuScenes 的 4 个 location 在场景分布上差异很大
  （boston-seaport 占 467/850），不分层会让某个 split 缺某座城市。
- 每个 location 内按 scene 数降序贪心，把每个 log 投给当前"欠额最大"的 split。
  降序是关键：先放大块，剩下的小块才有机会把比例填平。
- 完全确定性：不用随机数。同一份 manifest 永远得到同一份划分。
  （--seed 只在 --shuffle-ties 时用于打散 scene 数相同的 log 的顺序。）
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = ("train", "val", "test")


def load_logs(manifest_path: Path) -> Tuple[Dict[str, Dict], Dict]:
    """manifest -> {log_name: {location, scenes: [...], n_scenes}}。"""
    manifest = json.loads(manifest_path.read_text())
    logs: Dict[str, Dict] = {}
    for scene in manifest["scenes"]:
        entry = logs.setdefault(
            scene["log_name"], {"location": scene["location"], "scenes": []}
        )
        # 一个 log 横跨多个 location 会让分层失去意义，直接拦下来
        assert entry["location"] == scene["location"], (
            f"log {scene['log_name']} 同时属于 {entry['location']} 和 {scene['location']}"
        )
        entry["scenes"].append(scene["scene_name"])
    for entry in logs.values():
        entry["scenes"].sort()
        entry["n_scenes"] = len(entry["scenes"])
    return logs, manifest


def assign(
    logs: Dict[str, Dict], ratios: Dict[str, float], shuffle_ties: bool, seed: int
) -> Dict[str, List[str]]:
    """按 location 分层 + 贪心补欠额，返回 {split: [log_name, ...]}。"""
    by_location: Dict[str, List[str]] = defaultdict(list)
    for log_name, entry in logs.items():
        by_location[entry["location"]].append(log_name)

    assignment: Dict[str, List[str]] = {s: [] for s in SPLITS}
    rng = random.Random(seed)

    for location in sorted(by_location):
        names = by_location[location]
        if shuffle_ties:
            rng.shuffle(names)
        else:
            names.sort()
        # 大块优先。sort 是稳定的，所以上面那轮排序决定了同 scene 数时的先后。
        names.sort(key=lambda n: -logs[n]["n_scenes"])

        total = sum(logs[n]["n_scenes"] for n in names)
        placed = {s: 0 for s in SPLITS}
        for log_name in names:
            # 欠额 = 目标份额 - 已放份额。取最欠的那个；并列时按 SPLITS 顺序。
            target = min(
                SPLITS,
                key=lambda s: (placed[s] - ratios[s] * total, SPLITS.index(s)),
            )
            assignment[target].append(log_name)
            placed[target] += logs[log_name]["n_scenes"]

    for split in SPLITS:
        assignment[split].sort()
    return assignment


def render_yaml(
    assignment: Dict[str, List[str]],
    logs: Dict[str, Dict],
    manifest: Dict,
    manifest_path: Path,
    ratios: Dict[str, float],
) -> str:
    scene_counts = {
        s: sum(logs[n]["n_scenes"] for n in assignment[s]) for s in SPLITS
    }
    total_scenes = sum(scene_counts.values())

    lines: List[str] = [
        # 必须是第一行。少了它，键会落进 cfg.train_val_test_log_split.* 而不是顶层，
        # run_training.py:37 读 cfg.train_logs 会直接 ConfigAttributeError。
        "# @package _global_",
        "# nuScenes trainval 的 train/val/test 划分（由 scripts/data_audit/"
        "build_nuscenes_log_split.py 生成，请勿手改）",
        f"# 源: {manifest_path}",
        f"# 生成时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"# nuScenes 版本: {manifest['version']}",
        "#",
        "# 列表里是 **scene 名**（= navsim_logs/<split>/ 下的 pkl 名），因为",
        "# run_training.py 是在 pkl 名的粒度上做 train/val 切分的。但分组边界是",
        "# **真实 nuScenes log**：同一个 log 的所有 scene 必定落在同一个 split，",
        "# 不会跨 split —— 这是为了避免同一段行程的相邻 scene 造成时序泄漏。",
        "#",
        f"# 真实 log {len(logs)} 个 -> scene {total_scenes} 个",
    ]
    for split in SPLITS:
        pct = 100.0 * scene_counts[split] / total_scenes if total_scenes else 0.0
        lines.append(
            f"#   {split:<5s}: {len(assignment[split]):>2d} logs / "
            f"{scene_counts[split]:>3d} scenes ({pct:5.1f}%, 目标 {ratios[split]*100:.0f}%)"
        )
    lines.append("#")
    lines.append("# 按 location 的分布（logs / scenes）:")
    for location in sorted({e["location"] for e in logs.values()}):
        cells = []
        for split in SPLITS:
            picked = [n for n in assignment[split] if logs[n]["location"] == location]
            cells.append(
                f"{split}={len(picked)}/{sum(logs[n]['n_scenes'] for n in picked)}"
            )
        lines.append(f"#   {location:<26s} " + "  ".join(cells))
    lines.append("")

    for split in SPLITS:
        lines.append(f"{split}_logs:")
        for log_name in assignment[split]:
            entry = logs[log_name]
            lines.append(f"  # {log_name}  ({entry['location']}, {entry['n_scenes']} scenes)")
            lines.extend(f"  - {scene}" for scene in entry["scenes"])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_nuscenes/_index/manifest.json"),
        help="build-index 产出的 manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "navsim/planning/script/config/training/train_val_test_log_split/nuscenes_trainval.yaml"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--shuffle-ties",
        action="store_true",
        help="打散 scene 数相同的 log 的先后（默认按名字，完全确定）",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "train+val 比例必须小于 1"
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": test_ratio}

    logs, manifest = load_logs(args.manifest)
    assignment = assign(logs, ratios, args.shuffle_ties, args.seed)

    # 不变式：每个 log 恰好出现一次，scene 总数守恒
    flat = [n for split in SPLITS for n in assignment[split]]
    assert len(flat) == len(set(flat)) == len(logs), "log 重复或遗漏"
    n_scenes = sum(len(e["scenes"]) for e in logs.values())
    assert sum(len(logs[n]["scenes"]) for n in flat) == n_scenes, "scene 数不守恒"

    text = render_yaml(assignment, logs, manifest, args.manifest, ratios)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    print(f"写出 {args.out}")
    for split in SPLITS:
        count = sum(logs[n]["n_scenes"] for n in assignment[split])
        print(
            f"  {split:<5s} {len(assignment[split]):>2d} logs / {count:>3d} scenes "
            f"({100.0 * count / n_scenes:5.1f}%)"
        )


if __name__ == "__main__":
    main()
