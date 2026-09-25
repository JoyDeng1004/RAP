#!/usr/bin/env python3
"""解析某个 data_split 的 log 全集，并按 log 做分片。

cache_shard.sh 和 make_kshot_splits.py 共用这一份"谁算 log 全集"的定义，
免得两边对 scene_filter 的理解各走各的。

全集 = ``navsim_logs/<split>/*.pkl`` 的 basename；若 scene_filter 里
``log_names`` 非 null，再与之求交 —— navtrain 用的 navall.yaml 正是这种
情况，直接 glob pkl 会把 navtrain 之外的 log 也算进来。

分片默认按 stride（round-robin）而不是连续切块：log 名排序后相邻的往往
来自同一车同一天，连续切块会让各分片的耗时差出好几倍。

    # 全集大小
    python tools/log_universe.py --logs-dir $DS/navsim_logs/trainval \
        --scene-filter navsim/planning/script/config/common/train_test_split/scene_filter/navall.yaml \
        --format count

    # 第 3 片（共 40 片），输出 hydra 列表字面量
    python tools/log_universe.py --logs-dir ... --shard 3 --num-shards 40 --format hydra
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml


def read_scene_filter_log_names(path: Optional[Path]) -> Optional[List[str]]:
    """读 scene_filter yaml 的 log_names；文件没给或 log_names 为 null 时返回 None。"""
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"FAIL scene_filter 不存在: {path}")
    # Hydra 的 _target_/defaults 这里用不到，safe_load 足够。
    data = yaml.safe_load(path.read_text()) or {}
    names = data.get("log_names")
    if names is None:
        return None
    return [str(n) for n in names]


def resolve_universe(
    logs_dir: Path, scene_filter: Optional[Path]
) -> Tuple[List[str], List[str]]:
    """返回 (全集, scene_filter 点名但磁盘上没有的 log)。"""
    if not logs_dir.is_dir():
        raise SystemExit(f"FAIL navsim_logs 目录不存在: {logs_dir}")

    on_disk = sorted(p.stem for p in logs_dir.glob("*.pkl"))
    if not on_disk:
        raise SystemExit(f"FAIL {logs_dir} 里没有任何 .pkl")

    wanted = read_scene_filter_log_names(scene_filter)
    if wanted is None:
        return on_disk, []

    on_disk_set = set(on_disk)
    universe = sorted(n for n in set(wanted) if n in on_disk_set)
    missing = sorted(n for n in set(wanted) if n not in on_disk_set)
    return universe, missing


def take_shard(
    logs: Sequence[str], shard: int, num_shards: int, mode: str = "stride"
) -> List[str]:
    """取第 shard 片（1-based），与 rasterize_*.sh 的 ``run <第几片> <总片数>`` 对齐。"""
    if num_shards < 1:
        raise SystemExit("FAIL --num-shards 需 >= 1")
    if not 1 <= shard <= num_shards:
        raise SystemExit(f"FAIL --shard 需在 [1, {num_shards}]，收到 {shard}")

    if mode == "stride":
        return list(logs[shard - 1 :: num_shards])
    if mode == "block":
        per = (len(logs) + num_shards - 1) // num_shards
        return list(logs[(shard - 1) * per : shard * per])
    raise SystemExit(f"FAIL 未知 --shard-mode: {mode}")


def as_hydra_list(names: Sequence[str]) -> str:
    """Hydra override 的列表字面量。

    log 名以数字开头且含点（2021.05.12...），不加引号会被 override 语法当成
    数字解析并报错，所以每个元素都必须带单引号。
    """
    for n in names:
        if "'" in n or "," in n or "[" in n or "]" in n:
            raise SystemExit(f"FAIL log 名含无法安全转义的字符: {n}")
    return "[" + ",".join(f"'{n}'" for n in names) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--logs-dir", type=Path, required=True,
                        help="$OPENSCENE_DATA_ROOT/navsim_logs/<data_split>")
    parser.add_argument("--scene-filter", type=Path, default=None,
                        help="scene_filter yaml；给了就与其 log_names 求交")
    parser.add_argument("--shard", type=int, default=None, help="第几片（1-based）")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-mode", choices=("stride", "block"), default="stride")
    parser.add_argument("--format", choices=("lines", "hydra", "count", "json"),
                        default="lines")
    parser.add_argument("--strict-missing", action="store_true",
                        help="scene_filter 点名的 log 有缺失时直接失败")
    parser.add_argument("--quiet", action="store_true",
                        help="不打缺失提示；批量循环调用时用，免得同一条刷屏")
    args = parser.parse_args()

    universe, missing = resolve_universe(args.logs_dir, args.scene_filter)
    if missing:
        msg = f"[log_universe] scene_filter 点名但磁盘缺失 {len(missing)} 个 log，例如 {missing[:3]}"
        if args.strict_missing:
            raise SystemExit("FAIL " + msg)
        if not args.quiet:
            print(msg, file=sys.stderr)

    selected = universe
    if args.shard is not None:
        selected = take_shard(universe, args.shard, args.num_shards, args.shard_mode)

    if args.format == "count":
        print(len(selected))
    elif args.format == "hydra":
        print(as_hydra_list(selected))
    elif args.format == "json":
        print(json.dumps(
            {"universe": len(universe), "selected": selected, "missing": missing},
            ensure_ascii=False,
        ))
    else:
        print("\n".join(selected))


if __name__ == "__main__":
    main()
