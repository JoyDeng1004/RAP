#!/usr/bin/env python3
"""列出 raw split 目录里"不在 scene_filter 白名单内"的多余条目及其体积。

=============================================================================
本脚本永远不会删除任何东西
=============================================================================
代码里没有 os.remove / shutil.rmtree / unlink 的调用路径，也没有 --execute 开关。
它只做三件事：列出、归类、量体积。真要删，你自己看完报告后手动执行。
这是刻意的：raw 数据删掉就没有了，而且本项目的 split 污染问题尚未收敛
（见 docs/audits/datasets/20260911/download_gap_list.md），
现在销毁 raw 数据会直接消灭重新核验的能力。

=============================================================================
归类逻辑：一个条目"多余"与否，取决于拿谁当白名单
=============================================================================
这是本脚本的核心。只拿 navtrain/navtest/navhard 三个 split 做白名单会误伤，
因为各 split 的 log 集合是**交叉的**：

  * navhard_two_stage 的 76 个 log **完全包含在** navtest 的 136 个里
    （已实测：76/76 命中），所以 navhard 不会给 test 增加额外 log 需求。
  * navmini 的 62 个 log 跨两个 raw split：52 个落在 trainval、10 个落在 test。
    这 52 个全都在 navtrain 的 1192 里，这 10 个全都在 navtest 的 136 里。
  * warmup / private_test_hard 等挑战侧 split 有各自的 log 集合。

所以脚本把每个"多余"条目再过一遍**所有能找到的 filter**，输出三档：

  [其他 split 需要]  不在目标白名单，但被别的 filter 引用 → 删了会破坏那个 split
  [无 filter 引用]   所有已知 filter 都没引用 → 目前看是纯余量
  [UNKNOWN]          缺少该 filter 的定义，无法判断 → 一律按"不可删"对待

只有第二档才谈得上"考虑删除"，而且仍需人工确认。

=============================================================================
体积统计
=============================================================================
Lustre 上对百万文件目录做 du 很慢。默认 --size-mode none 只列条目不算体积；
--size-mode du 用 `du -sb`（准确但慢）；--size-mode sample 只统计第一层文件。
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

FILTER_SUBPATH = "navsim/planning/script/config/common/train_test_split"

DEFAULT_DEVKITS = (
    ("official", "/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim"),
    ("rap", os.environ.get("NAVSIM_DEVKIT_ROOT") or "/gs/bs/tga-RLA/qdeng/RAP"),
)

# 目标 split → 它落在哪个 raw split 上（data_split）
TARGET_SPLITS = {
    "navtrain": "trainval",
    "navtest": "test",
    "navhard_two_stage": "test",
}

# 交叉核验时要额外加载的 filter：任何一个引用了某条目，该条目就不可删
CROSS_CHECK_FILTERS = (
    "navall", "navmini", "navtest", "navtrain", "navhard_two_stage",
    "navsafe_two_stage", "navtest_two_stage", "warmup_two_stage",
    "warmup_test_e2e", "warmup_navsafe_two_stage_extended",
    "private_test_hard_two_stage", "private_test_e2e",
)

_TTY = sys.stdout.isatty()


def paint(text, color):
    if not _TTY:
        return text
    codes = {"red": 31, "green": 32, "yellow": 33, "cyan": 36, "grey": 90, "bold": 1}
    return "\033[%dm%s\033[0m" % (codes[color], text)


def human(num_bytes):
    if num_bytes is None:
        return "—"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return "%.2f %s" % (value, unit)
        value /= 1024


# ---------------------------------------------------------------------------
# filter 解析（与 resolve_scene_filter.py 同款行级解析，零三方依赖）
# ---------------------------------------------------------------------------

_SCALAR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
_ITEM = re.compile(r"^\s*-\s*'?\"?([^'\"]+?)'?\"?\s*$")


def parse_filter(path):
    lists = {}
    current = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = _SCALAR.match(line)
            if match:
                value = re.sub(r"\s+#.*$", "", match.group(2)).strip()
                if value in ("", "|", ">"):
                    lists[match.group(1)] = []
                    current = match.group(1)
                else:
                    current = None
                continue
            if current is not None:
                match = _ITEM.match(line)
                if match:
                    lists[current].append(match.group(1))
    return set(lists.get("log_names", []))


def load_all_filters(devkit_specs):
    """返回 ({filter_name: set(logs)}, {filter_name: 来源说明}, [缺失的 filter 名])。"""
    found, origin, missing = {}, {}, []
    dirs = []
    for alias, root in devkit_specs:
        if not root:
            continue
        candidate = Path(root) / FILTER_SUBPATH / "scene_filter"
        if candidate.is_dir():
            dirs.append((alias, candidate))
    for name in CROSS_CHECK_FILTERS:
        for alias, directory in dirs:
            path = directory / (name + ".yaml")
            if path.is_file():
                logs = parse_filter(path)
                # 同名 filter 多处存在时取并集，做最保守的"不可删"判断
                found[name] = found.get(name, set()) | logs
                origin[name] = origin.get(name, "") + ("[%s]" % alias)
        if name not in found:
            missing.append(name)
    return found, origin, missing, dirs


# ---------------------------------------------------------------------------
# 体积
# ---------------------------------------------------------------------------

def measure(path, mode):
    """返回字节数；mode=none 时返回 None。"""
    if mode == "none":
        return None
    # 条目可能是文件（navsim_logs 下的 .pkl）而不是目录，直接 lstat
    if not path.is_dir():
        try:
            return path.lstat().st_size
        except OSError:
            return None
    if mode == "du":
        try:
            out = subprocess.run(["du", "-sb", str(path)], capture_output=True,
                                 text=True, timeout=1800)
            if out.returncode == 0:
                return int(out.stdout.split()[0])
        except (subprocess.SubprocessError, ValueError, IndexError):
            pass
        return None
    # sample：只统计第一层（含一层子目录里的文件），用于粗估
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
            if root.count(os.sep) - str(path).count(os.sep) >= 1:
                _dirs[:] = []
    except OSError:
        return None
    return total


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------

def strip_suffix(name):
    """navsim_logs 下是 <log>.pkl / .pkl.gz；sensor_blobs 下是目录名 <log>。"""
    for suffix in (".pkl.gz", ".pkl", ".pickle"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def scan_directory(path, whitelist, all_filters, target_names, args):
    """列出 path 下不在 whitelist 里的条目，并交叉核验它被谁引用。"""
    if not path.is_dir():
        return {"path": str(path), "status": "MISSING", "entries": []}

    rows = []
    try:
        with os.scandir(path) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                log_name = strip_suffix(entry.name)
                if log_name in whitelist:
                    continue
                # 交叉核验：还有谁引用它
                referenced_by = sorted(name for name, logs in all_filters.items()
                                       if name not in target_names and log_name in logs)
                if referenced_by:
                    verdict = "OTHER_SPLIT"
                else:
                    verdict = "UNREFERENCED"
                rows.append({
                    "name": entry.name,
                    "log": log_name,
                    "is_dir": entry.is_dir(),
                    "verdict": verdict,
                    "referenced_by": referenced_by,
                    "bytes": measure(Path(entry.path), args.size_mode),
                })
    except (OSError, PermissionError) as error:
        return {"path": str(path), "status": "UNREADABLE", "error": str(error), "entries": []}

    return {"path": str(path), "status": "OK", "whitelist_size": len(whitelist),
            "extra_count": len(rows), "entries": rows}


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def print_block(title, result, args):
    print()
    print(paint("─" * 78, "grey"))
    print(" %s" % paint(title, "bold"))
    print(paint("─" * 78, "grey"))
    print("   %s" % paint(result["path"], "grey"))

    if result["status"] != "OK":
        print("   %s %s" % (paint(result["status"], "red"), result.get("error", "")))
        return 0, 0

    rows = result["entries"]
    print("   白名单 %d 项；目录内不在白名单的条目 %s 项"
          % (result["whitelist_size"], paint(str(len(rows)), "yellow" if rows else "green")))
    if not rows:
        print("   %s 没有多余条目。" % paint("[ OK ]", "green"))
        return 0, 0

    other = [row for row in rows if row["verdict"] == "OTHER_SPLIT"]
    free = [row for row in rows if row["verdict"] == "UNREFERENCED"]

    if other:
        print()
        print("   %s 被其他 split 引用 —— %s" % (paint("[保留]", "red"), paint("删除会破坏那个 split", "red")))
        # 按"被谁引用"归并计数，避免几百行刷屏；明细用 --list-all 或 --json
        tally = {}
        for row in other:
            tally.setdefault(",".join(row["referenced_by"]), []).append(row)
        for owners, group in sorted(tally.items()):
            size = sum(item["bytes"] or 0 for item in group)
            print("     %-40s %5d 项   %10s" % ("← " + owners, len(group), human(size)))
        shown = other if args.list_all else other[:10]
        for row in shown:
            print("       - %-50s %10s   ← %s"
                  % (row["name"], human(row["bytes"]), ",".join(row["referenced_by"])))
        if len(other) > len(shown):
            print("       ... 另有 %d 项，用 --list-all 或 --json 查看全部"
                  % (len(other) - len(shown)))

    if free:
        print()
        print("   %s 所有已知 filter 都未引用" % paint("[无引用]", "yellow"))
        shown = free if args.list_all else free[:30]
        for row in shown:
            print("     - %-52s %10s" % (row["name"], human(row["bytes"])))
        if len(free) > len(shown):
            print("     ... 另有 %d 项，用 --list-all 全部打印" % (len(free) - len(shown)))

    other_bytes = sum(row["bytes"] or 0 for row in other)
    free_bytes = sum(row["bytes"] or 0 for row in free)
    if args.size_mode != "none":
        print()
        print("   小计: 保留 %s / 无引用 %s" % (human(other_bytes), human(free_bytes)))
    return other_bytes, free_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",
                        default=os.environ.get("OPENSCENE_DATA_ROOT",
                                               "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset"))
    parser.add_argument("--devkit", action="append", metavar="别名=路径")
    parser.add_argument("--size-mode", choices=("none", "sample", "du"), default="none",
                        help="none=只列条目(默认,最快) / sample=粗估 / du=准确但在 Lustre 上很慢")
    parser.add_argument("--list-all", action="store_true")
    parser.add_argument("--json", dest="json_out", metavar="PATH")
    args = parser.parse_args()

    specs = list(DEFAULT_DEVKITS)
    if args.devkit:
        specs = []
        for spec in args.devkit:
            alias, _, root = spec.partition("=")
            specs.append((alias or "devkit", root or alias))

    root = Path(args.root)
    if not root.is_dir():
        print("数据根不存在：%s" % root, file=sys.stderr)
        return 2

    filters, origin, missing, dirs = load_all_filters(specs)

    print()
    print(paint("=" * 78, "bold"))
    print(paint(" 多余条目排查（DRY-RUN：本脚本不会删除任何东西）", "bold"))
    print(paint("=" * 78, "bold"))
    print("   数据根: %s" % root)
    for alias, directory in dirs:
        print("   devkit [%s]: %s" % (alias, directory))
    print()
    print("   已加载 filter:")
    for name in sorted(filters):
        print("     %-36s %6d logs  %s" % (name, len(filters[name]), origin.get(name, "")))
    if missing:
        print()
        print("   %s 以下 filter 没有定义，凡涉及它们的判断一律记 UNKNOWN 并按不可删处理:"
              % paint("[警告]", "yellow"))
        for name in missing:
            print("     - %s" % name)

    # 按 raw split 组装白名单：该 raw split 上所有目标 filter 的 log 并集
    whitelists = {}
    unknown_targets = []
    for split_name, data_split in TARGET_SPLITS.items():
        if split_name not in filters:
            unknown_targets.append(split_name)
            continue
        whitelists.setdefault(data_split, set())
        whitelists[data_split] |= filters[split_name]

    if unknown_targets:
        print()
        print("   %s 目标 split %s 缺少定义，其白名单未计入 —— "
              "此时的「多余」判断不可作为删除依据"
              % (paint("[警告]", "red"), ", ".join(unknown_targets)))

    results = {}
    total_other = total_free = 0
    for data_split, whitelist in sorted(whitelists.items()):
        targets = {name for name, split in TARGET_SPLITS.items() if split == data_split}
        for kind in ("navsim_logs", "sensor_blobs"):
            path = root / kind / data_split
            title = "%s/%s   (白名单来自 %s)" % (kind, data_split, "+".join(sorted(targets)))
            result = scan_directory(path, whitelist, filters, targets, args)
            results["%s/%s" % (kind, data_split)] = result
            other_bytes, free_bytes = print_block(title, result, args)
            total_other += other_bytes
            total_free += free_bytes

    print()
    print(paint("=" * 78, "bold"))
    print(paint(" 总结", "bold"))
    print(paint("=" * 78, "bold"))
    if args.size_mode == "none":
        print("   未统计体积。要看体积加 --size-mode du（慢）或 --size-mode sample（粗估）。")
    else:
        print("   被其他 split 引用（不可删）: %s" % human(total_other))
        print("   所有已知 filter 均未引用  : %s" % human(total_free))
    print()
    print("   %s 本脚本没有删除任何文件，也不提供删除开关。" % paint("提醒:", "cyan"))
    print("   「无引用」不等于「可删」：navhard 的合成 blob、maps、两套 two-stage 目录")
    print("   都不在本次 log 级比对范围内；raw 数据删除后无法重新核验。")
    print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"root": str(root), "filters": {k: len(v) for k, v in filters.items()},
             "missing_filters": missing, "results": results},
            ensure_ascii=False, indent=2))
        print("   结构化结果已写入 %s\n" % args.json_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
