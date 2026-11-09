#!/usr/bin/env python3
"""验证「46 个 navtrain history camera 文件是路径错位而非真缺失」这一猜想。

只读脚本：不下载、不移动、不建 symlink、不改配置。

=============================================================================
猜想与反向证据
=============================================================================
猜想：这 46 个文件其实存在，只是代码/配置把期望路径指到了 dataset 根之外。
先例确实有：`private_test_hard` 的 blob 就在
`navsim_workspace/navsim/download/private_test_hard_navsim_sensor/` 而不是
`$OPENSCENE_DATA_ROOT/sensor_blobs/`，审计把它判为「不是下载缺口，是 path 未接合」。

但这 46 个的形态和那次不一样，有两条反向证据：

  1. 这 3 个 log 在 sensor_blobs/trainval 下**是存在的**，各自还有上千个文件在位；
     真的根错位会整段 log 或整个通道消失，不会只漏 46 个。
  2. 46 = 6 个时间戳 × 8 路 camera − 2 个 metadata 主动声明的 `missing_camera/*.png`。
     也就是**整帧缺失**，且缺的全是 history 帧（current 帧 826,304/826,304 全在）。
     这更像上游 `navtrain_history_*.tgz` 的打包缺口，而不是本地路径问题。

所以本脚本的目的不是"证明猜想成立"，而是**把它证伪或证实**，并把证据留下来。
L5 是最直接的一层：直接打印 metadata 里这些引用的原始 data_path 字符串。

=============================================================================
六层检查
=============================================================================
  L0 期望路径复核    46 个目标文件现在到底在不在（含零字节/断链判定）
  L1 同目录兄弟      这 46 个文件所在目录里还有多少别的文件
                     → 目录非空就说明"整目录错位"不成立
  L2 帧级形态        缺的是零散文件还是整帧（8 路一起缺）
  L3 候选根探测      在一批已知候选根下按相同相对路径 stat（快，46×N 次）
  L4 全盘 basename   文件名是内容寻址哈希、全局唯一，可直接按名字搜（慢，需 --deep）
  L5 metadata 原文   从 log pickle 里打印这些引用的原始 data_path
                     → 前缀异常才是真正的"路径错位"
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path, PurePosixPath

MANIFEST = Path(__file__).with_name("navtrain_camera_gaps.json")

# L3 的候选根。来源：docs/audits/datasets/20260911/{path_layout,NAVSIM_inventory}.md
# 里登记过的所有 sensor-blob 类根，以及已知的下载 staging 目录。
CANDIDATE_ROOTS = (
    "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/trainval",
    "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/mini",
    "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/test",
    "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/mini_sensor_blobs/mini",
    "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/navhard_two_stage/sensor_blobs",
    "/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim/download",
    "/gs/bs/tga-RLA/qdeng/RAP/dataset_norm/sensor_blobs",
    "/gs/bs/tga-RLA/qdeng/RAP/dataset_norm/rendered_sensor_blobs",
    "/gs/bs/tga-RLA/qdeng/RAP/dataset_norm/rendered_sensor_blobs_4cam_v1",
    "/gs/bs/tga-RLA/qdeng/RAP/dataset_aug",
    "/gs/bs/tga-RLA/qdeng/RAP/dataset_perturbed",
    "/gs/bs/tga-RLA/qdeng/RAP/navsim_one_scene/dataset/sensor_blobs/trainval",
    "/gs/bs/tga-RLA/qdeng/datasets",
    "/gs/bs/tga-RLA/qdeng/nuplan_dataset",
)

# L4 全盘搜索的边界。不搜 /，避免把整个 Lustre 扫一遍。
DEEP_SEARCH_ROOTS = (
    "/gs/bs/tga-RLA/qdeng",
)

# 解压 staging 常见的中间层目录名，L3 会把它们当额外前缀试一遍
STAGING_PREFIXES = tuple(
    ["", "trainval", "sensor_blobs/trainval"]
    + ["navtrain_history_%d" % i for i in range(1, 33)]
    + ["navtrain_current_%d" % i for i in range(1, 33)]
)

_TTY = sys.stdout.isatty()


def paint(text, color):
    if not _TTY:
        return text
    codes = {"red": 31, "green": 32, "yellow": 33, "cyan": 36, "grey": 90, "bold": 1}
    return "\033[%dm%s\033[0m" % (codes[color], text)


def state_of(path):
    """返回 PRESENT / EMPTY / BROKEN_LINK / ABSENT。"""
    if path.is_file():
        return "PRESENT" if path.stat().st_size > 0 else "EMPTY"
    if os.path.lexists(path):
        return "BROKEN_LINK"
    return "ABSENT"


# ---------------------------------------------------------------------------
# L0 / L1 / L2
# ---------------------------------------------------------------------------

def level0_expected(sensor_root, paths):
    print()
    print(paint(" L0  期望路径复核", "cyan"))
    rows = {}
    for relative in paths:
        rows[relative] = state_of(sensor_root / relative)
    tally = {}
    for value in rows.values():
        tally[value] = tally.get(value, 0) + 1
    for state, count in sorted(tally.items()):
        color = "green" if state == "PRESENT" else "red"
        print("     %-14s %d" % (paint(state, color), count))
    if tally.get("PRESENT"):
        print("     %s 有文件已经在位了 —— 说明缺口清单已过期，先重跑审计" % paint("注意:", "yellow"))
    return rows


def level1_siblings(sensor_root, paths):
    """列出这 46 个文件所在目录各有多少文件。目录非空 → 整目录错位不成立。"""
    print()
    print(paint(" L1  同目录兄弟文件", "cyan"))
    directories = sorted({str(PurePosixPath(p).parent) for p in paths})
    result = {}
    for relative in directories:
        directory = sensor_root / relative
        if not directory.is_dir():
            result[relative] = None
            print("     %-52s %s" % (relative, paint("目录不存在", "red")))
            continue
        try:
            with os.scandir(directory) as entries:
                count = sum(1 for _ in entries)
        except OSError:
            count = None
        result[relative] = count
        missing_here = sum(1 for p in paths if str(PurePosixPath(p).parent) == relative)
        print("     %-52s %6s 个文件（其中缺 %d）"
              % (relative, count if count is not None else "?", missing_here))
    populated = [v for v in result.values() if v]
    if populated:
        print()
        print("     %s 这些目录都非空 —— %s"
              % (paint("结论:", "bold"),
                 paint("「整个 sensor 根指错地方」不成立", "green")))
    return result


def level2_frame_shape(paths):
    """判断缺的是零散文件还是整帧（同一时间戳的 8 路一起缺）。"""
    print()
    print(paint(" L2  缺失形态", "cyan"))
    by_channel = {}
    for relative in paths:
        parts = PurePosixPath(relative).parts
        by_channel.setdefault(parts[1], []).append(parts[0])
    print("     按通道: %s" % json.dumps({k: len(v) for k, v in sorted(by_channel.items())}))
    channels = len(by_channel)
    per_channel = sorted({len(v) for v in by_channel.values()})
    print("     覆盖 %d 个通道，每通道缺 %s 个" % (channels, "~".join(map(str, per_channel))))
    if channels == 8 and max(per_channel) - min(per_channel) <= 1:
        print("     %s 八路近似均匀 → 形态是%s，不是零散文件"
              % (paint("结论:", "bold"), paint("整帧缺失", "yellow")))
        print("            这更像上游 navtrain_history_*.tgz 的打包缺口")
    return {k: len(v) for k, v in by_channel.items()}


# ---------------------------------------------------------------------------
# L3 候选根
# ---------------------------------------------------------------------------

def level3_candidates(paths, roots, args):
    print()
    print(paint(" L3  候选根按相同相对路径探测", "cyan"))
    hits = []
    for root_text in roots:
        root = Path(root_text)
        if not root.is_dir():
            if args.verbose:
                print("     %-62s %s" % (root_text, paint("根不存在", "grey")))
            continue
        for prefix in STAGING_PREFIXES:
            found = 0
            sample = None
            for relative in paths:
                candidate = root / prefix / relative if prefix else root / relative
                if candidate.is_file() and candidate.stat().st_size > 0:
                    found += 1
                    sample = sample or str(candidate)
            if found:
                label = "%s%s" % (root_text, ("/" + prefix) if prefix else "")
                color = "red" if found == len(paths) else "yellow"
                print("     %s %-58s %d/%d 命中"
                      % (paint("[命中]", color), label, found, len(paths)))
                print("            例: %s" % sample)
                hits.append({"root": root_text, "prefix": prefix, "found": found})
    if not hits:
        print("     %s 所有候选根 × %d 种前缀组合，一个都没命中"
              % (paint("[无命中]", "green"), len(STAGING_PREFIXES)))
    return hits


# ---------------------------------------------------------------------------
# L4 全盘 basename 搜索
# ---------------------------------------------------------------------------

def level4_deep_search(paths, roots, args):
    print()
    print(paint(" L4  全盘 basename 搜索", "cyan"))
    basenames = sorted({PurePosixPath(p).name for p in paths})
    print("     文件名是内容寻址哈希、全局唯一，可直接按名字搜")
    print("     搜索根: %s" % ", ".join(roots))

    # lfs find 在 Lustre 上比 GNU find 快得多；没有就退回 find
    use_lfs = subprocess.run(["which", "lfs"], capture_output=True).returncode == 0
    tool = "lfs find" if use_lfs else "find"
    print("     使用: %s（超时 %d 秒）" % (tool, args.deep_timeout))

    found = {}
    for root in roots:
        command = (["lfs", "find", root] if use_lfs else ["find", root])
        # 一次传入全部 46 个 -name，只遍历一遍
        expr = []
        for index, name in enumerate(basenames):
            if index:
                expr.append("-o")
            expr += ["-name", name]
        command += ["("] + expr + [")"]
        try:
            out = subprocess.run(command, capture_output=True, text=True,
                                 timeout=args.deep_timeout)
        except subprocess.TimeoutExpired:
            print("     %s %s 超时（%d 秒）—— 换 qsub 跑或调大 --deep-timeout"
                  % (paint("[超时]", "yellow"), root, args.deep_timeout))
            continue
        for line in out.stdout.splitlines():
            line = line.strip()
            if line:
                found.setdefault(PurePosixPath(line).name, []).append(line)

    if not found:
        print("     %s 46 个文件名在搜索范围内一个都不存在"
              % paint("[无命中]", "green"))
        print("            → 猜想被证伪：它们是真的没有，不是路径错位")
    else:
        print("     %s 命中 %d/%d 个文件名:"
              % (paint("[命中]", "red"), len(found), len(basenames)))
        for name, locations in sorted(found.items()):
            for location in locations:
                print("       %s  →  %s" % (name, location))
    return found


# ---------------------------------------------------------------------------
# L5 metadata 原文
# ---------------------------------------------------------------------------

def walk_for_strings(obj, targets, hits, depth=0):
    """在反序列化后的对象里递归找包含目标 basename 的字符串。

    不假设 pickle 的具体 schema——navsim 各版本的 dataclass 结构不同，
    通用遍历比写死字段名可靠。
    """
    if depth > 12:
        return
    if isinstance(obj, str):
        for name in targets:
            if name in obj:
                hits.setdefault(name, set()).add(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            walk_for_strings(value, targets, hits, depth + 1)
        return
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            walk_for_strings(value, targets, hits, depth + 1)
        return
    if hasattr(obj, "__dict__"):
        walk_for_strings(vars(obj), targets, hits, depth + 1)


def level5_metadata(log_root, paths, args):
    """打印 metadata 里这些引用的原始 data_path 字符串。"""
    print()
    print(paint(" L5  metadata 里的原始 data_path", "cyan"))
    if not log_root.is_dir():
        print("     %s %s" % (paint("日志根不存在", "red"), log_root))
        return {}

    logs = sorted({PurePosixPath(p).parts[0] for p in paths})
    targets = {PurePosixPath(p).name for p in paths}
    all_hits = {}
    for log_name in logs:
        candidate = None
        for suffix in (".pkl", ".pkl.gz", ".pickle"):
            probe = log_root / (log_name + suffix)
            if probe.is_file():
                candidate = probe
                break
        if candidate is None:
            print("     %-46s %s" % (log_name, paint("找不到 metadata", "red")))
            continue
        try:
            opener = open
            if candidate.suffix == ".gz":
                import gzip
                opener = gzip.open
            with opener(candidate, "rb") as handle:
                data = pickle.load(handle)
        except Exception as error:                     # pickle 可能是任意异常
            print("     %-46s %s %s" % (log_name, paint("读取失败", "red"), error))
            continue

        hits = {}
        walk_for_strings(data, targets, hits)
        all_hits.update({k: sorted(v) for k, v in hits.items()})
        print("     %-46s 命中 %d 条引用" % (log_name, len(hits)))

    if not all_hits:
        print("     %s metadata 里找不到这些文件名" % paint("[异常]", "yellow"))
        return {}

    # 把引用的目录前缀归并，异常前缀一眼可见
    prefixes = {}
    for name, strings in all_hits.items():
        for text in strings:
            prefixes.setdefault(str(PurePosixPath(text).parent), []).append(name)
    print()
    print("     引用字符串的目录前缀分布:")
    for prefix, names in sorted(prefixes.items()):
        normal = len(PurePosixPath(prefix).parts) == 2      # <log>/<CAM>
        mark = paint("正常", "green") if normal else paint("异常前缀 ←", "red")
        print("       %-58s %3d 条  %s" % (prefix, len(names), mark))
    if all(len(PurePosixPath(p).parts) == 2 for p in prefixes):
        print()
        print("     %s 所有引用都是标准的 <log>/<CAM>/<hash>.jpg 两级相对路径"
              % paint("结论:", "bold"))
        print("            metadata 侧%s，路径错位猜想在这一层不成立"
              % paint("没有异常前缀", "green"))
    if args.list_all:
        print()
        for name, strings in sorted(all_hits.items()):
            for text in strings:
                print("       %s" % text)
    return all_hits


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root",
                        default=os.environ.get("OPENSCENE_DATA_ROOT",
                                               "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset"))
    parser.add_argument("--sensor-root", help="默认 <data-root>/sensor_blobs/trainval")
    parser.add_argument("--log-root", help="默认 <data-root>/navsim_logs/trainval")
    parser.add_argument("--deep", action="store_true", help="启用 L4 全盘搜索（慢）")
    parser.add_argument("--deep-timeout", type=int, default=3600)
    parser.add_argument("--skip-metadata", action="store_true", help="跳过 L5（读 pickle 较慢）")
    parser.add_argument("--list-all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", dest="json_out", metavar="PATH")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    paths = manifest["paths"]

    data_root = Path(args.data_root)
    sensor_root = Path(args.sensor_root) if args.sensor_root else data_root / "sensor_blobs" / "trainval"
    log_root = Path(args.log_root) if args.log_root else data_root / "navsim_logs" / "trainval"

    print()
    print(paint("=" * 78, "bold"))
    print(paint(" 46 个 navtrain history camera 文件 —— 路径错位猜想验证（只读）", "bold"))
    print(paint("=" * 78, "bold"))
    print("   缺口清单 : %s（%d 个文件）" % (MANIFEST.name, len(paths)))
    print("   sensor 根: %s" % sensor_root)
    print("   log 根   : %s" % log_root)

    result = {"expected": level0_expected(sensor_root, paths)}
    result["siblings"] = level1_siblings(sensor_root, paths)
    result["shape"] = level2_frame_shape(paths)
    result["candidates"] = level3_candidates(paths, CANDIDATE_ROOTS, args)
    if args.deep:
        result["deep_search"] = level4_deep_search(paths, DEEP_SEARCH_ROOTS, args)
    else:
        print()
        print(paint(" L4  全盘 basename 搜索", "cyan"))
        print("     %s 未启用。加 --deep 开启（建议走 qsub）" % paint("[跳过]", "grey"))
    if not args.skip_metadata:
        result["metadata"] = level5_metadata(log_root, paths, args)

    print()
    print(paint("=" * 78, "bold"))
    print(paint(" 判读指引", "bold"))
    print(paint("=" * 78, "bold"))
    print("   L1 目录非空 + L3 无命中 + L5 前缀正常  →  猜想不成立，是真缺失")
    print("   L3 或 L4 有命中                        →  猜想成立，去接合路径（先报批，别直接建链）")
    print("   L5 出现异常前缀                        →  是 metadata 侧的路径问题，不是文件缺失")
    print()
    print("   %s 本脚本没有移动任何文件，也没有创建任何 symlink。" % paint("提醒:", "cyan"))
    print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        print("   结构化结果已写入 %s\n" % args.json_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
