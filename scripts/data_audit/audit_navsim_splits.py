#!/usr/bin/env python3
"""审计 $OPENSCENE_DATA_ROOT 下 navtrain / navtest / navhard_two_stage 三个 split 的完整性。

只读脚本：不下载、不解压、不创建、不删除、不修改任何文件。

=============================================================================
一、判定标准的来源（不是靠目录名猜的）
=============================================================================

官方 splits 文档 https://github.com/autonomousvision/navsim/blob/main/docs/splits.md
里最关键的一句是：训练/测试 split 是 **scene filter**，不是可以单独下载的数据集。
所以这三个 split 的物理落盘位置是不对称的：

  navtrain            → train_test_split/navtrain.yaml           data_split: trainval
                        实际读 navsim_logs/trainval + sensor_blobs/trainval
                        （目录里**不会**出现名为 navtrain 的文件夹，这是正常的）

  navtest             → train_test_split/navtest.yaml            data_split: test
                        实际读 navsim_logs/test + sensor_blobs/test
                        （同理，不会有 navtest 文件夹）

  navhard_two_stage   → train_test_split/navhard_two_stage.yaml  data_split: test
                        既要 navsim_logs/test + sensor_blobs/test（原始/第一阶段），
                        又要 navhard_two_stage/ 下的合成场景（第二阶段）

路径公式来自 devkit 的 config/common/default_dataset_paths.yaml：

    navsim_log_path        = $OPENSCENE_DATA_ROOT/navsim_logs/${data_split}
    original_sensor_path   = $OPENSCENE_DATA_ROOT/sensor_blobs/${data_split}
    synthetic_sensor_path  = $OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
    synthetic_scenes_path  = $OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

注意最后一行：官方 main 分支叫 `synthetic_scene_pickles`，但部分 v2.2 发布物解出来
是 `openscene_meta_datas`。本脚本两个名字都接受，并且会明确告诉你实际是哪一个
——因为如果和你 devkit 版本对不上，eval 会在路径解析处直接失败。

=============================================================================
二、四层检查
=============================================================================
  L1 目录存在性   —— 该 split 依赖的每个目录在不在
  L2 目录非空     —— 存在但空目录 = 下载/解压中断的典型现场
  L3 log 级覆盖   —— 用官方 scene_filter 的 log_names 清单逐个核对：
                     metadata pickle 在不在？sensor_blobs 下对应 log 目录在不在？
  L4 通道级抽查   —— 命中的 log 目录里 8 路 camera 通道齐不齐（默认抽样，--deep 全查）

L3 是核心：只有它能区分「整个 split 没下」和「下了但少了几个 log」。

=============================================================================
三、性能
=============================================================================
sensor_blobs/trainval 下有百万级文件，脚本**不做全目录遍历**。
只在 log 层和 camera 通道层 scandir，通道内只探"有没有第一个文件"，不数总数。
默认 --sample 32 个 log 做通道级抽查；--deep 才对全部命中 log 做通道级检查。
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# OpenScene 的 8 路环视相机通道
CAMERAS = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2")

# navhard 第二阶段合成场景目录的两种已知命名（官方 main vs 部分 v2.2 发布物）
SYNTHETIC_SCENE_DIRS = ("synthetic_scene_pickles", "openscene_meta_datas")

# scene_filter yaml 的候选搜索位置，按优先级排列。
# 之所以要搜多处：RAP 内置的 devkit 是 two-stage 之前的版本，没有 navhard 配置，
# 真正的 v2.2 定义在 navsim_workspace/navsim 那个 checkout 里。
SCENE_FILTER_SUBPATH = "navsim/planning/script/config/common/train_test_split"

DEVKIT_CANDIDATES = (
    os.environ.get("NAVSIM_DEVKIT_ROOT", ""),
    "/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim",   # 官方 checkout（v2.2）
    "/gs/bs/tga-RLA/qdeng/RAP",                       # RAP 内置 devkit（较旧）
)

# 终端着色；非 tty（重定向到文件、qsub 日志）时自动关闭
_TTY = sys.stdout.isatty()


def paint(text, color):
    if not _TTY:
        return text
    codes = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "grey": 90, "bold": 1}
    return "\033[%dm%s\033[0m" % (codes[color], text)


def display_width(text):
    """CJK 字符在终端占两列；按显示宽度算，才能让表格对齐。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text, width):
    """按显示宽度左对齐补空格（超宽则原样返回）。"""
    return text + " " * max(0, width - display_width(text))


OK = paint("[ OK ]", "green")
WARN = paint("[WARN]", "yellow")
FAIL = paint("[FAIL]", "red")
INFO = paint("[INFO]", "blue")


# ---------------------------------------------------------------------------
# 极简 YAML 读取：只取需要的几个 key，避免强依赖 PyYAML
# ---------------------------------------------------------------------------

def read_yaml(path):
    """优先用 PyYAML；没装则退回只解析本脚本需要的字段的行级解析器。

    需要的字段很有限：data_split（标量）、log_names / tokens（字符串列表）。
    scene_filter yaml 的这两个列表都是扁平的 `  - 'xxx'` 形式，行级解析足够。
    """
    try:
        import yaml                                    # noqa: F401  (可选依赖)
        with path.open("r") as handle:
            return yaml.safe_load(handle) or {}
    except ImportError:
        pass

    data = {}
    current_key = None
    scalar = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
    item = re.compile(r"^\s*-\s*'?\"?([^'\"]+)'?\"?\s*$")
    with path.open("r") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = scalar.match(line)
            if match:                                   # 顶格的 key
                key, value = match.group(1), match.group(2).strip()
                if value in ("", "|", ">"):
                    data[key] = []
                    current_key = key
                else:
                    data[key] = value.strip("'\"")
                    current_key = None
                continue
            # 列表项可能缩进（navtest.yaml 的 "  - 'x'"），也可能顶格
            # （navall.yaml 的 "- x"）——YAML 两种都合法，必须都接受
            if current_key is not None:
                match = item.match(line)
                if match and isinstance(data.get(current_key), list):
                    data[current_key].append(match.group(1))
    return data


def locate_split_config_dir(explicit=None):
    """找到 train_test_split 配置目录，返回 (路径, 说明)。找不到返回 (None, 原因)。"""
    if explicit:
        candidate = Path(explicit)
        if candidate.is_dir():
            return candidate, "由 --split-config 指定"
        return None, "--split-config 指定的目录不存在：%s" % explicit
    for root in DEVKIT_CANDIDATES:
        if not root:
            continue
        candidate = Path(root) / SCENE_FILTER_SUBPATH
        if candidate.is_dir():
            return candidate, str(candidate)
    return None, "未找到 train_test_split 配置目录（试过 NAVSIM_DEVKIT_ROOT 及默认候选）"


def load_scene_filter(config_dir, split_name):
    """读取 split 顶层 yaml + 其 scene_filter，返回 (data_split, log_names, 说明)。

    读不到时返回 (None, None, 原因)——绝不编造期望值。
    """
    if config_dir is None:
        return None, None, "无 devkit 配置目录"
    top = config_dir / (split_name + ".yaml")
    if not top.is_file():
        return None, None, "缺少 %s（该 devkit 版本可能不含此 split 定义）" % top.name

    top_data = read_yaml(top)
    data_split = top_data.get("data_split")

    # defaults 里 `- scene_filter: xxx` 指向具体的 filter 文件；行级解析拿不到嵌套结构，
    # 这里直接用文本匹配，两种解析路径下都能工作
    text = top.read_text()
    match = re.search(r"scene_filter:\s*([A-Za-z0-9_]+)", text)
    filter_name = match.group(1) if match else split_name

    filter_path = config_dir / "scene_filter" / (filter_name + ".yaml")
    if not filter_path.is_file():
        return data_split, None, "缺少 scene_filter/%s.yaml" % filter_name

    filter_data = read_yaml(filter_path)
    log_names = filter_data.get("log_names") or None
    if isinstance(log_names, list):
        log_names = [str(name) for name in log_names]

    note = "scene_filter/%s.yaml" % filter_name
    # 关键陷阱：split 顶层 yaml 引用的 filter 名字可能被改过（例如 navtrain.yaml
    # 指到 navall），这会让"navtrain"实际覆盖的 log 集合完全不同。必须显式提示。
    if filter_name != split_name:
        note += "  ⚠ 注意：%s.yaml 引用的是 '%s' 而非同名 filter" % (split_name, filter_name)
    return data_split, log_names, note


# ---------------------------------------------------------------------------
# 文件系统探测（全部只读）
# ---------------------------------------------------------------------------

def dir_state(path):
    """返回 (状态, 条目数)。状态取值 MISSING / NOT_A_DIR / EMPTY / OK。

    条目数用 scandir 数第一层，不递归——百万级目录也只花一次目录读。
    """
    if not path.exists():
        return "MISSING", 0
    if not path.is_dir():
        return "NOT_A_DIR", 0
    count = 0
    try:
        with os.scandir(path) as entries:
            for _ in entries:
                count += 1
    except PermissionError:
        return "NO_PERMISSION", 0
    return ("EMPTY" if count == 0 else "OK"), count


def list_child_names(path):
    """列出第一层子项名字集合；路径不可用时返回空集合。"""
    try:
        with os.scandir(path) as entries:
            return {entry.name for entry in entries}
    except (OSError, PermissionError):
        return set()


def log_pickle_stems(log_dir):
    """navsim_logs/<split>/ 下的 metadata 文件名（去掉扩展名）集合。

    devkit 用 .pkl / .pkl.gz 两种；这里都接受，统一归一到 log 名。
    """
    stems = set()
    try:
        with os.scandir(log_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name
                for suffix in (".pkl.gz", ".pkl", ".pickle"):
                    if name.endswith(suffix):
                        stems.add(name[: -len(suffix)])
                        break
    except (OSError, PermissionError):
        pass
    return stems


def check_camera_channels(log_dir):
    """检查某个 log 的 sensor 目录里 8 路 camera 通道齐不齐。

    返回 (缺失通道列表, 空通道列表)。只探"有没有第一个文件"，不数总数。
    """
    present = list_child_names(log_dir)
    missing = [camera for camera in CAMERAS if camera not in present]
    empty = []
    for camera in CAMERAS:
        if camera in missing:
            continue
        channel = log_dir / camera
        try:
            with os.scandir(channel) as entries:
                if next(iter(entries), None) is None:
                    empty.append(camera)
        except (OSError, PermissionError):
            empty.append(camera)
    return missing, empty


# ---------------------------------------------------------------------------
# 单个 split 的审计
# ---------------------------------------------------------------------------

def audit_filter_split(root, split_name, data_split, log_names, filter_note, args):
    """审计 navtrain / navtest 这类"scene filter over 某个 raw split"的 split。"""
    report = {
        "split": split_name,
        "kind": "scene_filter",
        "data_split": data_split,
        "filter_source": filter_note,
        "expected_logs": len(log_names) if log_names else None,
        "problems": [],
        "dirs": {},
    }

    if not data_split:
        report["problems"].append("无法确定 data_split：devkit 配置缺失或无法解析")
        report["verdict"] = "UNKNOWN"
        return report

    log_root = root / "navsim_logs" / data_split
    sensor_root = root / "sensor_blobs" / data_split

    for label, path in (("navsim_logs", log_root), ("sensor_blobs", sensor_root)):
        state, count = dir_state(path)
        report["dirs"][label] = {"path": str(path), "state": state, "entries": count}
        if state != "OK":
            report["problems"].append("%s 目录 %s：%s" % (label, state, path))

    # 两个根目录只要有一个不可用，就没法做 log 级核对了
    if any(item["state"] != "OK" for item in report["dirs"].values()):
        report["verdict"] = "MISSING"
        return report

    if not log_names:
        report["problems"].append("拿不到 scene_filter 的 log_names（%s），只能做到目录级判定" % filter_note)
        report["verdict"] = "UNKNOWN"
        return report

    # ---- L3：逐 log 核对 metadata 与 sensor 目录 ----
    metadata_stems = log_pickle_stems(log_root)
    sensor_dirs = list_child_names(sensor_root)

    missing_metadata = sorted(name for name in log_names if name not in metadata_stems)
    missing_sensor = sorted(name for name in log_names if name not in sensor_dirs)
    report["missing_metadata"] = missing_metadata
    report["missing_sensor_dirs"] = missing_sensor

    # ---- L4：通道级检查（默认抽样） ----
    hit_logs = [name for name in log_names if name in sensor_dirs]
    inspect = hit_logs if args.deep else hit_logs[: args.sample]
    channel_problems = {}
    for name in inspect:
        missing, empty = check_camera_channels(sensor_root / name)
        if missing or empty:
            channel_problems[name] = {"missing_channels": missing, "empty_channels": empty}
    report["channel_checked"] = len(inspect)
    report["channel_mode"] = "deep(全部命中 log)" if args.deep else "sample(前 %d 个)" % args.sample
    report["channel_problems"] = channel_problems

    if missing_metadata:
        report["problems"].append("%d/%d 个 log 缺 metadata pickle" % (len(missing_metadata), len(log_names)))
    if missing_sensor:
        report["problems"].append("%d/%d 个 log 在 sensor_blobs 下没有目录" % (len(missing_sensor), len(log_names)))
    if channel_problems:
        report["problems"].append("%d 个受检 log 的 camera 通道不全" % len(channel_problems))

    report["verdict"] = "PASS" if not report["problems"] else "PARTIAL"
    return report


def audit_navhard(root, data_split, log_names, filter_note, args):
    """审计 navhard_two_stage：它同时依赖第一阶段(raw test)和第二阶段(合成场景)。"""
    report = {
        "split": "navhard_two_stage",
        "kind": "two_stage",
        "data_split": data_split,
        "filter_source": filter_note,
        "expected_logs": len(log_names) if log_names else None,
        "problems": [],
        "dirs": {},
    }

    # ---- 第二阶段：navhard_two_stage/ 自己的目录 ----
    base = root / "navhard_two_stage"
    state, count = dir_state(base)
    report["dirs"]["navhard_two_stage"] = {"path": str(base), "state": state, "entries": count}
    if state != "OK":
        report["problems"].append("navhard_two_stage 根目录 %s：%s" % (state, base))
        report["verdict"] = "MISSING"
        return report

    sensor_state, sensor_count = dir_state(base / "sensor_blobs")
    report["dirs"]["navhard_two_stage/sensor_blobs"] = {
        "path": str(base / "sensor_blobs"), "state": sensor_state, "entries": sensor_count}
    if sensor_state != "OK":
        report["problems"].append(
            "第二阶段 sensor_blobs %s：%s（对应 config 的 synthetic_sensor_path）"
            % (sensor_state, base / "sensor_blobs"))

    # 合成场景目录：两种命名二选一即可，所以合并成一行展示，
    # 不要因为"另一个名字不存在"就报 FAIL
    found_scene_dir = None
    scene_entries = 0
    for name in SYNTHETIC_SCENE_DIRS:
        scene_state, scene_count = dir_state(base / name)
        if scene_state == "OK" and found_scene_dir is None:
            found_scene_dir, scene_entries = name, scene_count
    report["dirs"]["navhard_two_stage/<合成场景> (%s 二选一)" % "|".join(SYNTHETIC_SCENE_DIRS)] = {
        "path": str(base / SYNTHETIC_SCENE_DIRS[0]),
        "state": "OK" if found_scene_dir else "MISSING",
        "entries": scene_entries,
        "resolved_as": found_scene_dir,
    }
    report["synthetic_scene_dir"] = found_scene_dir
    if found_scene_dir is None:
        report["problems"].append(
            "合成场景目录缺失：%s 下找不到 %s 任何一个"
            % (base, " 或 ".join(SYNTHETIC_SCENE_DIRS)))
    elif found_scene_dir != "synthetic_scene_pickles":
        # 不算缺数据，但会让官方 main 的 synthetic_scenes_path 解析不到，必须提示
        report["problems"].append(
            "合成场景目录名是 '%s'，而官方 default_dataset_paths.yaml 期望 "
            "'synthetic_scene_pickles'；devkit 版本不一致会在 eval 时路径解析失败" % found_scene_dir)

    # ---- 第一阶段：navhard 的 data_split 是 test，所以还要有 raw test ----
    if data_split:
        for label, path in (("navsim_logs", root / "navsim_logs" / data_split),
                            ("sensor_blobs", root / "sensor_blobs" / data_split)):
            state, count = dir_state(path)
            key = "%s/%s (第一阶段依赖)" % (label, data_split)
            report["dirs"][key] = {"path": str(path), "state": state, "entries": count}
            if state != "OK":
                report["problems"].append("第一阶段依赖 %s 目录 %s：%s" % (label, state, path))
    else:
        report["problems"].append("拿不到 navhard 的 data_split，无法核对第一阶段依赖")

    if log_names:
        # navhard 的 scene_filter 也带 log_names，核对它们在 raw test metadata 里在不在
        log_root = root / "navsim_logs" / (data_split or "test")
        if log_root.is_dir():
            stems = log_pickle_stems(log_root)
            missing = sorted(name for name in log_names if name not in stems)
            report["missing_metadata"] = missing
            if missing:
                report["problems"].append(
                    "%d/%d 个 navhard log 在 %s 下缺 metadata" % (len(missing), len(log_names), log_root))
    else:
        report["problems"].append("拿不到 navhard scene_filter 的 log_names（%s）" % filter_note)

    report["verdict"] = "PASS" if not report["problems"] else "PARTIAL"
    return report


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------

VERDICT_STYLE = {
    "PASS": (OK, "green"), "PARTIAL": (FAIL, "red"),
    "MISSING": (FAIL, "red"), "UNKNOWN": (WARN, "yellow"),
}


def print_report(report, args):
    badge, color = VERDICT_STYLE.get(report["verdict"], (WARN, "yellow"))
    print()
    print(paint("─" * 78, "grey"))
    print(" %s  %s" % (paint(report["split"], "bold"), badge + " " + paint(report["verdict"], color)))
    print(paint("─" * 78, "grey"))

    if report["kind"] == "scene_filter":
        print("  %s 该 split 是 data_split='%s' 上的 scene filter，"
              "目录下**不应该**有同名文件夹" % (INFO, report["data_split"]))
    print("  %s scene_filter 来源: %s   期望 log 数: %s"
          % (INFO, report["filter_source"], report["expected_logs"] if report["expected_logs"] else "UNKNOWN"))

    print()
    print("  目录检查:")
    for label, info in report["dirs"].items():
        mark = OK if info["state"] == "OK" else FAIL
        if info["state"] == "OK":
            detail = "%d 个条目" % info["entries"]
            if info.get("resolved_as"):
                detail += "  (实际为 %s)" % info["resolved_as"]
        else:
            detail = info["state"]
        print("    %s %s %s" % (mark, pad(label, 46), detail))
        if info["state"] != "OK":
            print("           %s" % paint(info["path"], "grey"))

    if report["verdict"] == "PASS":
        print()
        print("  %s 未发现缺失。" % OK)
        return

    print()
    print("  %s 问题汇总:" % paint("缺失/异常", "red"))
    for problem in report["problems"]:
        print("    • %s" % problem)

    # 具体缺了什么——这是用户最需要的部分
    for key, title in (("missing_metadata", "缺 metadata pickle 的 log"),
                       ("missing_sensor_dirs", "缺 sensor_blobs 目录的 log")):
        items = report.get(key) or []
        if not items:
            continue
        print()
        print("    %s（%d 个）:" % (title, len(items)))
        shown = items if args.list_all else items[:20]
        for name in shown:
            print("      - %s" % name)
        if len(items) > len(shown):
            print("      ... 另有 %d 个，用 --list-all 全部打印" % (len(items) - len(shown)))

    problems = report.get("channel_problems") or {}
    if problems:
        print()
        print("    camera 通道不全的 log（检查方式: %s）:" % report.get("channel_mode"))
        for name, detail in list(problems.items())[: (None if args.list_all else 20)]:
            parts = []
            if detail["missing_channels"]:
                parts.append("缺通道 " + ",".join(detail["missing_channels"]))
            if detail["empty_channels"]:
                parts.append("空通道 " + ",".join(detail["empty_channels"]))
            print("      - %s: %s" % (name, "; ".join(parts)))


def print_header(root, config_dir, config_note):
    print()
    print(paint("=" * 78, "bold"))
    print(paint(" NAVSIM split 审计（只读）", "bold"))
    print(paint("=" * 78, "bold"))
    print("  数据根 : %s" % root)
    print("  devkit : %s" % config_note)
    if config_dir is None:
        print("  %s 没有 devkit 配置就拿不到官方 log 清单，只能做目录级判定。" % WARN)
        print("       可用 --split-config 指定 train_test_split 目录。")
    print("  %s 判定标准来自 docs/splits.md + config/common/default_dataset_paths.yaml" % INFO)


def print_summary(reports):
    print()
    print(paint("=" * 78, "bold"))
    print(paint(" 总结", "bold"))
    print(paint("=" * 78, "bold"))
    for report in reports:
        badge, color = VERDICT_STYLE.get(report["verdict"], (WARN, "yellow"))
        head = report["problems"][0] if report["problems"] else "未发现缺失"
        print("  %s %-22s %s" % (badge, report["split"], paint(head, color)))
    print()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="审计 navtrain / navtest / navhard_two_stage 三个 split 的完整性（只读）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        default=os.environ.get("OPENSCENE_DATA_ROOT",
                               "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset"),
        help="数据根，默认取 $OPENSCENE_DATA_ROOT")
    parser.add_argument("--split-config",
                        help="train_test_split 配置目录；不给则自动搜 devkit")
    parser.add_argument("--splits", default="navtrain,navtest,navhard_two_stage",
                        help="要审计的 split，逗号分隔")
    parser.add_argument("--deep", action="store_true",
                        help="对全部命中 log 做 camera 通道检查（慢，百万文件目录慎用）")
    parser.add_argument("--sample", type=int, default=32,
                        help="非 deep 模式下做通道检查的 log 数，默认 32")
    parser.add_argument("--list-all", action="store_true", help="打印全部缺失项，不截断")
    parser.add_argument("--json", dest="json_out", metavar="PATH",
                        help="把结构化结果另存为 JSON（唯一会写盘的选项）")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print("%s 数据根不存在或不是目录：%s" % (FAIL, root), file=sys.stderr)
        return 2

    config_dir, config_note = locate_split_config_dir(args.split_config)
    print_header(root, config_dir, config_note)

    reports = []
    for split_name in [item.strip() for item in args.splits.split(",") if item.strip()]:
        data_split, log_names, filter_note = load_scene_filter(config_dir, split_name)
        if split_name == "navhard_two_stage":
            # navhard 的 data_split 在官方是 test；配置读不到时退回 test 但会标注
            report = audit_navhard(root, data_split or "test", log_names, filter_note, args)
        else:
            report = audit_filter_split(root, split_name, data_split, log_names, filter_note, args)
        reports.append(report)
        print_report(report, args)

    print_summary(reports)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"root": str(root), "devkit": config_note, "reports": reports},
                       ensure_ascii=False, indent=2))
        print("  结构化结果已写入 %s\n" % args.json_out)

    # 退出码：0 全部 PASS；1 有 PARTIAL/MISSING/UNKNOWN；2 脚本自身无法运行
    return 0 if all(report["verdict"] == "PASS" for report in reports) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
