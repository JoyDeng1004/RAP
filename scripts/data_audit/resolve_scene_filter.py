#!/usr/bin/env python3
"""查清一个 scene_filter（如 navall）到底指向什么，并与所有评测侧 filter 求交。

只读脚本：不下载、不写数据、不改配置。--json 是唯一会写盘的选项。

=============================================================================
为什么需要这个脚本
=============================================================================
RAP 的 `train_test_split/navtrain.yaml` 引用的是 `scene_filter: navall`，
而官方 navsim 的同名文件引用的是 `scene_filter: navtrain`。两者规模差约 3.9 倍，
所以"navtrain"这个名字在 RAP 里的实际含义必须实测，不能按名字推断。

本仓库已有的 `scripts/audit/a1_clean_subset.py` 做过这件事，但它有一个硬性盲区：
它在 RAP 自带的 scene_filter 目录里找 `navhard_two_stage.yaml`，而 **RAP 内置的
devkit 是 two-stage 之前的版本，根本没有这个文件**，于是它只能记
`MISSING_SPLIT_DEFINITION` 而跳过。v2.2 的 navhard / warmup / private 定义只存在于
`/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim` 那个官方 checkout 里。

本脚本补的就是这个洞：**同时挂载多个 devkit 根**，把两边的 filter 合并成一个
命名空间，再做 log 级 + token 级的两两求交。

=============================================================================
输出什么
=============================================================================
  1. 目标 filter 的身份：被谁引用、规模、SceneFilter 参数（num_history_frames 等）
  2. 与每个评测侧 filter 的交集（log 级和 token 级分别给，因为 SD-15a 未裁决）
  3. 同名 filter 在不同 devkit 间是否一致（RAP 的副本是否被改过）
  4. 可选 --verify-metadata：目标 filter 的 token 在 raw metadata 里真的存在吗
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# scene_filter 相对 devkit 根的位置
FILTER_SUBPATH = "navsim/planning/script/config/common/train_test_split"

# 默认挂载的 devkit 根：RAP 内置（较旧）+ 官方 v2.2 checkout
DEFAULT_DEVKITS = (
    ("rap", os.environ.get("NAVSIM_DEVKIT_ROOT") or "/gs/bs/tga-RLA/qdeng/RAP"),
    ("official", "/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim"),
)

# 评测/挑战侧：任何与训练池的交集都是污染信号
EVAL_FILTERS = (
    "navtest", "navhard_two_stage", "navsafe_two_stage", "navtest_two_stage",
    "warmup_two_stage", "warmup_test_e2e", "warmup_navsafe_two_stage_extended",
    "private_test_hard_two_stage", "private_test_e2e", "competition_private_tokens",
    "navmini",
)

# SceneFilter 的标量参数，用于判断"同名 filter 是否真的同一个东西"
SCALAR_KEYS = ("num_history_frames", "num_future_frames", "frame_interval",
               "has_route", "max_scenes", "_target_")

_TTY = sys.stdout.isatty()


def paint(text, color):
    if not _TTY:
        return text
    codes = {"red": 31, "green": 32, "yellow": 33, "cyan": 36, "grey": 90, "bold": 1}
    return "\033[%dm%s\033[0m" % (codes[color], text)


# ---------------------------------------------------------------------------
# YAML：只解析需要的结构，不依赖 PyYAML
#
# scene_filter 文件很大（navall.yaml 41 万行），但结构极简：顶格 `key:` 之后跟
# 若干 `- value` 或 `  - 'value'`。行级解析比 PyYAML 快且零依赖。
# ---------------------------------------------------------------------------

_SCALAR = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
_ITEM = re.compile(r"^\s*-\s*'?\"?([^'\"]+?)'?\"?\s*$")


def parse_filter_yaml(path):
    """返回 {scalars: {...}, log_names: [...], tokens: [...]}。"""
    scalars = {}
    lists = {}
    current = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = _SCALAR.match(line)
            if match:
                key, value = match.group(1), match.group(2).strip()
                # 去掉行尾注释（all_scenes.yaml 每行都有注释）
                value = re.sub(r"\s+#.*$", "", value).strip()
                if value in ("", "|", ">"):
                    lists[key] = []
                    current = key
                else:
                    scalars[key] = value.strip("'\"")
                    current = None
                continue
            if current is not None:
                match = _ITEM.match(line)
                if match:
                    lists[current].append(match.group(1))
    return {
        "scalars": {k: scalars.get(k) for k in SCALAR_KEYS if k in scalars},
        "log_names": lists.get("log_names", []),
        "tokens": lists.get("tokens", []),
    }


def resolve_reference(config_dir, split_name):
    """读 split 顶层 yaml，返回 (data_split, 它引用的 scene_filter 名)。"""
    top = config_dir / (split_name + ".yaml")
    if not top.is_file():
        return None, None
    text = top.read_text(encoding="utf-8")
    data_split = None
    match = re.search(r"^data_split:\s*(\S+)", text, re.M)
    if match:
        data_split = match.group(1).strip("'\"")
    match = re.search(r"scene_filter:\s*([A-Za-z0-9_]+)", text)
    return data_split, (match.group(1) if match else None)


# ---------------------------------------------------------------------------
# devkit 挂载
# ---------------------------------------------------------------------------

def mount_devkits(specs):
    """specs 是 [(别名, 根路径)]；返回 [(别名, config_dir)]，跳过不存在的。"""
    mounted = []
    for alias, root in specs:
        if not root:
            continue
        config_dir = Path(root) / FILTER_SUBPATH
        if config_dir.is_dir():
            mounted.append((alias, config_dir))
    return mounted


def collect_filters(mounted, names):
    """把各 devkit 的同名 filter 收集成 {name: {alias: parsed}}。"""
    table = {}
    for name in names:
        for alias, config_dir in mounted:
            path = config_dir / "scene_filter" / (name + ".yaml")
            if path.is_file():
                table.setdefault(name, {})[alias] = parse_filter_yaml(path)
    return table


def find_referrers(mounted, filter_name):
    """找出哪些顶层 split yaml 引用了这个 filter。"""
    referrers = []
    for alias, config_dir in mounted:
        for top in sorted(config_dir.glob("*.yaml")):
            data_split, referenced = resolve_reference(config_dir, top.stem)
            if referenced == filter_name:
                referrers.append({"devkit": alias, "split": top.stem,
                                  "data_split": data_split})
    return referrers


# ---------------------------------------------------------------------------
# metadata 校验（可选，需要真实数据根）
# ---------------------------------------------------------------------------

def verify_against_metadata(log_root, log_names):
    """目标 filter 的 log 在 navsim_logs/<split> 下有没有 metadata pickle。"""
    if not log_root.is_dir():
        return {"status": "MISSING_LOG_ROOT", "path": str(log_root)}
    stems = set()
    try:
        with os.scandir(log_root) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                for suffix in (".pkl.gz", ".pkl", ".pickle"):
                    if entry.name.endswith(suffix):
                        stems.add(entry.name[: -len(suffix)])
                        break
    except (OSError, PermissionError) as error:
        return {"status": "UNREADABLE", "error": str(error)}
    missing = sorted(set(log_names) - stems)
    return {
        "status": "OK",
        "path": str(log_root),
        "metadata_logs": len(stems),
        "filter_logs": len(log_names),
        "present": len(log_names) - len(missing),
        "missing": len(missing),
        "missing_sample": missing[:20],
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def report(args, target, table, referrers, verification):
    result = {"target_filter": args.filter, "devkits": {}, "referrers": referrers,
              "intersections": [], "verification": verification}

    print()
    print(paint("=" * 78, "bold"))
    print(paint(" scene_filter 解析: %s" % args.filter, "bold"))
    print(paint("=" * 78, "bold"))

    # ---- 1. 身份 ----
    print()
    print(paint(" 1. 这个 filter 是什么", "cyan"))
    for alias, parsed in target.items():
        logs, tokens = len(parsed["log_names"]), len(parsed["tokens"])
        result["devkits"][alias] = {"logs": logs, "tokens": tokens,
                                    "scalars": parsed["scalars"]}
        print("    [%s] %s logs / %s tokens" % (alias, format(logs, ","), format(tokens, ",")))
        for key, value in parsed["scalars"].items():
            print("         %-20s %s" % (key, value))

    # 同名 filter 跨 devkit 是否一致——RAP 的副本被改过没有
    if len(target) > 1:
        aliases = list(target)
        first = target[aliases[0]]
        same = all(set(target[a]["log_names"]) == set(first["log_names"])
                   and set(target[a]["tokens"]) == set(first["tokens"])
                   for a in aliases[1:])
        result["identical_across_devkits"] = same
        print()
        mark = paint("一致", "green") if same else paint("不一致 —— RAP 的副本与官方不同！", "red")
        print("    跨 devkit 内容比对: %s" % mark)

    # ---- 2. 谁在用它 ----
    print()
    print(paint(" 2. 谁引用了它", "cyan"))
    if not referrers:
        print("    %s 没有任何顶层 split yaml 引用它" % paint("(无)", "yellow"))
    for item in referrers:
        print("    [%s] %s.yaml  →  data_split: %s"
              % (item["devkit"], item["split"], item["data_split"]))

    # ---- 3. 与评测侧的交集 ----
    print()
    print(paint(" 3. 与评测/挑战侧 filter 的交集（污染核验）", "cyan"))
    print("    %s log 级和 token 级分别给：两种粒度结论可能不同" % paint("注意:", "yellow"))
    print()
    print("    %-34s %-18s %s" % ("filter", "log 交集", "token 交集"))
    print("    " + paint("-" * 70, "grey"))

    # 目标取并集：同名 filter 在多个 devkit 都有时，用并集做最保守的判断
    target_logs = set()
    target_tokens = set()
    for parsed in target.values():
        target_logs |= set(parsed["log_names"])
        target_tokens |= set(parsed["tokens"])

    for name in EVAL_FILTERS:
        if name not in table:
            print("    %-34s %s" % (name, paint("未找到定义（该 devkit 版本不含）", "yellow")))
            result["intersections"].append({"filter": name, "status": "NOT_FOUND"})
            continue
        for alias, parsed in table[name].items():
            logs = set(parsed["log_names"])
            tokens = set(parsed["tokens"])
            log_hit = len(target_logs & logs)
            token_hit = len(target_tokens & tokens)
            label = "%s [%s]" % (name, alias)
            log_text = "%d / %d" % (log_hit, len(logs))
            token_text = "%d / %d" % (token_hit, len(tokens)) if tokens else "(无 token 列表)"
            color = "red" if token_hit else ("yellow" if log_hit else "green")
            print("    %-34s %-18s %s" % (label, paint(log_text, color), paint(token_text, color)))
            result["intersections"].append({
                "filter": name, "devkit": alias,
                "eval_logs": len(logs), "eval_tokens": len(tokens),
                "log_intersection": log_hit, "token_intersection": token_hit,
            })

    # ---- 4. metadata 校验 ----
    if verification:
        print()
        print(paint(" 4. 对 raw metadata 的校验", "cyan"))
        if verification["status"] != "OK":
            print("    %s %s" % (paint(verification["status"], "red"),
                                 verification.get("path", verification.get("error", ""))))
        else:
            print("    %s" % verification["path"])
            print("    filter 的 %s 个 log 中，%s 个有 metadata，%s 个缺"
                  % (format(verification["filter_logs"], ","),
                     format(verification["present"], ","),
                     format(verification["missing"], ",")))
            for name in verification["missing_sample"]:
                print("      - %s" % name)
            if verification["missing"] > len(verification["missing_sample"]):
                print("      ... 另有 %d 个" % (verification["missing"] - len(verification["missing_sample"])))

    print()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", default="navall", help="要解析的 scene_filter 名，默认 navall")
    parser.add_argument("--devkit", action="append", metavar="别名=路径",
                        help="额外挂载 devkit 根，可重复；不给则用默认的 rap + official")
    parser.add_argument("--verify-metadata", metavar="SPLIT",
                        help="对 $OPENSCENE_DATA_ROOT/navsim_logs/<SPLIT> 校验 log 是否有 metadata")
    parser.add_argument("--data-root",
                        default=os.environ.get("OPENSCENE_DATA_ROOT",
                                               "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset"))
    parser.add_argument("--json", dest="json_out", metavar="PATH", help="另存结构化结果")
    args = parser.parse_args()

    specs = list(DEFAULT_DEVKITS)
    if args.devkit:
        specs = []
        for spec in args.devkit:
            alias, _, root = spec.partition("=")
            specs.append((alias or "devkit%d" % len(specs), root or alias))

    mounted = mount_devkits(specs)
    if not mounted:
        print("未挂载到任何 devkit；用 --devkit 别名=路径 指定", file=sys.stderr)
        return 2
    print()
    print("  已挂载 devkit:")
    for alias, config_dir in mounted:
        print("    [%s] %s" % (alias, config_dir))

    wanted = sorted(set(EVAL_FILTERS) | {args.filter})
    table = collect_filters(mounted, wanted)
    if args.filter not in table:
        print("\n找不到 scene_filter/%s.yaml" % args.filter, file=sys.stderr)
        return 2

    referrers = find_referrers(mounted, args.filter)

    verification = None
    if args.verify_metadata:
        logs = set()
        for parsed in table[args.filter].values():
            logs |= set(parsed["log_names"])
        verification = verify_against_metadata(
            Path(args.data_root) / "navsim_logs" / args.verify_metadata, sorted(logs))

    result = report(args, table[args.filter], table, referrers, verification)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("  结构化结果已写入 %s\n" % args.json_out)

    # 退出码：token 级出现任何交集 → 1（污染信号）
    contaminated = any(item.get("token_intersection") for item in result["intersections"])
    return 1 if contaminated else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
