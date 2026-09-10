#!/usr/bin/env python3
"""限量递归代码搜索。"""
import argparse
import json
import os
import re
import sys

SKIP_DIRS = {".git", "__pycache__", ".agent_jobs"}
SKIP_SUFFIXES = {".pth", ".ckpt", ".npy", ".pkl", ".db", ".sqlite"}


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        fail(f"参数错误：{message}。")
        raise SystemExit(2)


def fail(message):
    print(json.dumps({"error": message, "next_action": "检查正则、路径和参数后重试。"}, ensure_ascii=False))
    return 1


def main():
    parser = JsonParser(description="限量搜索代码")
    parser.add_argument("pattern")
    parser.add_argument("path")
    parser.add_argument("--max-matches", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()
    if args.max_matches < 1 or args.max_matches > 1000 or args.max_depth < 0 or args.max_depth > 8:
        return fail("参数超出安全范围。")
    try:
        matcher = re.compile(args.pattern)
    except re.error as exc:
        return fail(f"正则表达式无效：{exc}。")
    target = os.path.abspath(args.path)
    if not os.path.exists(target):
        return fail("路径不存在。")
    matches = []
    truncated = False

    def inspect_file(path):
        nonlocal truncated
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line_no, line in enumerate(handle, 1):
                    if matcher.search(line):
                        matches.append({"file": path, "line_no": line_no,
                                        "text": line.rstrip("\r\n")[:200]})
                        if len(matches) >= args.max_matches:
                            truncated = True
                            return True
        except (OSError, UnicodeError):
            return False
        return False

    if os.path.isfile(target):
        inspect_file(target)
    else:
        queue = [(target, 0)]
        while queue and not truncated:
            current, depth = queue.pop(0)
            try:
                entries = list(os.scandir(current))
            except OSError:
                continue
            for entry in entries:
                if entry.name in SKIP_DIRS:
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if depth < args.max_depth:
                            queue.append((entry.path, depth + 1))
                    elif entry.is_file(follow_symlinks=False) and os.path.splitext(entry.name)[1].lower() not in SKIP_SUFFIXES:
                        if inspect_file(entry.path):
                            break
                except OSError:
                    continue
    result = {"pattern": args.pattern, "matches": matches, "match_count": len(matches),
              "truncated": truncated,
              "next_action": "如需更多命中，增大 --max-matches 或缩小 PATH 范围。"}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit(fail(f"执行失败：{exc}。"))
