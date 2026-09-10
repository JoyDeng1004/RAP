#!/usr/bin/env python3
"""按范围安全读取文本文件。"""
import argparse
import json
import os
import re
import sys

MAX_LINES = 400
MAX_BYTES = 64 * 1024


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        fail(f"参数错误：{message}。")
        raise SystemExit(2)


def fail(message):
    print(json.dumps({"error": message, "next_action": "检查文件和参数后重试。"}, ensure_ascii=False))
    return 1


def main():
    parser = JsonParser(description="按行范围读取源码")
    parser.add_argument("file")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--lines", type=int, default=200)
    parser.add_argument("--grep")
    args = parser.parse_args()
    if args.start < 1 or args.lines < 1 or args.lines > MAX_LINES:
        return fail(f"行号必须从 1 开始，单次最多 {MAX_LINES} 行。")
    if not os.path.isfile(args.file):
        return fail("文件不存在或不是普通文件。")
    try:
        with open(args.file, "r", encoding="utf-8", errors="replace") as handle:
            all_lines = handle.readlines()
    except OSError as exc:
        return fail(f"无法读取文件：{exc.strerror or '权限不足'}。")
    total = len(all_lines)
    truncated = False
    if args.grep is not None:
        try:
            pattern = re.compile(args.grep)
        except re.error as exc:
            return fail(f"正则表达式无效：{exc}。")
        segments = []
        last_end = -1
        for index, line in enumerate(all_lines):
            if pattern.search(line):
                begin = max(0, index - 3)
                end = min(total, index + 4)
                if begin <= last_end:
                    segments[-1][1] = max(segments[-1][1], end)
                else:
                    segments.append([begin, end])
                last_end = end
                if len(segments) >= 50:
                    truncated = any(pattern.search(item) for item in all_lines[index + 1:])
                    break
        selected = []
        for begin, end in segments:
            selected.extend(all_lines[begin:end])
        start = (segments[0][0] + 1) if segments else 1
        finish = segments[-1][1] if segments else 0
    else:
        start = args.start
        finish = min(total, args.start + args.lines - 1)
        selected = all_lines[start - 1:finish]
        truncated = finish < args.start + args.lines - 1
    content = "".join(selected)
    if len(content.encode("utf-8")) > MAX_BYTES:
        raw = content.encode("utf-8")[:MAX_BYTES]
        content = raw.decode("utf-8", errors="ignore")
        truncated = True
    result = {"file": args.file, "total_lines": total, "start": start, "end": finish,
              "content": content, "truncated": truncated,
              "next_action": "如需后续内容，调整 --start；搜索请使用 --grep。"}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit(fail(f"执行失败：{exc}。"))
