#!/usr/bin/env python3
"""从日志末尾生成确定性摘要。"""
import argparse
import json
import os
import sys

MAX_READ_BYTES = 65536
MAX_OUTPUT_LINES = 50
KEYWORDS = {
    "errors": ("error", "exception", "traceback", "failed", "fatal", "oom", "out of memory", "killed"),
    "warnings": ("warning", "warn"),
    "metrics": ("loss", "epoch", "step", "metric", "accuracy", "dms", "epdms"),
}


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        fail(f"参数错误：{message}。")
        raise SystemExit(2)


def fail(message):
    print(json.dumps({"error": message, "next_action": "检查日志路径和参数后重试。"}, ensure_ascii=False))
    return 1


def main():
    parser = JsonParser(description="读取日志末尾并分类摘要")
    parser.add_argument("file")
    parser.add_argument("--tail-bytes", type=int, default=MAX_READ_BYTES)
    parser.add_argument("--max-lines", type=int, default=MAX_OUTPUT_LINES)
    args = parser.parse_args()
    if args.tail_bytes < 1 or args.max_lines < 1 or args.max_lines > MAX_OUTPUT_LINES:
        return fail(f"参数范围无效，最多输出 {MAX_OUTPUT_LINES} 行。")
    if not os.path.isfile(args.file):
        return fail("日志文件不存在或不是普通文件。")
    try:
        size = os.path.getsize(args.file)
        read_bytes = min(args.tail_bytes, MAX_READ_BYTES)
        with open(args.file, "rb") as handle:
            handle.seek(max(0, size - read_bytes))
            raw = handle.read(read_bytes)
    except OSError as exc:
        return fail(f"无法读取日志：{exc.strerror or '权限不足'}。")
    text = raw.decode("utf-8", errors="replace").replace("\r", "\n")
    lines = text.splitlines()
    truncated = size > read_bytes
    if len(lines) > args.max_lines:
        lines = lines[-args.max_lines:]
        truncated = True
    groups = {key: [] for key in KEYWORDS}
    for line in lines:
        lower = line.lower()
        short = line[:300]
        for key, words in KEYWORDS.items():
            if any(word in lower for word in words) and len(groups[key]) < 15:
                groups[key].append(short)
    result = {"file": args.file, "size_bytes": size, "read_bytes": len(raw),
              "errors": groups["errors"], "warnings": groups["warnings"],
              "metrics": groups["metrics"], "tail": [line[:300] for line in lines[-20:]],
              "truncated": truncated,
              "next_action": "如需扩大范围，增加 --tail-bytes（最多 65536）或调整日志文件。"}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit(fail(f"执行失败：{exc}。"))
