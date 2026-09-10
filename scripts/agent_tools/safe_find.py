#!/usr/bin/env python3
"""受限目录结构探查。"""
import argparse
import json
import os
import sys
import time

MAX_LIMITS = {"max_depth": 5, "samples": 100, "max_dirs": 500}
ENTRY_LIMIT = 500
TIMEOUT_SEC = 10.0


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        fail(f"参数错误：{message}。")
        raise SystemExit(2)


def fail(message):
    print(json.dumps({"error": message, "next_action": "检查路径和参数后重试。"}, ensure_ascii=False))
    return 1


def main():
    parser = JsonParser(description="受限探查目录结构")
    parser.add_argument("path")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-dirs", type=int, default=100)
    args = parser.parse_args()
    for key, limit in MAX_LIMITS.items():
        value = getattr(args, key)
        if value < 0 or value > limit:
            return fail(f"{key} 超出允许范围上限 {limit}。")
    root = os.path.abspath(args.path)
    if not os.path.exists(root):
        return fail("路径不存在。")
    if not os.path.isdir(root):
        return fail("路径不是目录。")
    started = time.monotonic()
    queue = [(root, 0)]
    scanned_dirs = 0
    visible_files = 0
    visible_dirs = 0
    structure = []
    sample_files = []
    truncated = False
    timed_out = False
    while queue and scanned_dirs < args.max_dirs:
        if time.monotonic() - started >= TIMEOUT_SEC:
            timed_out = True
            break
        current, depth = queue.pop(0)
        scanned_dirs += 1
        try:
            entries = os.scandir(current)
        except OSError as exc:
            if current == root:
                return fail(f"无法读取目录：{exc.strerror or '权限不足'}。")
            continue
        count = 0
        try:
            for entry in entries:
                if count >= ENTRY_LIMIT:
                    truncated = True
                    break
                count += 1
                if time.monotonic() - started >= TIMEOUT_SEC:
                    timed_out = True
                    break
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError:
                    continue
                rel = os.path.relpath(entry.path, root)
                if is_dir:
                    visible_dirs += 1
                    structure.append({"path": rel, "type": "dir", "depth": depth + 1})
                    if depth < args.max_depth:
                        queue.append((entry.path, depth + 1))
                elif is_file:
                    visible_files += 1
                    structure.append({"path": rel, "type": "file", "depth": depth + 1})
                    if len(sample_files) < args.samples:
                        sample_files.append(rel)
            if timed_out:
                break
        finally:
            entries.close()
    if queue or scanned_dirs >= args.max_dirs:
        truncated = True
    result = {"root": root, "max_depth": args.max_depth, "scanned_dirs": scanned_dirs,
              "visible_files": visible_files, "visible_dirs": visible_dirs,
              "structure": structure, "sample_files": sample_files,
              "truncated": truncated, "timed_out": timed_out,
              "next_action": "如需更多内容，增大 --max-depth、--samples 或 --max-dirs（遵守上限）。"}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit(fail(f"执行失败：{exc}。"))
