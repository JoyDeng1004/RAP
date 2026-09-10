#!/usr/bin/env python3
"""受限执行命令并把完整输出落盘。"""
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

DEFAULT_MAX_LOG = 256 * 1024 * 1024


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        fail(f"参数错误：{message}。")
        raise SystemExit(2)


def emit(data, code=0):
    print(json.dumps(data, ensure_ascii=False))
    return code


def fail(message):
    return emit({"error": message, "next_action": "修改命令或参数后重试。"}, 1)


def safe_command(parts):
    text = shlex.join(parts)
    if re.search(r"(?:^|\s)rm\s+-[^\s]*r[^\s]*f|(?:^|\s)rm\s+-[^\s]*f[^\s]*r", text):
        return "命令包含禁止的 rm -rf。"
    if re.search(r"\b(?:mv|chmod|chown)\b[^\n]*(?:/gs/bs|/gs/fs)", text) and re.search(r"\b(?:chmod|chown)\b[^\n]*-R|\bmv\b", text):
        return "禁止对 /gs/bs 或 /gs/fs 执行危险移动或递归权限操作。"
    return None


def proc_starttime(pid):
    try:
        fields = open(f"/proc/{pid}/stat", encoding="utf-8").read().split()
        return fields[21]
    except (OSError, IndexError):
        return None


def make_job(args, command):
    base = os.environ.get("AGENT_JOBS_DIR", ".agent_jobs")
    os.makedirs(base, exist_ok=True)
    clean_name = re.sub(r"[^A-Za-z0-9_-]", "", args.job_name or "job")[:40] or "job"
    job_id = f"{clean_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    job_dir = os.path.join(base, job_id)
    os.makedirs(job_dir)
    exit_path = shlex.quote(os.path.abspath(os.path.join(job_dir, "exit_code")))
    script = shlex.join(command) + f"\nstatus=$?\necho $status > {exit_path}\nexit $status\n"
    with open(os.path.join(job_dir, "cmd.sh"), "w", encoding="utf-8") as handle:
        handle.write("#!/bin/sh\nset +e\n" + script)
    os.chmod(os.path.join(job_dir, "cmd.sh"), 0o700)
    start = time.time()
    meta = {"command": command, "pid": None, "pgid": None, "start_time": start,
            "proc_starttime": None, "max_log_bytes": args.max_log_bytes}
    with open(os.path.join(job_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)
    return job_id, job_dir, meta


def update_meta(job_dir, meta):
    with open(os.path.join(job_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)


def foreground(args, command):
    job_id, job_dir, meta = make_job(args, command)
    stdout_path, stderr_path = os.path.join(job_dir, "stdout.log"), os.path.join(job_dir, "stderr.log")
    process = subprocess.Popen(["sh", os.path.join(job_dir, "cmd.sh")], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    meta.update(pid=process.pid, pgid=os.getpgid(process.pid), proc_starttime=proc_starttime(process.pid))
    update_meta(job_dir, meta)
    counters = {"stdout": 0, "stderr": 0}
    trunc = {"stdout": False, "stderr": False}

    def drain(stream, path, key):
        with open(path, "wb") as handle:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                counters[key] += len(chunk)
                if handle.tell() < args.max_log_bytes:
                    allowed = args.max_log_bytes - handle.tell()
                    handle.write(chunk[:allowed])
                    if len(chunk) > allowed:
                        trunc[key] = True
                else:
                    trunc[key] = True

    threads = [threading.Thread(target=drain, args=(process.stdout, stdout_path, "stdout")),
               threading.Thread(target=drain, args=(process.stderr, stderr_path, "stderr"))]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    for thread in threads:
        thread.join()
    exit_code = process.returncode
    with open(os.path.join(job_dir, "exit_code"), "w", encoding="utf-8") as handle:
        handle.write(str(exit_code))
    elapsed = round(time.time() - meta["start_time"], 3)
    state = "SUCCEEDED" if exit_code == 0 and not timed_out else "FAILED"
    result = {"job_id": job_id, "state": state, "exit_code": exit_code, "elapsed_sec": elapsed,
              "stdout_bytes": counters["stdout"], "stderr_bytes": counters["stderr"],
              "job_dir": job_dir, "next_action": "用 job_status.py 查询状态；用 log_summary.py 查看日志摘要。"}
    if trunc["stdout"] or trunc["stderr"]:
        result["log_truncated"] = True
    if state == "FAILED":
        try:
            with open(stderr_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 8192))
                result["stderr_tail"] = handle.read().decode("utf-8", errors="replace")
        except OSError:
            result["stderr_tail"] = ""
    return emit(result, 0 if state == "SUCCEEDED" else 1)


def background(args, command):
    if not os.environ.get("PBS_JOBID"):
        return fail("后台模式只允许在计算节点（存在 PBS_JOBID）运行，请改用 qsub。")
    job_id, job_dir, meta = make_job(args, command)
    out = open(os.path.join(job_dir, "stdout.log"), "ab")
    err = open(os.path.join(job_dir, "stderr.log"), "ab")
    process = subprocess.Popen(["sh", os.path.join(job_dir, "cmd.sh")], stdout=out, stderr=err,
                               start_new_session=True)
    out.close()
    err.close()
    meta.update(pid=process.pid, pgid=os.getpgid(process.pid), proc_starttime=proc_starttime(process.pid))
    update_meta(job_dir, meta)
    return emit({"job_id": job_id, "state": "RUNNING", "pid": process.pid, "job_dir": job_dir,
                 "next_action": "稍后用 job_status.py 查询，不要反复读取完整日志。"})


def main():
    parser = JsonParser(description="安全执行高输出命令")
    parser.add_argument("--job-name", default="job")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--max-log-bytes", type=int, default=DEFAULT_MAX_LOG)
    parser.add_argument("--", dest="separator", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        return fail("缺少要执行的命令，请使用 -- COMMAND ...。")
    if args.timeout <= 0 or args.max_log_bytes <= 0:
        return fail("timeout 和 max-log-bytes 必须为正数。")
    blocked = safe_command(command)
    if blocked:
        return fail(blocked)
    return background(args, command) if args.background else foreground(args, command)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit(fail(f"执行失败：{exc}。"))
