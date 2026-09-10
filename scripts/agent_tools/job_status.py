#!/usr/bin/env python3
"""查询本地或 PBS 任务状态，并清理旧任务。"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        fail(f"参数错误：{message}。")
        raise SystemExit(2)


def emit(data, code=0):
    print(json.dumps(data, ensure_ascii=False))
    return code


def fail(message):
    return emit({"error": message, "next_action": "检查任务编号或参数后重试。"}, 1)


def starttime(pid):
    try:
        return open(f"/proc/{pid}/stat", encoding="utf-8").read().split()[21]
    except (OSError, IndexError):
        return None


def local_status(identifier):
    base = os.environ.get("AGENT_JOBS_DIR", ".agent_jobs")
    job_dir = identifier if os.path.isdir(identifier) else os.path.join(base, identifier)
    meta_path = os.path.join(job_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return fail("找不到任务目录或 meta.json。")
    try:
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return fail("任务元数据损坏或不可读。")
    exit_path = os.path.join(job_dir, "exit_code")
    exit_code = None
    if os.path.isfile(exit_path):
        try:
            with open(exit_path, encoding="utf-8") as handle:
                exit_code = int(handle.read().strip())
        except (OSError, ValueError):
            exit_code = None
    pid = meta.get("pid")
    if exit_code is not None:
        state = "SUCCEEDED" if exit_code == 0 else "FAILED"
    elif pid and starttime(pid) is not None and str(starttime(pid)) == str(meta.get("proc_starttime")):
        state = "RUNNING"
    else:
        state = "UNKNOWN"
    result = {"job_id": os.path.basename(os.path.normpath(job_dir)), "state": state,
              "job_dir": job_dir, "pid": pid, "next_action": "完成后可用 log_summary.py 查看日志摘要。"}
    if exit_code is not None:
        result["exit_code"] = exit_code
    stdout_path = os.path.join(job_dir, "stdout.log")
    try:
        if os.path.getsize(stdout_path) > 100 * 1024 * 1024:
            result["warning"] = "stdout.log 超过 100 MiB，请及时清理。"
    except OSError:
        pass
    return emit(result)


def pbs_status(jobid):
    raw = subprocess.run(["qstat", "-x", "-f", "-F", "json", jobid], capture_output=True, text=True)
    info = {}
    if raw.returncode == 0:
        try:
            payload = json.loads(raw.stdout)
            jobs = payload.get("Jobs", payload.get("jobs", {}))
            record = next(iter(jobs.values())) if isinstance(jobs, dict) and jobs else {}
            info = {"pbs_state": record.get("job_state"), "exit_code": record.get("Exit_status"),
                    "walltime": record.get("resources_used.walltime")}
        except ValueError:
            pass
    if not info.get("pbs_state"):
        text = raw.stdout + raw.stderr
        for line in text.splitlines():
            if "job_state" in line or "job_state =" in line:
                info["pbs_state"] = line.split("=")[-1].strip()
            elif "Exit_status" in line:
                try:
                    info["exit_code"] = int(line.split("=")[-1].strip())
                except ValueError:
                    pass
            elif "resources_used.walltime" in line:
                info["walltime"] = line.split("=")[-1].strip()
    state_code = info.get("pbs_state")
    if state_code in {"R", "Q", "H"}:
        state = "RUNNING"
    elif state_code == "F":
        state = "SUCCEEDED" if str(info.get("exit_code", 1)) == "0" else "FAILED"
    else:
        state = "UNKNOWN"
    result = {"job_id": jobid, "state": state, "pbs_state": state_code,
              "next_action": "PBS 任务运行中请稍后查询；结束后可查看日志摘要。"}
    if info.get("exit_code") is not None:
        result["exit_code"] = info["exit_code"]
    if info.get("walltime") is not None:
        result["walltime"] = info["walltime"]
    if raw.returncode != 0 and not state_code:
        result["error"] = "qstat 查询失败。"
        return emit(result, 1)
    return emit(result)


def prune(args):
    base = os.environ.get("AGENT_JOBS_DIR", ".agent_jobs")
    if not os.path.isdir(base):
        return emit({"removed": [], "next_action": "暂无任务目录。"})
    candidates = []
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "meta.json")):
            candidates.append((os.path.getmtime(path), path))
    candidates.sort(reverse=True)
    keep = max(0, args.keep_last)
    cutoff = time.time() - args.older_than * 86400 if args.older_than is not None else None
    removed = []
    for index, (mtime, path) in enumerate(candidates):
        if (index >= keep) or (cutoff is not None and mtime < cutoff):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return emit({"removed": removed, "next_action": "保留的任务可用 job_status.py 查询。"})


def main():
    parser = JsonParser(description="查询任务状态或清理任务")
    parser.add_argument("identifier", nargs="?")
    parser.add_argument("--pbs")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--keep-last", type=int, default=5)
    parser.add_argument("--older-than", type=float)
    args = parser.parse_args()
    if args.prune:
        if args.keep_last < 0 or (args.older_than is not None and args.older_than < 0):
            return fail("清理参数不能为负数。")
        return prune(args)
    if args.pbs:
        return pbs_status(args.pbs)
    if not args.identifier:
        return fail("请提供任务编号、--pbs PBS_JOB_ID 或 --prune。")
    return local_status(args.identifier)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit(fail(f"执行失败：{exc}。"))
