#!/usr/bin/env python3
"""按 log 精准补齐 NAVSIM / OpenScene trainval sensor blobs（camera / lidar 分片）。

依据：docs/audits/datasets/download_gap_list.md（snapshot 20260910T092936+0900）。
该审计里 NAVSIM 侧真正属于"下载缺口"的只有两项：

  * P0：navtrain 4-frame history 缺 46 个 camera JPEG
        —— 已有专用脚本 scripts/data_audit/fill_navtrain_camera_gaps.py，
           本脚本不重复实现，plan 阶段会提示改用它。
  * P3：full OpenScene trainval sensor blobs 只覆盖 20.85% 引用
        —— 缺 4,632,115 个 camera 引用，官方全量 camera+lidar 约 1.93 TiB。
           本脚本处理的就是这一项，并且只取"你点名的 log 所在的分片"。

其余 NAVSIM 条目（dataset_norm / dataset_aug 污染、private_test_hard 路径未接合、
navtest / mini / two-stage / maps 已 PRESENT）都不是下载问题，本脚本一律不碰。

设计原则
--------
1. 不伪造任何 size / checksum：分片的字节数与 SHA-256 全部在运行时从官方
   Hugging Face 目录 API 的 LFS 元数据取回（已核对与仓库内既有 manifest 一致）。
2. 只写 staging，不碰 live 数据根；install 阶段只新建缺失文件，永不覆盖。
3. 大文件健壮性：curl 断点续传 + 多层重试 + 超时 + 逐分片 SHA-256 校验 +
   磁盘余量预检 + `--part i/n` 分卷切片（可拆成 UGE array job 并行跑）。
4. 默认 dry-run（plan），下载/解包/安装必须显式指定阶段。

阶段
----
  plan      只读：算出需要哪些分片、多少字节、本地已有多少，不发生任何写入
  download  下载所需分片到 <staging>/archives，断点续传 + 校验
  extract   从已校验的分片里只提取目标 log，落到 <staging>/sensor_blobs
  install   把 staging 的文件补进 live sensor root（只补缺失，不覆盖）

典型用法
--------
  # 1) 先看计划（不联网下载，只读 HF 索引元数据）
  python scripts/data_audit/download_navsim_shards.py plan --logs-file missing_logs.txt

  # 2) 下载（走 qsub；可用 --part 拆到多个 array task）
  qsub -g tga-RLA scripts/data_audit/download_navsim_shards.qsub \
      download --logs-file /abs/path/missing_logs.txt

  # 3) 解包 + 安装
  qsub -g tga-RLA scripts/data_audit/download_navsim_shards.qsub \
      extract --logs-file /abs/path/missing_logs.txt
  qsub -g tga-RLA scripts/data_audit/download_navsim_shards.qsub \
      install  --logs-file /abs/path/missing_logs.txt
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

# ---------------------------------------------------------------------------
# 官方常量：来源 https://huggingface.co/datasets/OpenDriveLab/OpenScene
# ---------------------------------------------------------------------------

HF_REPO = "OpenDriveLab/OpenScene"
# 目录列举 API：返回每个 .tgz 的 size 与 lfs.oid（= 文件 SHA-256）
HF_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}?limit=1000"
# 单文件直链
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
# shard -> log 名列表 的官方索引（trainval 共 200 个分片，编号 0..199）
SHARD_INDEX = "openscene-v1.1/openscene_sensor_trainval_0-199.json"
# 每种模态的分片目录；两种模态共用上面那个 shard->log 索引
MODALITY_DIR = {
    "camera": "openscene-v1.1/openscene_sensor_trainval_camera",
    "lidar": "openscene-v1.1/openscene_sensor_trainval_lidar",
}
# 该脚本不负责的 P0 缺口，plan 阶段用来提示
P0_SCRIPT = "scripts/data_audit/fill_navtrain_camera_gaps.py"

# OpenScene 的 8 路环视 camera 通道；用于判断某个 log 目录是否"看起来已展开"
CAMERAS = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2")

# 网络参数：HF 大文件在共享集群上容易断，所以既给 curl 自身重试，也做外层重投
CONNECT_TIMEOUT = 30          # 建连超时（秒）
MAX_TIME = 24 * 3600          # 单次 curl 最长存活（秒），大分片留足余量
CURL_RETRY = 5                # curl 内部重试次数
OUTER_ATTEMPTS = 6            # 外层重投次数；每次都带 --continue-at - 续传
BACKOFF_BASE = 15             # 外层重投退避基数（秒），指数增长


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def log(message, **fields):
    """统一的单行 JSON 日志，方便 qsub 输出被 grep / 解析。"""
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "msg": message}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def nonempty(path):
    """存在且非零字节。零字节文件一律当作"没有"，但不会被本脚本悄悄删掉。"""
    return path.is_file() and path.stat().st_size > 0


def checked_path(root, relative):
    """把相对路径接到 root 下，并确认结果没有通过 `..`/绝对路径逃出 root。"""
    target = root / relative
    target.resolve().relative_to(root.resolve())
    return target


def sha256_of(path, chunk=8 * 1024 * 1024):
    """流式算 SHA-256；分片有 4~9 GB，不能一次读进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def human(num_bytes):
    """给人看的体积字符串（GiB / TiB）。"""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return "%.2f %s" % (value, unit)
        value /= 1024


def http_json(url, cache_file=None, timeout=120):
    """取一段 JSON；可选落盘缓存，避免每个 array task 都去打官方 API。"""
    if cache_file is not None and nonempty(cache_file):
        return json.loads(cache_file.read_text())
    request = urllib.request.Request(url, headers={"User-Agent": "RAP-dataset-audit/1.0"})
    last_error = None
    for attempt in range(1, 4):                       # 元数据请求也做少量重试
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            log("metadata_retry", url=url, attempt=attempt, error=str(error))
            time.sleep(BACKOFF_BASE * attempt)
    else:
        raise RuntimeError("无法取回元数据 %s：%s" % (url, last_error))
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 官方元数据：shard -> log 列表、shard -> (url, size, sha256)
# ---------------------------------------------------------------------------

def load_shard_index(cache_dir):
    """返回 (log -> shard_id, 全部 log 的集合)。"""
    index = http_json(HF_FILE.format(repo=HF_REPO, path=SHARD_INDEX),
                      cache_dir / "openscene_sensor_trainval_0-199.json")
    log_to_shard = {}
    for shard_name, logs in index.items():
        # key 形如 "openscene_sensor_trainval_137"
        shard_id = int(shard_name.rsplit("_", 1)[1])
        for name in logs:
            log_to_shard[name] = shard_id
    return log_to_shard, set(log_to_shard)


def load_shard_files(modality, cache_dir):
    """返回 shard_id -> {"url":..., "size":..., "sha256":...}。

    size / sha256 直接来自官方 LFS 元数据，本脚本不内置任何硬编码校验值。
    """
    directory = MODALITY_DIR[modality]
    entries = http_json(HF_TREE.format(repo=HF_REPO, path=directory),
                        cache_dir / ("tree_%s.json" % modality))
    shards = {}
    for entry in entries:
        name = entry["path"].rsplit("/", 1)[-1]
        if not name.endswith(".tgz"):
            continue
        shard_id = int(name[: -len(".tgz")].rsplit("_", 1)[1])
        lfs = entry.get("lfs") or {}
        checksum = lfs.get("oid")
        size = lfs.get("size", entry.get("size"))
        if not checksum or len(checksum) != 64 or not size:
            raise RuntimeError("官方元数据缺少 size/SHA-256：" + name)
        shards[shard_id] = {
            "name": name,
            "url": HF_FILE.format(repo=HF_REPO, path=entry["path"]),
            "size": int(size),
            "sha256": checksum,
        }
    if not shards:
        raise RuntimeError("未能列举分片目录：" + directory)
    return shards


# ---------------------------------------------------------------------------
# 目标 log 的确定
# ---------------------------------------------------------------------------

def resolve_target_logs(args, known_logs, sensor_root):
    """决定要补哪些 log。三种来源，互斥：

    --logs-file   显式清单（推荐：直接来自审计产物 navtrain_missing_refs.jsonl）
    --logs        命令行逗号分隔
    --scan-missing 启发式：官方索引里有、但本地 sensor_root 下没有对应目录
                   （目录级判断，不等于引用级完整性，仅用于 P3 全量补齐的粗筛）
    """
    if args.logs_file:
        text = Path(args.logs_file).read_text()
        names = [line.strip() for line in text.splitlines()]
        # 同时兼容 jsonl：每行是对象时，取其中的 log 字段
        parsed = []
        for line in names:
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                record = json.loads(line)
                value = record.get("log") or record.get("log_name") or record.get("path")
                if value:
                    # path 形如 "<log>/CAM_F0/xxxx.jpg"
                    parsed.append(PurePosixPath(value).parts[0])
            else:
                parsed.append(PurePosixPath(line).parts[0])
        targets = sorted(set(parsed))
    elif args.logs:
        targets = sorted({name.strip() for name in args.logs.split(",") if name.strip()})
    elif args.scan_missing:
        targets = sorted(name for name in known_logs
                         if not _log_looks_present(sensor_root / name))
    else:
        raise ValueError("必须给出 --logs-file / --logs / --scan-missing 之一")

    unknown = [name for name in targets if name not in known_logs]
    if unknown:
        raise ValueError("以下 log 不在官方 trainval 索引里（%d 个），先核对来源：%s"
                         % (len(unknown), ", ".join(unknown[:5])))
    if not targets:
        raise ValueError("目标 log 集合为空：没有需要下载的分片")
    return targets


def _log_looks_present(log_dir):
    """目录级启发式：8 路 camera 子目录都在且各自非空，才算"已展开"。

    这只排除"整段 log 完全没下"的情况，**不能**替代引用级完整性检查
    （引用级结论以 docs/audits/datasets/ 下的审计产物为准）。
    """
    if not log_dir.is_dir():
        return False
    for camera in CAMERAS:
        channel = log_dir / camera
        if not channel.is_dir():
            return False
        if not any(channel.iterdir()):
            return False
    return True


def slice_part(items, spec):
    """把有序列表按 "i/n" 切片，配合 UGE array job 做分卷并行下载。

    i 从 1 开始；采用步进切片（items[i-1::n]），让大小分片自然错开。
    """
    if not spec:
        return items
    index_text, total_text = spec.split("/", 1)
    index, total = int(index_text), int(total_text)
    if not (1 <= index <= total):
        raise ValueError("--part 必须形如 i/n 且 1<=i<=n")
    return items[index - 1::total]


# ---------------------------------------------------------------------------
# 下载：断点续传 + 重试 + 校验
# ---------------------------------------------------------------------------

def download_shard(shard, archive_dir):
    """下载单个分片到 archive_dir，返回已校验的文件路径。

    健壮性：
      * `.part` 临时文件 + `--continue-at -`：中断后重跑只补剩余字节
      * curl 自身 --retry，外层再做指数退避重投
      * 完成后必须 size + SHA-256 双校验通过，才改名为正式文件
      * 已存在的正式文件先校验，通过就直接跳过（幂等，可反复重跑）
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    final = archive_dir / shard["name"]
    partial = final.with_suffix(final.suffix + ".part")

    if final.exists():
        if final.stat().st_size == shard["size"] and sha256_of(final) == shard["sha256"]:
            log("shard_cached", shard=shard["name"], bytes=shard["size"])
            return final
        raise RuntimeError("已存在的分片校验失败，请人工确认后删除：" + str(final))

    if not shard["url"].startswith("https://"):
        raise RuntimeError("分片 URL 必须是 HTTPS：" + shard["url"])

    for attempt in range(1, OUTER_ATTEMPTS + 1):
        have = partial.stat().st_size if partial.exists() else 0
        log("shard_download", shard=shard["name"], attempt=attempt,
            have=have, total=shard["size"])
        result = subprocess.run([
            "curl",
            "--fail", "--location", "--show-error",
            "--proto", "=https", "--proto-redir", "=https",   # 拒绝降级到明文
            "--continue-at", "-",                             # 断点续传
            "--retry", str(CURL_RETRY),
            "--retry-delay", "10",
            "--retry-all-errors",
            "--connect-timeout", str(CONNECT_TIMEOUT),
            "--max-time", str(MAX_TIME),
            "--speed-limit", "10240", "--speed-time", "300",  # 连续 5 分钟 <10KB/s 判死
            "--output", str(partial),
            shard["url"],
        ], check=False)
        if result.returncode == 0 and partial.exists() and partial.stat().st_size == shard["size"]:
            break
        if attempt == OUTER_ATTEMPTS:
            raise RuntimeError("分片下载重试耗尽：%s（curl exit %d）"
                               % (shard["name"], result.returncode))
        delay = BACKOFF_BASE * (2 ** (attempt - 1))
        log("shard_retry", shard=shard["name"], sleep=delay, curl_exit=result.returncode)
        time.sleep(delay)

    log("shard_verify", shard=shard["name"])
    if partial.stat().st_size != shard["size"]:
        raise RuntimeError("分片大小不符：" + shard["name"])
    if sha256_of(partial) != shard["sha256"]:
        # 不自动删除：留给人工判断是源端更新还是传输损坏
        raise RuntimeError("分片 SHA-256 不符，保留 .part 供排查：" + str(partial))
    partial.rename(final)
    log("shard_ok", shard=shard["name"], bytes=shard["size"])
    return final


def ensure_space(directory, needed_bytes, margin=1.10):
    """下载前预检磁盘余量（默认留 10% 余量），免得跑到一半把配额打满。"""
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    required = int(needed_bytes * margin)
    log("space_check", path=str(directory), free=human(free), required=human(required))
    if free < required:
        raise RuntimeError("磁盘余量不足：需要约 %s，可用 %s" % (human(required), human(free)))


# ---------------------------------------------------------------------------
# 解包：只取目标 log
# ---------------------------------------------------------------------------

def extract_logs(archive, wanted_logs, staging_blobs):
    """流式扫描 tgz，只提取属于 wanted_logs 的成员，其余整包跳过。

    用 "r|gz" 顺序流模式：分片有几 GB，不做随机 seek，内存占用恒定。
    每个文件先写同目录临时文件再 link，避免半截文件被当成有效数据。
    """
    wanted = set(wanted_logs)
    extracted = 0
    skipped = 0
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if not member.isfile() or member.size <= 0:
                continue
            parts = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in parts:
                skipped += 1                       # 防 tar 路径穿越
                continue
            # 归档内前缀层数不固定，按 log 名在路径中的位置来定位相对路径
            position = next((i for i, part in enumerate(parts) if part in wanted), None)
            if position is None or position + 1 >= len(parts):
                continue
            relative = "/".join(parts[position:])
            target = checked_path(staging_blobs, relative)
            if nonempty(target):
                continue                            # 幂等：已提取过就跳过
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                continue
            with source, tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=".navsim-dl-", delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    shutil.copyfileobj(source, handle, length=4 * 1024 * 1024)
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.chmod(temporary, 0o644)
                    os.replace(temporary, target)
                    extracted += 1
                finally:
                    temporary.unlink(missing_ok=True)
    log("extract_done", archive=archive.name, files=extracted, unsafe_members=skipped)
    return extracted


# ---------------------------------------------------------------------------
# 安装：staging -> live sensor root（只补缺失，绝不覆盖）
# ---------------------------------------------------------------------------

def install_logs(staging_blobs, sensor_root, wanted_logs):
    installed = 0
    existing = 0
    for name in wanted_logs:
        source_root = staging_blobs / name
        if not source_root.is_dir():
            continue
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(staging_blobs).as_posix()
            target = checked_path(sensor_root, relative)
            if nonempty(target):
                existing += 1
                continue
            if os.path.lexists(target):
                raise RuntimeError("目标位置存在无效文件/断链，拒绝替换：" + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            # 同文件系统优先硬链接（零拷贝）；跨盘则退回复制
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            installed += 1
    log("install_done", installed=installed, already_present=existing)
    return installed


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(args):
    sensor_root = Path(args.sensor_root).resolve()
    staging = Path(args.staging).resolve()
    # staging 与 live 根必须完全分离，防止半成品污染现有数据
    if sensor_root == staging or sensor_root in staging.parents or staging in sensor_root.parents:
        raise ValueError("--staging 必须与 --sensor-root 完全分离且互不嵌套")

    cache_dir = staging / "index"
    archive_dir = staging / "archives"
    staging_blobs = staging / "sensor_blobs"

    log_to_shard, known_logs = load_shard_index(cache_dir)
    targets = resolve_target_logs(args, known_logs, sensor_root)

    # 目标 log -> 需要的分片编号
    needed_shards = sorted({log_to_shard[name] for name in targets})
    needed_shards = slice_part(needed_shards, args.part)

    shard_files = {}
    total_bytes = 0
    for modality in args.modalities:
        shard_files[modality] = load_shard_files(modality, cache_dir)
        total_bytes += sum(shard_files[modality][sid]["size"] for sid in needed_shards)

    # 已经下好的分片不重复计入"还要传多少"
    pending_bytes = 0
    for modality in args.modalities:
        for shard_id in needed_shards:
            shard = shard_files[modality][shard_id]
            if not (archive_dir / shard["name"]).exists():
                pending_bytes += shard["size"]

    log("plan",
        mode=args.mode,
        target_logs=len(targets),
        shards=needed_shards if len(needed_shards) <= 40 else "%d shards" % len(needed_shards),
        modalities=args.modalities,
        part=args.part or "1/1",
        transfer_total=human(total_bytes),
        transfer_pending=human(pending_bytes),
        sensor_root=str(sensor_root),
        staging=str(staging))

    if args.mode == "plan":
        log("note", p0_gap="navtrain 的 46 个 history camera 缺口请用 " + P0_SCRIPT,
            next_action="确认体积后再以 download 阶段执行（建议走 qsub）")
        return

    # 下载/解包/安装都属于重 IO，强制走 UGE，避免在登录节点上跑
    if not args.allow_local and not os.environ.get("JOB_ID"):
        raise RuntimeError("download/extract/install 请通过 qsub 执行（或显式 --allow-local）")

    if args.mode == "download":
        ensure_space(archive_dir, pending_bytes)
        for modality in args.modalities:
            for shard_id in needed_shards:
                download_shard(shard_files[modality][shard_id], archive_dir)
        log("download_complete", shards=len(needed_shards) * len(args.modalities))
        return

    if args.mode == "extract":
        staging_blobs.mkdir(parents=True, exist_ok=True)
        total_files = 0
        for modality in args.modalities:
            for shard_id in needed_shards:
                shard = shard_files[modality][shard_id]
                archive = archive_dir / shard["name"]
                if not archive.exists():
                    raise RuntimeError("分片尚未下载：" + shard["name"])
                # 解包前重新校验，确保用的是完整未损坏的归档
                if sha256_of(archive) != shard["sha256"]:
                    raise RuntimeError("分片校验失败，拒绝解包：" + shard["name"])
                shard_logs = [name for name in targets if log_to_shard[name] == shard_id]
                total_files += extract_logs(archive, shard_logs, staging_blobs)
        log("extract_complete", files=total_files,
            next_action="确认 staging 内容后再执行 install")
        return

    if args.mode == "install":
        if not staging_blobs.is_dir():
            raise RuntimeError("staging 里没有可安装内容，先执行 extract")
        installed = install_logs(staging_blobs, sensor_root, targets)
        log("install_complete", installed=installed,
            note="只补了缺失文件；补 raw 缺图不代表 processed 侧合规问题已解决")
        return


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="plan",
                        choices=("plan", "download", "extract", "install"),
                        help="阶段，默认 plan（只读，不写任何文件）")
    parser.add_argument("--sensor-root",
                        default=os.environ.get("OPENSCENE_DATA_ROOT",
                                               "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset")
                                + "/sensor_blobs/trainval",
                        help="live sensor 根（默认取 $OPENSCENE_DATA_ROOT/sensor_blobs/trainval）")
    parser.add_argument("--staging", default="outputs/navsim_shard_staging",
                        help="staging 目录，必须与 live 根分离")
    parser.add_argument("--logs-file",
                        help="目标 log 清单：每行一个 log 名，或审计产出的 jsonl")
    parser.add_argument("--logs", help="目标 log，逗号分隔")
    parser.add_argument("--scan-missing", action="store_true",
                        help="启发式扫描：官方索引有、本地目录缺的 log（目录级，非引用级）")
    parser.add_argument("--modalities", default="camera",
                        help="camera / lidar / camera,lidar（默认只取 camera）")
    parser.add_argument("--part",
                        help='分卷切片 "i/n"，配合 UGE array job 并行下载')
    parser.add_argument("--allow-local", action="store_true",
                        help="允许不在 UGE 作业里执行重 IO 阶段（不推荐）")
    args = parser.parse_args()
    args.modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    for modality in args.modalities:
        if modality not in MODALITY_DIR:
            parser.error("未知 modality：" + modality)

    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError,
            tarfile.TarError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
