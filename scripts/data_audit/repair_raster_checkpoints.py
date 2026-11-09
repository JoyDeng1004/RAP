#!/usr/bin/env python3
"""按磁盘实际内容校正 rasterize 的 checkpoint.txt（磁盘写满后的恢复工具）。

默认 dry-run：只报告，不改任何文件。加 --apply 才写盘。

=============================================================================
为什么 checkpoint 在磁盘写满后不可信
=============================================================================
create_openscene_metadata.py:460 写渲染图用的是：

    cv2.imwrite(full_rendered_path, rendered[:, :, ::-1])

cv2.imwrite 失败时【返回 False，不抛异常】，而这里没有检查返回值。于是磁盘满时：

    渲染循环   -> imwrite 静默失败，图没落盘
    pkl 写入   -> pkl 只有约 10MB，而该 log 的 raster 约 214MB，
                  所以常常 pkl 还写得进去
    函数返回   -> append_done(idx) 把这个 log 记成"已完成"

结果就是一批"标记完成、实际缺图"的 log。直接重投作业会按 checkpoint 跳过它们，
留下永久的黑图 —— 而加载侧 navsim/common/dataclasses.py:81 对缺失 raster 同样
是裸 except 兜零，且 raster 分支没有 validity mask，训练时不会报任何错。

本脚本的做法：不信 checkpoint，逐 log 拿 pkl 里声明的相机文件去磁盘核对，
只保留 100% 齐全的索引，其余从 checkpoint 中剔除，让下次重投重新渲染它们。

=============================================================================
索引语义
=============================================================================
checkpoint.txt 存的是 db 列表下标，该列表来自
    sorted(f[:-3] for f in os.listdir(NUPLAN_DB_PATH) if isfile)
本脚本用同样的规则重建这个列表来做 下标 <-> log 名 的映射。

**前提：DB 目录自作业提交以来没有增删过文件。** 增删会整体平移下标，使所有
既有 checkpoint 失效 —— 那种情况下唯一安全的做法是清空 checkpoint 全量重跑。
脚本会打印当前 db 数量，请和作业日志里 "out of NNNNN" 的数字核对。
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, List, Set, Tuple

DEFAULT_CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0")  # ego/perturbed 的默认渲染相机
MISSING_CAMERA_PREFIX = "missing_camera"


def build_db_index(db_path: Path) -> List[str]:
    """复刻 create_openscene_metadata.py:536-542 的 db 列表构造与排序。"""
    names = [
        f[:-3]
        for f in os.listdir(db_path)
        if os.path.isfile(os.path.join(db_path, f))
    ]
    names.sort()
    return names


def scan_camera_dir(root: Path, log_dir: str, camera: str) -> Dict[str, int]:
    entries: Dict[str, int] = {}
    try:
        with os.scandir(root / log_dir / camera) as it:
            for e in it:
                if e.is_file(follow_symlinks=True):
                    try:
                        entries[e.name] = e.stat(follow_symlinks=True).st_size
                    except OSError:
                        entries[e.name] = -1
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        pass
    return entries


def verify_log(
    log_name: str, logs_dir: Path, raster_root: Path, cameras: Tuple[str, ...]
) -> Tuple[bool, str, int, int]:
    """返回 (是否完整, 原因, 期望文件数, 实有文件数)。"""
    pkl = logs_dir / f"{log_name}.pkl"
    if not pkl.is_file():
        return False, "pkl 缺失", 0, 0
    try:
        with pkl.open("rb") as fh:
            frames = pickle.load(fh)
    except Exception as err:  # noqa: BLE001 - 截断的 pkl 也是磁盘满的典型产物
        return False, f"pkl 读取失败（{type(err).__name__}）", 0, 0
    if not isinstance(frames, list):
        return False, "pkl 内容不是 list", 0, 0

    expected = present = 0
    cache: Dict[Tuple[str, str], Dict[str, int]] = {}
    for frame in frames:
        cams = frame.get("cams") or {}
        for camera in cameras:
            entry = cams.get(camera)
            if entry is None or not entry.get("data_path"):
                continue
            rel = PurePosixPath(str(entry["data_path"]))
            if rel.parts and rel.parts[0] == MISSING_CAMERA_PREFIX:
                continue  # 数据集自己声明无图，渲染侧同样不会有，不计入期望
            if len(rel.parts) < 3:
                continue
            log_dir, cam_dir, filename = rel.parts[0], rel.parts[1], rel.parts[-1]
            key = (log_dir, cam_dir)
            if key not in cache:
                cache[key] = scan_camera_dir(raster_root, log_dir, cam_dir)
            expected += 1
            size = cache[key].get(filename)
            if size is not None and size > 0:
                present += 1

    if expected == 0:
        return False, "pkl 内无可核对的相机条目", 0, 0
    if present == expected:
        return True, "完整", expected, present
    return False, f"渲染图缺失 {expected - present}", expected, present


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--stage", default="ego", help="ego | perturbed | aug")
    p.add_argument("--db-path", type=Path,
                   default=Path("/gs/bs/tga-RLA/qdeng/nuplan_dataset/nuplan-v1.1/splits/trainval"))
    p.add_argument("--data-root", type=Path,
                   default=Path("/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset"))
    p.add_argument("--split", default="trainval")
    p.add_argument("--work-root", type=Path,
                   default=Path("/gs/bs/tga-RLA/qdeng/RAP/exp/rasterize"))
    p.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    p.add_argument("--apply", action="store_true", help="真的改写 checkpoint.txt（默认只报告）")
    args = p.parse_args()

    cameras = tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    logs_dir = args.data_root / "navsim_logs" / args.split
    raster_sub = "rendered_sensor_blobs_augmented" if args.stage == "aug" else "rendered_sensor_blobs"
    raster_root = args.data_root / raster_sub / args.split
    work_dir = args.work_root / args.stage

    assert args.db_path.is_dir(), f"db 目录不存在: {args.db_path}"
    assert logs_dir.is_dir(), f"navsim_logs 不存在: {logs_dir}"
    assert work_dir.is_dir(), f"workdir 不存在: {work_dir}"

    db_names = build_db_index(args.db_path)
    print(f"db 列表      : {args.db_path}  ({len(db_names)} 项)")
    print(f"              ↑ 请和作业日志里 'out of NNNNN' 核对，不一致说明 DB 目录变过，")
    print(f"                所有既有 checkpoint 的下标都已失效，只能清空全量重跑。")
    print(f"navsim_logs  : {logs_dir}")
    print(f"渲染根       : {raster_root}")
    print(f"相机         : {list(cameras)}")
    print(f"模式         : {'APPLY（会改写 checkpoint）' if args.apply else 'DRY-RUN（只报告）'}\n")

    checkpoints = sorted(work_dir.glob("*/checkpoint.txt"))
    if not checkpoints:
        print(f"没有找到任何 checkpoint.txt（{work_dir}/*/checkpoint.txt）")
        return

    total_kept = total_dropped = 0
    dropped_detail: List[str] = []

    for cp in checkpoints:
        try:
            idxs = sorted({int(l.strip()) for l in cp.read_text().splitlines() if l.strip().isdigit()})
        except OSError as err:
            print(f"[{cp.parent.name}] 读取失败: {err}")
            continue

        kept: List[int] = []
        dropped: List[int] = []
        for i in idxs:
            if i >= len(db_names):
                dropped.append(i)
                dropped_detail.append(f"{cp.parent.name}\t{i}\t下标越界")
                continue
            name = db_names[i]
            ok, reason, exp, got = verify_log(name, logs_dir, raster_root, cameras)
            if ok:
                kept.append(i)
            else:
                dropped.append(i)
                dropped_detail.append(f"{cp.parent.name}\t{i}\t{name}\t{reason} ({got}/{exp})")

        total_kept += len(kept)
        total_dropped += len(dropped)
        flag = "" if not dropped else "  <- 有需要重渲染的"
        print(f"[{cp.parent.name:24s}] 原 {len(idxs):5d} -> 保留 {len(kept):5d}，剔除 {len(dropped):5d}{flag}")

        if args.apply and dropped:
            backup = cp.with_suffix(".txt.bak")
            if not backup.exists():           # 只备份一次，别让二次运行覆盖掉原始状态
                backup.write_text(cp.read_text(), encoding="utf-8")
            cp.write_text("".join(f"{i}\n" for i in kept), encoding="utf-8")

    print(f"\n合计：保留 {total_kept}，剔除 {total_dropped}")
    if dropped_detail:
        out = work_dir / "repair_dropped.tsv"
        out.write_text("".join(l + "\n" for l in dropped_detail), encoding="utf-8")
        print(f"剔除明细：{out}")
        print("前 10 条：")
        for line in dropped_detail[:10]:
            print("  " + line)

    if not args.apply and total_dropped:
        print("\n以上为 DRY-RUN。确认无误后加 --apply 真正改写（会先存 .txt.bak 备份）。")
    elif args.apply:
        print("\ncheckpoint 已校正。现在重投作业，被剔除的 log 会重新渲染。")


if __name__ == "__main__":
    main()
