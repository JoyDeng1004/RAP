#!/usr/bin/env python3
"""Rendered-raster coverage audit: real x raster 配对矩阵。

只读脚本：不下载、不写数据、不改配置。只往 --out-dir 写报告。

=============================================================================
为什么需要这个脚本（sensor_blob_coverage.py 不够用）
=============================================================================
`sensor_blob_coverage.py` 审计的是真实 sensor（sensor_blobs/）。但 RAP 的训练输入
是**两路**：真实图 + rasterized view，而后者的缺失后果和前者完全不同。

navsim/common/dataclasses.py:77 处的加载逻辑：

    rendered_image_path = Path(str(image_path).replace('sensor_blobs',
                                                       'rendered_sensor_blobs'))
    try:    rendered_image = np.array(Image.open(rendered_image_path))[20:-20]
    except: rendered_image = np.zeros((1080, 1920, 3), dtype=np.float32)
    try:    real_image = np.array(Image.open(image_path)); real_valid = True
    except: real_image = np.zeros_like(rendered_image); real_valid = False

两路都是裸 except 兜零，**缺文件不报错**。但下游待遇是不对称的：

  real 缺失   -> real_valid=False -> camera_valid -> agent_lightning_module.py:149
                 用它做 mask，把该样本从 real 分支剔除。**被正确处理。**
  raster 缺失 -> 没有任何 validity 标记。rap_dino/bevformer/bev_feature_build.py:93
                 只导出 real_image_result["validity"]，synthetic 那份算了但丢掉。
                 全黑图带全权重进 loss。**静默污染。**

（对照：navsim/agents/diffusiondrive/transfuser_features.py:44 反而导出了
`rendered_camera_valid`。所以这个不对称是 rap_dino/pad 两个 agent 的特性。）

因此 raster 覆盖率必须单独审计，而且判读标准比 real 严得多：
real 覆盖 ~20% 是 navtrain 在全量 trainval 里的正常占比；raster 覆盖必须接近 100%。

=============================================================================
它能区分什么
=============================================================================
配对矩阵 both / real_only / raster_only / neither 直接区分两个假设：

  raster 覆盖 ~0%          -> 官方 OpenScene 原版 metadata（没做过 raster），
                              或 raster 根路径不对（见下面 sensor_blobs 替换陷阱）
  raster 覆盖 ~100%        -> RAP rasterization 覆盖了这批 log
  raster 覆盖居中          -> RAP 只跑了一部分；logs_raster_absent.txt 给出未覆盖清单

=============================================================================
raster 根的 sensor_blobs 替换陷阱
=============================================================================
raster 根不是配置项，是 str.replace 出来的，而 str.replace 替换**全部**出现位置。
本脚本因此显式报告 sensor_blobs 在 blob_root 里出现了几次：

  0 次 -> 替换是 no-op，raster 根 == 真实根，一定测不出 raster
  2+次 -> 路径被多处改写，得到一个不存在的目录（例如经由某些 symlink 层时）

同理，名字不叫 `rendered_sensor_blobs` 的 raster 根（如 `..._4cam_v1`）永远不会被
读到，必须以该名字暴露（symlink 即可）。

=============================================================================
两种数据集布局
=============================================================================
本脚本不对 data_path 的层数做假设，nuPlan 与 nuScenes 都能直接跑：

  nuPlan/NAVSIM   <log_id>/CAM_F0/<token>.jpg
  nuScenes        samples/CAM_FRONT/<file>.jpg     （create_nuscenes_metadata.py）

nuScenes 侧的两个额外注意点：

  1. 所有 scene 共用同一批 samples/CAM_* 目录，所以 --probe-sample 的目录列表
     必须去重（已做），否则会把几万文件的目录反复 listdir。
  2. camera-mode=navsim 时 raster 裁剪后是 1080x1920，与 dataclasses.py 的兜零
     形状【完全相同】—— 整张缺失在形状上看不出来。必须靠 --probe-sample 的
     全黑率来探测，光看"文件存在"是不够的。

  用法示例：
      python scripts/data_audit/raster_coverage.py \
          --openscene-data-root <...>/dataset_nuscenes \
          --data-split nuscenes_trainval \
          --group-manifest <...>/dataset_nuscenes/_index/manifest.json \
          --probe-sample 500 --fail-under 99 --out-dir <...>
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

DEFAULT_CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")
MISSING_CAMERA_PREFIX = "missing_camera"
SENSOR_TOKEN = "sensor_blobs"
RASTER_TOKEN = "rendered_sensor_blobs"

# dataclasses.py 的兜零形状。raster 实测形状与它不一致时，[20:-20] 裁剪后会和
# 兜零样本 stack 不起来 —— 属于会真正崩掉的那一类，必须报出来。
FALLBACK_SHAPE = (1080, 1920)
RASTER_CROP = 20


def derive_raster_root(blob_root: Path) -> Tuple[Path, int]:
    """按 dataclasses.py 的同一规则推导 raster 根，并返回 token 出现次数。"""
    text = str(blob_root)
    return Path(text.replace(SENSOR_TOKEN, RASTER_TOKEN)), text.count(SENSOR_TOKEN)


def scan_dir(root: Path, parent: str) -> Dict[str, int]:
    """一次 scandir 换掉一堆 stat —— Lustre 上这个差别是分钟 vs 小时级。

    `parent` 是 data_path 去掉文件名后的那一段，两种数据集布局都能直接套：

        nuPlan/NAVSIM   <log_id>/CAM_F0/<token>.jpg      -> parent = <log_id>/CAM_F0
        nuScenes        samples/CAM_FRONT/<file>.jpg     -> parent = samples/CAM_FRONT

    返回 {filename: size}；目录不存在返回空 dict，这让"整个 log 缺失"变得极廉价。
    注意 nuScenes 布局下所有 scene 共用同一批 parent（只有相机数那么多个），
    所以调用方的缓存在那边尤其关键 —— 见 audit_log 里的 listing()。
    """
    entries: Dict[str, int] = {}
    try:
        with os.scandir(root / parent) as iterator:
            for entry in iterator:
                if entry.is_file(follow_symlinks=True):
                    try:
                        entries[entry.name] = entry.stat(follow_symlinks=True).st_size
                    except OSError:
                        entries[entry.name] = -1
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        pass
    return entries


def audit_log(
    log_path: Path,
    blob_root: Path,
    raster_root: Path,
    cameras: Tuple[str, ...],
    frames_per_log: int,
    rng: random.Random,
) -> Dict[str, Any]:
    with log_path.open("rb") as handle:
        frames = pickle.load(handle)
    assert isinstance(frames, list), f"metadata is not a list: {log_path}"

    n_frames_total = len(frames)
    if frames_per_log and n_frames_total > frames_per_log:
        frames = rng.sample(frames, frames_per_log)

    pair: Counter = Counter()  # both / real_only / raster_only / neither
    declared_missing = 0
    zero_byte_raster = 0
    blob_dirs: set[str] = set()
    render_shapes: Counter = Counter()
    raster_absent_paths: List[str] = []
    cache: Dict[Tuple[str, str], Dict[str, int]] = {}

    def listing(root: Path, tag: str, parent: str) -> Dict[str, int]:
        key = (tag, parent)
        if key not in cache:
            cache[key] = scan_dir(root, parent)
        return cache[key]

    for frame in frames:
        cams = frame.get("cams") or {}
        for camera in cameras:
            entry = cams.get(camera)
            if entry is None or not entry.get("data_path"):
                declared_missing += 1
                continue
            rel = PurePosixPath(str(entry["data_path"]))
            if rel.parts and rel.parts[0] == MISSING_CAMERA_PREFIX:
                # 数据集自己声明无图。raster 侧同样不会有 —— 不算"缺失"，
                # 但也意味着这一帧两路都是黑的，单列出来。
                declared_missing += 1
                continue
            # 不对 parts 的层数做假设：nuPlan 是 <log>/<cam>/<file>，
            # nuScenes 是 samples/<cam>/<file>，更深的布局也照样成立。
            assert len(rel.parts) >= 2, f"unexpected data_path layout: {rel}"
            parent, filename = rel.parent.as_posix(), rel.parts[-1]
            blob_dirs.add(parent)

            # create_nuscenes_metadata.py 会把渲染画布尺寸写进 render_image_shape。
            # 有它就不用猜 expected height（见 probe_raster_geometry）。
            shape = entry.get("render_image_shape")
            if shape:
                render_shapes[f"{int(shape[0])}x{int(shape[1])}"] += 1

            real_size = listing(blob_root, "real", parent).get(filename)
            raster_size = listing(raster_root, "raster", parent).get(filename)
            has_real = real_size is not None and real_size > 0
            has_raster = raster_size is not None and raster_size > 0
            if raster_size == 0:
                zero_byte_raster += 1

            if has_real and has_raster:
                pair["both"] += 1
            elif has_real:
                pair["real_only"] += 1
                raster_absent_paths.append(rel.as_posix())
            elif has_raster:
                pair["raster_only"] += 1
            else:
                pair["neither"] += 1
                raster_absent_paths.append(rel.as_posix())

    checked = sum(pair.values())
    raster_present = pair["both"] + pair["raster_only"]
    if checked == 0:
        status = "no_expected_files"
    elif raster_present == checked:
        status = "raster_complete"
    elif raster_present == 0:
        status = "raster_absent"
    else:
        status = "raster_partial"

    return {
        "log_name": log_path.stem,
        "status": status,
        "n_frames_total": n_frames_total,
        "n_frames_checked": len(frames),
        "blob_dirs": sorted(blob_dirs),
        "render_shapes": dict(render_shapes),
        "checked_files": checked,
        "raster_present": raster_present,
        "real_present": pair["both"] + pair["real_only"],
        "pair_both": pair["both"],
        "pair_real_only": pair["real_only"],
        "pair_raster_only": pair["raster_only"],
        "pair_neither": pair["neither"],
        "declared_missing_paths": declared_missing,
        "zero_byte_raster": zero_byte_raster,
        "raster_absent_paths": raster_absent_paths,
    }


def probe_raster_geometry(
    raster_root: Path, reports: List[Dict[str, Any]], sample: int, seed: int
) -> Dict[str, Any]:
    """真开若干 raster 图：抓截断、抓全黑、抓形状不一致。

    形状很关键：加载侧做 [20:-20]，兜零形状是 (1080,1920)。若实测 raster 高度
    不是 <期望高度>+2*20，裁剪结果就和兜零样本对不齐。

    期望高度优先取 metadata 里的 render_image_shape（create_nuscenes_metadata.py
    会写），没有就退回 nuPlan 那条链路的 1080+40。这一点对 nuScenes 是必须的：
    camera-mode=native 的画布是 1600x940，硬套 1120 会把正常数据全报成 mismatch。

    【全黑判据在 nuScenes 上比 nuPlan 更要紧】：navsim mode 下 raster 裁剪后正好是
    1080x1920 —— 和 dataclasses.py 的兜零形状【逐位相同】。也就是说 raster 整张
    缺失时，形状检查看不出任何异常，只有 all_black 能抓到。
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return {"skipped": "Pillow/numpy not available"}

    # 每个相机目录随机取 2 帧，而不是排序后的前 2 帧。取前 2 帧会系统性地命中
    # 每个 log 的开头 —— 那里常是车库/场站，视野内没有地图要素，渲染成全黑是
    # 正常的，于是 all_black 会被虚高。随机取样才反映整条 log 的真实情况。
    picker = random.Random(seed)
    candidates: List[str] = []
    # blob_dirs 现在已经是「含相机层」的完整父目录，直接用即可，不必再 listdir 一层。
    # 必须先去重：nuScenes 布局下 850 个 scene 共用同一批 samples/CAM_* 目录，
    # 不去重就会把同一个几万文件的目录 listdir 上千遍（Lustre 上是小时级）。
    probe_dirs: List[str] = sorted(
        {d for report in reports if report["raster_present"] for d in report["blob_dirs"]}
    )
    # 每目录配额：至少 2 张（nuPlan 有成百上千个目录时维持原行为），但目录很少时
    # 要按 sample 摊开 —— nuScenes 去重后只剩 4 个 samples/CAM_* 目录，
    # 固定取 2 张会让 --probe-sample 500 实际只查到 8 张，probe 形同虚设。
    per_dir = max(2, -(-sample // max(len(probe_dirs), 1))) if sample else 2
    for parent in probe_dirs:
        try:
            names = os.listdir(raster_root / parent)
        except OSError:
            continue
        if not names:
            continue
        picked = picker.sample(names, min(per_dir, len(names)))
        candidates.extend(f"{parent}/{n}" for n in picked)
    if not candidates:
        return {"checked": 0, "note": "raster 根下没有可抽查的文件"}

    random.Random(seed).shuffle(candidates)
    shapes: Counter = Counter()
    corrupt: List[str] = []
    black: List[str] = []
    # 按【完整父目录】分组，两种布局都有意义：
    #   nuPlan   -> <log_id>/CAM_F0   （能看出某个 log 的某一路整批是黑的）
    #   nuScenes -> samples/CAM_FRONT （能看出某一路相机整批是黑的）
    # 原来取 rel.split("/")[0]，在 nuScenes 下恒为 "samples"，没有信息量。
    black_by_dir: Counter = Counter()

    declared: Counter = Counter()
    for report in reports:
        declared.update(report.get("render_shapes") or {})
    if declared:
        # metadata 自己声明的画布高度（可能有多种，各自减去 2*crop）
        expected_heights = sorted(
            {int(s.split("x")[0]) - 2 * RASTER_CROP for s in declared}
        )
    else:
        expected_heights = [FALLBACK_SHAPE[0]]
    expected_h = expected_heights[0] + 2 * RASTER_CROP
    expected_raw = sorted(h + 2 * RASTER_CROP for h in expected_heights)
    for rel in candidates[:sample]:
        try:
            with Image.open(raster_root / rel) as image:
                image.load()
                shapes[f"{image.height}x{image.width}"] += 1
                if not np.asarray(image).any():
                    black.append(rel)
                    black_by_dir[rel.rsplit("/", 1)[0]] += 1
        except Exception as error:  # noqa: BLE001 - 任何解码失败都是发现
            corrupt.append(f"{rel}\t{type(error).__name__}: {error}")
    return {
        "checked": min(sample, len(candidates)),
        "probe_dirs": len(probe_dirs),
        "shapes": dict(shapes),
        "expected_height_for_crop": expected_h,
        "expected_heights_from_metadata": expected_raw if declared else None,
        "shape_mismatch": {
            s: n for s, n in shapes.items() if int(s.split("x")[0]) not in expected_raw
        },
        "all_black_files": len(black),
        # 判读要点：黑图【分散】在很多 log 里 = 多半是场站/空旷路段，正常；
        # 【集中】在少数 log 里 = 那些 log 的地图要素没取到，需要单独查。
        "all_black_paths": black[:50],
        "all_black_by_dir": dict(black_by_dir.most_common(20)),
        "corrupt": corrupt,
    }


def verdict(summary: Dict[str, Any]) -> List[str]:
    """把数字翻成结论。只陈述能由数字支持的部分。"""
    lines: List[str] = []
    if not summary["raster_root_exists"]:
        lines.append(
            "BLOCKING：raster 根不存在。当前配置下每一个样本的 raster 都会走兜零分支"
            "变成全黑，且 raster 侧无 validity mask，训练不会报错但信号全废。"
        )
    occurrences = summary["sensor_blobs_token_occurrences"]
    if occurrences != 1:
        lines.append(
            f"BLOCKING：blob_root 里 '{SENSOR_TOKEN}' 出现 {occurrences} 次"
            f"（需要恰好 1 次）。str.replace 会改写全部出现位置，推导出的 raster 根"
            f"不是你想要的那个。"
        )
    pct = summary["raster_coverage_pct"]
    if summary["checked_files"] == 0:
        lines.append("无法判定：没有可检查的文件（metadata 里全是 declared_missing？）")
    elif pct < 1.0:
        lines.append(
            f"raster 覆盖 {pct}% ≈ 0：这批 metadata 未经 RAP rasterization（官方 "
            f"OpenScene 原版），或 raster 根位置/命名不对。查 checkpoint.txt 可佐证。"
        )
    elif pct > 99.0:
        lines.append(f"raster 覆盖 {pct}%：RAP rasterization 覆盖了这批 log，raster 侧就绪。")
    else:
        lines.append(
            f"raster 覆盖 {pct}%：RAP rasterization 只跑了一部分。"
            f"logs_raster_absent.txt / logs_raster_partial.txt 给出未覆盖清单；"
            f"在补齐或改用只含已覆盖 log 的 scene_filter 之前，不要开全量训练。"
        )
    real_pct = summary["real_coverage_pct"]
    lines.append(
        f"real 覆盖 {real_pct}%（nuPlan/navtrain 参考：全量 trainval 下 ~20% 属正常，"
        f"由 camera_valid 正确屏蔽；raster 没有这个保护。nuScenes 侧 sensor_blobs 是"
        f"整份 symlink，这里应当接近 100%，低了说明 symlink 或 data_path 有问题）。"
    )

    probe = summary.get("raster_probe") or {}
    if probe.get("checked"):
        if probe.get("shape_mismatch"):
            lines.append(
                f"BLOCKING：raster 形状与 metadata 声明不符 {probe['shape_mismatch']}"
                f"（期望 {probe.get('expected_heights_from_metadata') or probe['expected_height_for_crop']}）。"
                f"[20:-20] 裁完会和兜零样本对不齐。"
            )
        black_pct = 100.0 * probe["all_black_files"] / probe["checked"]
        lines.append(
            f"抽查 {probe['checked']} 张：全黑 {probe['all_black_files']} 张（{black_pct:.1f}%）。"
            f"【注意】navsim camera-mode 下 raster 裁剪后是 1080x1920，与 dataclasses.py 的"
            f"兜零形状完全相同 —— 整张缺失在形状上看不出来，全黑率是唯一的探测手段。"
            f"黑图分散在很多 log 里多半正常（场站/空旷路段）；集中在少数 log 要单独查。"
        )
    elif summary["raster_coverage_pct"] > 99.0:
        lines.append(
            "未跑 --probe-sample：只验证了文件【存在】。存在不等于有内容，"
            "建议至少 --probe-sample 500 再下结论。"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-split", default="trainval", help="trainval | test | mini")
    parser.add_argument("--openscene-data-root", type=Path, default=os.environ.get("OPENSCENE_DATA_ROOT"))
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--blob-root", type=Path, default=None)
    parser.add_argument(
        "--raster-root",
        type=Path,
        default=None,
        help="默认按 dataclasses.py 的规则从 blob-root 推导。显式传入可用于"
        "在建 symlink 之前先验证某个候选 raster 根。",
    )
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-logs", type=int, default=0, help="随机抽 N 个 log（0=全部）")
    parser.add_argument("--frames-per-log", type=int, default=0, help="每 log 抽 M 帧（0=全部）")
    parser.add_argument("--sample-seed", type=int, default=20260915)
    parser.add_argument("--dump-missing-files", action="store_true")
    parser.add_argument("--probe-sample", type=int, default=0, help="真开 N 张 raster 查形状/全黑/截断")
    parser.add_argument(
        "--group-manifest",
        type=Path,
        default=None,
        help="nuScenes 专用：dataset_nuscenes/_index/manifest.json。"
        "用它把 850 个 scene 聚合回 68 个真实 log 再报覆盖率。",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="raster 覆盖率低于该百分比时以退出码 1 结束（给 CI / qsub 用）。"
        "不给则永远退 0。",
    )
    args = parser.parse_args()

    if args.log_root is None or args.blob_root is None:
        assert args.openscene_data_root is not None, "set OPENSCENE_DATA_ROOT or pass --log-root/--blob-root"
    log_root = args.log_root or Path(args.openscene_data_root) / "navsim_logs" / args.data_split
    blob_root = args.blob_root or Path(args.openscene_data_root) / SENSOR_TOKEN / args.data_split
    derived_raster, occurrences = derive_raster_root(blob_root)
    raster_root = args.raster_root or derived_raster
    cameras = tuple(name.strip() for name in args.cameras.split(",") if name.strip())

    assert log_root.is_dir(), f"log root not found: {log_root}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_paths = sorted(path for path in log_root.iterdir() if path.suffix == ".pkl")
    assert log_paths, f"no .pkl logs under {log_root}"
    n_logs_on_disk = len(log_paths)
    rng = random.Random(args.sample_seed)
    if args.max_logs and n_logs_on_disk > args.max_logs:
        log_paths = sorted(rng.sample(log_paths, args.max_logs))

    print(f"log_root      {log_root}  ({n_logs_on_disk} pkl)")
    print(f"blob_root     {blob_root}")
    print(f"raster_root   {raster_root}  exists={raster_root.is_dir()}"
          f"{'  [显式指定]' if args.raster_root else '  [按 dataclasses.py 规则推导]'}")
    print(f"审计 {len(log_paths)} 个 log，每 log {'全部' if not args.frames_per_log else args.frames_per_log} 帧\n")

    reports: List[Dict[str, Any]] = []
    for index, log_path in enumerate(log_paths, start=1):
        reports.append(audit_log(log_path, blob_root, raster_root, cameras, args.frames_per_log, rng))
        if index % 50 == 0 or index == len(log_paths):
            print(f"[{index}/{len(log_paths)}] {log_path.stem}", flush=True)

    by_status: Counter = Counter(r["status"] for r in reports)
    checked = sum(r["checked_files"] for r in reports)
    raster_present = sum(r["raster_present"] for r in reports)
    real_present = sum(r["real_present"] for r in reports)

    # 分组维度。nuPlan 的 pkl 名前 10 位是采集日期，按它聚合能看出"某天的数据整批没渲"。
    # nuScenes 的 pkl 名是 scene-XXXX，前 10 位就是它自己 —— 聚合退化成 850 个单元素桶，
    # 没有信息量。给了 --group-manifest 就按【真实 nuScenes log】聚合（68 个桶），
    # 这才是和 nuPlan 按 log 聚合对等的粒度。
    group_of: Dict[str, str] = {}
    if args.group_manifest:
        manifest = json.loads(Path(args.group_manifest).read_text())
        group_of = {s["scene_name"]: s["log_name"] for s in manifest["scenes"]}
    by_date: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for report in reports:
        key = group_of.get(report["log_name"], report["log_name"][:10])
        by_date[key][0] += report["checked_files"]
        by_date[key][1] += report["raster_present"]

    summary = {
        "data_split": args.data_split,
        "log_root": str(log_root),
        "blob_root": str(blob_root),
        "raster_root": str(raster_root),
        "raster_root_explicit": args.raster_root is not None,
        "raster_root_exists": raster_root.is_dir(),
        "sensor_blobs_token_occurrences": occurrences,
        "cameras": list(cameras),
        "logs_on_disk": n_logs_on_disk,
        "logs_audited": len(reports),
        "logs_raster_complete": by_status.get("raster_complete", 0),
        "logs_raster_partial": by_status.get("raster_partial", 0),
        "logs_raster_absent": by_status.get("raster_absent", 0),
        "logs_no_expected_files": by_status.get("no_expected_files", 0),
        "checked_files": checked,
        "raster_present": raster_present,
        "real_present": real_present,
        "raster_coverage_pct": round(100.0 * raster_present / checked, 4) if checked else 0.0,
        "real_coverage_pct": round(100.0 * real_present / checked, 4) if checked else 0.0,
        "pair_matrix": {
            k: sum(r[f"pair_{k}"] for r in reports)
            for k in ("both", "real_only", "raster_only", "neither")
        },
        "declared_missing_paths": sum(r["declared_missing_paths"] for r in reports),
        "zero_byte_raster": sum(r["zero_byte_raster"] for r in reports),
        "raster_coverage_by_date": {
            key: {
                "checked": value[0],
                "raster_present": value[1],
                "coverage_pct": round(100.0 * value[1] / value[0], 2) if value[0] else 0.0,
            }
            for key, value in sorted(by_date.items())
        },
    }
    if args.probe_sample:
        summary["raster_probe"] = probe_raster_geometry(
            raster_root, reports, args.probe_sample, args.sample_seed
        )

    absent = [r["log_name"] for r in reports if r["status"] == "raster_absent"]
    partial = [r for r in reports if r["status"] == "raster_partial"]
    complete = [r["log_name"] for r in reports if r["status"] == "raster_complete"]

    (args.out_dir / "logs_raster_absent.txt").write_text(
        "".join(f"{n}\n" for n in absent), encoding="utf-8"
    )
    (args.out_dir / "logs_raster_partial.txt").write_text(
        "".join(
            f"{r['log_name']}\t{r['raster_present']}/{r['checked_files']}\n"
            for r in sorted(partial, key=lambda r: r["raster_present"] / max(r["checked_files"], 1))
        ),
        encoding="utf-8",
    )
    # raster 就绪的 log 清单：可直接用来收窄 scene_filter 的 log_names
    (args.out_dir / "logs_raster_complete.txt").write_text(
        "".join(f"{n}\n" for n in complete), encoding="utf-8"
    )
    if args.dump_missing_files:
        with (args.out_dir / "raster_missing_files.txt").open("w", encoding="utf-8") as handle:
            for report in reports:
                for rel in report["raster_absent_paths"]:
                    handle.write(rel + "\n")

    for report in reports:
        report.pop("raster_absent_paths")
    with (args.out_dir / "raster_coverage.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_log": reports}, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    matrix = summary["pair_matrix"]
    print(
        f"\n审计 log {summary['logs_audited']}/{summary['logs_on_disk']}"
        f" | raster 完整 {summary['logs_raster_complete']}"
        f" | 部分 {summary['logs_raster_partial']}"
        f" | 全缺 {summary['logs_raster_absent']}"
        f"\n配对矩阵  both={matrix['both']}  real_only={matrix['real_only']}"
        f"  raster_only={matrix['raster_only']}  neither={matrix['neither']}"
        f"\nraster 覆盖 {summary['raster_coverage_pct']}%"
        f" | real 覆盖 {summary['real_coverage_pct']}%"
        f" | 数据集声明无图 {summary['declared_missing_paths']} 条"
    )
    print("\n结论：")
    for line in verdict(summary):
        print(f"  - {line}")

    if args.fail_under is not None and summary["raster_coverage_pct"] < args.fail_under:
        raise SystemExit(
            f"\nFAIL raster 覆盖 {summary['raster_coverage_pct']}% < --fail-under {args.fail_under}"
        )


if __name__ == "__main__":
    main()
