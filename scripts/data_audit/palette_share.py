#!/usr/bin/env python3
"""按基色统计 raster 的像素占比，用来定量对比两条渲染链路。

只读脚本：不写数据，只往 stdout 和 --json-out 写报告。

=============================================================================
为什么按颜色统计就够了
=============================================================================
nuPlan 与 nuScenes 两条链路调的是【同一个】ScenarioRenderer.observe，
调色板是共享常量（renderer.py 的 COLOR_TABLE + draw_cuboids_with_occlusion
内部的 base_face_colors）。元数据侧给不了颜色参数。

所以 raster 里每个非黑像素都是"某个基色 × 深度衰减 alpha"。按基色反查占比，
等价于统计"每一类要素画了多少"—— 这正是跨脚本一致性要量的东西，
而且不需要任何标注、不需要两张图对应同一个场景。

=============================================================================
怎么读这份报告
=============================================================================
拿 nuPlan raster 和 nuScenes raster 各跑一次，对比三个数：

  lanelines        两边都是 lane 多边形边界 -> 占比应当量级相当
  crosswalks       两边都是人行横道       -> 占比应当量级相当
  road_boundaries  【语义不同】
                     nuPlan   = intersection ∪ roadblock 的外轮廓（路缘）
                     nuScenes = lane_divider + road_divider（路中央分道线）
                   两边都画成 radius=10 的深红粗线，但位置完全不同。
                   预期 nuScenes 侧显著更高。这个差值就是该项不一致的量化值。

box 面色（box_face_*）占比反映画面里有多少车 —— 场景不同就不可比，
只用来确认"确实画了框"，不要拿来做一致性判断。

=============================================================================
两个判读陷阱
=============================================================================
1. **JPEG 振铃会制造调色板外的像素，还会伪造小占比的颜色。**
   raster 是 cv2.imwrite 存的 .jpg，锐利色边处会产生中间色。实测合成图
   （quality=95、纯色硬边）上 unmatched 约 1.8%，并凭空出现 0.22% 的
   traffic_light_red。所以：
     - unmatched 在 0-3% 属正常本底，超过 5% 才值得查；
     - **任何低于 0.5% 的单色占比都不可当作"该要素存在"的证据**，
       尤其 traffic_light_* —— nuScenes 侧恒不画灯，出现小占比必是振铃。
   真正可信的是占比大的那几项之间的【比值】，而不是绝对值。

2. raster 全黑时本脚本会报 black=100%，但这**既可能**是渲染缺失（文件没生成、
   被 dataclasses.py 兜零），**也可能**是该帧本来就在空旷路段。单张图分不清，
   要看一批图的黑占比分布 —— 见 scripts/data_audit/raster_coverage.py。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers.renderer import COLOR_TABLE  # noqa: E402

# draw_cuboids_with_occlusion 内部硬编码的六个面色（renderer.py:374-382）。
# top 与 bottom 是同一个值，去重后是 5 个。
BOX_FACE_COLORS = {
    "box_face_0": (247, 37, 133),
    "box_face_1": (76, 201, 240),
    "box_face_2": (114, 9, 183),
    "box_face_3": (67, 97, 238),
    "box_face_4": (58, 12, 163),
}

# 亮度低于此值的像素，方向受 uint8 量化影响已不可靠，计入 dim 而不强行归类。
MIN_BRIGHTNESS = 40
# 余弦相似度阈值。基色只被 alpha 线性缩放，方向不变，所以可以卡得很紧。
COS_THRESHOLD = 0.99
# unmatched 的告警线。JPEG 振铃本底实测 1-3%，卡 0.5% 会天天误报。
UNMATCHED_ALARM_PCT = 5.0
# 低于此占比的单色不可信（振铃伪造），打印时标注出来。
NOISE_FLOOR_PCT = 0.5


def build_palette() -> tuple[List[str], np.ndarray]:
    names: List[str] = []
    colors: List[np.ndarray] = []
    for name, value in COLOR_TABLE.items():
        names.append(name)
        colors.append(np.asarray(value, dtype=float))
    for name, value in BOX_FACE_COLORS.items():
        names.append(name)
        colors.append(np.asarray(value, dtype=float))
    matrix = np.stack(colors)
    return names, matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def share_of(path: Path, names: List[str], unit: np.ndarray) -> Dict[str, float]:
    """单张图的基色占比。返回值对全部像素归一，各项之和为 1。"""
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    # 与加载侧 dataclasses.py:79 一致地裁掉上下各 20 行，统计的才是模型真看到的。
    if image.shape[0] > 2 * 20:
        image = image[20:-20]
    flat = image.reshape(-1, 3).astype(float)
    total = len(flat)

    out: Dict[str, float] = {name: 0.0 for name in names}
    brightness = flat.max(axis=1)
    out["black"] = float((brightness == 0).sum()) / total
    out["dim"] = float(((brightness > 0) & (brightness <= MIN_BRIGHTNESS)).sum()) / total

    bright = flat[brightness > MIN_BRIGHTNESS]
    if len(bright):
        cos = (bright / np.linalg.norm(bright, axis=1, keepdims=True)) @ unit.T
        best = cos.argmax(axis=1)
        hit = cos.max(axis=1) >= COS_THRESHOLD
        # 落在调色板外。JPEG 振铃本底约 1-3%，见模块 docstring 的陷阱 1。
        out["unmatched"] = float((~hit).sum()) / total
        for index in np.nonzero(hit)[0]:
            out[names[best[index]]] += 1.0 / total
    else:
        out["unmatched"] = 0.0
    return out


def collect(root: Path, sample: int, seed: int) -> List[Path]:
    if root.is_file():
        return [root]
    paths = [p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not paths:
        raise SystemExit(f"FAIL {root} 下没有图片")
    paths.sort()
    if sample and len(paths) > sample:
        paths = random.Random(seed).sample(paths, sample)
    return paths


def summarize(shares: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for s in shares for k in s})
    return {k: float(np.mean([s.get(k, 0.0) for s in shares])) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", type=Path, nargs="+",
                        help="一个或多个 raster 目录/文件。给两个就并排对比。")
    parser.add_argument("--labels", default=None,
                        help="逗号分隔的标签，与 paths 一一对应（默认用路径名）")
    parser.add_argument("--sample", type=int, default=50, help="每个路径抽 N 张（0=全部）")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    labels = (args.labels.split(",") if args.labels
              else [p.name or str(p) for p in args.paths])
    assert len(labels) == len(args.paths), "--labels 数量与 paths 不一致"

    names, unit = build_palette()
    report: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for label, root in zip(labels, args.paths):
        files = collect(root, args.sample, args.seed)
        report[label] = summarize([share_of(p, names, unit) for p in files])
        counts[label] = len(files)
        print(f"[palette] {label}: {len(files)} 张  <- {root}", flush=True)

    rows = ["black", "dim", "unmatched", "lanelines", "crosswalks", "road_boundaries",
            "traffic_light_red", "traffic_light_green"]
    rows += sorted(BOX_FACE_COLORS)

    # 下限 10：占比最大能到 100.000（7 字符），标签短时列宽必须由数字定，
    # 否则各列会挤成一片（"99.21887.180..."）。
    width = max(max(len(label) for label in labels) + 2, 10)
    print()
    print("基色占比（%，已按 dataclasses.py 裁掉上下各 20 行）")
    print(f"{'':22s}" + "".join(f"{label:>{width}s}" for label in labels))
    print("-" * (22 + width * len(labels)))
    for key in rows:
        values = [report[label].get(key, 0.0) * 100 for label in labels]
        if all(v < 1e-4 for v in values):
            continue
        # 全部低于本底的行标 (噪声)，免得把振铃当成"画了这类要素"
        flag = "  (噪声)" if max(values) < NOISE_FLOOR_PCT and key not in (
            "black", "dim", "unmatched") else ""
        print(f"{key:22s}" + "".join(f"{v:>{width}.3f}" for v in values) + flag)

    if len(labels) == 2:
        a, b = labels
        print()
        print("判读：")
        for key, note in (
            ("lanelines", "lane 多边形边界，两边语义相同，应当量级相当"),
            ("crosswalks", "人行横道，两边语义相同，应当量级相当"),
            ("road_boundaries",
             "【语义不同】nuPlan=路缘轮廓 / nuScenes=路中央 divider，预期后者更高"),
        ):
            va, vb = report[a].get(key, 0.0) * 100, report[b].get(key, 0.0) * 100
            ratio = (vb / va) if va > 1e-6 else float("inf")
            print(f"  {key:18s} {a}={va:.3f}%  {b}={vb:.3f}%  比值={ratio:.2f}x   {note}")
        for label in labels:
            um = report[label].get("unmatched", 0.0) * 100
            if um > UNMATCHED_ALARM_PCT:
                print(f"  ⚠ {label} 有 {um:.2f}% 像素不在已知调色板上（JPEG 振铃本底"
                      f"约 1-3%，超过 {UNMATCHED_ALARM_PCT:.0f}% 才异常）—— "
                      f"多半是 observe 加了新的绘制分支")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"files": counts, "share": report}, indent=2, sort_keys=True))
        print(f"\n写出 {args.json_out}")


if __name__ == "__main__":
    main()
