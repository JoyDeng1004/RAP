#!/usr/bin/env python3
"""判别 nuScenes 侧 BOUNDARY 分支装的到底是"分道线"还是"路缘"。

只读脚本：只读 _index/_maps/*.npz，不渲染、不需要 nuPlan 环境、不写数据。

=============================================================================
要回答什么
=============================================================================
renderer.py:750 的 BOUNDARY/SOLID 分支把要素画成 radius=10 的深红粗线。
两条链路往这个分支里塞的东西语义不同：

    nuPlan   LINE_SOLID_SINGLE_WHITE = intersection ∪ roadblock 多边形并集的
             【外轮廓】。按构造它一定在所有可行驶区域【之外】，只有一侧有路。
    nuScenes （v3 起）drivable_area 区域去掉内部接缝后的外轮廓，语义已对齐。
             v1 装的是 lane_divider + road_divider（路中央的分道线，语义相反）。

判别标准：**线的两侧是不是都有 lane**。

⚠️ 这个判据有一个【已知的假阳性模式】，读数前必须知道：
   偏移取 1.2-2.4m 且"任一命中即算"，所以任何窄于约 2*max(offsets) 的
   非可行驶条带（中央分隔带、交通岛）都会让采样点跨过去落到对面车道，
   把一条真边界判成 divider。这类条带在 nuScenes 里以 drivable_area 的
   【内环/hole】出现。同样的几何在 nuPlan 的 union boundary 上也会中招，
   所以 divider% 的绝对值不该拿来和 0 比 —— 要用 --by-source 分环型看：
   exterior 的 divider 才是真问题，hole 的 divider 多半是这个假阳性。

=============================================================================
怎么判
=============================================================================
沿每条 BOUNDARY 线按 --step 米重采样，在每个采样点取局部切向，
朝左右法向各偏移 --offsets 里的若干距离，检查偏移点是否落在某个 lane 多边形内：

    两侧都有 lane -> divider（分道线）
    只有一侧      -> edge（路缘/路肩）
    两侧都没有    -> detached（离路面较远，多半是停车场边界之类）

偏移距离要覆盖典型车道半宽。nuScenes 城区车道宽约 3-3.5m，所以默认
1.2/1.8/2.4m —— 任一距离命中就算该侧有路，避免车道窄时误判。

=============================================================================
判读
=============================================================================
divider 占压倒多数 -> 证实"nuScenes 的 BOUNDARY 装的是分道线"，与 nuPlan 的
                     路缘语义不一致。要对齐，应把该分支换成 drivable_area /
                     road_segment 多边形并集的外轮廓，divider 整个不画
                     （nuPlan 侧的虚线分隔线 ROAD_LINE_BROKEN_SINGLE_WHITE
                      本来就不命中任何分支）。
edge 占多数        -> 语义其实已经接近 nuPlan，这条不一致不成立，之前的判断要推翻。

本脚本只给事实，不改任何东西。现状被
tests/rasterization/test_render_contract.py::test_boundary_semantics_divergence_is_pinned
钉住，改 MapCache 时那个测试会失败。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers.nuscenes_adapter import MapCache  # noqa: E402


def resample(points: np.ndarray, step: float) -> np.ndarray:
    """按弧长等距重采样一条折线。点少于 2 个时原样返回。

    不重采样直接用原始节点会有偏差：nuScenes 的 line 节点密度不均匀，
    弯道处密、直道处疏，不重采样等于给弯道更高的权重。
    """
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total < step:
        return points[[0, -1]]
    targets = np.arange(0.0, total, step)
    out = np.empty((len(targets), 2), dtype=np.float64)
    for axis in (0, 1):
        out[:, axis] = np.interp(targets, arc, points[:, axis])
    return out


def tangents(points: np.ndarray) -> np.ndarray:
    """每个采样点的单位切向（中心差分，端点用单边）。"""
    diff = np.gradient(points, axis=0)
    norm = np.linalg.norm(diff, axis=1, keepdims=True)
    return diff / np.where(norm < 1e-9, 1.0, norm)


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """射线法，向量化到 (N 点 × M 边)。返回 (N,) bool。"""
    x = pts[:, None, 0]
    y = pts[:, None, 1]
    xi, yi = poly[None, :, 0], poly[None, :, 1]
    rolled = np.roll(poly, -1, axis=0)
    xj, yj = rolled[None, :, 0], rolled[None, :, 1]
    straddles = (yi > y) != (yj > y)
    denom = np.where(np.abs(yj - yi) < 1e-12, 1e-12, yj - yi)
    x_cross = xi + (y - yi) * (xj - xi) / denom
    return (straddles & (x < x_cross)).sum(axis=1) % 2 == 1


class LaneIndex:
    """lane 多边形的质心+半径预筛，避免每个采样点扫全图。"""

    def __init__(self, layer):
        self.polys = [
            layer.points[layer.offsets[i]: layer.offsets[i + 1]]
            for i in range(len(layer.ids))
        ]
        self.centroids = layer.centroids
        self.radii = layer.radii

    def which_contains(self, pts: np.ndarray) -> np.ndarray:
        """(N,2) -> (N,) int：命中的 lane 多边形下标，没命中给 -1。

        返回下标而不是 bool，是为了能区分两种 divider：
          左右命中【同一个】多边形 -> 边界线横穿了一片车道
                                     （drivable_area 与 lane 两层标注错位）
          左右命中【不同】多边形   -> 线确实夹在两条车道之间（真 divider 形态）
        这两者的成因和处置完全不同，混在一个数字里就看不出来了。
        """
        owner = np.full(len(pts), -1, dtype=np.int64)
        if not self.polys:
            return owner
        for index, poly in enumerate(self.polys):
            todo = owner < 0
            if not todo.any():
                break
            # 粗筛：点到该多边形质心的距离超过外接半径就不可能在里面
            near = todo.copy()
            near[todo] = (
                np.linalg.norm(pts[todo] - self.centroids[index], axis=1)
                <= self.radii[index] + 1e-6
            )
            if not near.any():
                continue
            inside = points_in_polygon(pts[near], poly)
            idx = np.nonzero(near)[0][inside]
            owner[idx] = index
        return owner


def classify(
    layers, step: float, offsets: List[float], max_lines: Optional[int],
    seed: int = 20260917,
) -> Tuple[Counter, Dict[str, int], int]:
    lane_index = LaneIndex(layers["lane"])
    boundary = layers["boundary"]

    line_ids = list(range(len(boundary.ids)))
    if max_lines and len(line_ids) > max_lines:
        # 必须随机抽，不能取前 N 条。索引顺序就是 expansion json 的文件顺序，
        # 空间上高度聚集 —— 取前 N 条等于只看地图的一角，会让不同 location
        # 的数字不可比（boston-seaport 线最多、截断最重，最容易被带偏）。
        line_ids = sorted(random.Random(seed).sample(line_ids, max_lines))

    verdicts: Counter = Counter()
    by_source: Dict[str, Counter] = {}
    n_samples = 0

    for i in line_ids:
        pts = boundary.points[boundary.offsets[i]: boundary.offsets[i + 1]]
        pts = resample(np.asarray(pts, dtype=np.float64), step)
        if len(pts) < 2:
            continue
        tan = tangents(pts)
        normal = np.stack([-tan[:, 1], tan[:, 0]], axis=1)  # 左法向

        left = np.full(len(pts), -1, dtype=np.int64)
        right = np.full(len(pts), -1, dtype=np.int64)
        for distance in offsets:
            for side, sign in ((left, 1.0), (right, -1.0)):
                todo = side < 0
                if not todo.any():
                    continue
                side[todo] = lane_index.which_contains(pts[todo] + sign * normal[todo] * distance)

        n_samples += len(pts)
        # 外环 vs 内环分开统计。内环是区域里挖掉的洞 —— 交通岛、中央分隔带。
        # 洞窄于约 2*max(offsets) 时，判别器朝两侧偏移会直接跨过去落到对面
        # 车道里，把一条真边界误判成 divider。这是判别器【已知的假阳性模式】，
        # 同样的窄分隔带在 nuPlan 的 union boundary 上也会中招 ——
        # 所以 divider% 的绝对值不该拿来和 0 比，要分环型看。
        fid = boundary.ids[i]
        source = f"{fid.split(':', 1)[0]}/{'hole' if ':hole' in fid else 'exterior'}"
        bucket = by_source.setdefault(source, Counter())
        for l, r in zip(left, right):
            if l >= 0 and r >= 0:
                # 同一片车道 = 边界横穿车道，两层标注错位；不同片 = 真夹在中间
                verdict = "divider_same_lane" if l == r else "divider_two_lanes"
            elif l >= 0 or r >= 0:
                verdict = "edge"
            else:
                verdict = "detached"
            verdicts[verdict] += 1
            bucket[verdict] += 1

    flat = {f"{src}/{k}": v for src, c in by_source.items() for k, v in c.items()}
    return verdicts, flat, n_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index-dir", type=Path,
                        default=Path("dataset_nuscenes/_index"))
    parser.add_argument("--locations", nargs="*", default=None,
                        help="默认跑 _maps/ 下的全部 location")
    parser.add_argument("--step", type=float, default=2.0, help="沿线重采样间隔（米）")
    parser.add_argument("--offsets", default="1.2,1.8,2.4",
                        help="左右法向偏移距离（米），任一命中即算该侧有路")
    parser.add_argument("--max-lines", type=int, default=400,
                        help="每个 location 最多取 N 条线（0=全部）")
    parser.add_argument("--seed", type=int, default=20260917,
                        help="--max-lines 抽样用；改它可看结果稳不稳")
    parser.add_argument("--by-source", action="store_true",
                        help="按环型（exterior / hole）分开报。"
                             "hole 是区域里挖掉的洞（交通岛、中央分隔带），"
                             "窄洞会触发判据的假阳性，必须和外轮廓分开看")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    pack_dir = args.index_dir / "_maps"
    if not pack_dir.is_dir():
        raise SystemExit(f"FAIL 找不到打包地图: {pack_dir}")
    locations = args.locations or sorted(p.stem for p in pack_dir.glob("*.npz"))
    if not locations:
        raise SystemExit(f"FAIL {pack_dir} 下没有 npz")
    offsets = [float(v) for v in args.offsets.split(",") if v.strip()]

    # map_root 传 pack_dir 只是占位：有 npz 就不会去读原始 expansion json
    cache = MapCache(pack_dir, pack_dir=pack_dir)

    report: Dict[str, Dict] = {}
    grand: Counter = Counter()
    print(f"step={args.step}m  offsets={offsets}m  max_lines={args.max_lines or '全部'}\n")
    print(f"{'location':26s}{'lines':>7s}{'samples':>9s}"
          f"{'同片车道':>11s}{'跨两车道':>11s}{'edge':>9s}{'detached':>10s}")
    print("-" * 83)
    for location in locations:
        layers = cache.get(location)
        verdicts, by_source, n_samples = classify(
            layers, args.step, offsets, args.max_lines or None, args.seed
        )
        total = max(sum(verdicts.values()), 1)
        grand.update(verdicts)
        report[location] = {
            "n_boundary_lines": len(layers["boundary"].ids),
            "n_lane_polygons": len(layers["lane"].ids),
            "n_samples": n_samples,
            "counts": dict(verdicts),
            "by_source": by_source,
        }
        print(f"{location:26s}{len(layers['boundary'].ids):>7d}{n_samples:>9d}"
              f"{100.0 * verdicts['divider_same_lane'] / total:>10.1f}%"
              f"{100.0 * verdicts['divider_two_lanes'] / total:>10.1f}%"
              f"{100.0 * verdicts['edge'] / total:>8.1f}%"
              f"{100.0 * verdicts['detached'] / total:>9.1f}%")
        if args.by_source:
            sources = sorted({k.rsplit("/", 1)[0] for k in by_source})
            for src in sources:
                sub = {k.rsplit("/", 1)[1]: v for k, v in by_source.items()
                       if k.rsplit("/", 1)[0] == src}
                sub_total = max(sum(sub.values()), 1)
                print(f"  └ {src:22.22s}{'':>7s}{sub_total:>9d}"
                      f"{100.0 * sub.get('divider_same_lane', 0) / sub_total:>10.1f}%"
                      f"{100.0 * sub.get('divider_two_lanes', 0) / sub_total:>10.1f}%"
                      f"{100.0 * sub.get('edge', 0) / sub_total:>8.1f}%"
                      f"{100.0 * sub.get('detached', 0) / sub_total:>9.1f}%")

    total = max(sum(grand.values()), 1)
    same_pct = 100.0 * grand["divider_same_lane"] / total
    two_pct = 100.0 * grand["divider_two_lanes"] / total
    divider_pct = same_pct + two_pct
    edge_pct = 100.0 * grand["edge"] / total
    print("-" * 83)
    print(f"{'合计':24s}{'':>7s}{total:>9d}"
          f"{same_pct:>10.1f}%{two_pct:>10.1f}%{edge_pct:>8.1f}%"
          f"{100.0 * grand['detached'] / total:>9.1f}%")

    print("\n结论：")
    print(f"  两侧都有车道的点共 {divider_pct:.1f}%，拆开看：")
    print(f"    同一片车道 {same_pct:.1f}% —— 边界线【横穿】一片 lane 多边形。")
    print("      这是 nuScenes 的 drivable_area 与 lane 两层标注不贴合，")
    print("      不是取层逻辑的问题，我们这边改不动。")
    print(f"    跨两片车道 {two_pct:.1f}% —— 线确实夹在两条车道【之间】。")
    print("      这一项才是取层问题：说明 BOUNDARY 里还混着分道线形态的几何。")
    if two_pct < 5.0:
        print("  -> 跨两车道的比例很低，BOUNDARY 的语义已与 nuPlan 对齐；")
        print("     剩余部分是两层标注错位，记为已知差异即可。")
    else:
        print("  -> 跨两车道的比例仍然可观，取层还有残留问题，值得继续查。")
    print(f"  edge {edge_pct:.1f}% + detached {100.0 * grand['detached'] / total:.1f}%"
          f" = {edge_pct + 100.0 * grand['detached'] / total:.1f}% 是形态正确的边界。")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"step": args.step, "offsets": offsets, "per_location": report,
             "total": dict(grand)}, indent=2, sort_keys=True))
        print(f"\n写出 {args.json_out}")


if __name__ == "__main__":
    main()
