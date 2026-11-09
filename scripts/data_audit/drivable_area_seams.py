#!/usr/bin/env python3
"""查清 nuScenes drivable_area 的相邻多边形之间是怎么共享边界的。

只读脚本：只读 maps/expansion/*.json，不写数据、不改配置。

=============================================================================
为什么要查这个
=============================================================================
nuScenes 的一条 drivable_area 记录带的是 polygon_tokens（复数）——区域由
多块相邻多边形拼成。相邻两块共享的那条边是【内部接缝】，不是区域外边界。

nuPlan 侧靠 unary_union 先把多边形焊成一块再取 boundary，接缝自然消失。
nuscenes_adapter 目前直接取每块的环，接缝被当成边界画了出来 ——
实测 38.2% 的边界采样点两侧都有车道（scripts/data_audit/boundary_semantics.py），
量级与接缝占比吻合。

修法取决于一件事：**相邻多边形共不共用同一批 node token**。

    共用   -> 每条无向边数出现次数，出现 >=2 次的是接缝，丢掉即可。
              精确、零新依赖、O(E)。
    不共用 -> 边计数失效（同一位置有多个 token），得几何去重，
              或直接上 shapely.unary_union（仓库里本来就有 shapely）。

本脚本同时测这两件事，不做任何猜测：
  1. 无向边的出现次数分布（按条数和按长度两种口径）
  2. 坐标重合但 token 不同的节点有多少 —— 这是判断"共不共用"的直接证据
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 判定两个节点"坐标重合"的容差（米）。nuScenes 地图坐标是米制。
COINCIDENT_TOL = 1e-6


def rings_of(data: dict) -> List[Tuple[str, List[str]]]:
    """返回 [(ring_id, node_tokens), ...]，含每块多边形的外环与内环。"""
    polygons = {r["token"]: r for r in data["polygon"]}
    out: List[Tuple[str, List[str]]] = []
    for record in data.get("drivable_area", []):
        for poly_token in record.get("polygon_tokens", []):
            poly = polygons.get(poly_token)
            if poly is None:
                continue
            out.append((f"{poly_token}/ext", list(poly["exterior_node_tokens"])))
            for index, hole in enumerate(poly.get("holes") or []):
                out.append((f"{poly_token}/hole{index}", list(hole.get("node_tokens", []))))
    return out


def _regions(data: dict, nodes: Dict[str, Tuple[float, float]]):
    """把每块 drivable_area 多边形还原成 (外环, [内环...], bbox)。

    含洞的多边形：点在区域内 <=> 在外环内 且 不在任何内环内。
    """
    polygons = {r["token"]: r for r in data["polygon"]}
    out = []
    for record in data.get("drivable_area", []):
        for poly_token in record.get("polygon_tokens", []):
            poly = polygons.get(poly_token)
            if poly is None:
                continue
            ext = [nodes[t] for t in poly["exterior_node_tokens"] if t in nodes]
            if len(ext) < 3:
                continue
            ext_arr = np.asarray(ext, dtype=np.float64)
            holes = []
            for hole in poly.get("holes") or []:
                pts = [nodes[t] for t in hole.get("node_tokens", []) if t in nodes]
                if len(pts) >= 3:
                    holes.append(np.asarray(pts, dtype=np.float64))
            out.append((
                poly_token, ext_arr, holes,
                (ext_arr[:, 0].min(), ext_arr[:, 1].min(),
                 ext_arr[:, 0].max(), ext_arr[:, 1].max()),
            ))
    return out


def _inside(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """射线法。落在边上的点结果不确定 —— 这正是为什么共享边要靠边计数
    单独处理，而这里只用来抓【面积重叠】。"""
    x, y = pts[:, None, 0], pts[:, None, 1]
    xi, yi = poly[None, :, 0], poly[None, :, 1]
    rolled = np.roll(poly, -1, axis=0)
    xj, yj = rolled[None, :, 0], rolled[None, :, 1]
    straddles = (yi > y) != (yj > y)
    denom = np.where(np.abs(yj - yi) < 1e-12, 1e-12, yj - yi)
    x_cross = xi + (y - yi) * (xj - xi) / denom
    return (straddles & (x < x_cross)).sum(axis=1) % 2 == 1


def measure_overlap(data: dict, nodes: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """边界边的中点有多少落在【另一块】多边形内部。

    ⚠️ 这个数【包含】共享边，不是纯粹的重叠指标。共享边的中点恰好落在两块的
    边界线上，射线法对这种点的结果不稳定，实测会被判成 "inside"
    （合成验证：纯贴边共享的两块报 12.5%，而真实重叠的两块报 37.5%）。

    所以要这样读：

        面积重叠 ≈ 本指标 − 共享边占比

    对 boston-seaport（共享边仅 2.7%）这个减法基本可以忽略，读数直接就是重叠；
    对 hollandvillage / queenstown（共享边 80-89%）必须先减掉才有意义。

    区分这两者很关键：共享边能用边计数消掉（已实现），面积重叠不能 ——
    那部分只有真正的 union 才处理得了。
    """
    regions = _regions(data, nodes)
    if len(regions) < 2:
        return {"midpoints": 0, "inside_other_pct": 0.0, "length_inside_other_pct": 0.0,
                "regions": len(regions)}

    total = 0
    inside = 0
    total_len = 0.0
    inside_len = 0.0
    for index, (_, ext, holes, _) in enumerate(regions):
        rings = [ext] + holes
        for ring in rings:
            closed = np.vstack([ring, ring[:1]])
            mids = 0.5 * (closed[:-1] + closed[1:])
            lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            hit = np.zeros(len(mids), dtype=bool)
            for other, (_, o_ext, o_holes, bbox) in enumerate(regions):
                if other == index:
                    continue
                cand = ~hit
                cand &= ((mids[:, 0] >= bbox[0]) & (mids[:, 1] >= bbox[1])
                         & (mids[:, 0] <= bbox[2]) & (mids[:, 1] <= bbox[3]))
                if not cand.any():
                    continue
                verdict = _inside(mids[cand], o_ext)
                for o_hole in o_holes:
                    verdict &= ~_inside(mids[cand], o_hole)
                hit[cand] = verdict
            total += len(mids)
            inside += int(hit.sum())
            total_len += float(lens.sum())
            inside_len += float(lens[hit].sum())
    return {
        "regions": len(regions),
        "midpoints": total,
        "inside_other_pct": 100.0 * inside / max(total, 1),
        "length_inside_other_pct": 100.0 * inside_len / max(total_len, 1e-9),
    }


def analyse(path: Path, with_overlap: bool = False) -> Dict[str, object]:
    data = json.loads(path.read_text())
    nodes = {r["token"]: (float(r["x"]), float(r["y"])) for r in data["node"]}
    rings = rings_of(data)

    # --- 1) 无向边的出现次数 -------------------------------------------------
    edge_count: Counter = Counter()
    edge_length: Dict[Tuple[str, str], float] = {}
    edge_rings: Dict[Tuple[str, str], set] = defaultdict(set)
    for ring_id, tokens in rings:
        if len(tokens) < 2:
            continue
        closed = tokens + [tokens[0]] if tokens[0] != tokens[-1] else tokens
        for a, b in zip(closed, closed[1:]):
            if a == b or a not in nodes or b not in nodes:
                continue
            key = (a, b) if a < b else (b, a)
            edge_count[key] += 1
            edge_rings[key].add(ring_id.split("/")[0])
            if key not in edge_length:
                (x0, y0), (x1, y1) = nodes[a], nodes[b]
                edge_length[key] = math.hypot(x1 - x0, y1 - y0)

    by_multiplicity: Counter = Counter()
    length_by_multiplicity: Dict[int, float] = defaultdict(float)
    shared_across_polygons = 0
    for key, count in edge_count.items():
        bucket = min(count, 3)  # 3 表示 ">=3"
        by_multiplicity[bucket] += 1
        length_by_multiplicity[bucket] += edge_length[key]
        if count >= 2 and len(edge_rings[key]) >= 2:
            shared_across_polygons += 1

    total_edges = max(sum(by_multiplicity.values()), 1)
    total_length = max(sum(length_by_multiplicity.values()), 1e-9)

    # --- 2) 坐标重合但 token 不同的节点 --------------------------------------
    # 只统计参与了 drivable_area 环的节点 —— 其它层的节点与本问题无关。
    used = {t for _, tokens in rings for t in tokens if t in nodes}
    buckets: Dict[Tuple[int, int], set] = defaultdict(set)
    scale = 1.0 / COINCIDENT_TOL
    for token in used:
        x, y = nodes[token]
        buckets[(int(round(x * scale)), int(round(y * scale)))].add(token)
    duplicated_positions = sum(1 for s in buckets.values() if len(s) > 1)

    overlap = (measure_overlap(data, nodes) if with_overlap
               else {"regions": 0, "midpoints": 0,
                     "inside_other_pct": 0.0, "length_inside_other_pct": 0.0})

    return {
        "overlap_midpoint_pct": overlap["inside_other_pct"],
        "overlap_length_pct": overlap["length_inside_other_pct"],
        "rings": len(rings),
        "polygons": len({r.split("/")[0] for r, _ in rings}),
        "nodes_used": len(used),
        "edges_unique": total_edges,
        "edges_once_pct": 100.0 * by_multiplicity[1] / total_edges,
        "edges_twice_pct": 100.0 * by_multiplicity[2] / total_edges,
        "edges_3plus_pct": 100.0 * by_multiplicity[3] / total_edges,
        "length_once_pct": 100.0 * length_by_multiplicity[1] / total_length,
        "length_shared_pct": 100.0
        * (length_by_multiplicity[2] + length_by_multiplicity[3]) / total_length,
        "shared_across_polygons": shared_across_polygons,
        "coincident_positions": duplicated_positions,
        "coincident_pct": 100.0 * duplicated_positions / max(len(used), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map-root", type=Path, default=None,
                        help="默认 $NUSCENES_ROOT/maps/expansion")
    parser.add_argument("--locations", nargs="*", default=None)
    parser.add_argument("--check-overlap", action="store_true",
                        help="额外测【面积重叠】：边界边中点落在另一块多边形内部的比例。"
                             "边计数只能消共享边，消不掉重叠 —— 这一项才是"
                             "boston/onenorth 那 25%% divider 的候选解释。")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    map_root = args.map_root
    if map_root is None:
        root = os.environ.get("NUSCENES_ROOT")
        if not root:
            raise SystemExit("FAIL 请设 NUSCENES_ROOT 或传 --map-root")
        map_root = Path(root) / "maps" / "expansion"
    if not map_root.is_dir():
        raise SystemExit(f"FAIL 地图目录不存在: {map_root}")

    locations = args.locations or sorted(p.stem for p in map_root.glob("*.json"))
    if not locations:
        raise SystemExit(f"FAIL {map_root} 下没有 json")

    report: Dict[str, Dict[str, object]] = {}
    print(f"{'location':26s}{'polys':>7s}{'rings':>7s}{'edges':>9s}"
          f"{'出现1次':>9s}{'出现2次':>9s}{'>=3次':>8s}{'接缝长度占比':>14s}")
    print("-" * 89)
    for location in locations:
        stats = analyse(map_root / f"{location}.json", args.check_overlap)
        report[location] = stats
        print(f"{location:26s}{stats['polygons']:>7d}{stats['rings']:>7d}"
              f"{stats['edges_unique']:>9d}"
              f"{stats['edges_once_pct']:>8.1f}%{stats['edges_twice_pct']:>8.1f}%"
              f"{stats['edges_3plus_pct']:>7.1f}%{stats['length_shared_pct']:>13.1f}%")

    if args.check_overlap:
        print("\n面积重叠（边计数消不掉的那部分，需要 union 才能处理）：")
        for location, stats in report.items():
            print(f"  {location:26s} 中点落在他块内部 "
                  f"{stats['overlap_midpoint_pct']:6.2f}%  "
                  f"按长度 {stats['overlap_length_pct']:6.2f}%")

    print("\n坐标重合但 token 不同的节点（判断能否用边计数的直接证据）：")
    for location, stats in report.items():
        print(f"  {location:26s} {stats['coincident_positions']:>7d} 处 "
              f"/ {stats['nodes_used']} 个节点 ({stats['coincident_pct']:.2f}%)")

    shared = sum(float(s["length_shared_pct"]) for s in report.values()) / len(report)
    coincident = sum(float(s["coincident_pct"]) for s in report.values()) / len(report)
    print("\n结论：")
    if shared < 1.0 and coincident < 1.0:
        print("  几乎没有共享边、也几乎没有坐标重合的异 token 节点 ——")
        print("  相邻多边形【不接触】，38% divider 另有原因，本假设不成立。")
    elif shared >= 1.0:
        print(f"  共享边占总边界长度 {shared:.1f}% —— 相邻多边形确实共用同一批 node")
        print("  token，内部接缝被当成边界画了出来。")
        print("  修法：在 _build 里只保留【出现 1 次】的无向边，出现 >=2 次的丢掉。")
        print("  精确、零新依赖，等价于 unary_union 后取 boundary。")
    else:
        print(f"  共享边很少（{shared:.1f}%）但有 {coincident:.2f}% 的节点坐标重合 ——")
        print("  相邻多边形接触但各用各的 token，边计数失效。")
        print("  修法：先按坐标把重合节点合并成同一个 id，再做边计数；")
        print("  或直接上 shapely.unary_union（仓库里已有 shapely）。")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\n写出 {args.json_out}")


if __name__ == "__main__":
    main()
