#!/usr/bin/env python3
"""WOMD scenario 探针：写 WOMD→RAP 适配器之前，先从真实数据确认字段。

只读脚本：只读 $WOMD_ROOT/scenario/<split>/*.tfrecord*，只写 --out 指定的 JSON。
不依赖 tensorflow / waymo-open-dataset：TFRecord 和 Scenario proto 都在本文件里
解析，只要 protobuf>=4（rap 环境已有 4.25.3）+ numpy。

=============================================================================
要回答什么
=============================================================================
1. 体积疑点：validation 每秒数据量约为 training_20s 的 2 倍。按顶层字段号统计
   字节占比（含 schema 外的未知字段号），看多出来的是哪一块。
   注意：v1.3.1 的 sdc_paths 不在 scenario proto 里，而在 tf_example 格式的
   path_samples/{xyz,valid,id,arc_length,on_route}（Waymax WOD_1_3_1_* 配置，
   45 条 × 800 点），所以本探针不会、也不该在这里看到它。
2. 地图能否直接喂 renderer.py:734 的三个分支：
   - LANE      读 polygon；WOMD 的 LaneCenter 只有中心线 polyline（schema 如此），
               这里统计左右边界引用的覆盖率，决定能不能由边界拼出车道。
   - CROSSWALK / SPEED_BUMP 读 polygon；WOMD 有。
   - BOUNDARY/SOLID 读 polyline；road_line 的 SOLID_* 类型会被子串误判成路沿，
               所以要看 road_line 各类型的占比。
3. 自车：sdc_track 有效率、尺寸、速度；按 0.5s 步长能切出多少个
   「当前帧 + 前一帧(算加速度) + 4s 未来全有效」的锚点。
4. 平地假设：renderer 把地图压到 z=0。统计自车周围地图点相对自车接地点的
   高差，估计坡地场景占比。
5. 红绿灯：动态状态的覆盖、状态分布、stop_point 是否齐全。
6. 地图覆盖：20s 片段末尾的锚点处，周围还有没有地图（地图是否只裁到开头）。

=============================================================================
用法
=============================================================================
    python scripts/data_audit/probe_womd.py \
        --root /gs/bs/tga-RLA/qdeng/womd_dataset/v_1_3_1 \
        --splits training_20s validation --shards 3 --max-scenarios 200

结果打印到终端，同时写 $WOMD_ROOT/_probe/probe_<split>.json。
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool

# =============================================================================
# Scenario / Map schema（摘自 waymo-open-dataset master 的 scenario.proto 与
# map.proto，2026-10-28 核对）。只声明探针用到的字段：其余字段（lidar 12、
# camera tokens 13、lane neighbors 11/12 等）按 proto2 规则进 unknown fields，
# 解析不受影响。enum 一律按 int32 声明，名字在下面的表里映射。
# =============================================================================

D, F, I64, I32, B, S, M = 1, 2, 3, 5, 8, 9, 11  # FieldDescriptorProto.Type
OPT, REP = 1, 3

SCHEMA = {
    "MapPoint": [("x", 1, D, OPT), ("y", 2, D, OPT), ("z", 3, D, OPT)],
    "TrafficSignalLaneState": [
        ("lane", 1, I64, OPT), ("state", 2, I32, OPT),
        ("stop_point", 3, "MapPoint", OPT),
    ],
    "DynamicMapState": [("lane_states", 1, "TrafficSignalLaneState", REP)],
    "BoundarySegment": [
        ("lane_start_index", 1, I32, OPT), ("lane_end_index", 2, I32, OPT),
        ("boundary_feature_id", 3, I64, OPT), ("boundary_type", 4, I32, OPT),
    ],
    "LaneCenter": [
        ("speed_limit_mph", 1, D, OPT), ("type", 2, I32, OPT),
        ("interpolating", 3, B, OPT), ("polyline", 8, "MapPoint", REP),
        ("entry_lanes", 9, I64, REP), ("exit_lanes", 10, I64, REP),
        ("left_boundaries", 13, "BoundarySegment", REP),
        ("right_boundaries", 14, "BoundarySegment", REP),
    ],
    "RoadEdge": [("type", 1, I32, OPT), ("polyline", 2, "MapPoint", REP)],
    "RoadLine": [("type", 1, I32, OPT), ("polyline", 2, "MapPoint", REP)],
    "StopSign": [("lane", 1, I64, REP), ("position", 2, "MapPoint", OPT)],
    "Crosswalk": [("polygon", 1, "MapPoint", REP)],
    "SpeedBump": [("polygon", 1, "MapPoint", REP)],
    "Driveway": [("polygon", 1, "MapPoint", REP)],
    "MapFeature": [
        ("id", 1, I64, OPT), ("lane", 3, "LaneCenter", OPT),
        ("road_line", 4, "RoadLine", OPT), ("road_edge", 5, "RoadEdge", OPT),
        ("stop_sign", 7, "StopSign", OPT), ("crosswalk", 8, "Crosswalk", OPT),
        ("speed_bump", 9, "SpeedBump", OPT), ("driveway", 10, "Driveway", OPT),
    ],
    "ObjectState": [
        ("center_x", 2, D, OPT), ("center_y", 3, D, OPT), ("center_z", 4, D, OPT),
        ("length", 5, F, OPT), ("width", 6, F, OPT), ("height", 7, F, OPT),
        ("heading", 8, F, OPT), ("velocity_x", 9, F, OPT),
        ("velocity_y", 10, F, OPT), ("valid", 11, B, OPT),
    ],
    "Track": [
        ("id", 1, I32, OPT), ("object_type", 2, I32, OPT),
        ("states", 3, "ObjectState", REP),
    ],
    "RequiredPrediction": [("track_index", 1, I32, OPT), ("difficulty", 2, I32, OPT)],
    "Scenario": [
        ("timestamps_seconds", 1, D, REP), ("tracks", 2, "Track", REP),
        ("objects_of_interest", 4, I32, REP), ("scenario_id", 5, S, OPT),
        ("sdc_track_index", 6, I32, OPT),
        ("dynamic_map_states", 7, "DynamicMapState", REP),
        ("map_features", 8, "MapFeature", REP),
        ("current_time_index", 10, I32, OPT),
        ("tracks_to_predict", 11, "RequiredPrediction", REP),
    ],
}

# Scenario 顶层字段号 -> 名字（含本探针没声明的，用于体积统计）。
TOP_LEVEL_NAMES = {
    1: "timestamps_seconds", 2: "tracks", 4: "objects_of_interest",
    5: "scenario_id", 6: "sdc_track_index", 7: "dynamic_map_states",
    8: "map_features", 10: "current_time_index", 11: "tracks_to_predict",
    12: "compressed_frame_laser_data", 13: "frame_camera_tokens",
}

TRACK_TYPE = {0: "UNSET", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}
LANE_TYPE = {0: "UNDEFINED", 1: "FREEWAY", 2: "SURFACE_STREET", 3: "BIKE_LANE"}
ROAD_EDGE_TYPE = {0: "UNKNOWN", 1: "ROAD_EDGE_BOUNDARY", 2: "ROAD_EDGE_MEDIAN"}
ROAD_LINE_TYPE = {
    0: "UNKNOWN", 1: "BROKEN_SINGLE_WHITE", 2: "SOLID_SINGLE_WHITE",
    3: "SOLID_DOUBLE_WHITE", 4: "BROKEN_SINGLE_YELLOW", 5: "BROKEN_DOUBLE_YELLOW",
    6: "SOLID_SINGLE_YELLOW", 7: "SOLID_DOUBLE_YELLOW", 8: "PASSING_DOUBLE_YELLOW",
}
SIGNAL_STATE = {
    0: "UNKNOWN", 1: "ARROW_STOP", 2: "ARROW_CAUTION", 3: "ARROW_GO",
    4: "STOP", 5: "CAUTION", 6: "GO", 7: "FLASHING_STOP", 8: "FLASHING_CAUTION",
}
MAP_KINDS = ("lane", "road_line", "road_edge", "stop_sign", "crosswalk",
             "speed_bump", "driveway")

PACKAGE = "womd_probe"


def build_scenario_class():
    fdp = descriptor_pb2.FileDescriptorProto(
        name="womd_probe.proto", package=PACKAGE, syntax="proto2")
    for msg_name, fields in SCHEMA.items():
        msg = fdp.message_type.add(name=msg_name)
        for name, number, ftype, label in fields:
            f = msg.field.add(name=name, number=number, label=label)
            if isinstance(ftype, str):
                f.type = M
                f.type_name = f".{PACKAGE}.{ftype}"
            else:
                f.type = ftype
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    desc = pool.FindMessageTypeByName(f"{PACKAGE}.Scenario")
    try:
        from google.protobuf import message_factory
        return message_factory.GetMessageClass(desc)
    except AttributeError:  # protobuf < 4.21
        from google.protobuf import message_factory
        return message_factory.MessageFactory(pool).GetPrototype(desc)


Scenario = build_scenario_class()


# =============================================================================
# TFRecord：<uint64 len><uint32 crc(len)><data><uint32 crc(data)>。不校验 CRC：
# 下载器已逐 shard 对过尺寸，探针只读少量 shard。
# =============================================================================

def iter_tfrecord(path: Path) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        while True:
            head = fh.read(12)
            if not head:
                return
            if len(head) < 12:
                raise IOError(f"{path}: 记录头被截断")
            (length,) = struct.unpack("<Q", head[:8])
            data = fh.read(length)
            if len(data) < length or len(fh.read(4)) < 4:
                raise IOError(f"{path}: 记录体被截断")
            yield data


def read_varint(buf: bytes, pos: int):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def top_level_bytes(buf: bytes) -> Dict[int, int]:
    """Scenario 顶层每个字段号占的字节数（含 key 和长度前缀）。"""
    out: Dict[int, int] = defaultdict(int)
    pos, n = 0, len(buf)
    while pos < n:
        start = pos
        key, pos = read_varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            _, pos = read_varint(buf, pos)
        elif wire == 1:
            pos += 8
        elif wire == 2:
            length, pos = read_varint(buf, pos)
            pos += length
        elif wire == 5:
            pos += 4
        else:
            raise ValueError(f"不支持的 wire type {wire}（字段 {number}）")
        out[number] += pos - start
    return out


# =============================================================================
# 统计
# =============================================================================

def pct(values, qs=(5, 50, 95)) -> Dict[str, float]:
    if len(values) == 0:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{q}": round(float(np.percentile(arr, q)), 3) for q in qs}


def points_xyz(pts) -> np.ndarray:
    return np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float64).reshape(-1, 3)


class SplitStats:
    def __init__(self, split: str):
        self.split = split
        self.n_scenarios = 0
        self.bytes_total = 0
        self.seconds_total = 0.0
        self.top_bytes: Dict[int, int] = defaultdict(int)
        self.n_steps = Counter()
        self.current_idx = Counter()
        self.dt = []
        # 自车
        self.sdc_valid_frac = []
        self.sdc_dims = defaultdict(list)
        self.sdc_speed_at_anchor = []
        self.anchors_05s = 0
        self.anchors_4s_disjoint = 0
        self.sdc_type = Counter()
        # agent
        self.track_type = Counter()
        self.dims_by_type = defaultdict(lambda: defaultdict(list))
        self.tracks_per_scenario = []
        # 地图
        self.map_kind = Counter()
        self.lane_type = Counter()
        self.road_line_type = Counter()
        self.road_edge_type = Counter()
        self.lane_has_left = 0
        self.lane_has_right = 0
        self.lane_spacing = []
        self.lane_pts_per_feature = []
        self.map_features_per_scenario = []
        self.map_radius = []  # 地图点到自车当前位置的最远水平距离
        # 平地
        self.dz_near = []   # 0-30m 内地图点相对自车接地点的 |dz| 的 p95（每场景一个）
        self.dz_far = []    # 30-60m
        self.hilly_scenarios = 0
        # 地图覆盖：锚点处 50m 内的地图点数，相对首个锚点的比例
        self.cover_ratio_last_anchor = []
        # 红绿灯
        self.dyn_len_match = 0
        self.signals_per_step = []
        self.signal_state = Counter()
        self.signal_with_stop_point = 0
        self.signal_total = 0
        self.scenarios_with_signal = 0
        self.tracks_to_predict = []
        self.errors = Counter()

    def add(self, raw: bytes) -> None:
        sc = Scenario()
        sc.ParseFromString(raw)
        self.n_scenarios += 1
        self.bytes_total += len(raw)
        for k, v in top_level_bytes(raw).items():
            self.top_bytes[k] += v

        ts = np.array(sc.timestamps_seconds)
        T = len(ts)
        self.n_steps[T] += 1
        self.current_idx[sc.current_time_index] += 1
        if T > 1:
            self.dt.extend(np.diff(ts).tolist())
            self.seconds_total += float(ts[-1] - ts[0])
        self.tracks_per_scenario.append(len(sc.tracks))
        self.tracks_to_predict.append(len(sc.tracks_to_predict))

        # ---- agent ----
        for tr in sc.tracks:
            tname = TRACK_TYPE.get(tr.object_type, str(tr.object_type))
            self.track_type[tname] += 1
            valid = [s for s in tr.states if s.valid]
            if valid:
                s = valid[len(valid) // 2]
                d = self.dims_by_type[tname]
                d["length"].append(s.length)
                d["width"].append(s.width)
                d["height"].append(s.height)
                d["z_rel_ground_ok"].append(1.0 if s.height > 0 else 0.0)

        # ---- 自车 ----
        if not (0 <= sc.sdc_track_index < len(sc.tracks)):
            self.errors["sdc_track_index 越界"] += 1
            return
        sdc = sc.tracks[sc.sdc_track_index]
        self.sdc_type[TRACK_TYPE.get(sdc.object_type, str(sdc.object_type))] += 1
        st = sdc.states
        valid = np.array([s.valid for s in st], dtype=bool)
        self.sdc_valid_frac.append(float(valid.mean()) if len(valid) else 0.0)
        xyz = np.array([(s.center_x, s.center_y, s.center_z) for s in st])
        speed = np.array([math.hypot(s.velocity_x, s.velocity_y) for s in st])
        heights = np.array([s.height for s in st])
        for s in st:
            if s.valid:
                self.sdc_dims["length"].append(s.length)
                self.sdc_dims["width"].append(s.width)
                self.sdc_dims["height"].append(s.height)
                break

        # 锚点：t-1（差分加速度）、t、t+1..t+40（4s @10Hz）全有效，步长 5 帧=0.5s。
        horizon = 40
        anchors = [t for t in range(1, T - horizon, 5) if valid[t - 1:t + horizon + 1].all()]
        self.anchors_05s += len(anchors)
        last = -10**9
        for t in anchors:
            if t - last >= horizon:
                self.anchors_4s_disjoint += 1
                last = t
        self.sdc_speed_at_anchor.extend(speed[anchors].tolist())

        # ---- 地图 ----
        self.map_features_per_scenario.append(len(sc.map_features))
        lane_pts_all = []
        all_pts = []
        for mf in sc.map_features:
            kind = None  # schema 里没声明 oneof，逐个 HasField
            for k in MAP_KINDS:
                if mf.HasField(k):
                    kind = k
                    break
            if kind is None:
                self.map_kind["<empty>"] += 1
                continue
            self.map_kind[kind] += 1
            if kind == "lane":
                ln = mf.lane
                self.lane_type[LANE_TYPE.get(ln.type, str(ln.type))] += 1
                self.lane_has_left += bool(len(ln.left_boundaries))
                self.lane_has_right += bool(len(ln.right_boundaries))
                p = points_xyz(ln.polyline)
                self.lane_pts_per_feature.append(len(p))
                if len(p) > 1 and len(self.lane_spacing) < 200000:
                    self.lane_spacing.extend(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).tolist())
                lane_pts_all.append(p)
                all_pts.append(p)
            elif kind == "road_line":
                self.road_line_type[ROAD_LINE_TYPE.get(mf.road_line.type, str(mf.road_line.type))] += 1
                all_pts.append(points_xyz(mf.road_line.polyline))
            elif kind == "road_edge":
                self.road_edge_type[ROAD_EDGE_TYPE.get(mf.road_edge.type, str(mf.road_edge.type))] += 1
                all_pts.append(points_xyz(mf.road_edge.polyline))
            elif kind in ("crosswalk", "speed_bump", "driveway"):
                all_pts.append(points_xyz(getattr(mf, kind).polygon))

        cur = sc.current_time_index
        if all_pts and 0 <= cur < T and valid[cur]:
            P = np.concatenate(all_pts)
            ego = xyz[cur]
            ground_z = ego[2] - heights[cur] / 2.0  # center_z 是框中心
            dxy = np.linalg.norm(P[:, :2] - ego[None, :2], axis=1)
            self.map_radius.append(float(dxy.max()))
            dz = np.abs(P[:, 2] - ground_z)
            near, far = dz[dxy < 30], dz[(dxy >= 30) & (dxy < 60)]
            if len(near):
                self.dz_near.append(float(np.percentile(near, 95)))
            if len(far):
                p95 = float(np.percentile(far, 95))
                self.dz_far.append(p95)
                # 60m 内高差 > 1.5m：约 2.5% 以上的坡，或立交。
                self.hilly_scenarios += p95 > 1.5

        if lane_pts_all and anchors:
            L = np.concatenate(lane_pts_all)[:, :2]
            counts = [int((np.linalg.norm(L - xyz[t, None, :2], axis=1) < 50).sum()) for t in anchors]
            if counts[0] > 0:
                self.cover_ratio_last_anchor.append(counts[-1] / counts[0])

        # ---- 红绿灯 ----
        self.dyn_len_match += len(sc.dynamic_map_states) == T
        has_signal = False
        for i, dms in enumerate(sc.dynamic_map_states):
            self.signals_per_step.append(len(dms.lane_states))
            for ls in dms.lane_states:
                has_signal = True
                if i == cur:
                    self.signal_state[SIGNAL_STATE.get(ls.state, str(ls.state))] += 1
                self.signal_total += 1
                self.signal_with_stop_point += ls.HasField("stop_point")
        self.scenarios_with_signal += has_signal

    def summary(self) -> dict:
        n = max(self.n_scenarios, 1)
        tb = sum(self.top_bytes.values()) or 1
        lanes = max(self.map_kind.get("lane", 0), 1)
        rl_total = max(sum(self.road_line_type.values()), 1)
        solid = sum(v for k, v in self.road_line_type.items() if "SOLID" in k)
        return {
            "split": self.split,
            "n_scenarios": self.n_scenarios,
            "errors": dict(self.errors),
            "size": {
                "mean_MB_per_scenario": round(self.bytes_total / n / 1e6, 3),
                "mean_seconds_per_scenario": round(self.seconds_total / n, 2),
                "MB_per_second": round(self.bytes_total / max(self.seconds_total, 1e-9) / 1e6, 4),
                "top_level_share": {
                    f"{k}:{TOP_LEVEL_NAMES.get(k, '<schema 外>')}": round(v / tb, 4)
                    for k, v in sorted(self.top_bytes.items(), key=lambda kv: -kv[1])
                },
            },
            "time": {
                "n_steps": dict(self.n_steps),
                "current_time_index": dict(self.current_idx),
                "dt": pct(self.dt),
            },
            "sdc": {
                "object_type": dict(self.sdc_type),
                "valid_frac": pct(self.sdc_valid_frac),
                "dims_first_valid": {k: pct(v) for k, v in self.sdc_dims.items()},
                "anchors_0p5s_stride_total": self.anchors_05s,
                "anchors_per_scenario": round(self.anchors_05s / n, 2),
                "anchors_disjoint_4s_total": self.anchors_4s_disjoint,
                "speed_at_anchor_mps": pct(self.sdc_speed_at_anchor),
                "stationary_frac_at_anchor(<0.5m/s)": round(
                    float(np.mean(np.array(self.sdc_speed_at_anchor) < 0.5)), 3)
                if self.sdc_speed_at_anchor else None,
            },
            "agents": {
                "tracks_per_scenario": pct(self.tracks_per_scenario),
                "tracks_to_predict_per_scenario": pct(self.tracks_to_predict),
                "type": dict(self.track_type),
                "dims_by_type": {
                    t: {k: pct(v) if k != "z_rel_ground_ok" else round(float(np.mean(v)), 3)
                        for k, v in d.items()}
                    for t, d in self.dims_by_type.items()
                },
            },
            "map": {
                "features_per_scenario": pct(self.map_features_per_scenario),
                "kind": dict(self.map_kind),
                "lane_type": dict(self.lane_type),
                "lane_with_left_boundary_frac": round(self.lane_has_left / lanes, 3),
                "lane_with_right_boundary_frac": round(self.lane_has_right / lanes, 3),
                "lane_points_per_feature": pct(self.lane_pts_per_feature),
                "lane_point_spacing_m": pct(self.lane_spacing),
                "road_line_type": dict(self.road_line_type),
                "road_line_SOLID_share(会被 renderer 当路沿)": round(solid / rl_total, 3),
                "road_edge_type": dict(self.road_edge_type),
                "map_radius_from_ego_m": pct(self.map_radius),
                "lane_pts_within_50m_last/first_anchor": pct(self.cover_ratio_last_anchor),
                "frac_last_anchor_cover_below_0.3": round(
                    float(np.mean(np.array(self.cover_ratio_last_anchor) < 0.3)), 3)
                if self.cover_ratio_last_anchor else None,
            },
            "flat_ground": {
                "p95_abs_dz_0_30m": pct(self.dz_near),
                "p95_abs_dz_30_60m": pct(self.dz_far),
                "frac_scenarios_dz_far_gt_1.5m": round(self.hilly_scenarios / max(len(self.dz_far), 1), 3),
            },
            "traffic_signals": {
                "dynamic_states_len_equals_T_frac": round(self.dyn_len_match / n, 3),
                "scenarios_with_any_signal_frac": round(self.scenarios_with_signal / n, 3),
                "signals_per_step": pct(self.signals_per_step),
                "state_at_current_index": dict(self.signal_state),
                "with_stop_point_frac": round(self.signal_with_stop_point / max(self.signal_total, 1), 4),
            },
        }


def pick_shards(split_dir: Path, k: int) -> List[Path]:
    shards = sorted(p for p in split_dir.iterdir() if ".tfrecord" in p.name)
    if not shards:
        raise FileNotFoundError(f"{split_dir} 下没有 tfrecord")
    if k >= len(shards):
        return shards
    idx = np.linspace(0, len(shards) - 1, k).round().astype(int)
    return [shards[i] for i in sorted(set(idx.tolist()))]


def print_summary(s: dict) -> None:
    print(f"\n==================== {s['split']}（{s['n_scenarios']} 个场景） ====================")
    for section in ("size", "time", "sdc", "agents", "map", "flat_ground", "traffic_signals"):
        print(f"--- {section}")
        for k, v in s[section].items():
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
    if s["errors"]:
        print(f"--- errors: {s['errors']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("/gs/bs/tga-RLA/qdeng/womd_dataset/v_1_3_1"))
    ap.add_argument("--splits", nargs="+", default=["training_20s", "validation"])
    ap.add_argument("--shards", type=int, default=3, help="每个 split 均匀抽几个 shard")
    ap.add_argument("--max-scenarios", type=int, default=200, help="每个 split 最多读几个场景")
    ap.add_argument("--out", type=Path, default=None, help="默认 <root>/_probe/")
    args = ap.parse_args()

    out_dir = args.out or args.root / "_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        shards = pick_shards(args.root / "scenario" / split, args.shards)
        per_shard = math.ceil(args.max_scenarios / len(shards))
        stats = SplitStats(split)
        for shard in shards:
            for i, raw in enumerate(iter_tfrecord(shard)):
                if i >= per_shard:
                    break
                stats.add(raw)
            print(f"[{split}] {shard.name}: 累计 {stats.n_scenarios} 个场景", file=sys.stderr)
        summary = stats.summary()
        summary["shards"] = [p.name for p in shards]
        print_summary(summary)
        path = out_dir / f"probe_{split}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"-> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
