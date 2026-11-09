"""WOMD 适配器契约测试：全部用合成 Scenario proto，不需要 WOMD 数据。

钉住 womd_adapter.py 文档里写的映射：
  * 只输出 LANE / CROSSWALK / BOUNDARY 三种类型串，road_line（含 SOLID_*）永不进 renderer；
  * 车道多边形宽度来自边界引用，自行车道丢弃；
  * 原点 = 自车后轴，框 z 相对自车接地点；
  * 红绿灯沿车道离停止点 8 m，UNKNOWN 默认丢弃。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers import womd_adapter as wa  # noqa: E402
from process_data.helpers.renderer import ScenarioRenderer  # noqa: E402

T = 91
DT = 0.1
SPEED = 5.0
GROUND_Z = 10.0  # 全局高程不为 0，检验 z 是否减掉了自车接地点


def _line(mf_points, xs, y, z=GROUND_Z):
    for x in xs:
        p = mf_points.add()
        p.x, p.y, p.z = float(x), float(y), z


def make_scenario(heading: float = 0.0):
    """直路沿 heading 方向：自车 5 m/s，前方 20 m 一辆车，一个行人。"""
    sc = wa.ScenarioProto()
    sc.scenario_id = "synthetic"
    sc.current_time_index = 10
    sc.timestamps_seconds.extend([i * DT for i in range(T)])
    c, s = math.cos(heading), math.sin(heading)

    def add_track(tid, otype, x0, y0, vx, dims):
        tr = sc.tracks.add()
        tr.id, tr.object_type = tid, otype
        for i in range(T):
            x, y = x0 + vx * i * DT, y0
            st = tr.states.add()
            st.center_x, st.center_y = c * x - s * y, s * x + c * y
            st.center_z = GROUND_Z + dims[2] / 2
            st.length, st.width, st.height = dims
            st.heading = heading
            st.velocity_x, st.velocity_y = c * vx, s * vx
            st.valid = True

    add_track(0, 1, 0.0, 0.0, SPEED, (5.286, 2.332, 2.33))  # SDC
    add_track(1, 1, 20.0, 0.0, SPEED, (4.6, 2.0, 1.6))
    add_track(2, 2, 15.0, 4.0, 0.0, (0.8, 0.8, 1.7))
    add_track(3, 4, 30.0, 0.0, 0.0, (1.0, 1.0, 1.0))  # OTHER -> dropped
    sc.sdc_track_index = 0

    def rot_line(points_field, xs, y):
        for x in xs:
            p = points_field.add()
            p.x, p.y, p.z = c * x - s * y, s * x + c * y, GROUND_Z

    xs = np.arange(-50.0, 150.0, 0.5)
    lane = sc.map_features.add(id=100)
    rot_line(lane.lane.polyline, xs, 0.0)
    lane.lane.type = 2
    lb = lane.lane.left_boundaries.add(lane_start_index=0, lane_end_index=len(xs) - 1,
                                       boundary_feature_id=200, boundary_type=7)
    rb = lane.lane.right_boundaries.add(lane_start_index=0, lane_end_index=len(xs) - 1,
                                        boundary_feature_id=300)
    del lb, rb
    yellow = sc.map_features.add(id=200)
    yellow.road_line.type = 7  # SOLID_DOUBLE_YELLOW
    rot_line(yellow.road_line.polyline, xs, 1.6)
    edge = sc.map_features.add(id=300)
    edge.road_edge.type = 1
    rot_line(edge.road_edge.polyline, xs, -2.0)
    bike = sc.map_features.add(id=400)
    bike.lane.type = 3
    rot_line(bike.lane.polyline, xs, -3.0)
    nobound = sc.map_features.add(id=500)  # 无边界引用 -> 默认半宽
    nobound.lane.type = 2
    rot_line(nobound.lane.polyline, xs, 6.0)
    cw = sc.map_features.add(id=600)
    for x, y in ((30, -2), (34, -2), (34, 2), (30, 2)):
        p = cw.crosswalk.polygon.add()
        p.x, p.y, p.z = c * x - s * y, s * x + c * y, GROUND_Z

    for _ in range(T):
        dms = sc.dynamic_map_states.add()
        red = dms.lane_states.add(lane=100, state=4)  # STOP
        red.stop_point.x, red.stop_point.y = 40.0 * c, 40.0 * s
        dms.lane_states.add(lane=500, state=0)        # UNKNOWN
        dms.lane_states.add(lane=999, state=6)        # lane 不在地图里
    return sc


class WomdAdapterTest(unittest.TestCase):
    def setUp(self):
        self.sc = wa.parse_scenario(make_scenario().SerializeToString())
        self.ego = wa.ego_track(self.sc)
        self.map = wa.build_map(self.sc)
        self.t = 10
        self.scenario, self.extras = wa.build_scenario(self.sc, self.t, self.map, self.ego)

    def test_only_three_type_strings(self):
        types = {f["type"] for f in self.scenario["map_features"].values()}
        self.assertEqual(types, {"LANE", "BOUNDARY", "CROSSWALK"})
        self.assertFalse(any(k.startswith("road_line") or "200" in k
                             for k in self.scenario["map_features"]))

    def test_lane_width_and_bike_lane(self):
        self.assertNotIn(400, self.map.lanes)
        self.assertEqual((self.map.half_width_measured, self.map.half_width_default), (1, 1))
        poly = self.map.lanes[100]
        self.assertAlmostEqual(poly[:, 1].max(), 1.6, places=3)
        self.assertAlmostEqual(poly[:, 1].min(), -2.0, places=3)
        np.testing.assert_allclose(poly[0], poly[-1])
        ys = self.map.lanes[500][:, 1]
        self.assertAlmostEqual(ys.max() - ys.min(), 2 * wa.DEFAULT_LANE_HALF_WIDTH, places=3)

    def test_rear_axle_origin_and_box_z(self):
        ego_center_x = SPEED * self.t * DT
        self.assertAlmostEqual(self.extras["ego_rear_xy"][0], ego_center_x - wa.REAR_AXLE_TO_CENTER, places=5)
        anns = self.scenario["anns"]
        self.assertEqual(sorted(anns["gt_names"].tolist()), ["pedestrian", "vehicle"])
        car = anns["gt_boxes_world"][anns["gt_names"] == "vehicle"][0]
        self.assertAlmostEqual(car[0], 20.0 + wa.REAR_AXLE_TO_CENTER, places=4)
        self.assertAlmostEqual(car[2], 1.6 / 2, places=4)  # 相对接地点，不是全局高程

    def test_traffic_lights(self):
        lights = self.scenario["traffic_lights"]
        self.assertEqual(len(lights), 1)
        lane_id, is_red, xy = lights[0]
        self.assertEqual((lane_id, is_red), (100, True))
        self.assertAlmostEqual(xy[0], 48.0 - self.extras["ego_rear_xy"][0], places=3)
        self.assertEqual(self.extras["lights"],
                         {"red": 1, "not_red": 0, "unknown_dropped": 1, "missing_lane": 1,
                          "into_exit_lane": 0})

    def test_light_continues_into_straightest_exit(self):
        """信号挂在止于停止线的进口车道上时，沿最直的出口车道再走够 8 m。"""
        xs = np.arange(0.0, 10.5, 0.5)
        m = wa.WomdMap()
        m.lane_centerlines = {
            1: np.stack([xs, np.zeros_like(xs)], 1),                      # 进口道，止于 x=10
            2: np.stack([10 + xs, np.zeros_like(xs)], 1),                 # 直行出口
            3: np.stack([10 + xs * 0.1, xs], 1),                          # 左转出口
        }
        m.lane_exits = {1: [3, 2]}
        pos, hops = wa.walk_lanes(m, 1, np.array([10.0, 0.0]), wa.TRAFFIC_LIGHT_OFFSET)
        self.assertEqual(hops, 1)
        np.testing.assert_allclose(pos, [18.0, 0.0], atol=1e-6)

    def test_command_and_flat_ground(self):
        self.assertEqual(self.extras["driving_command"].tolist(), [0, 1, 0, 0])
        self.assertAlmostEqual(self.extras["slope_p95_30_60m"], 0.0, places=6)

    def test_heading_invariance(self):
        """整个场景旋转后，自车系下的框与 gt_boxes 不变。"""
        sc = wa.parse_scenario(make_scenario(heading=1.0).SerializeToString())
        ego = wa.ego_track(sc)
        scen, _ = wa.build_scenario(sc, self.t, wa.build_map(sc), ego)
        np.testing.assert_allclose(scen["anns"]["gt_boxes"], self.scenario["anns"]["gt_boxes"], atol=1e-4)

    def test_renderer_accepts_scenario(self):
        renderer = ScenarioRenderer(camera_channel_list=["CAM_F0"])  # K 对应 1920x1120
        canvas = renderer.observe(self.scenario)["CAM_F0"]
        self.assertGreater(int((canvas.sum(axis=2) > 0).sum()), 1000)


if __name__ == "__main__":
    unittest.main()
