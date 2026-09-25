"""渲染契约测试：把 create_nuscenes_metadata 与 create_openscene_metadata 的
一致性约束固化下来。

全部用合成数据，不需要 nuScenes / nuPlan 任何一个字节，可以进 CI。

=============================================================================
测什么、不测什么
=============================================================================
**不测颜色和线宽。** 两条链路调的是同一个 ScenarioRenderer.observe，
COLOR_TABLE 是共享常量，元数据侧根本给不了颜色参数 —— 样式必然一致，
测它没有信息量。

**测元素落进哪个分支。** renderer.py:736-750 是对 feat['type'] 做子串匹配，
再按分支去取 'polygon' 或 'polyline'。这是唯一会静默出错的地方：
类型串写错 -> 落错分支 -> 取错几何键 -> KeyError 或画成另一种东西。

=============================================================================
BOUNDARY 语义（2026-09-18 修正后）
=============================================================================
BOUNDARY 分支两边现在装的是同一类东西：

    nuPlan   : LINE_SOLID_SINGLE_WHITE = intersection ∪ roadblock 的外轮廓
    nuScenes : drivable_area 多边形的外环 + 内环

两者都表示"可行驶区域到此为止"，都只有一侧有路。

修正前 nuScenes 侧装的是 lane_divider / road_divider（路中央的分道线），
和 nuPlan 的含义相反 —— 同一条深红粗线，一边说"别过去"，一边说"可以变道"。
实测 81.3% 的采样点两侧都有 lane，确认是分道线
（scripts/data_audit/boundary_semantics.py，基线存于 exp/smoke/）。
test_boundary_uses_drivable_area 现在钉的是修正【后】的语义。
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.helpers import nuscenes_adapter as adapter  # noqa: E402
from process_data.helpers.renderer import (  # noqa: E402
    COLOR_TABLE,
    ScenarioRenderer,
    camera_params,
)


def load_boundary_semantics():
    """按路径加载判别脚本（scripts/ 不是包）。与既有测试的加载方式一致。"""
    script = REPO_ROOT / "scripts/data_audit/boundary_semantics.py"
    spec = importlib.util.spec_from_file_location("boundary_semantics", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# renderer.py:736-750 的分支判断，逐字镜像
# =============================================================================
def branch_of(type_string: str):
    """返回 (分支名, 该分支会去读的几何键)。None 表示不画。"""
    if "LANE" in type_string:
        return "LANE", "polygon"
    if "CROSSWALK" in type_string or "SPEED_BUMP" in type_string:
        return "CROSSWALK", "polygon"
    if "BOUNDARY" in type_string or "SOLID" in type_string:
        return "BOUNDARY", "polyline"
    return None, None


# 2026-09-17 在 TSUBAME 的 rap 环境实测。MetaDriveType 的【常量名和字符串值
# 不一样】，所以必须钉住值而不是名。metadrive 升级改了这些串，这里会先炸，
# 而不是等到 raster 里少画了一整类要素才发现。
METADRIVE_TYPE_VALUES = {
    "LANE_SURFACE_STREET": "LANE_SURFACE_STREET",
    "LANE_SURFACE_UNSTRUCTURE": "LANE_SURFACE_UNSTRUCTURE",
    "CROSSWALK": "CROSSWALK",
    "SPEED_BUMP": "SPEED_BUMP",
    "BOUNDARY_SIDEWALK": "ROAD_EDGE_SIDEWALK",
    "BOUNDARY_LINE": "ROAD_EDGE_BOUNDARY",
    "LINE_BROKEN_SINGLE_WHITE": "ROAD_LINE_BROKEN_SINGLE_WHITE",
    "LINE_SOLID_SINGLE_WHITE": "ROAD_LINE_SOLID_SINGLE_WHITE",
}

# create_openscene_metadata.py 实际写进 map_features 的类型，及其归宿。
# 注意 walkway 和虚线车道分隔线【都不画】—— 这不是 bug，是上游既成行为。
OPENSCENE_ELEMENT_FATE = {
    "LANE_SURFACE_STREET": ("LANE", "polygon"),          # roadblock 的 interior_edges
    "LANE_SURFACE_UNSTRUCTURE": ("LANE", "polygon"),     # roadblock_connector
    "CROSSWALK": ("CROSSWALK", "polygon"),
    "LINE_SOLID_SINGLE_WHITE": ("BOUNDARY", "polyline"),  # 可行驶区域外轮廓
    "BOUNDARY_SIDEWALK": (None, None),                    # walkway：不画
    "LINE_BROKEN_SINGLE_WHITE": (None, None),             # lane 左边界虚线：不画
}

# 画布上可能出现的全部基色（都会被深度衰减系数 alpha ∈ [0,1] 线性压暗）。
# 地图要素来自 COLOR_TABLE；box 的六个面来自 draw_cuboids_with_occlusion
# 内部硬编码的 base_face_colors（renderer.py:374-382）。
BOX_FACE_COLORS = [
    (247, 37, 133), (76, 201, 240), (114, 9, 183),
    (67, 97, 238), (58, 12, 163),
]


def closed_palette() -> np.ndarray:
    colors = [np.asarray(v, dtype=float) for v in COLOR_TABLE.values()]
    colors += [np.asarray(c, dtype=float) for c in BOX_FACE_COLORS]
    return np.stack(colors)


# =============================================================================
# 合成夹具
# =============================================================================
def rect_ring(x0, y0, x1, y1, per_side=1):
    """矩形周边采样点。per_side>1 时加密，用来触发 _chunk_ring 的切片。"""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    points = []
    for i in range(4):
        a = np.asarray(corners[i], dtype=float)
        b = np.asarray(corners[(i + 1) % 4], dtype=float)
        for t in np.linspace(0.0, 1.0, per_side, endpoint=False):
            points.append(tuple(a + (b - a) * t))
    return points


def fake_expansion_map(path: Path) -> None:
    """最小可用的 nuScenes expansion 地图，几何是刻意设计过的：

        drivable_area 外环  x∈[-2,102]  y∈[-4,4]   （加密到 200 点，触发切片）
          └ hole           x∈[40,50]   y∈[3.4,3.8]（车道上方的非可行驶小岛）
        lane A             x∈[0,100]   y∈[0.05,3.2]
        lane B             x∈[0,100]   y∈[-3.2,-0.05]
        lane_divider       y=0         （夹在 A/B 之间 —— 【不该】进 BOUNDARY）

    这样每一段 drivable_area 轮廓都只有【一侧】有 lane，boundary_semantics.py
    必须把它判成 edge；而那条 divider 两侧都有 lane，会被判成 divider ——
    所以只要结果里出现 divider，就说明 divider 又混进 BOUNDARY 了。
    """
    nodes = []

    def add_nodes(name, pts):
        tokens = []
        for i, (x, y) in enumerate(pts):
            token = f"{name}_n{i}"
            nodes.append({"token": token, "x": float(x), "y": float(y)})
            tokens.append(token)
        return tokens

    lane_a = add_nodes("laneA", rect_ring(0.0, 0.05, 100.0, 3.2))
    lane_b = add_nodes("laneB", rect_ring(0.0, -3.2, 100.0, -0.05))
    cross = add_nodes("cross", rect_ring(20.0, 0.05, 26.0, 3.2))
    # per_side=250 -> 1000 点。真实 drivable_area 的环就是这个量级；
    # 点太稀（比如 200 点）时一片 50 点会覆盖矩形的一整条边，
    # 切片带来的预筛收益体现不出来，测不出这个性质。
    drivable = add_nodes("drivable", rect_ring(-2.0, -4.0, 102.0, 4.0, per_side=250))
    hole = add_nodes("hole", rect_ring(40.0, 3.4, 50.0, 3.8))
    divider = add_nodes("div", [(5.0, 0.0), (95.0, 0.0)])

    # 另起一对【相邻】多边形，共用中缝的两个节点 —— 真实 nuScenes 就是这么拼的
    # （hollandvillage / queenstown 有 80-89% 的边界长度是这种共享边）。
    # 中缝 SEAM_A-SEAM_B 在两块里各出现一次，必须被过滤掉。
    seam = add_nodes("seam", [(200.0, 0.0), (200.0, 10.0)])       # 共享的两个节点
    west = add_nodes("west", [(190.0, 0.0), (190.0, 10.0)])
    east = add_nodes("east", [(210.0, 0.0), (210.0, 10.0)])

    path.write_text(json.dumps({
        "node": nodes,
        "polygon": [
            {"token": "poly_laneA", "exterior_node_tokens": lane_a},
            {"token": "poly_laneB", "exterior_node_tokens": lane_b},
            {"token": "poly_cross", "exterior_node_tokens": cross},
            {"token": "poly_drivable", "exterior_node_tokens": drivable,
             "holes": [{"node_tokens": hole}]},
            # west:  (190,0) -> seam0(200,0) -> seam1(200,10) -> (190,10)
            {"token": "poly_west",
             "exterior_node_tokens": [west[0], seam[0], seam[1], west[1]]},
            # east:  seam0(200,0) -> (210,0) -> (210,10) -> seam1(200,10)
            {"token": "poly_east",
             "exterior_node_tokens": [seam[0], east[0], east[1], seam[1]]},
        ],
        "line": [{"token": "line_div", "node_tokens": divider}],
        "lane": [{"token": "LA", "polygon_token": "poly_laneA"},
                 {"token": "LB", "polygon_token": "poly_laneB"}],
        "lane_connector": [],
        "ped_crossing": [{"token": "P1", "polygon_token": "poly_cross"}],
        "drivable_area": [
            {"token": "DA1", "polygon_tokens": ["poly_drivable"]},
            {"token": "DA2", "polygon_tokens": ["poly_west", "poly_east"]},
        ],
        # 故意留着：改对之后它们必须被忽略
        "lane_divider": [{"token": "D1", "line_token": "line_div"}],
        "road_divider": [],
    }))


def fake_scene_index(sample_token: str = "tok0") -> dict:
    """够 build_annotations 用的最小 scene index。"""
    return {
        "anns": {
            sample_token: [
                {
                    "token": "ann0",
                    "instance_token": "inst0",
                    "category": "vehicle.car",
                    # 全局坐标：ego 正前方 12m
                    "translation": [112.0, 200.0, 0.8],
                    # nuScenes 的 size 顺序是 [width, length, height]
                    "size": [1.9, 4.6, 1.5],
                    "rotation": [1.0, 0.0, 0.0, 0.0],   # yaw = 0
                    "velocity": [1.0, 0.0, 0.0],
                    "num_pts": 50,
                },
                {
                    "token": "ann1",
                    "instance_token": "inst1",
                    "category": "movable_object.trafficcone",  # 应被过滤
                    "translation": [105.0, 203.0, 0.3],
                    "size": [0.4, 0.4, 0.8],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "num_pts": 3,
                },
            ]
        }
    }


def synthetic_scenario() -> dict:
    """前后左右都放要素，保证四路相机各自都能看到东西。"""

    def rect(cx, cy, half=5.0):
        return np.array([[cx - half, cy - half], [cx + half, cy - half],
                         [cx + half, cy + half], [cx - half, cy + half]],
                        dtype=np.float32)

    map_features = {
        "lane_front": {"type": "LANE", "polygon": rect(14.0, 0.0)},
        "lane_back": {"type": "LANE", "polygon": rect(-14.0, 0.0)},
        "lane_left": {"type": "LANE", "polygon": rect(0.0, 12.0)},
        "lane_right": {"type": "LANE", "polygon": rect(0.0, -12.0)},
        "cross_front": {"type": "CROSSWALK", "polygon": rect(26.0, 0.0, 6.0)},
        "bound_front": {
            "type": "BOUNDARY",
            "polyline": np.array([[8.0, 6.0], [40.0, 6.0]], dtype=np.float32),
        },
        "bound_back": {
            "type": "BOUNDARY",
            "polyline": np.array([[-8.0, -6.0], [-40.0, -6.0]], dtype=np.float32),
        },
    }
    # gt_boxes_world: [dx, dy, dz, length, width, height, yaw_global]
    boxes = np.array([
        [12.0, 0.0, 0.8, 4.6, 1.9, 1.5, 0.0],
        [-12.0, 0.0, 0.8, 4.6, 1.9, 1.5, 0.0],
        [0.0, 10.0, 0.8, 4.6, 1.9, 1.5, math.pi / 2],
        [0.0, -10.0, 0.8, 4.6, 1.9, 1.5, math.pi / 2],
    ], dtype=np.float32)
    return {
        "ego_pos": [0.0, 0.0],
        "ego_heading": 0.0,
        "traffic_lights": [],
        "map_features": map_features,
        "anns": {"gt_boxes_world": boxes,
                 "gt_names": np.array(["vehicle"] * 4)},
    }


def render_four_views() -> dict:
    renderer = ScenarioRenderer(
        camera_channel_list=list(adapter.RAP_DINO_CHANNELS),
        width=1920, height=1120, depth_max=120.0,
    )
    renderer.camera_models = {
        ch: camera_params[ch] for ch in adapter.RAP_DINO_CHANNELS
    }
    return renderer.observe(synthetic_scenario())


# =============================================================================
# 1. 分支基准（跨脚本一致性的地基）
# =============================================================================
class BranchBaselineTests(unittest.TestCase):
    def test_metadrive_type_values_are_pinned(self):
        """MetaDriveType 的字符串值变了要立刻知道 —— 它决定谁被画。"""
        try:
            from metadrive.type import MetaDriveType
        except ImportError:
            self.skipTest("metadrive 不可用（本测试只在装了 metadrive 的环境跑）")
        for name, expected in METADRIVE_TYPE_VALUES.items():
            with self.subTest(constant=name):
                self.assertTrue(hasattr(MetaDriveType, name), f"{name} 不存在了")
                self.assertEqual(str(getattr(MetaDriveType, name)), expected)

    def test_openscene_element_fate_matches_branch_logic(self):
        """把"nuPlan 侧每类要素的归宿"钉死，包括两个【不画】的。"""
        for name, expected in OPENSCENE_ELEMENT_FATE.items():
            with self.subTest(element=name):
                self.assertEqual(branch_of(METADRIVE_TYPE_VALUES[name]), expected)

    def test_sidewalk_and_broken_line_are_not_drawn(self):
        """单独拎出来：这两类在 nuPlan 侧不画，所以 nuScenes 侧也不该补。

        ROAD_EDGE_SIDEWALK / ROAD_LINE_BROKEN_SINGLE_WHITE 都不含
        'LANE' / 'CROSSWALK' / 'BOUNDARY' / 'SOLID'，三个分支全不命中。
        """
        for name in ("BOUNDARY_SIDEWALK", "LINE_BROKEN_SINGLE_WHITE"):
            with self.subTest(element=name):
                self.assertEqual(branch_of(METADRIVE_TYPE_VALUES[name]), (None, None))


# =============================================================================
# 2. nuScenes 侧喂进去的东西
# =============================================================================
class MapFeatureContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.map_root = Path(self.tmp.name)
        fake_expansion_map(self.map_root / "unit-test-town.json")
        self.cache = adapter.MapCache(self.map_root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_query_emits_only_renderable_types(self):
        """每个要素的 type 必须命中某个分支，且带着该分支要读的那个几何键。

        这是防 KeyError 的那一条：BOUNDARY 分支去读 feat['polyline']，
        只写了 'polygon' 就会在渲染中途炸掉整个 Pool。
        """
        feats = self.cache.query("unit-test-town", np.array([10.0, 0.0]), 200.0)
        self.assertTrue(feats, "合成地图应当查得到要素")
        seen = set()
        for fid, feat in feats.items():
            with self.subTest(feature=fid):
                name, key = branch_of(feat["type"])
                self.assertIsNotNone(name, f"{feat['type']} 不命中任何分支，画不出来")
                self.assertIn(key, feat, f"{feat['type']} 落 {name} 分支但缺 '{key}' 键")
                self.assertEqual(feat[key].ndim, 2)
                self.assertEqual(feat[key].shape[1], 2)
                seen.add(name)
        self.assertEqual(seen, {"LANE", "CROSSWALK", "BOUNDARY"})

    def test_query_returns_ego_centered_geometry(self):
        """几何必须是"全局轴系、ego 为原点"——renderer 的 lidar_pos 恒为 0。

        不假设任何具体 id/坐标：直接拿未偏移的原始层做对照，
        断言 query 的输出恰好等于"原始点 - center"。
        """
        center = np.array([37.0, -1.5])
        feats = self.cache.query("unit-test-town", center, 500.0)
        layers = self.cache.get("unit-test-town")
        checked = 0
        for layer_key, geom_key in (("lane", "polygon"), ("crosswalk", "polygon"),
                                    ("boundary", "polyline")):
            layer = layers[layer_key]
            for i, fid in enumerate(layer.ids):
                if fid not in feats:
                    continue
                raw = layer.points[layer.offsets[i]: layer.offsets[i + 1]]
                np.testing.assert_allclose(
                    feats[fid][geom_key], (raw - center).astype(np.float32), atol=1e-3
                )
                checked += 1
        self.assertGreater(checked, 0, "没有任何要素被查出来，对照无从谈起")

    def test_boundary_uses_drivable_area_not_dividers(self):
        """BOUNDARY 必须来自 drivable_area，而不是分道线。

        divider 画出来的粗线在路中央，nuPlan 的同一条线在路缘，含义相反。
        合成地图里刻意留了一条 lane_divider，它不该出现在这里。
        """
        feats = self.cache.query("unit-test-town", np.array([50.0, 0.0]), 200.0)
        boundary_ids = [k for k, v in feats.items() if v["type"] == "BOUNDARY"]
        self.assertTrue(boundary_ids, "BOUNDARY 层是空的")
        self.assertTrue(
            all(k.startswith("drivable_area:") for k in boundary_ids),
            f"BOUNDARY 里混进了非 drivable_area 的来源: "
            f"{[k for k in boundary_ids if not k.startswith('drivable_area:')]}",
        )
        self.assertFalse(
            any("divider" in k for k in feats),
            "divider 仍然在被送进渲染",
        )

    def test_boundary_includes_interior_holes(self):
        """内环（区域中间挖掉的部分）也必须画。

        nuPlan 那边 unary_union(...).boundary 会一并给出内外环；漏掉内环，
        非可行驶的小岛在 raster 里就没有边界，模型会以为那块也能开。
        """
        feats = self.cache.query("unit-test-town", np.array([45.0, 3.6]), 200.0)
        boundary_ids = [k for k, v in feats.items() if v["type"] == "BOUNDARY"]
        # id 形如 drivable_area:<poly>[:holeN]:runM#K —— 接缝过滤会把一个环
        # 切成若干 run，所以只能按子串判，不能假设 ":hole0#" 这种紧邻格式。
        self.assertTrue(any(":hole0:" in k for k in boundary_ids),
                        f"内环没进 BOUNDARY: {boundary_ids}")
        self.assertTrue(any(":hole" not in k for k in boundary_ids),
                        f"外环没进 BOUNDARY: {boundary_ids}")

    def test_long_rings_are_chunked_for_prefilter(self):
        """长环必须被切片，否则 query 的质心+半径预筛失效。

        区域级轮廓上千个点、外接半径覆盖半张图，不切片的话每帧都会把整条环
        拉进来逐点减 center —— 功能上不报错，只是慢到不可用。
        """
        layer = self.cache.get("unit-test-town")["boundary"]
        sizes = [int(layer.offsets[i + 1] - layer.offsets[i])
                 for i in range(len(layer.ids))]
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), adapter.RING_CHUNK_POINTS)
        exterior = [i for i, k in enumerate(layer.ids)
                    if k.startswith("drivable_area:poly_drivable:run")]
        self.assertGreater(len(exterior), 1, "200 点的外环应当被切成多片")
        # 切片后半径必须远小于整条环的跨度（否则预筛还是没用）
        span = float(np.linalg.norm(
            layer.points.max(axis=0) - layer.points.min(axis=0)))
        self.assertLess(float(layer.radii[exterior].max()), span / 4)

    def test_chunks_overlap_so_the_outline_has_no_gap(self):
        """相邻片共享一个端点。不重叠的话轮廓上会每 50 点豁一个口。"""
        line = np.asarray(rect_ring(0.0, 0.0, 100.0, 50.0, per_side=40))  # 160 点
        chunks = adapter._chunk_polyline(line, chunk=50)
        self.assertGreater(len(chunks), 1)
        for a, b in zip(chunks, chunks[1:]):
            np.testing.assert_allclose(a[-1], b[0])
        # 覆盖完整：所有片拼起来等于原折线
        merged = np.vstack([chunks[0]] + [c[1:] for c in chunks[1:]])
        np.testing.assert_allclose(merged, line)

    def test_chunker_never_closes_an_open_polyline(self):
        """切片器不得自动闭合。

        接缝过滤后剩下的是【开口】折线。早先的版本会把首点补到末尾，
        于是两个隔着半张地图的端点被连起来，图上多出一条横贯的假边界。
        """
        line = np.asarray([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        chunks = adapter._chunk_polyline(line, chunk=50)
        self.assertEqual(len(chunks), 1)
        np.testing.assert_allclose(chunks[0], line)
        self.assertFalse(np.allclose(chunks[0][-1], chunks[0][0]))

    def test_internal_seams_between_adjacent_polygons_are_dropped(self):
        """相邻多边形共享的那条边是内部接缝，不是边界，必须丢掉。

        合成地图里 poly_west 与 poly_east 共用 (200,0)-(200,10) 这条中缝。
        不过滤的话它会被画成深红粗线横在可行驶区域中间 —— 两侧都是路，
        含义与 nuPlan 的路缘正好相反。真实 nuScenes 上这类接缝占
        hollandvillage / queenstown 边界总长的 80-89%。
        """
        layer = self.cache.get("unit-test-town")["boundary"]
        pairs = set()
        for i in range(len(layer.ids)):
            pts = layer.points[layer.offsets[i]: layer.offsets[i + 1]]
            for a, b in zip(pts, pts[1:]):
                pairs.add((tuple(np.round(a, 6)), tuple(np.round(b, 6))))
                pairs.add((tuple(np.round(b, 6)), tuple(np.round(a, 6))))

        seam = ((200.0, 0.0), (200.0, 10.0))
        self.assertNotIn(seam, pairs, "内部接缝没有被过滤掉")
        # 外侧边必须还在，否则是把整块都删了
        self.assertIn(((190.0, 0.0), (190.0, 10.0)), pairs, "西侧外边界被误删")
        self.assertIn(((210.0, 0.0), (210.0, 10.0)), pairs, "东侧外边界被误删")

    def test_boundary_is_classified_as_road_edge_not_divider(self):
        """跨脚本验证：判别脚本必须把这批 BOUNDARY 判成 edge。

        这是和 nuPlan 对齐与否的最终判据 —— 路缘只有一侧有路，分道线两侧都有。
        合成地图的几何保证了正确答案是 edge（见 fake_expansion_map 的 docstring）。
        """
        semantics = load_boundary_semantics()
        layers = self.cache.get("unit-test-town")
        verdicts, by_source, n_samples = semantics.classify(
            layers, step=2.0, offsets=[1.2, 1.8, 2.4], max_lines=None
        )
        self.assertGreater(n_samples, 0)
        # divider 现在拆成两档：same_lane（边界横穿一片车道，是两层标注错位）
        # 与 two_lanes（真夹在两条车道之间，才是取层问题）。合成地图两者都该为 0。
        self.assertEqual(verdicts["divider_two_lanes"], 0,
                         f"有采样点夹在两条车道之间，说明混进了分道线: {dict(verdicts)}")
        self.assertEqual(verdicts["divider_same_lane"], 0,
                         f"有采样点横穿了同一片车道: {dict(verdicts)}")
        self.assertGreater(verdicts["edge"], 0, f"一个 edge 都没有: {dict(verdicts)}")


class MapPackVersionTests(unittest.TestCase):
    """旧的 _maps/*.npz 必须被拒绝，不能静默拿旧语义继续渲染。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fake_expansion_map(self.root / "unit-test-town.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_of_current_version(self):
        pack_dir = self.root / "_maps"
        adapter.MapCache(self.root).pack_to_npz("unit-test-town", pack_dir)
        cached = adapter.MapCache(self.root, pack_dir=pack_dir)
        fresh = adapter.MapCache(self.root)
        for key in adapter.MAP_LAYER_KEYS:
            with self.subTest(layer=key):
                self.assertEqual(cached.get("unit-test-town")[key].ids,
                                 fresh.get("unit-test-town")[key].ids)

    def test_stale_v1_pack_is_refused(self):
        """v1 的 boundary 装的是 divider，语义与 v2 相反。

        没有这个闸门，改完代码但忘了重跑 build-index 的人会拿着旧包继续渲染，
        而且【不报任何错】—— 渲出来的图看着也正常，只是红线画错了地方。
        """
        pack_dir = self.root / "_maps"
        adapter.MapCache(self.root).pack_to_npz("unit-test-town", pack_dir)
        path = pack_dir / "unit-test-town.npz"
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files if k != "pack_version"}
        np.savez_compressed(path, **payload)   # 退化成 v1（无版本字段）

        with self.assertRaises(SystemExit) as caught:
            adapter.MapCache(self.root, pack_dir=pack_dir).get("unit-test-town")
        self.assertIn("v1", str(caught.exception))
        self.assertIn("build-index", str(caught.exception))


class AnnotationContractTests(unittest.TestCase):
    def test_box_column_order_is_length_width_height(self):
        """nuScenes 的 size 是 [w,l,h]，写进 gt_boxes 必须重排成 [l,w,h]。

        这一位搞错不会报错，只会让所有框在 raster 里转 90 度。
        """
        anns = adapter.build_annotations(
            fake_scene_index(), "tok0",
            ego_translation=np.array([100.0, 200.0, 0.0]), ego_yaw=0.0,
        )
        self.assertEqual(anns["gt_boxes"].shape, (1, 7))   # 锥桶被过滤掉
        np.testing.assert_allclose(anns["gt_boxes"][0, 3:6], [4.6, 1.9, 1.5])
        np.testing.assert_allclose(anns["gt_boxes_world"][0, 3:6], [4.6, 1.9, 1.5])

    def test_filtered_classes_match_openscene(self):
        """过滤类别必须与 create_openscene_metadata.py:47 逐项相同。"""
        self.assertEqual(
            tuple(adapter.FILTERED_CLASSES),
            ("traffic_cone", "barrier", "czone_sign", "generic_object"),
        )

    def test_world_boxes_are_ego_translated_but_not_rotated(self):
        """gt_boxes_world 只减平移、不转朝向 —— renderer 吃的就是这个约定。"""
        anns = adapter.build_annotations(
            fake_scene_index(), "tok0",
            ego_translation=np.array([100.0, 200.0, 0.0]), ego_yaw=math.pi / 2,
        )
        np.testing.assert_allclose(anns["gt_boxes_world"][0, :2], [12.0, 0.0], atol=1e-4)
        # yaw 是全局的，不减 ego_yaw
        self.assertAlmostEqual(float(anns["gt_boxes_world"][0, 6]), 0.0, places=4)
        # 而 gt_boxes 是车体系：位移被转了，yaw 也减了
        np.testing.assert_allclose(anns["gt_boxes"][0, :2], [0.0, -12.0], atol=1e-4)
        self.assertAlmostEqual(float(anns["gt_boxes"][0, 6]), -math.pi / 2, places=4)


# =============================================================================
# 3. 四视角渲染
# =============================================================================
class FourViewRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.views = render_four_views()

    def test_all_four_channels_rendered_at_expected_shape(self):
        self.assertEqual(set(self.views), set(adapter.RAP_DINO_CHANNELS))
        for ch, canvas in self.views.items():
            with self.subTest(camera=ch):
                self.assertEqual(canvas.shape, (1120, 1920, 3))
                self.assertEqual(canvas.dtype, np.uint8)

    def test_every_view_has_content(self):
        """场景四周都放了要素，四路都该画出东西。

        某一路恒为空，多半是 camera_models 没覆盖到它，或该路的外参搞错了。
        """
        for ch, canvas in self.views.items():
            with self.subTest(camera=ch):
                frac = float(np.count_nonzero(canvas)) / canvas.size
                self.assertGreater(frac, 1e-4, f"{ch} 几乎全黑")
                self.assertLess(frac, 0.6, f"{ch} 画满了，多半投影崩了")

    def test_views_are_mutually_distinct(self):
        """四路必须互不相同。

        全相同 = renderer.camera_models 被同一组参数覆盖了 —— 这正是
        create_nuscenes_metadata.py:195 整体覆盖 camera_models 时最容易犯的错。
        """
        names = list(adapter.RAP_DINO_CHANNELS)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                with self.subTest(pair=(a, b)):
                    same = float((self.views[a] == self.views[b]).all(axis=2).mean())
                    self.assertLess(same, 0.995, f"{a} 与 {b} 几乎逐像素相同")

    def test_palette_is_closed(self):
        """非黑像素必须落在某个基色的深度衰减射线上。

        观察到画布外的颜色，说明有人往 observe 里加了新的绘制分支，
        或 COLOR_TABLE 被改了 —— 两者都会让两条链路的样式就此分叉。
        """
        palette = closed_palette()
        unit = palette / np.linalg.norm(palette, axis=1, keepdims=True)
        for ch, canvas in self.views.items():
            flat = canvas.reshape(-1, 3).astype(float)
            # 低亮度像素受 uint8 量化影响方向不可靠，只看足够亮的
            bright = flat[flat.max(axis=1) > 40]
            if len(bright) == 0:
                continue
            sample = bright[:: max(1, len(bright) // 20000)]
            cos = (sample / np.linalg.norm(sample, axis=1, keepdims=True)) @ unit.T
            with self.subTest(camera=ch):
                off = float((cos.max(axis=1) < 0.99).mean())
                self.assertLess(off, 0.02, f"{ch} 有 {off:.1%} 像素不在已知调色板上")


if __name__ == "__main__":
    unittest.main(verbosity=2)
