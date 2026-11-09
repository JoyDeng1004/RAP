"""WOMD metadata 生成测试：合成 Scenario 写成真 tfrecord，端到端跑 run / audit。

钉住：
  * 2 Hz 网格经过 current_time_index，帧字典字段齐全（Scene.from_scene_dict_list 读的那些）；
  * 位姿 = 后轴 + 接地 z，ego_dynamic_state 在自车系；
  * 只有被选中的 current 帧 is_valid=True 且四路渲染图都在；
  * 静止 anchor 按 --stationary-keep 保留；
  * 按 SceneLoader 的 4+10 窗口切出来的未来轨迹是 2.5 m/步。
"""

from __future__ import annotations

import pickle
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_data import create_womd_metadata as cwm  # noqa: E402
from process_data.helpers import womd_adapter as wa  # noqa: E402
from test_womd_adapter import DT, GROUND_Z, SPEED, make_scenario  # noqa: E402

H, F = 4, 10


def write_tfrecord(path: Path, scenarios) -> None:
    with open(path, "wb") as fh:
        for sc in scenarios:
            data = sc.SerializeToString()
            fh.write(struct.pack("<Q", len(data)) + b"\0" * 4 + data + b"\0" * 4)  # CRC 不校验


def stopped(sc):
    for st in sc.tracks[0].states:
        st.center_x, st.center_y = 0.0, 0.0
        st.velocity_x = st.velocity_y = 0.0
    sc.scenario_id = "stopped"
    return sc


class WomdMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        split_dir = root / "womd" / "scenario" / "validation"
        split_dir.mkdir(parents=True)
        write_tfrecord(split_dir / "validation.tfrecord-00000-of-00001",
                       [make_scenario(), stopped(make_scenario())])
        cls.ds = root / "ds"
        common = ["--womd-root", str(root / "womd"), "--womd-split", "validation", "--ds-root", str(cls.ds)]
        cwm.cmd_run(cwm.parse_args(["run", *common, "--thread-num", "1", "--anchor-stride", "1",
                                    "--stationary-keep", "0", "--camera-channels", "CAM_F0,CAM_B0",
                                    "--render-channels", "CAM_F0"]))
        cls.common = common
        log_dir = cls.ds / "navsim_logs" / "womd_validation"
        cls.infos = pickle.load(open(log_dir / "womd-validation-synthetic.pkl", "rb"))
        cls.stopped = pickle.load(open(log_dir / "womd-validation-stopped.pkl", "rb"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_grid_and_fields(self):
        # 91 帧、current=10：网格 5..85，17 帧
        frames = [i["rap_source"]["womd_frame_index"] for i in self.infos]
        self.assertEqual(frames, list(range(5, 90, 5)))
        self.assertIn(10, frames)
        self.assertTrue(np.all(np.diff([i["timestamp"] for i in self.infos]) == 500_000))
        info = self.infos[3]
        for key in ("token", "timestamp", "roadblock_ids", "traffic_lights", "log_name", "scene_token",
                    "map_location", "ego2global_translation", "ego2global_rotation",
                    "ego_dynamic_state", "driving_command", "cams"):
            self.assertIn(key, info)
        for key in ("gt_boxes", "gt_names", "gt_velocity_3d", "instance_tokens", "track_tokens"):
            self.assertIn(key, info["anns"])
        cam = info["cams"]["CAM_F0"]
        for key in ("data_path", "sensor2lidar_rotation", "sensor2lidar_translation",
                    "cam_intrinsic", "distortion"):
            self.assertIn(key, cam)
        self.assertEqual(len(set(i["token"] for i in self.infos)), len(self.infos))

    def test_pose_and_dynamics(self):
        info = self.infos[3]
        t = info["rap_source"]["womd_frame_index"]
        np.testing.assert_allclose(info["ego2global_translation"],
                                   [SPEED * t * DT - wa.REAR_AXLE_TO_CENTER, 0.0, GROUND_Z], atol=1e-6)
        np.testing.assert_allclose(info["ego2global_rotation"], [1, 0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(info["ego_dynamic_state"], [SPEED, 0, 0, 0], atol=1e-6)
        self.assertEqual(info["driving_command"].tolist(), [0, 1, 0, 0])
        self.assertEqual(info["roadblock_ids"], ["100"])  # 只沿着车道 100 走

    def test_window_future_trajectory(self):
        """按 SceneLoader 的 4+10 窗口切，未来 10 个点每步 2.5 m。"""
        window = self.infos[0:H + F]
        cur = window[H - 1]["ego2global_translation"][:2]
        future = np.array([w["ego2global_translation"][:2] for w in window[H:]]) - cur
        np.testing.assert_allclose(future[:, 0], SPEED * 0.5 * np.arange(1, F + 1), atol=1e-6)

    def test_only_anchors_valid_and_rendered(self):
        valid = [k for k, i in enumerate(self.infos) if i["is_valid"]]
        self.assertEqual(valid, list(range(H - 1, len(self.infos) - F)))  # stride 1：全部 current 帧
        render_root = self.ds / "rendered_sensor_blobs" / "womd_validation"
        for k, info in enumerate(self.infos):
            # B0 is listed but never rendered, like the official NAVSIM rasters
            self.assertEqual(set(info["cams"]), {"CAM_F0", "CAM_B0"})
            self.assertEqual((render_root / info["cams"]["CAM_F0"]["data_path"]).exists(), k in valid)
            self.assertFalse((render_root / info["cams"]["CAM_B0"]["data_path"]).exists())

    def test_render_channels_default_and_subset(self):
        args = cwm.parse_args(["run", "--ds-root", "x"])
        self.assertEqual(args.render_channels, ("CAM_F0", "CAM_L0", "CAM_R0"))
        self.assertIn("CAM_B0", args.camera_channels)
        with self.assertRaises(SystemExit):
            cwm.parse_args(["run", "--ds-root", "x", "--camera-channels", "CAM_F0", "--render-channels", "CAM_R0"])

    def test_stationary_dropped(self):
        self.assertFalse(any(i["is_valid"] for i in self.stopped))

    def test_audit_and_resume(self):
        with self.assertRaises(SystemExit) as cm:
            cwm.cmd_audit(cwm.parse_args(["audit", *self.common, "--render-channels", "CAM_F0"]))
        self.assertEqual(cm.exception.code, 0)
        # the default F0/L0/R0 is not in this pkl, so the audit must flag it
        with self.assertRaises(SystemExit) as cm:
            cwm.cmd_audit(cwm.parse_args(["audit", *self.common]))
        self.assertEqual(cm.exception.code, 1)
        # 已在 checkpoint 里的文件不再处理
        before = sorted(p.stat().st_mtime_ns for p in (self.ds / "navsim_logs").rglob("*.pkl"))
        cwm.cmd_run(cwm.parse_args(["run", *self.common, "--thread-num", "1", "--anchor-stride", "1",
                                    "--stationary-keep", "0", "--camera-channels", "CAM_F0,CAM_B0",
                                    "--render-channels", "CAM_F0"]))
        after = sorted(p.stat().st_mtime_ns for p in (self.ds / "navsim_logs").rglob("*.pkl"))
        self.assertEqual(before, after)

    def test_short_timestamps_fall_back_to_10hz(self):
        """时间戳比轨迹状态少（training_20s 实测有）时按 10 Hz 补，不崩。"""
        sc = make_scenario()
        del sc.timestamps_seconds[-20:]
        sc = wa.parse_scenario(sc.SerializeToString())
        stats = cwm.new_stats()
        args = cwm.parse_args(["run", "--ds-root", "x", "--anchor-stride", "1", "--dry-run"])
        infos = cwm.process_scenario(sc, "validation", None, args, stats)
        self.assertEqual(stats["timestamps_synthesized"], 1)
        self.assertTrue(np.all(np.diff([i["timestamp"] for i in infos]) == 500_000))
        np.testing.assert_allclose(infos[3]["ego_dynamic_state"], [SPEED, 0, 0, 0], atol=1e-6)

    def test_bad_scenario_counted_not_fatal(self):
        root = Path(self.tmp.name) / "bad"
        split_dir = root / "scenario" / "validation"
        split_dir.mkdir(parents=True)
        broken = make_scenario()
        broken.sdc_track_index = 99  # 越界 -> 该场景抛异常
        broken.scenario_id = "broken"
        write_tfrecord(split_dir / "validation.tfrecord-00000-of-00001", [broken, make_scenario()])
        args = cwm.parse_args(["run", "--womd-root", str(root), "--womd-split", "validation",
                               "--ds-root", str(root / "ds"), "--no-render"])
        cwm._init_worker(args.camera_channels, args.depth_max, render=False)
        result = cwm.process_file((str(split_dir / "validation.tfrecord-00000-of-00001"), args))
        self.assertEqual((result["scenarios"], result["scenarios_failed"], result["pkls"]), (2, 1, 1))
        self.assertIn("IndexError", result["failed"][0])

    def test_stride_selects_subset(self):
        sc = wa.parse_scenario(make_scenario().SerializeToString())
        ego = wa.ego_track(sc)
        frames = cwm.grid_frames(sc, ego)
        args = cwm.parse_args(["run", "--ds-root", "x", "--anchor-stride", "2"])
        stats = cwm.new_stats()
        picked = cwm.select_anchors(sc, ego, frames, wa.build_map(sc), args, stats)
        self.assertEqual(stats["current_frames"], 4)
        self.assertEqual(len(picked), 2)
        self.assertEqual(np.diff(picked).tolist(), [2])


if __name__ == "__main__":
    unittest.main()
