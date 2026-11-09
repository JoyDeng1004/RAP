"""Self-test for rap_eval_agents.rig_transform (numpy + OpenCV only, no data needed).

    python3 eval/tools/test_rig_transform.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

from dataclasses import replace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))
from rap_eval_agents import rig_transform as rt  # noqa: E402
from rap_eval_agents.rig_transform import CAMS, PRESETS, RigSpec, apply_rig, get_spec, new_calibration  # noqa: E402

W, H = 1920, 1080
K0 = np.array([[1545.0, 0, 960.0], [0, 1545.0, 560.0], [0, 0, 1]])
YAW = {"cam_f0": 0.0, "cam_l0": 55.0, "cam_r0": -55.0, "cam_b0": 180.0}  # lidar frame: x fwd, y left, z up


def s2l_rotation(yaw_deg):
    a = np.deg2rad(yaw_deg)
    fwd, right, down = np.array([np.cos(a), np.sin(a), 0]), np.array([np.sin(a), -np.cos(a), 0]), np.array([0, 0, -1.0])
    return np.stack([right, down, fwd], axis=1)  # columns = camera x, y, z in the lidar frame


def rap_lidar2img(K, rot, trans):
    """Verbatim from rap_dino/bevformer/bev_feature_build.py:LoadMultiViewImageFromFiles."""
    lidar2cam_r = np.linalg.inv(rot)
    lidar2cam_t = trans @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4)
    viewpad[:3, :3] = K
    return viewpad @ lidar2cam_rt.T


def project(l2i, pts):
    x = (l2i @ np.c_[pts, np.ones(len(pts))].T).T
    return x[:, :2] / x[:, 2:3], x[:, 2]


def make_camera(name, img=None, dist=None):
    return SimpleNamespace(image=np.zeros((H, W, 3), np.uint8) if img is None else img,
                           intrinsics=K0.astype(np.float32), sensor2lidar_rotation=s2l_rotation(YAW[name]).astype(np.float32),
                           sensor2lidar_translation=np.array([1.5, 0.0, 1.5], np.float32), distortion=dist)


def make_rig(**kw):
    return SimpleNamespace(**{n: make_camera(n, **kw) for n in CAMS})


def points_in_view(name, n, rng):
    """Random lidar-frame points 5-40 m in front of camera ``name``, projecting inside the image."""
    cam = make_camera(name)
    l2i = rap_lidar2img(cam.intrinsics.astype(np.float64), cam.sensor2lidar_rotation.astype(np.float64),
                        cam.sensor2lidar_translation.astype(np.float64))
    pts = []
    while len(pts) < n:
        u, v, d = rng.uniform(100, W - 100), rng.uniform(100, H - 100), rng.uniform(5, 40)
        p_cam = d * np.linalg.inv(K0) @ np.array([u, v, 1.0])
        pts.append(cam.sensor2lidar_rotation @ p_cam + cam.sensor2lidar_translation)
    pts = np.array(pts)
    uv, z = project(l2i, pts)
    assert (z > 0).all() and np.allclose(uv[:, 0].clip(0, W), uv[:, 0])
    return pts


def test_presets_listed():
    expected = 1 + 4 + 4 + 2 + 4 + 2 + 4 + 14 + 3 + 8 + 11 + 10  # undistort_only, pitch, yaw, roll, focal, pp, drop, mask controls, resample + 2 gray, 4 ppy + 4 masks, 11 crop controls, 5 whole-row ppy + 5 crops
    assert len(PRESETS) == expected, (len(PRESETS), sorted(PRESETS))


def test_raw_is_bit_identical():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    rig = make_rig(img=img.copy())
    before = {n: (getattr(rig, n).image.copy(), getattr(rig, n).intrinsics.copy(), getattr(rig, n).sensor2lidar_rotation.copy())
              for n in CAMS}
    apply_rig(rig, get_spec("raw", undistort=True))  # raw ignores the undistort flag
    for n in CAMS:
        c = getattr(rig, n)
        assert np.array_equal(c.image, before[n][0]) and np.array_equal(c.intrinsics, before[n][1])
        assert np.array_equal(c.sensor2lidar_rotation, before[n][2])


def test_projection_matches_homography():
    """For every geometric preset: RAP's lidar2img from the new calibration == H applied to old pixels."""
    rng = np.random.default_rng(1)
    for preset, spec in PRESETS.items():
        if spec.is_geometric_identity() or spec.mask_only:
            continue
        for name in CAMS:
            cam = make_camera(name)
            pts = points_in_view(name, 50, rng)
            K, rot, t = (np.asarray(x, np.float64) for x in (cam.intrinsics, cam.sensor2lidar_rotation,
                                                             cam.sensor2lidar_translation))
            K2, rot2, Hm = new_calibration(name, K, rot, W, spec)
            uv_old, _ = project(rap_lidar2img(K, rot, t), pts)
            uv_new, z_new = project(rap_lidar2img(K2, rot2, t), pts)
            h = (Hm @ np.c_[uv_old, np.ones(len(uv_old))].T).T
            uv_h = h[:, :2] / h[:, 2:3]
            ok = z_new > 0
            err = np.abs(uv_h[ok] - uv_new[ok]).max()
            assert err < 1e-6, (preset, name, err)


def test_direction_conventions():
    """pitch_up moves content down, yaw_out moves l0 content right / r0 content left, focal>1 spreads content."""
    def shift(name, spec, uv=(960.0, 560.0)):
        _, _, Hm = new_calibration(name, K0, s2l_rotation(YAW[name]), W, spec)
        x = Hm @ np.array([uv[0], uv[1], 1.0])
        return x[:2] / x[2] - np.array(uv)
    assert shift("cam_f0", RigSpec(pitch_up_deg=5))[1] > 0   # camera looks up -> scene moves down in the image
    assert shift("cam_l0", RigSpec(yaw_out_deg=5))[0] > 0    # l0 turns left -> scene moves right
    assert shift("cam_r0", RigSpec(yaw_out_deg=5))[0] < 0
    assert np.allclose(shift("cam_f0", RigSpec(yaw_out_deg=5)), 0)
    assert shift("cam_f0", RigSpec(focal_scale=1.25), uv=(1500.0, 560.0))[0] > 0


def test_warped_image_follows_projection():
    """A dot drawn at a 3D point's pixel lands where RAP's new lidar2img projects that point."""
    rng = np.random.default_rng(2)
    checked = 0
    for preset in ("pitch_up_m10", "yaw_out_p10", "roll_p3", "focal_0.8", "pp_p5"):
        for name in ("cam_f0", "cam_l0"):
            cam0 = make_camera(name)
            K, rot, t = (np.asarray(x, np.float64) for x in (cam0.intrinsics, cam0.sensor2lidar_rotation,
                                                             cam0.sensor2lidar_translation))
            for p in points_in_view(name, 5, rng):
                # Put the 3D point exactly on an integer pixel so the drawn dot has no rounding offset.
                uv, _ = project(rap_lidar2img(K, rot, t), p[None])
                ui, vi = np.round(uv[0])
                p_cam = np.linalg.inv(rot) @ (p - t)
                p = rot @ (p_cam[2] * np.linalg.inv(K) @ np.array([ui, vi, 1.0])) + t
                img = np.zeros((H, W, 3), np.uint8)
                cv2.circle(img, (int(ui), int(vi)), 4, (255, 255, 255), -1)
                rig = make_rig()
                getattr(rig, name).image = img
                apply_rig(rig, get_spec(preset, undistort=False))
                c = getattr(rig, name)
                uv2 = project(rap_lidar2img(c.intrinsics.astype(np.float64), c.sensor2lidar_rotation.astype(np.float64), t),
                              p[None])[0][0]
                if not (20 < uv2[0] < W - 20 and 20 < uv2[1] < H - 20):
                    continue
                ys, xs = np.nonzero(c.image[..., 0] > 128)
                assert len(xs), (preset, name)
                err = np.hypot(xs.mean() - uv2[0], ys.mean() - uv2[1])
                assert err < 0.5, (preset, name, err)
                checked += 1
    assert checked >= 30, checked


def test_mask_only_keeps_calibration():
    rng = np.random.default_rng(3)
    img = rng.integers(1, 256, (H, W, 3), dtype=np.uint8)
    rig_t, rig_m = make_rig(img=img.copy()), make_rig(img=img.copy())
    st = apply_rig(rig_t, get_spec("pitch_up_m10", undistort=False))
    sm = apply_rig(rig_m, get_spec("mask_pitch_up_m10", undistort=False))
    for n in CAMS:
        cm = getattr(rig_m, n)
        assert np.array_equal(cm.intrinsics, K0.astype(np.float32))
        assert np.array_equal(cm.sensor2lidar_rotation, s2l_rotation(YAW[n]).astype(np.float32))
        kept = (cm.image > 0).any(-1)
        assert np.array_equal(cm.image[kept], img[kept])
        assert abs(st[n]["invalid_frac"] - sm[n]["invalid_frac"]) < 1e-9 and st[n]["invalid_frac"] > 0.05


def test_invalid_fraction_and_drop():
    rig = make_rig()
    st = apply_rig(rig, get_spec("focal_1.25", undistort=False))
    assert all(v["invalid_frac"] == 0 for v in st.values())
    st = apply_rig(make_rig(), get_spec("focal_0.8", undistort=False))
    assert all(0.3 < v["invalid_frac"] < 0.4 for v in st.values())  # 1 - 0.8**2 = 0.36
    rng = np.random.default_rng(4)
    rig = make_rig(img=rng.integers(1, 256, (H, W, 3), dtype=np.uint8))
    st = apply_rig(rig, get_spec("drop_black_l0", undistort=False))
    assert not rig.cam_l0.image.any() and rig.cam_f0.image.all() and st["cam_l0"]["invalid_frac"] == 1


def test_undistort_straightens_projection():
    """A point drawn at its distorted pixel moves back to the pinhole projection."""
    D = np.array([-0.356, 0.12, 0.0, 0.0, 0.0])
    p_cam = np.array([[6.0, 2.0, 10.0]])  # far off-centre: large distortion
    uv_d = cv2.projectPoints(p_cam, np.zeros(3), np.zeros(3), K0, D)[0][0, 0]
    uv_p = (K0 @ p_cam[0])[:2] / p_cam[0, 2]
    img = np.zeros((H, W, 3), np.uint8)
    cv2.circle(img, (int(round(uv_d[0])), int(round(uv_d[1]))), 3, (255, 255, 255), -1)
    rig = make_rig(dist=D)
    rig.cam_f0.image = img
    apply_rig(rig, get_spec("undistort_only"))
    ys, xs = np.nonzero(rig.cam_f0.image[..., 0] > 64)
    assert np.hypot(xs.mean() - uv_p[0], ys.mean() - uv_p[1]) < 3.0, (xs.mean(), ys.mean(), uv_p, uv_d)


D_NAVSIM = np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231])  # renderer.py camera_params


def lens_pixel(K, D, ray):
    x, y = ray[0] / ray[2], ray[1] / ray[2]
    xd, yd = rt.distort_normalized(np.array([x]), np.array([y]), D)
    return np.array([xd[0] * K[0, 0] + K[0, 2], yd[0] * K[1, 1] + K[1, 2]])


def test_monotone_radius():
    r = rt.monotone_radius(D_NAVSIM)
    assert 1.2 < r < 1.45, r  # NAVSIM lens folds back at ~1.3 (ideal normalized radius)
    assert rt.monotone_radius(np.zeros(5)) == 3.0


def test_lens_kept_follows_lens_model():
    """Lens kept: a dot drawn at an old distorted pixel lands where the new camera (K_new, same lens) sees that ray."""
    rng = np.random.default_rng(5)
    checked = 0
    for preset in ("pitch_up_m10", "pitch_up_p5", "yaw_out_p10", "roll_p3", "focal_0.8", "focal_1.25", "pp_p5", "ppy_m272"):
        spec = get_spec(preset)
        for name in ("cam_f0", "cam_l0", "cam_r0"):
            K2, rot2, _ = new_calibration(name, K0, s2l_rotation(YAW[name]), W, spec)
            dR = rt.rotation_delta(name, spec)
            for _ in range(6):
                u, v = int(rng.integers(150, W - 150)), int(rng.integers(150, H - 150))
                xy = cv2.undistortPointsIter(np.array([[[u, v]]], np.float64), K0, D_NAVSIM, None, None,
                                             (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12))[0, 0]
                ray_new = dR.T @ np.array([xy[0], xy[1], 1.0])
                if ray_new[2] <= 0:
                    continue
                exp = lens_pixel(K2, D_NAVSIM, ray_new)
                if not (20 < exp[0] < W - 20 and 20 < exp[1] < H - 20):
                    continue
                img = np.zeros((H, W, 3), np.uint8)
                cv2.circle(img, (u, v), 4, (255, 255, 255), -1)
                rig = make_rig(dist=D_NAVSIM)
                getattr(rig, name).image = img
                apply_rig(rig, spec)
                c = getattr(rig, name)
                assert np.allclose(c.intrinsics, K2) and np.allclose(c.sensor2lidar_rotation, rot2, atol=1e-6)
                ys, xs = np.nonzero(c.image[..., 0] > 128)
                assert len(xs), (preset, name, exp)
                err = np.hypot(xs.mean() - exp[0], ys.mean() - exp[1])
                assert err < 1.0, (preset, name, err)  # dot is reshaped by the lens; centroid stays within 1 px
                checked += 1
    assert checked >= 70, checked


def test_lens_kept_identity_paths():
    """Lens kept, no geometric change: the image stays as loaded (drop only blacks out its camera)."""
    rng = np.random.default_rng(6)
    img = rng.integers(1, 256, (H, W, 3), dtype=np.uint8)
    rig = make_rig(img=img.copy(), dist=D_NAVSIM)
    apply_rig(rig, get_spec("drop_black_f0"))
    assert np.array_equal(rig.cam_l0.image, img) and not rig.cam_f0.image.any()
    assert np.array_equal(rig.cam_l0.intrinsics, K0.astype(np.float32))


def test_lens_kept_mask_and_cache():
    rng = np.random.default_rng(7)
    img = rng.integers(1, 256, (H, W, 3), dtype=np.uint8)
    rt._MAP_CACHE.clear()
    n0 = 0
    rig_t, rig_m = make_rig(img=img.copy(), dist=D_NAVSIM), make_rig(img=img.copy(), dist=D_NAVSIM)
    st = apply_rig(rig_t, get_spec("focal_0.8"))
    sm = apply_rig(rig_m, get_spec("mask_focal_0.8"))
    assert len(rt._MAP_CACHE) == n0 + 4  # one map per camera, shared by the transform and its mask control
    for n in CAMS:
        assert 0.3 < st[n]["invalid_frac"] < 0.6 and abs(st[n]["invalid_frac"] - sm[n]["invalid_frac"]) < 1e-9
        cm = getattr(rig_m, n)
        kept = (cm.image > 0).any(-1)
        assert np.array_equal(cm.image[kept], img[kept])
        assert np.array_equal(cm.intrinsics, K0.astype(np.float32))


def test_resample_ctrl_is_half_pixel_bilinear():
    """Interpolation-only control: content moves by exactly (-0.5, -0.5) px, a single pixel spreads to 2x2,
    and the principal point follows, both with and without a lens model."""
    for dist in (None, D_NAVSIM):
        img = np.zeros((H, W, 3), np.uint8)
        img[500, 900] = 200
        rig = make_rig(img=img.copy(), dist=dist)
        rig.cam_f0.image = img
        st = apply_rig(rig, get_spec("resample_ctrl"))
        c = rig.cam_f0
        ys, xs = np.nonzero(c.image[..., 0] > 0)
        w = c.image[ys, xs, 0].astype(float)
        assert len(xs) == 4 and np.allclose(w, 50, atol=1), (dist is None, xs, ys, w)
        assert np.isclose((xs * w).sum() / w.sum(), 899.5) and np.isclose((ys * w).sum() / w.sum(), 499.5)
        assert np.isclose(c.intrinsics[0, 2], 959.5) and np.isclose(c.intrinsics[1, 2], 559.5)
        assert all(v["invalid_frac"] < 0.002 for v in st.values())


def test_ppy_matches_pitch_centre_shift():
    """ppy_* moves the image centre exactly as far as the matching pitch preset does (pinhole)."""
    for ppy, pitch in (("ppy_p67", "pitch_up_p2.5"), ("ppy_p135", "pitch_up_p5"),
                       ("ppy_m135", "pitch_up_m5"), ("ppy_m272", "pitch_up_m10")):
        c = np.array([960.0, 560.0, 1.0])
        _, _, Hp = new_calibration("cam_f0", K0, s2l_rotation(0.0), W, get_spec(pitch))
        _, _, Hy = new_calibration("cam_f0", K0, s2l_rotation(0.0), W, get_spec(ppy))
        a, b = Hp @ c, Hy @ c
        assert np.allclose(a[:2] / a[2], b[:2] / b[2], atol=0.06), (ppy, a[:2] / a[2], b[:2] / b[2])
        assert np.allclose(Hy[:2, :2], np.eye(2)) and abs(Hy[0, 2]) < 1e-9  # vertical translation only


def test_crop_control_keeps_position():
    """crop_*: calibration unchanged; for a pure vertical shift exactly the rows that leave the view go black,
    and about as much area as the transform blackens."""
    rng = np.random.default_rng(9)
    img = rng.integers(1, 256, (H, W, 3), dtype=np.uint8)
    rig = make_rig(img=img.copy())
    st = apply_rig(rig, get_spec("crop_ppy_m272"))
    c = rig.cam_f0
    assert np.array_equal(c.intrinsics, K0.astype(np.float32))
    black_rows = np.nonzero(~(c.image > 0).any(-1).any(1))[0]
    assert black_rows.min() == 0 and 270 <= black_rows.max() <= 273, (black_rows.min(), black_rows.max())
    assert np.array_equal(c.image[300:], img[300:])
    st_t = apply_rig(make_rig(img=img.copy()), get_spec("ppy_m272"))
    assert abs(st["cam_f0"]["invalid_frac"] - st_t["cam_f0"]["invalid_frac"]) < 0.003
    # lens kept: lost area close to the transform's black area; zoom-out loses no content
    st_l = apply_rig(make_rig(img=img.copy(), dist=D_NAVSIM), get_spec("crop_pitch_up_m10"))
    st_lt = apply_rig(make_rig(img=img.copy(), dist=D_NAVSIM), get_spec("pitch_up_m10"))
    assert abs(st_l["cam_f0"]["invalid_frac"] - st_lt["cam_f0"]["invalid_frac"]) < 0.06, (st_l, st_lt)
    rig = make_rig(img=img.copy(), dist=D_NAVSIM)
    apply_rig(rig, replace(get_spec("focal_0.8"), crop_only=True))
    assert np.array_equal(rig.cam_f0.image, img)


def test_whole_row_shift_is_exact_translation():
    """ppy_*r with the lens kept copies the image k*40 rows up/down; after RAP's 0.4 resize that is
    exactly k*16 rows, i.e. k feature rows. feature_shift reports the same."""
    rng = np.random.default_rng(10)
    img = rng.integers(1, 256, (H, W, 3), dtype=np.uint8)
    small = cv2.resize(img, (768, 432), interpolation=cv2.INTER_LINEAR)  # mmcv.imresize default
    for name, k in (("ppy_m7r", -7), ("ppy_m3r", -3), ("ppy_p1r", 1), ("ppy_p3r", 3)):
        rig = make_rig(img=img.copy(), dist=D_NAVSIM)
        apply_rig(rig, get_spec(name))
        out, s = rig.cam_f0.image, 40 * abs(k)
        moved, src = (out[:H - s], img[s:]) if k < 0 else (out[s:], img[:H - s])
        n_bad = int((moved != src).any(-1).sum())  # single edge pixels where undistortPointsIter fails
        assert n_bad <= 4, (name, n_bad)
        assert rt.feature_shift(get_spec(name)) == (k, 0.0)
        o_s, t = cv2.resize(out, (768, 432), interpolation=cv2.INTER_LINEAR), 16 * abs(k)
        a, b = (o_s[:432 - t - 1], small[t:431]) if k < 0 else (o_s[t + 1:], small[1:432 - t])
        # one border row blends with black; the few failed edge pixels blur into <= 2x2 neighbours each
        assert int((a != b).any(-1).sum()) <= 16, (name, int((a != b).any(-1).sum()))
    assert rt.feature_shift(get_spec("ppy_m272")) == (-272.4 / 40, 0.0)
    assert rt.feature_shift(get_spec("pp_p5")) == (0.0, 96.0 / 40)
    for name in ("raw", "pitch_up_m10", "focal_0.8", "drop_black_f0", "mask_ppy_m272"):
        assert rt.feature_shift(get_spec(name)) is None, name


def test_gray_fill_is_rap_zero():
    """gray_* presets: same geometry as the black version; empty pixels normalise to exactly 0 in RAP."""
    rng = np.random.default_rng(8)
    img = rng.integers(1, 256, (H, W, 3), dtype=np.uint8)
    for name in ("pitch_up_m10", "focal_0.8"):
        rig_b, rig_g = make_rig(img=img.copy(), dist=D_NAVSIM), make_rig(img=img.copy(), dist=D_NAVSIM)
        sb = apply_rig(rig_b, get_spec(name))
        sg = apply_rig(rig_g, get_spec(f"gray_{name}"))
        for n in CAMS:
            b, g = getattr(rig_b, n), getattr(rig_g, n)
            assert g.image.dtype == np.float32 and np.array_equal(b.intrinsics, g.intrinsics)
            assert abs(sb[n]["invalid_frac"] - sg[n]["invalid_frac"]) < 1e-12
            empty = ~(b.image > 0).any(-1)
            normed = (g.image[empty] - rt.GRAY_RGB) / np.array([58.395, 57.12, 57.375], np.float32)
            assert empty.mean() > 0.2 and np.abs(normed).max() < 1e-6
            assert np.array_equal(g.image[~empty], b.image[~empty].astype(np.float32))


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")
