"""Exp-0 / T1 step 4: check camera calibration before any sweep.

For a few navtest tokens and presets, project the current frame's GT boxes with
the lidar2img that RAP itself derives (LoadMultiViewImageFromFiles, before the
432x768 resize) and draw them on the image RAP would see. Outputs one .webp per
(token, preset) -- webp because the mutagen sync ignores png/jpg -- plus a
text report with each camera's calibration, distortion and invalid-pixel share.

Every tile shows two box overlays:
  green  full lens model (K, distortion, extrinsics of the camera after the change);
         must fit the objects everywhere if the transform is right
  red    RAP's own pure-pinhole lidar2img; off near the edges on distorted images,
         exactly as in RAP's training (undistort_only removes the gap)
It also checks that all navtest tokens share one calibration per camera, which
lets the lens-kept remap tables be computed once.

    cd /gs/bs/tga-RLA/qdeng/RAP/eval && PYTHONPATH=py python tools/audit_rig_t1.py
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR / "py"))
sys.path.insert(0, str(EVAL_DIR / "v1"))
from rap_eval_agents.rig_transform import (CAMS, apply_rig, distort_normalized, get_spec, has_distortion,  # noqa: E402
                                          monotone_radius)

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
GRID = (("cam_l0", "cam_f0", "cam_r0"), ("cam_b0", None, None))


def box_corners(box):
    """gt_boxes row [x, y, z, l, w, h, yaw] (lidar frame, z = box centre) -> (8, 3)."""
    x, y, z, l, w, h, yaw = box[:7]
    dx, dy, dz = np.array([1, 1, -1, -1]) * l / 2, np.array([1, -1, -1, 1]) * w / 2, h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.stack([c * dx - s * dy, s * dx + c * dy], 1) + [x, y]
    return np.concatenate([np.c_[xy, np.full(4, z - dz)], np.c_[xy, np.full(4, z + dz)]])


def pinhole_projector(l2i):
    def project(pts):
        p = (l2i @ np.c_[pts, np.ones(len(pts))].T).T
        return p[:, :2] / np.where(p[:, 2:3] > 0, p[:, 2:3], 1), p[:, 2]
    return project


def lens_projector(cam, use_distortion):
    K = np.asarray(cam.intrinsics, np.float64)
    R, t = np.asarray(cam.sensor2lidar_rotation, np.float64), np.asarray(cam.sensor2lidar_translation, np.float64)
    D = cam.distortion if use_distortion and has_distortion(cam.distortion) else np.zeros(5)

    r_mono = monotone_radius(D)

    def project(pts):
        pc = (pts - t) @ R  # lidar -> camera: R^T (p - t)
        z = pc[:, 2]
        zs = np.where(z > 0, z, 1)
        x, y = pc[:, 0] / zs, pc[:, 1] / zs
        z = np.where(np.hypot(x, y) < r_mono, z, -1.0)  # beyond the fold the polynomial maps rays back into view
        xd, yd = distort_normalized(x, y, D)
        return np.stack([xd * K[0, 0] + K[0, 2], yd * K[1, 1] + K[1, 2]], -1), z
    return project


def draw_boxes(img, project, boxes, max_range, color, thickness):
    h, w = img.shape[:2]
    n = 0
    for box in boxes:
        if np.hypot(box[0], box[1]) > max_range:
            continue
        uv, z = project(box_corners(box))
        if (z <= 0.5).any():
            continue
        if not ((uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)).all():
            continue
        for a, b in EDGES:
            cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), color, thickness, cv2.LINE_AA)
        n += 1
    return n


def calibration_consistency(loader, cur):
    """Distinct calibrations per camera over every token's current frame (log dicts only, no images)."""
    seen = {name: {} for name in CAMS}
    for token in loader.tokens:
        cams = loader.scene_frames_dicts[token][cur]["cams"]
        for name in CAMS:
            c = cams[name.upper()]
            key = (tuple(np.round(np.asarray(c["cam_intrinsic"], np.float64).ravel(), 3)),
                   tuple(np.round(np.asarray(c["distortion"], np.float64).ravel(), 6)),
                   tuple(np.round(np.asarray(c["sensor2lidar_rotation"], np.float64).ravel(), 4)),
                   tuple(np.round(np.asarray(c["sensor2lidar_translation"], np.float64).ravel(), 3)))
            seen[name][key] = seen[name].get(key, 0) + 1
    lines = [f"calibration over {len(loader.tokens)} tokens (current frame):"]
    for name, d in seen.items():
        lines.append(f"  {name}: {len(d)} distinct (intrinsics, distortion, rotation, translation)")
        if len(d) > 1:
            for key, n in sorted(d.items(), key=lambda kv: -kv[1])[:5]:
                lines.append(f"    {n:6d} tokens: fx={key[0][0]} cx={key[0][2]} cy={key[0][5]} k1={key[1][0]} t={key[3]}")
    return lines


def distortion_shift_px(K, D, w, h):
    """How far distortion moves pixels at the image edge midpoints and corners (px, full resolution)."""
    pts = np.array([[w - 1, h / 2], [0, h / 2], [w / 2, 0], [w / 2, h - 1], [0, 0], [w - 1, h - 1]], np.float64)
    rays = np.c_[(pts - K[:2, 2]) / np.diag(K)[:2], np.ones(len(pts))]
    proj = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, np.asarray(D, np.float64).reshape(-1))[0][:, 0]
    return np.linalg.norm(proj - pts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, default=EVAL_DIR.parent)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--num-tokens", type=int, default=3, help="first token of N evenly spaced logs")
    ap.add_argument("--tokens", nargs="*", default=None)
    ap.add_argument("--presets", nargs="*", default=["raw", "undistort_only", "pitch_up_m10", "yaw_out_p10", "focal_0.8", "pp_p5"])
    ap.add_argument("--undistort", type=int, default=0, help="RIG_UNDISTORT for every preset (0 = lens kept)")
    ap.add_argument("--max-range", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR / "audit_t1")
    args = ap.parse_args()

    sys.path.append(str(args.rap_root / "navsim" / "agents"))
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    from rap_dino.bevformer import bev_feature_build as bfb
    from run_pdm_score_v1 import load_split

    data_root = Path(os.environ.get("OPENSCENE_DATA_ROOT", "/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset"))
    data_split, scene_filter = load_split(args.rap_root, args.split)
    sensor_config = SensorConfig(cam_f0=[3], cam_l0=[3], cam_l1=False, cam_l2=False,
                                 cam_r0=[3], cam_r1=False, cam_r2=False, cam_b0=[3], lidar_pc=False)
    loader = SceneLoader(data_path=data_root / "navsim_logs" / data_split,
                         sensor_blobs_path=data_root / "sensor_blobs" / data_split,
                         scene_filter=scene_filter, sensor_config=sensor_config)
    if args.tokens:
        tokens = args.tokens
    else:
        per_log = loader.get_tokens_list_per_log()
        logs = sorted(per_log)
        pick = np.linspace(0, len(logs) - 1, args.num_tokens).round().astype(int)
        tokens = [sorted(per_log[logs[i]])[0] for i in pick]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = [f"data_root={data_root} split={data_split} tokens={tokens} undistort={args.undistort}"]
    cur = scene_filter.num_history_frames - 1
    report += calibration_consistency(loader, cur)

    for ti, token in enumerate(tokens):
        boxes = np.asarray(loader.scene_frames_dicts[token][cur]["anns"]["gt_boxes"], np.float64)
        report.append(f"\n## token {token} ({len(boxes)} gt boxes)")
        for preset in args.presets:
            agent_input = loader.get_agent_input_from_token(token)  # fresh: transforms edit cameras in place
            cams = agent_input.cameras[-1]
            if ti == 0 and preset == args.presets[0]:
                for name in CAMS:
                    c = getattr(cams, name)
                    K, D = np.asarray(c.intrinsics, np.float64), c.distortion
                    h, w = c.image.shape[:2]
                    shift = "n/a" if D is None else np.round(distortion_shift_px(K, D, w, h), 1).tolist()
                    report.append(f"{name}: image={c.image.shape} {c.image.dtype} fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                                  f"cx={K[0,2]:.1f} cy={K[1,2]:.1f} distortion={None if D is None else np.round(np.asarray(D).reshape(-1), 5).tolist()} "
                                  f"t={np.round(np.asarray(c.sensor2lidar_translation), 3).tolist()} "
                                  f"distortion_shift_px[right,left,top,bottom,tl,br]={shift}")
            spec = get_spec(preset, bool(args.undistort))
            stats = apply_rig(cams, spec)
            for name in CAMS:
                getattr(cams, name).real_valid = True
            r = bfb.LoadMultiViewImageFromFiles(agent_input, synthetic=False)  # order cam_b0, f0, l0, r0
            tiles, drawn = {}, {}
            for name, img, l2i in zip(CAMS, r["img"], r["lidar2img"]):
                canvas = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8)[..., ::-1])  # RGB -> BGR for cv2
                draw_boxes(canvas, pinhole_projector(np.asarray(l2i, np.float64)), boxes, args.max_range, (0, 0, 255), 2)
                drawn[name] = draw_boxes(canvas, lens_projector(getattr(cams, name), not spec.undistort), boxes,
                                         args.max_range, (0, 255, 0), 3)
                cv2.putText(canvas, f"{name} {preset}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                tiles[name] = cv2.resize(canvas, (960, int(960 * canvas.shape[0] / canvas.shape[1])))
            blank = np.zeros_like(tiles["cam_f0"])
            grid = np.vstack([np.hstack([tiles[n] if n else blank for n in row]) for row in GRID])
            out = args.out_dir / f"{ti:02d}_{token[:8]}_{preset}.webp"
            cv2.imwrite(str(out), grid, [cv2.IMWRITE_WEBP_QUALITY, 80])
            report.append(f"{preset:>14}: boxes_drawn={drawn} invalid_frac="
                          f"{ {k: round(v['invalid_frac'], 3) for k, v in stats.items()} } -> {out.name}")

    text = "\n".join(report)
    (args.out_dir / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
