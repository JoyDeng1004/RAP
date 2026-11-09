"""Exact test-time camera-rig changes for Exp-0 / T1.

Only changes that need no depth are allowed: rotation of a camera about its own
optical centre, focal length, principal point, plus camera dropout. The image,
``intrinsics`` and ``sensor2lidar_rotation`` of each Camera are rewritten
together and RAP's own loader then derives lidar2img, so image and projection
cannot disagree. ``sensor2lidar_translation`` and ``distortion`` are never touched.

Two image models:
- default (lens kept): the new camera has the same lens distortion as the old
  one. Each output pixel is undistorted with (K_new, D), rotated into the old
  camera and re-distorted with (K_old, D), so the model sees distorted images
  exactly as in training, and lidar2img stays pure pinhole as in training.
- undistort=True: every image is first undistorted onto the pinhole K, then the
  change is a plain homography. Used only for the ``undistort_only`` setting,
  which measures RAP under corrected (pinhole-consistent) input.
Cameras without distortion coefficients take the homography path either way.

Conventions (camera frame: x right, y down, z forward):
- pitch_up_deg > 0 tilts every camera up (about its own x axis);
- yaw_out_deg > 0 turns the side cameras away from the front camera
  (cam_l0 further left, cam_r0 further right); cam_f0 / cam_b0 are unchanged;
- roll_deg > 0 rotates every camera about its optical axis, x towards y;
- focal_scale scales fx, fy about the principal point (> 1 zooms in);
- pp_shift_frac moves cx by that fraction of the image width;
- pp_shift_px moves (cx, cy) by whole pixels; resample_ctrl uses (-0.5, -0.5),
  i.e. every output pixel is a bilinear average of 4 input pixels while the
  geometry moves by only half a pixel -- the interpolation-only control.
Pixels with no source content are black (raw 0, the value RAP's own loader uses
for missing images) or, with fill="gray", the normalisation mean, which RAP's
NormalizeMultiviewImage maps to exactly 0 -- the value of RAP's own bottom padding.
"""
from dataclasses import dataclass, replace
from typing import Dict, Tuple

import cv2
import numpy as np

CAMS = ("cam_b0", "cam_f0", "cam_l0", "cam_r0")  # RAP's slot order
# rap_dino/bevformer/bev_feature_build.py: NormalizeMultiviewImage(mean=..., to_rgb=False) on RGB images
GRAY_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SIDE_SIGN = {"cam_l0": -1.0, "cam_r0": 1.0}  # outward yaw, in camera-frame yaw (turn right > 0)


@dataclass(frozen=True)
class RigSpec:
    pitch_up_deg: float = 0.0
    yaw_out_deg: float = 0.0
    roll_deg: float = 0.0
    focal_scale: float = 1.0
    pp_shift_frac: float = 0.0
    pp_shift_px: Tuple[float, float] = (0.0, 0.0)
    drop: Tuple[str, ...] = ()
    fill: str = "black"  # "black" (raw 0) or "gray" (normalisation mean)
    mask_only: bool = False  # control: keep geometry, only black out what the transform would leave empty
    crop_only: bool = False  # control: keep geometry, only black out the content the transform pushes out of view
    undistort: bool = False

    def is_geometric_identity(self) -> bool:
        return (self.pitch_up_deg == 0 and self.yaw_out_deg == 0 and self.roll_deg == 0
                and self.focal_scale == 1 and self.pp_shift_frac == 0 and tuple(self.pp_shift_px) == (0.0, 0.0))


PRESETS: Dict[str, RigSpec] = {"undistort_only": RigSpec(undistort=True)}
PRESETS.update({f"pitch_up_{t}": RigSpec(pitch_up_deg=v) for t, v in
                [("p2.5", 2.5), ("p5", 5.0), ("m5", -5.0), ("m10", -10.0)]})
PRESETS.update({f"yaw_out_{t}": RigSpec(yaw_out_deg=v) for t, v in
                [("p5", 5.0), ("p10", 10.0), ("m5", -5.0), ("m10", -10.0)]})
PRESETS.update({f"roll_{t}": RigSpec(roll_deg=v) for t, v in [("p3", 3.0), ("m3", -3.0)]})
PRESETS.update({f"focal_{t}": RigSpec(focal_scale=v) for t, v in
                [("0.8", 0.8), ("0.9", 0.9), ("1.1", 1.1), ("1.25", 1.25)]})
PRESETS.update({f"pp_{t}": RigSpec(pp_shift_frac=v) for t, v in [("p5", 0.05), ("m5", -0.05)]})
PRESETS.update({f"drop_black_{c[4:]}": RigSpec(drop=(c,)) for c in CAMS})
# Mask-only controls for every transform that leaves part of the image empty
# (zoom-in to 1.1 / 1.25 fills the frame, so it needs none).
for _name, _spec in list(PRESETS.items()):
    if _spec.is_geometric_identity() or (_spec.focal_scale > 1 and _spec.pp_shift_frac == 0
                                          and _spec.pitch_up_deg == _spec.yaw_out_deg == _spec.roll_deg == 0):
        continue
    PRESETS[f"mask_{_name}"] = replace(_spec, mask_only=True)
# Interpolation-only control and fill-value controls (no mask_* counterparts).
PRESETS["resample_ctrl"] = RigSpec(pp_shift_px=(-0.5, -0.5))
for _name in ("pitch_up_m10", "focal_0.8"):
    PRESETS[f"gray_{_name}"] = replace(PRESETS[_name], fill="gray")
# Content-loss controls: same visible content as the transform, left where it was. For a pure
# translation this differs from the transform only in pixel position (same content, same black area).
for _name in ("pitch_up_p2.5", "pitch_up_p5", "pitch_up_m5", "pitch_up_m10", "pp_p5", "pp_m5", "focal_1.25"):
    PRESETS[f"crop_{_name}"] = replace(PRESETS[_name], crop_only=True)
# Vertical principal-point shifts matching the image-centre displacement of each pitch preset
# (fx * tan(pitch), fx = 1545 for every NAVSIM camera): pure translation, geometry exact.
# If they reproduce the pitch drops, the cause is pixel position, not perspective / content.
for _name, _deg in (("p67", 2.5), ("p135", 5.0), ("m135", -5.0), ("m272", -10.0)):
    _spec = RigSpec(pp_shift_px=(0.0, round(1545.0 * float(np.tan(np.deg2rad(_deg))), 1)))
    PRESETS[f"ppy_{_name}"] = _spec
    PRESETS[f"mask_ppy_{_name}"] = replace(_spec, mask_only=True)
    PRESETS[f"crop_ppy_{_name}"] = replace(_spec, crop_only=True)
# Step 8: vertical shifts by whole feature rows. RAP resizes 1080 -> 432 (x0.4) and DINOv3 uses
# 16 px patches, so 40 px here = exactly one row of the 28 x 48 feature map. A whole-row shift
# keeps the patch grid phase, so an equivariant backbone would just move its features by k rows.
for _k in (-7, -3, -1, 1, 3):
    _spec = RigSpec(pp_shift_px=(0.0, 40.0 * _k))
    _name = f"ppy_{'m' if _k < 0 else 'p'}{abs(_k)}r"
    PRESETS[_name] = _spec
    PRESETS[f"crop_{_name}"] = replace(_spec, crop_only=True)

ROW_PX = 40.0  # source pixels per feature-map row / column (16 px patch / 0.4 resize)


def feature_shift(spec: RigSpec, width: int = 1920) -> Tuple[float, float]:
    """(rows, cols) by which a pure principal-point shift moves the content on RAP's feature map
    (> 0: down / right); None for anything that is not a pure translation."""
    if (spec.pitch_up_deg or spec.yaw_out_deg or spec.roll_deg or spec.focal_scale != 1 or spec.drop
            or spec.undistort or spec.mask_only):
        return None
    dx = spec.pp_shift_frac * width + spec.pp_shift_px[0]
    dy = spec.pp_shift_px[1]
    if dx == 0 and dy == 0:
        return None
    return dy / ROW_PX, dx / ROW_PX


def get_spec(preset: str, undistort: bool = False) -> RigSpec:
    """``raw`` is the unmodified input (must reproduce existing CSVs). ``undistort`` switches every
    other preset to the pinhole image model; ``undistort_only`` always uses it."""
    if preset == "raw":
        return RigSpec()
    if preset not in PRESETS:
        raise KeyError(f"unknown RIG_PRESET {preset!r}; known: raw, {', '.join(PRESETS)}")
    spec = PRESETS[preset]
    return replace(spec, undistort=spec.undistort or undistort)


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_delta(cam_name: str, spec: RigSpec) -> np.ndarray:
    """New camera axes expressed in the old camera frame."""
    yaw = spec.yaw_out_deg * SIDE_SIGN.get(cam_name, 0.0)
    # _rot("x", a) sends +z to (0, -sin a, cos a), i.e. towards -y (up) for a > 0.
    return _rot("y", yaw) @ _rot("x", spec.pitch_up_deg) @ _rot("z", spec.roll_deg)


def new_calibration(cam_name: str, K: np.ndarray, s2l_rot: np.ndarray, width: int,
                    spec: RigSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (K_new, sensor2lidar_rotation_new, H) with pinhole pixel_new ~ H @ pinhole pixel_old."""
    dR = rotation_delta(cam_name, spec)
    K_new = np.array(K, dtype=np.float64, copy=True)
    K_new[0, 0] *= spec.focal_scale
    K_new[1, 1] *= spec.focal_scale
    K_new[0, 2] += spec.pp_shift_frac * width + spec.pp_shift_px[0]
    K_new[1, 2] += spec.pp_shift_px[1]
    H = K_new @ dR.T @ np.linalg.inv(np.asarray(K, dtype=np.float64))
    return K_new, np.asarray(s2l_rot, dtype=np.float64) @ dR, H


# ---------- lens model (OpenCV k1, k2, p1, p2, k3) ----------

def _coeffs(D) -> np.ndarray:
    d = np.zeros(5)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    if len(D) > 5 and np.any(D[5:] != 0):
        raise NotImplementedError(f"only k1, k2, p1, p2, k3 are supported, got {D}")
    d[: min(5, len(D))] = D[:5]
    return d


def has_distortion(D) -> bool:
    return D is not None and np.any(np.asarray(D, dtype=np.float64) != 0)


def distort_normalized(x: np.ndarray, y: np.ndarray, D) -> Tuple[np.ndarray, np.ndarray]:
    k1, k2, p1, p2, k3 = _coeffs(D)
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def monotone_radius(D, r_max: float = 3.0) -> float:
    """Largest ideal radius up to which r * radial(r) still increases.

    Beyond it the polynomial folds back, so far-off rays would land inside the
    image at wrong places; such rays are treated as outside the lens.
    """
    k1, k2, _, _, k3 = _coeffs(D)
    r = np.linspace(0.0, r_max, 300001)
    g = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    bad = np.nonzero(np.diff(g) <= 0)[0]
    return float(r[bad[0]]) if len(bad) else r_max


_MAP_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def lens_kept_map(cam_name: str, K: np.ndarray, D, h: int, w: int,
                  spec: RigSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cv2.remap maps (map_x, map_y) and validity for the lens-kept transform; cached per calibration."""
    key = (cam_name, np.asarray(K, np.float64).tobytes(), _coeffs(D).tobytes(), h, w,
           spec.pitch_up_deg, spec.yaw_out_deg, spec.roll_deg, spec.focal_scale, spec.pp_shift_frac,
           tuple(spec.pp_shift_px))
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    K = np.asarray(K, np.float64)
    K_new, _, _ = new_calibration(cam_name, K, np.eye(3), w, spec)
    dR = rotation_delta(cam_name, spec)
    r_mono = monotone_radius(D)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([u.ravel(), v.ravel()], -1)

    # 1. output pixel -> ideal ray of the new camera (inverse lens, iterative)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    xy = cv2.undistortPointsIter(pix.reshape(-1, 1, 2), K_new, _coeffs(D), None, None, crit).reshape(-1, 2)
    xd, yd = distort_normalized(xy[:, 0], xy[:, 1], D)
    back = np.stack([xd * K_new[0, 0] + K_new[0, 2], yd * K_new[1, 1] + K_new[1, 2]], -1)
    ok = (np.abs(back - pix).max(1) < 1e-3) & (np.hypot(xy[:, 0], xy[:, 1]) < r_mono)

    # 2. rotate into the old camera, 3. old lens -> source pixel
    p_old = np.c_[xy, np.ones(len(xy))] @ dR.T
    ok &= p_old[:, 2] > 1e-6
    z = np.where(ok, p_old[:, 2], 1.0)
    xo, yo = p_old[:, 0] / z, p_old[:, 1] / z
    ok &= np.hypot(xo, yo) < r_mono
    xd, yd = distort_normalized(xo, yo, D)
    su, sv = xd * K[0, 0] + K[0, 2], yd * K[1, 1] + K[1, 2]
    ok &= (su >= 0) & (su <= w - 1) & (sv >= 0) & (sv <= h - 1)

    map_x = np.where(ok, su, -1).astype(np.float32).reshape(h, w)
    map_y = np.where(ok, sv, -1).astype(np.float32).reshape(h, w)
    out = (map_x, map_y, ok.reshape(h, w))
    _MAP_CACHE[key] = out
    return out


def _undistort(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort onto the same pinhole K (RAP's lidar2img is pure pinhole); returns (image, valid)."""
    D = np.asarray(dist, dtype=np.float64).reshape(-1)
    out = cv2.undistort(img, K, D)
    valid = cv2.undistort(np.ones(img.shape[:2], np.uint8), K, D) > 0
    return out, valid


def source_visible(map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Source pixels that some valid output pixel samples (bilinear footprint), i.e. content kept in view."""
    vis = np.zeros((h, w), np.uint8)
    xs, ys = map_x[valid].astype(np.float64), map_y[valid].astype(np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            xi = np.clip(np.floor(xs).astype(np.int64) + dx, 0, w - 1)
            yi = np.clip(np.floor(ys).astype(np.int64) + dy, 0, h - 1)
            vis[yi, xi] = 1
    return cv2.morphologyEx(vis, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0  # zoom-out leaves no gaps; this is for safety


def _mask(img: np.ndarray, keep: np.ndarray, fill: str = "black") -> np.ndarray:
    if fill == "black":
        return np.where(keep[..., None] if img.ndim == 3 else keep, img, 0).astype(img.dtype)
    if fill == "gray":  # float32 so the mean survives exactly; RAP's loader casts to float32 anyway
        return np.where(keep[..., None], img.astype(np.float32), GRAY_RGB).astype(np.float32)
    raise ValueError(f"unknown fill {fill!r}")


def transform_camera(cam, cam_name: str, spec: RigSpec) -> Dict[str, float]:
    """Rewrite one Camera in place; returns the fraction of pixels left without content."""
    img = cam.image
    if img is None:
        raise RuntimeError(f"{cam_name}: no image to transform")
    h, w = img.shape[:2]
    K = np.asarray(cam.intrinsics, dtype=np.float64)
    valid = np.ones((h, w), bool)
    lens_kept = has_distortion(cam.distortion) and not spec.undistort
    if spec.undistort:
        if cam.distortion is None:
            raise RuntimeError(f"{cam_name}: undistort requested but Camera.distortion is None")
        img, valid = _undistort(img, K, cam.distortion)
    if not spec.is_geometric_identity():
        K_new, rot_new, H = new_calibration(cam_name, K, cam.sensor2lidar_rotation, w, spec)
        if lens_kept:
            map_x, map_y, warped_valid = lens_kept_map(cam_name, K, cam.distortion, h, w, spec)
        else:
            map_x = map_y = None
            warped_valid = cv2.warpPerspective(valid.astype(np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0
        if spec.mask_only:
            valid = valid & warped_valid
        elif spec.crop_only:
            if lens_kept:
                kept = source_visible(map_x, map_y, warped_valid, h, w)
            else:
                kept = cv2.warpPerspective(valid.astype(np.uint8), np.linalg.inv(H), (w, h),
                                           flags=cv2.INTER_NEAREST) > 0  # kept(s) = valid(H s)
            valid = valid & kept
        else:
            if lens_kept:
                img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                img = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = warped_valid
            cam.intrinsics = K_new.astype(np.asarray(cam.intrinsics).dtype)
            cam.sensor2lidar_rotation = rot_new.astype(np.asarray(cam.sensor2lidar_rotation).dtype)
    if cam_name in spec.drop:
        valid = np.zeros_like(valid)
    if not valid.all() or spec.fill != "black":
        img = _mask(img, valid, spec.fill)
    cam.image = img
    return {"invalid_frac": float(1.0 - valid.mean())}


def apply_rig(cameras, spec: RigSpec) -> Dict[str, Dict[str, float]]:
    """Apply ``spec`` to the four RAP cameras of one ``Cameras`` frame, in place."""
    if spec == RigSpec():
        return {name: {"invalid_frac": 0.0} for name in CAMS}  # leave the input bit-identical
    return {name: transform_camera(getattr(cameras, name), name, spec) for name in CAMS}
