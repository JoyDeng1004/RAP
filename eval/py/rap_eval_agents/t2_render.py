"""T2: raster renders of one NAVSIM frame under a changed camera rig, for official RAP (no training).

Rendering reuses RAP's own ``process_data/helpers/renderer.py`` unchanged: ``ScenarioRenderer.observe``
runs with its camera models replaced by the changed rig, so the identity rig reproduces the official
training rasters. The official convention is kept throughout:
- the renderer moves every camera by (x -2 m, z +0.8 m) inside ``observe`` (renderer.py:710-712),
  while the model is given the per-log NAVSIM calibration (dataclasses.py falls back to it for
  ``render_*``), so raster and calibration disagree by that offset in every condition alike;
- the 1920 x 1120 canvas goes through JPEG (cv2.imwrite default quality, as the official files) and
  RAP's loader drops 20 rows at the top and bottom (dataclasses.py: ``[20:-20]``);
- cam_b0 is not rendered and stays black, as in training.

A rig change ``RigDelta`` moves the whole rig by ``t`` (metres, lidar frame: x forward, y left, z up)
and scales fx, fy by ``focal`` about the principal point, in the renderer's cameras and, unless
``calib == "stale"``, in the calibration given to the model (``update_model_camera``).

``view_mask`` blacks out every pixel whose drawn surface the reference rig (identity) could not see
inside its own frustum (front of the camera, inside the cropped 1920 x 1080 view); occlusion in the
reference view is ignored. The surface under a pixel is found per layer in the official drawing
order (traffic lights, then map, then boxes on top): box pixels by ray / oriented-box intersection,
map pixels on the z = 0 plane where the renderer draws the map, traffic-light pixels on their boxes.
Pixels whose surface cannot be resolved (anti-aliasing fringe) are kept.

``surface_map`` exposes that per-pixel surface (point, kind, box id) for any rig, and ``surface_pairs``
uses it for the surface-correspondence diagnostic: pixels of two rigs that show the same surface point.
"""
import importlib.util
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

CHANNELS = {"cam_f0": "CAM_F0", "cam_l0": "CAM_L0", "cam_r0": "CAM_R0"}  # rendered; cam_b0 stays black
CAMS = ("cam_b0", "cam_f0", "cam_l0", "cam_r0")  # RAP's slot order
CANVAS_H, CANVAS_W = 1120, 1920
CROP = 20  # rows RAP's loader drops at the top and bottom of a raster
OFFICIAL_SHIFT = np.array([-2.0, 0.0, 0.8])  # renderer.py:710-712, applied inside ScenarioRenderer.observe
JPEG_QUALITY = 95  # cv2.imwrite default for .jpg
TL_Z, TL_HALF = 5.0, np.array([0.25, 0.25, 0.5])  # renderer.py: lights at z_base 5, dims (0.5, 0.5, 1.0)


@dataclass(frozen=True)
class RigDelta:
    t: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    focal: float = 1.0
    calib: str = "updated"  # "updated": the model gets the moved calibration; "stale": the reference one
    view_mask: bool = False
    drop: Tuple[str, ...] = ()  # RAP slot names rendered black
    scene: str = "full"  # "full", "map_only" (boxes and traffic lights removed: planar ground scene) or
    # "map_scaled" (map_only, map scaled about CAM_F0's foot point so that the unmoved rig sees what a rig
    # moved up by world_scale_dh would see: the H_scale equivalence, see scene_view)
    world_scale_dh: float = 0.0

    def is_identity_geometry(self) -> bool:
        return tuple(self.t) == (0.0, 0.0, 0.0) and self.focal == 1.0


SCENES = ("full", "map_only", "map_scaled")


def reference_of(delta: "RigDelta") -> "RigDelta":
    """The identity rig the c1 of ``delta`` is rendered with: same scene, except that map_scaled is taken
    against the unscaled map_only scene."""
    return RigDelta(scene="map_only" if delta.scene == "map_scaled" else delta.scene)


def equivalent_of(delta: "RigDelta") -> "RigDelta":
    """For a map_scaled preset, the map_only rig change it is geometrically equivalent to (rig moved by dh)."""
    if delta.scene != "map_scaled":
        raise ValueError("only map_scaled presets have a height-change equivalent")
    return RigDelta(t=(0.0, 0.0, delta.world_scale_dh), scene="map_only")


def f0_height(rmod) -> float:
    """Height of the raster CAM_F0 above the map plane: renderer translation z + OFFICIAL_SHIFT z."""
    return float(rmod.camera_params["CAM_F0"]["sensor2lidar_translation"][2] + OFFICIAL_SHIFT[2])


def scale_map(rmod, scenario: dict, dh: float) -> dict:
    """Map features scaled by k = H / (H + dh) about CAM_F0's foot point (H = f0_height).

    A planar scene seen from height H + dh equals, up to a uniform scale about the camera foot point, the
    scene scaled by k seen from height H with the same orientation: exact for CAM_F0. L0 / R0 sit ~0.5 m
    from F0's foot point, so for them the equivalence is off by ~(k - 1) * 0.5 m of camera position. The
    renderer's depth shading (colour from camera depth) and its 120 m cull are not scale invariant."""
    H = f0_height(rmod)
    k = H / (H + dh)
    c, s_ = np.cos(scenario["ego_heading"]), np.sin(scenario["ego_heading"])
    t = np.asarray(rmod.camera_params["CAM_F0"]["sensor2lidar_translation"], np.float64) + OFFICIAL_SHIFT
    foot = np.array([c * t[0] - s_ * t[1], s_ * t[0] + c * t[1]])  # world xy (global axes, lidar at 0)
    feats = {}
    for key, f in scenario["map_features"].items():
        g = dict(f)
        for kk in ("polygon", "polyline"):
            if kk in g:
                arr = np.asarray(g[kk])
                g[kk] = (foot + k * (arr[:, :2].astype(np.float64) - foot)).astype(arr.dtype)
        feats[key] = g
    return dict(scenario, map_features=feats)


def scene_view(scenario: dict, scene: str, rmod=None, world_scale_dh: float = 0.0) -> dict:
    """``scenario`` restricted to the layers of ``scene`` (map_only: no boxes, no traffic lights;
    map_scaled: map_only with the map scaled by ``scale_map``, which needs ``rmod``)."""
    if scene == "full":
        return scenario
    if scene in ("map_only", "map_scaled"):
        anns = dict(scenario["anns"], gt_boxes_world=np.zeros((0, 7)), gt_names=np.array([]))
        out = dict(scenario, traffic_lights=[], anns=anns)
        return scale_map(rmod, out, world_scale_dh) if scene == "map_scaled" else out
    raise ValueError(f"unknown scene {scene!r}; known: {SCENES}")


IDENT = RigDelta()
_BASE = {
    "h_p1.0": RigDelta(t=(0.0, 0.0, 1.0)),
    "h_m0.7": RigDelta(t=(0.0, 0.0, -0.7)),
    "d_p1.0": RigDelta(t=(1.0, 0.0, 0.0)),
    "d_m1.0": RigDelta(t=(-1.0, 0.0, 0.0)),
    "l_p0.5": RigDelta(t=(0.0, 0.5, 0.0)),
    "l_m0.5": RigDelta(t=(0.0, -0.5, 0.0)),
    "f_0.8": RigDelta(focal=0.8),
    "f_1.25": RigDelta(focal=1.25),
}
PRESETS: Dict[str, RigDelta] = {"ident": IDENT, "mask_ident": RigDelta(view_mask=True),
                                "drop_f0": RigDelta(drop=("cam_f0",))}
PRESETS.update(_BASE)
PRESETS.update({f"stale_{k}": RigDelta(t=v.t, focal=v.focal, calib="stale") for k, v in _BASE.items()})
PRESETS.update({f"mask_{k}": RigDelta(t=_BASE[k].t, focal=_BASE[k].focal, view_mask=True)
                for k in ("h_m0.7", "h_p1.0", "d_m1.0", "f_0.8")})
# H_par confirmation: height dose curve (with h_m0.7 / h_p1.0 above) and the planar map-only scene.
PRESETS.update({f"h_{'p' if v > 0 else 'm'}{abs(v):.1f}": RigDelta(t=(0.0, 0.0, v)) for v in (0.3, -0.3, 0.5, -0.5)})
PRESETS.update({f"map_{k}": RigDelta(t=PRESETS[k].t, focal=PRESETS[k].focal, scene="map_only")
                for k in ("ident", "h_m0.7", "h_p1.0", "d_m1.0", "h_m0.3")})
# H_scale test (registered 2026-10-23): the unmoved rig sees the map scaled as if moved by dh; compare with
# map_h_<same>.
PRESETS.update({f"mapscale_{k}": RigDelta(scene="map_scaled", world_scale_dh=PRESETS[k].t[2])
                for k in ("h_m0.3", "h_m0.7", "h_p1.0")})


def get_delta(preset: str) -> RigDelta:
    if preset not in PRESETS:
        raise KeyError(f"unknown T2 preset {preset!r}; known: {', '.join(PRESETS)}")
    return PRESETS[preset]


_RENDERER = None


def renderer_module(rap_root: Path):
    """RAP's process_data/helpers/renderer.py, imported by path (it needs only cv2 / numpy / tqdm)."""
    global _RENDERER
    if _RENDERER is None:
        path = Path(rap_root) / "process_data" / "helpers" / "renderer.py"
        spec = importlib.util.spec_from_file_location("rap_raster_renderer", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RENDERER = mod
    return _RENDERER


def camera_models(rmod, delta: RigDelta) -> Dict[str, dict]:
    """The renderer's camera models with the rig change (before observe adds OFFICIAL_SHIFT)."""
    out = {}
    for ch in CHANNELS.values():
        cp = rmod.camera_params[ch]
        K = np.array(cp["intrinsics"], dtype=np.float64, copy=True)
        K[0, 0] *= delta.focal
        K[1, 1] *= delta.focal
        t = np.asarray(cp["sensor2lidar_translation"], dtype=np.float64) + np.asarray(delta.t, dtype=np.float64)
        out[ch] = dict(cp, intrinsics=K, sensor2lidar_translation=t)
    return out


def _observe(rmod, scenario: dict, models: Dict[str, dict]) -> Dict[str, np.ndarray]:
    r = rmod.ScenarioRenderer(camera_channel_list=list(CHANNELS.values()), width=CANVAS_W, height=CANVAS_H)
    r.camera_models = models
    return r.observe(scenario)


def jpeg_roundtrip(canvas_rgb: np.ndarray) -> np.ndarray:
    """What RAP reads back from an official raster file: cv2 JPEG write of the RGB canvas, PIL read."""
    ok, buf = cv2.imencode(".jpg", canvas_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return np.array(Image.open(io.BytesIO(buf.tobytes())))


def world_to_cam(rmod, scenario: dict, model: dict) -> np.ndarray:
    """T_w2c exactly as observe builds it (lidar at the origin, ego heading, OFFICIAL_SHIFT added)."""
    t = np.asarray(model["sensor2lidar_translation"], dtype=np.float64) + OFFICIAL_SHIFT
    return rmod.world_to_camera_T(np.zeros(3), scenario["ego_heading"], t, model["sensor2lidar_rotation"]).astype(np.float64)


def _pixel_rays(T_w2c: np.ndarray, K: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Camera centre (3,) and world ray directions (n, 3) through pixel centres."""
    R, t = T_w2c[:3, :3], T_w2c[:3, 3]
    centre = -R.T @ t
    pix = np.stack([cols + 0.5, rows + 0.5, np.ones_like(rows, dtype=np.float64)], -1)
    d_cam = pix @ np.linalg.inv(K).T
    return centre, d_cam @ R  # (R.T @ d).T == d.T @ R


def _ray_box(centre: np.ndarray, dirs: np.ndarray, box_centre: np.ndarray, R_box: np.ndarray,
             half: np.ndarray) -> np.ndarray:
    """Entry distance along each ray into an oriented box; inf where it misses or lies behind."""
    o = (centre - box_centre) @ R_box  # into box axes
    d = dirs @ R_box
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d
        t2 = (half - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin)
    t = np.where(tmin > 0, tmin, tmax)
    return np.where(hit, t, np.inf)


def _boxes(scenario: dict):
    """(centre, R, half extents) of every box the renderer draws (renderer.py draw_cuboids_with_occlusion)."""
    out = []
    for b in np.asarray(scenario["anns"]["gt_boxes_world"], dtype=np.float64).reshape(-1, 7):
        c, s = np.cos(b[6]), np.sin(b[6])
        out.append((b[:3], np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), b[3:6] / 2))
    return out


def _lights(scenario: dict):
    out = []
    for feat in scenario["traffic_lights"]:
        x, y = feat[2]
        out.append((np.array([x, y, TL_Z + TL_HALF[2]]), np.eye(3), TL_HALF))
    return out


def _hit_boxes(centre, dirs, boxes, T_w2c, K, rows, cols, return_id: bool = False):
    """Nearest box entry distance per ray; only rays inside a box's projected footprint are tested.
    With ``return_id`` also the index of that box (-1 where no box is hit)."""
    t_best = np.full(len(dirs), np.inf)
    id_best = np.full(len(dirs), -1, dtype=np.int32)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    for i, (bc, R, half) in enumerate(boxes):
        pw = (corners * half) @ R.T + bc
        pc = pw @ T_w2c[:3, :3].T + T_w2c[:3, 3]
        if np.all(pc[:, 2] <= 0):
            continue
        if np.all(pc[:, 2] > 1e-3):  # footprint test; near / behind-camera boxes test every ray
            uv = (pc / pc[:, 2:3]) @ K.T
            sel = ((cols + 0.5 >= uv[:, 0].min() - 2) & (cols + 0.5 <= uv[:, 0].max() + 2)
                   & (rows + 0.5 >= uv[:, 1].min() - 2) & (rows + 0.5 <= uv[:, 1].max() + 2))
        else:
            sel = np.ones(len(dirs), bool)
        if not sel.any():
            continue
        t = _ray_box(centre, dirs[sel], bc, R, half)
        idx = np.nonzero(sel)[0]
        closer = t < t_best[idx]
        t_best[idx[closer]] = t[closer]
        id_best[idx[closer]] = i
    return (t_best, id_best) if return_id else t_best


def _visible(points: np.ndarray, T_ref: np.ndarray, K_ref: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """Inside the reference camera's cropped view (in front, 1920 x 1080 after RAP's crop)."""
    pc = points @ T_ref[:3, :3].T + T_ref[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc / z[:, None]) @ K_ref.T
    return ((z > 1e-3) & (uv[:, 0] >= -tol) & (uv[:, 0] < CANVAS_W + tol)
            & (uv[:, 1] >= CROP - tol) & (uv[:, 1] < CANVAS_H - CROP + tol))


KIND_NONE, KIND_BOX, KIND_MAP, KIND_LIGHT = 0, 1, 2, 3


def surface_map(rmod, scenario: dict, delta: RigDelta, channels: Optional[Tuple[str, ...]] = None) -> Dict[str, dict]:
    """Per channel, the drawn surface under every canvas pixel (layer order as in the module docstring):
    pts (CANVAS_H, CANVAS_W, 3) world points, NaN where nothing is drawn or the surface is unresolved;
    kind (int8, KIND_*); obj (int32) index of the box / traffic light hit, -1 otherwise; drawn (bool);
    T (world -> camera, as observe) and K of the changed rig."""
    scenario = scene_view(scenario, delta.scene, rmod, delta.world_scale_dh)  # what render draws
    models = camera_models(rmod, delta)
    if channels is not None:
        models = {ch: models[ch] for ch in channels}
    empty_anns = dict(scenario["anns"], gt_boxes_world=np.zeros((0, 7)), gt_names=np.array([]))
    layers = {
        "light": _observe(rmod, dict(scenario, map_features={}, anns=empty_anns), models),
        "map": _observe(rmod, dict(scenario, traffic_lights=[], anns=empty_anns), models),
        "box": _observe(rmod, dict(scenario, traffic_lights=[], map_features={}), models),
    }
    boxes, lights = _boxes(scenario), _lights(scenario)
    out = {}
    for ch, model in models.items():
        T, K = world_to_cam(rmod, scenario, model), model["intrinsics"]
        on = {k: layers[k][ch].any(-1) for k in layers}
        owner_box = on["box"]
        owner_map = on["map"] & ~owner_box
        owner_light = on["light"] & ~owner_box & ~on["map"]
        pts = np.full((CANVAS_H, CANVAS_W, 3), np.nan)
        kind = np.zeros((CANVAS_H, CANVAS_W), np.int8)
        obj = np.full((CANVAS_H, CANVAS_W), -1, np.int32)
        for owner, k in ((owner_box, KIND_BOX), (owner_map, KIND_MAP), (owner_light, KIND_LIGHT)):
            rows, cols = np.nonzero(owner)
            if len(rows) == 0:
                continue
            centre, dirs = _pixel_rays(T, K, rows.astype(np.float64), cols.astype(np.float64))
            if k == KIND_MAP:
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(dirs[:, 2] < 0, -centre[2] / dirs[:, 2], np.inf)
                ids = np.full(len(rows), -1, np.int32)
            else:
                t, ids = _hit_boxes(centre, dirs, boxes if k == KIND_BOX else lights, T, K,
                                    rows.astype(np.float64), cols.astype(np.float64), return_id=True)
            ok = np.isfinite(t) & (t > 0)
            r, c = rows[ok], cols[ok]
            pts[r, c] = centre + t[ok, None] * dirs[ok]
            kind[r, c] = k
            obj[r, c] = ids[ok]
        out[ch] = {"pts": pts, "kind": kind, "obj": obj, "drawn": on["box"] | on["map"] | on["light"], "T": T, "K": K}
    return out


def view_keep_masks(rmod, scenario: dict, delta: RigDelta) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """Per channel: (CANVAS_H, CANVAS_W) bool, True where the drawn surface lies in the reference view."""
    surf, ref_models = surface_map(rmod, scenario, delta), camera_models(rmod, IDENT)
    keep_all, info = {}, {}
    for ch, s in surf.items():
        T_ref, K_ref = world_to_cam(rmod, scenario, ref_models[ch]), ref_models[ch]["intrinsics"]
        resolved = np.isfinite(s["pts"][..., 0])
        keep = np.ones((CANVAS_H, CANVAS_W), bool)
        keep[resolved] = _visible(s["pts"][resolved], T_ref, K_ref)
        drawn = s["drawn"]
        keep_all[ch] = keep
        view = slice(CROP, CANVAS_H - CROP)  # what RAP keeps of the canvas
        info[ch] = {"drawn_frac": float(drawn[view].mean()), "masked_frac": float((drawn & ~keep)[view].mean()),
                    "unresolved_frac": float((drawn.sum() - resolved.sum()) / max(drawn.sum(), 1))}
    return keep_all, info


def surface_pairs(surf_a: dict, surf_b: dict, stride: int = 20, tol_abs: float = 0.3,
                  tol_rel: float = 0.05) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Same-surface correspondences of one channel from rig a to rig b (``surface_map`` entries).

    a's canvas is sampled on a ``stride`` px grid inside RAP's crop. A sample with a resolved surface
    point X is matched when X projects into b's cropped view onto a pixel that shows the same surface:
    same kind, same box / light, and b's surface point there within tol_abs + tol_rel * depth of X
    (depth in a's camera). The distance test also rejects X occluded in b. Returns, for the matched
    samples, uv_a / uv_b (n, 2) continuous canvas coordinates (pixel (r, c) spans [c, c+1) x [r, r+1)),
    kind and depth (n,), and the counts n_grid / n_surface / n_out / n_mismatch / n_matched."""
    rows = np.arange(CROP + stride // 2, CANVAS_H - CROP, stride)
    cols = np.arange(stride // 2, CANVAS_W, stride)
    rr, cc = (g.ravel() for g in np.meshgrid(rows, cols, indexing="ij"))
    n_grid = len(rr)
    X = surf_a["pts"][rr, cc]
    has = np.isfinite(X[:, 0])
    rr, cc, X = rr[has], cc[has], X[has]
    kind_a, obj_a = surf_a["kind"][rr, cc], surf_a["obj"][rr, cc]
    Ta, Tb, Kb = surf_a["T"], surf_b["T"], surf_b["K"]
    depth = X @ Ta[2, :3] + Ta[2, 3]
    pc = X @ Tb[:3, :3].T + Tb[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc / z[:, None]) @ Kb.T
    inside = (z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < CANVAS_W) & (uv[:, 1] >= CROP) & (uv[:, 1] < CANVAS_H - CROP)
    same = np.zeros(len(X), bool)
    ui = np.floor(uv[inside, 0]).astype(int)
    vi = np.floor(uv[inside, 1]).astype(int)
    Xb = surf_b["pts"][vi, ui]
    with np.errstate(invalid="ignore"):
        close = np.linalg.norm(Xb - X[inside], axis=1) <= tol_abs + tol_rel * depth[inside]
    same[inside] = (surf_b["kind"][vi, ui] == kind_a[inside]) & (surf_b["obj"][vi, ui] == obj_a[inside]) & close
    pairs = {"uv_a": np.stack([cc + 0.5, rr + 0.5], -1)[same], "uv_b": uv[same, :2],
             "kind": kind_a[same], "depth": depth[same]}
    counts = {"n_grid": n_grid, "n_surface": int(has.sum()), "n_out": int((~inside).sum()),
              "n_mismatch": int((inside & ~same).sum()), "n_matched": int(same.sum())}
    return pairs, counts


def surface_rgb_diff(img_a: np.ndarray, img_b: np.ndarray, pairs: Dict[str, np.ndarray]) -> float:
    """Mean absolute RGB difference (0-255) of the raster inputs at matched pixels (``render`` images,
    already cropped): how much the renderer itself changes the look of the same surface (depth shading)."""
    if len(pairs["kind"]) == 0:
        return float("nan")

    def at(img, uv):
        r = np.clip(np.floor(uv[:, 1]).astype(int) - CROP, 0, img.shape[0] - 1)
        c = np.clip(np.floor(uv[:, 0]).astype(int), 0, img.shape[1] - 1)
        return img[r, c].astype(np.float64)

    return float(np.abs(at(img_a, pairs["uv_a"]) - at(img_b, pairs["uv_b"])).mean())


def render(rmod, scenario: dict, delta: RigDelta) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """RAP slot name -> 1080 x 1920 RGB uint8 as RAP's loader would read it; per-slot info."""
    models = camera_models(rmod, delta)
    canvases = _observe(rmod, scene_view(scenario, delta.scene, rmod, delta.world_scale_dh), models)
    info: Dict[str, dict] = {}
    if delta.view_mask:
        keep, info = view_keep_masks(rmod, scenario, delta)  # applies the scene itself (surface_map)
        canvases = {ch: np.where(keep[ch][..., None], c, 0).astype(np.uint8) for ch, c in canvases.items()}
    images = {}
    for slot in CAMS:
        ch = CHANNELS.get(slot)
        if ch is None or slot in delta.drop:
            images[slot] = np.zeros((CANVAS_H - 2 * CROP, CANVAS_W, 3), np.uint8)
            continue
        img = jpeg_roundtrip(canvases[ch])[CROP:-CROP]
        images[slot] = img
        info.setdefault(ch, {})["nonblack_frac"] = float(img.any(-1).mean())
    return images, info


def image_equivalence(img_a: np.ndarray, img_b: np.ndarray, thresh: int = 20) -> Dict[str, float]:
    """How close two raster inputs (render images) are: IoU of the drawn pixels (any channel > thresh, so
    JPEG ringing does not count), the same with 1 px tolerance (3 x 3 dilation of each mask), and the mean
    max-channel |difference| over the union of drawn pixels."""
    da, db = (img_a > thresh).any(-1), (img_b > thresh).any(-1)
    union = da | db
    k = np.ones((3, 3), np.uint8)
    ga, gb = cv2.dilate(da.astype(np.uint8), k) > 0, cv2.dilate(db.astype(np.uint8), k) > 0
    tol_hit = (da & gb) | (db & ga)
    n = max(int(union.sum()), 1)
    mad = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16)).max(-1)[union]
    return {"iou": float((da & db).sum() / n), "iou_tol1": float(tol_hit.sum() / n),
            "mad": float(mad.mean()) if mad.size else 0.0}


def update_model_camera(cam, delta: RigDelta) -> None:
    """Give the model the changed calibration (in place); no-op for the stale-calibration conditions."""
    if delta.calib == "stale" or delta.is_identity_geometry():
        return
    t = np.asarray(cam.sensor2lidar_translation)
    cam.sensor2lidar_translation = (t + np.asarray(delta.t, dtype=np.float64)).astype(t.dtype)
    K = np.array(cam.intrinsics, copy=True)
    K[0, 0] *= delta.focal
    K[1, 1] *= delta.focal
    cam.intrinsics = K


def selfcheck_offset(rmod, tol_px: float = 10.0) -> float:
    """observe must place a box where world_to_cam projects it, i.e. OFFICIAL_SHIFT matches renderer.py.
    A wrong or missing shift moves the box by ~90 px at this range; the silhouette centroid of a
    0.6 m cube sits within a few px of its projected centre."""
    heading = 0.3  # world axes are global (lidar-centred): put the box 12 m ahead along the heading
    box = np.array([[12.0 * np.cos(heading), 12.0 * np.sin(heading), 1.0, 0.6, 0.6, 0.6, heading]])
    scen = {"ego_heading": heading, "map_features": {}, "traffic_lights": [],
            "anns": {"gt_boxes_world": box, "gt_names": np.array(["vehicle"])}}
    models = camera_models(rmod, IDENT)
    canvas = _observe(rmod, scen, {"CAM_F0": models["CAM_F0"]})["CAM_F0"]
    rows, cols = np.nonzero(canvas.any(-1))
    if len(rows) == 0:
        raise RuntimeError("selfcheck: the test box was not drawn")
    T = world_to_cam(rmod, scen, models["CAM_F0"])
    pc = T[:3, :3] @ box[0, :3] + T[:3, 3]
    uv = models["CAM_F0"]["intrinsics"] @ (pc / pc[2])
    err = float(np.hypot(cols.mean() - uv[0], rows.mean() - uv[1]))
    if err > tol_px:
        raise RuntimeError(f"selfcheck: drawn box centroid is {err:.1f} px from the projection; "
                           f"OFFICIAL_SHIFT no longer matches renderer.py")
    return err
