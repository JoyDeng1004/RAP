"""T2 stage A: renderer input (``scenario`` dicts) for every token of a metric cache, from NAVSIM logs.

ScenarioRenderer.observe needs map polylines / polygons, traffic-light positions and world boxes. The
official rasters build them from the nuPlan DB (process_data/create_openscene_metadata.py); here they
are rebuilt from the NAVSIM log frame of each token plus the nuPlan maps, with the same functions
(extract_map_features, set_light_position) and the inverse of the same box transform:
  gt_boxes (ego frame) = [R_ego^T (b - ego), dims, yaw_b - yaw_ego]  ->  gt_boxes_world = [R_ego @ loc, dims, rot + yaw_ego]
Logs written by RAP's own metadata script already carry ``gt_boxes_world`` and light positions; those
are used as they are.

Outputs <out-dir>/<log>.pkl = {token: scenario}. Run in RAP's ``rap`` env (nuplan-devkit, scenarionet,
metadrive), CPU only.

  python build_t2_scenarios.py --rap-root $RAP --metric-cache <navtest subset cache> --out-dir <dir>

Validation (V0) against official rasters on disk: --compare-official N renders N trainval tokens that
have official raster files with the identity rig and reports the pixel difference per camera.
"""
import argparse
import logging
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("build_t2_scenarios")
KEEP_KEYS = ("type", "polygon", "polyline")  # all the renderer reads from a map feature
_META = None
_ARGS = None


def metadata_module(rap_root: Path):
    """process_data/create_openscene_metadata.py (needs NUPLAN_MAPS_ROOT and process_data on sys.path)."""
    global _META
    if _META is None:
        # helpers/nuplan_cameras_utils.py reads these at import; the functions used here never touch
        # the nuPlan DB or sensor files, so placeholders are enough.
        os.environ.setdefault("NUPLAN_DB_PATH", "/nonexistent/nuplan_db")
        os.environ.setdefault("NUPLAN_SENSOR_PATH", "/nonexistent/nuplan_sensor")
        sys.path.insert(0, str(rap_root / "process_data"))
        import create_openscene_metadata as meta
        _META = meta
    return _META


def scenario_from_frame(frame: dict, meta, maps_root: str, map_version: str) -> dict:
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from pyquaternion import Quaternion

    q = Quaternion(*frame["ego2global_rotation"])
    yaw = q.yaw_pitch_roll[0]
    xy = [float(frame["ego2global_translation"][0]), float(frame["ego2global_translation"][1])]
    map_api = get_maps_api(maps_root, map_version, frame["map_location"])  # lru cached per map
    feats = meta.extract_map_features(map_api, xy, radius=200)
    feats = {str(k): {kk: (np.asarray(v[kk], np.float32) if kk != "type" else v[kk])
                      for kk in KEEP_KEYS if kk in v} for k, v in feats.items()}
    lights, n_light_missing = [], 0
    for tl in frame["traffic_lights"]:
        lane_id, is_red = tl[0], bool(tl[1])
        if len(tl) > 2:
            pos = tl[2]
        else:
            try:
                pos = meta.set_light_position(map_api, lane_id, xy)
            except AssertionError:  # lane connector not in the map: the light cannot be placed
                n_light_missing += 1
                continue
        lights.append((lane_id, is_red, [float(pos[0]), float(pos[1])]))
    anns = frame["anns"]
    if "gt_boxes_world" in anns:
        boxes = np.asarray(anns["gt_boxes_world"], np.float64).reshape(-1, 7)
    else:
        boxes = boxes_world_from_ego(frame)
    return {"ego_heading": float(yaw), "map_features": feats, "traffic_lights": lights,
            "anns": {"gt_boxes_world": boxes, "gt_names": np.asarray(anns["gt_names"])},
            "n_light_missing": n_light_missing}


def boxes_world_from_ego(frame: dict) -> np.ndarray:
    """Inverse of create_openscene_metadata.py's gt_boxes: lidar-centred global axes, global yaw."""
    from pyquaternion import Quaternion
    q = Quaternion(*frame["ego2global_rotation"])
    b = np.asarray(frame["anns"]["gt_boxes"], np.float64).reshape(-1, 7)
    return np.concatenate([b[:, :3] @ q.rotation_matrix.T, b[:, 3:6], b[:, 6:7] + q.yaw_pitch_roll[0]], axis=1)


def _init(args):
    global _ARGS
    _ARGS = args
    metadata_module(args.rap_root)


def build_log(job) -> Dict[str, object]:
    log_name, tokens = job
    a = _ARGS
    meta = metadata_module(a.rap_root)
    out = a.out_dir / f"{log_name}.pkl"
    if out.exists() and not a.overwrite:
        return {"log": log_name, "n": 0, "skipped": True}
    with open(a.logs_dir / f"{log_name}.pkl", "rb") as f:
        frames = pickle.load(f)
    want = set(tokens)
    scen, missing = {}, []
    t0 = time.time()
    for fr in frames:
        if fr["token"] in want:
            scen[fr["token"]] = scenario_from_frame(fr, meta, a.maps_root, a.map_version)
    missing = sorted(want - set(scen))
    tmp = out.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(scen, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    return {"log": log_name, "n": len(scen), "missing": missing, "sec": time.time() - t0,
            "lights_missing": sum(s["n_light_missing"] for s in scen.values())}


def tokens_per_log(args) -> Dict[str, List[str]]:
    from navsim.common.dataloader import MetricCacheLoader
    want = set(MetricCacheLoader(args.metric_cache).metric_cache_paths)
    per_log: Dict[str, List[str]] = {}
    for p in sorted(args.logs_dir.glob("*.pkl")):
        with open(p, "rb") as f:
            toks = [fr["token"] for fr in pickle.load(f) if fr["token"] in want]
        if toks:
            per_log[p.stem] = sorted(toks)
    n = sum(len(t) for t in per_log.values())
    if n != len(want):
        logger.warning("metric cache has %d tokens, found %d in %s", len(want), n, args.logs_dir)
    return per_log


def compare_official(args) -> None:
    """V0: identity-rig renders of trainval tokens vs the official raster files of the same tokens."""
    sys.path.insert(0, str(args.rap_root / "eval" / "py"))
    from PIL import Image
    from rap_eval_agents import t2_render as tr
    rmod = tr.renderer_module(args.rap_root)
    logger.info("selfcheck offset: %.2f px", tr.selfcheck_offset(rmod))
    meta = metadata_module(args.rap_root)
    raster_root = args.data_root / "rendered_sensor_blobs" / args.split
    rows, n_frames = [], 0
    for p in sorted(args.logs_dir.glob("*.pkl")):
        with open(p, "rb") as f:
            frames = pickle.load(f)
        for fr in frames[:: max(1, len(frames) // 3)]:
            paths = {ch: raster_root / fr["cams"][ch]["data_path"] for ch in tr.CHANNELS.values()}
            if not all(q.exists() for q in paths.values()):
                continue
            scen = scenario_from_frame(fr, meta, args.maps_root, args.map_version)
            if "gt_boxes_world" in fr["anns"] and len(fr["anns"]["gt_boxes"]):
                rec = boxes_world_from_ego(fr)
                ref = np.asarray(fr["anns"]["gt_boxes_world"], np.float64).reshape(-1, 7)
                dyaw = np.abs((rec[:, 6] - ref[:, 6] + np.pi) % (2 * np.pi) - np.pi)
                logger.info("V0 %s box inverse transform: max |dpos|=%.2e m max |dyaw|=%.2e rad",
                            fr["token"], float(np.abs(rec[:, :6] - ref[:, :6]).max()), float(dyaw.max()))
            canv = tr._observe(rmod, scen, tr.camera_models(rmod, tr.IDENT))
            for ch, q in paths.items():
                off = np.asarray(Image.open(q)).astype(np.int16)
                ours = tr.jpeg_roundtrip(canv[ch]).astype(np.int16)
                if off.shape != ours.shape:
                    rows.append((fr["token"], ch, float("nan"), float("nan"), f"shape {off.shape} vs {ours.shape}"))
                    continue
                d = np.abs(off - ours).max(-1)
                rows.append((fr["token"], ch, float(d.mean()), float((d > 40).mean()), ""))
            n_frames += 1
            if n_frames >= args.compare_official:
                break
        if n_frames >= args.compare_official:
            break
    for r in rows:
        logger.info("V0 %s %s mean|diff|=%.2f frac>40=%.4f %s", *r)
    good = [r for r in rows if r[4] == "" and r[3] < 0.01]
    logger.info("V0 summary: %d / %d camera images agree (frac>40 < 1%%) over %d frames", len(good), len(rows), n_frames)
    if not rows:
        raise SystemExit(f"V0: no official rasters found under {raster_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--metric-cache", type=Path, help="tokens to build (stage A)")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--split", default="test", help="navsim_logs/<split> (test = navtest)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-log", type=int, default=0, help="smoke test: first N tokens of each log")
    ap.add_argument("--max-logs", type=int, default=0, help="smoke test: first N logs")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compare-official", type=int, default=0, help="V0 mode: number of trainval frames")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args.data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    args.maps_root = os.environ["NUPLAN_MAPS_ROOT"]
    args.map_version = os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    args.logs_dir = args.data_root / "navsim_logs" / args.split
    if args.compare_official:
        compare_official(args)
        return
    if args.metric_cache is None or args.out_dir is None:
        raise SystemExit("--metric-cache and --out-dir are required (or --compare-official N)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_log = tokens_per_log(args)
    if args.per_log:
        per_log = {k: v[: args.per_log] for k, v in per_log.items()}
    if args.max_logs:
        per_log = dict(list(per_log.items())[: args.max_logs])
    logger.info("logs=%d tokens=%d -> %s", len(per_log), sum(len(v) for v in per_log.values()), args.out_dir)
    n_tok, n_missing, n_lights = 0, 0, 0
    with Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        for i, r in enumerate(pool.imap_unordered(build_log, list(per_log.items()), chunksize=1)):
            n_tok += r["n"]
            n_lights += r.get("lights_missing", 0)
            n_missing += len(r.get("missing", []))
            if r.get("missing"):
                logger.warning("%s: %d tokens not found in the log", r["log"], len(r["missing"]))
            if i % 20 == 0:
                logger.info("%d / %d logs, %d tokens", i + 1, len(per_log), n_tok)
    logger.info("Finished: tokens=%d missing=%d unplaceable_lights=%d out=%s", n_tok, n_missing, n_lights, args.out_dir)
    if n_missing:
        raise SystemExit(f"{n_missing} tokens missing")


if __name__ == "__main__":
    main()
