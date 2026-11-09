"""T2 surface-correspondence diagnostic, NAVSIM v1, official RAP weights, forward only, no PDM scoring.

Question: under a rig change, does the encoder represent the *same surface* differently, or do the
round-1 3D reference points just land on *different surfaces*? T2's ``anchor_cos`` (feature cosine at
the projections of the same 3D reference points) mixes both. Per token and preset this run measures

  surf_cos          F0 features at pixels of c1 and of the changed rig that show the same surface point
                    (t2_render.surface_pairs on the per-pixel surfaces of the view-mask code)
  surf_samepix_cos  the same pairs, changed rig read at the c1 pixel (correspondence ignored)
  anchor_cos        as T2 (recomputed; must match the T2 CSV of the same preset when --t2-tag is given),
                    split into anchors on a drawn pixel (anchor_cos_drawn) and on black background
  surf_rgb_diff     how much the raster itself changes at the matched pixels (renderer depth shading)

c1 = raster under the identity rig (official convention); presets from t2_render.PRESETS, rendered with
the updated calibration as in T2. One job runs all presets: c1 is rendered and encoded once per token.

Checks (failure -> row invalid with assert_fail=True; run exits non-zero after the CSV):
  - decode(encode(c1)) == model(c1)
  - updated non-identity presets change lidar2img; ident does not and reproduces c1 exactly
  - ident: every surface sample matches itself, surf_cos == 1, zero shift
  - with --t2-tag: anchor_cos per token equals the T2 run of the same preset (|diff| <= --t2-atol)

    python run_t2_surface_v1.py --rap-root $RAP --agent-config rap_v2_official_agent.yaml \
        --metric-cache <subset cache> --scenario-dir <stage A dir> --output-dir <dir> [--t2-tag full]
"""
import argparse
import logging
import os
import pickle
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from run_pdm_score_v1 import load_split

logger = logging.getLogger("run_t2_surface_v1")

ATOL_MODEL = 1e-5
SLOT_F0 = 1  # RAP camera order b0, f0, l0, r0
CH_F0 = "CAM_F0"
PRESETS = ("ident", "h_p1.0", "h_m0.7", "d_p1.0", "d_m1.0", "l_p0.5", "l_m0.5", "f_0.8", "f_1.25")


def surface_logs(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import torch
    from navsim.common.dataloader import SceneLoader
    from rap_eval_agents import rig_probe as rp
    from rap_eval_agents import t2_probe as tp
    from rap_eval_agents import t2_render as tr
    from rap_eval_agents.rap_v2_agent import RAPV2Agent

    job = args[0]
    thread_id = str(uuid.uuid4())
    agent = instantiate(OmegaConf.load(job["agent_config"]))
    if type(agent) is not RAPV2Agent:
        raise SystemExit(f"use the plain RAPV2Agent config (the probe renders the rig itself), got {type(agent)}")
    agent.initialize()
    model = agent._model

    rmod = tr.renderer_module(Path(job["rap_root"]))
    tr.selfcheck_offset(rmod)
    deltas = {p: tr.get_delta(p) for p in job["presets"]}
    scenarios: Dict[str, dict] = {}
    for a in args:
        with open(Path(job["scenario_dir"]) / f"{a['log_file']}.pkl", "rb") as f:
            scenarios.update(pickle.load(f))

    scene_filter = job["scene_filter"]
    scene_filter.log_names = [a["log_file"] for a in args]
    scene_filter.tokens = [t for a in args for t in a["tokens"]]
    scene_loader = SceneLoader(
        data_path=Path(job["navsim_log_path"]),
        sensor_blobs_path=Path(job["sensor_blobs_path"]),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    tokens = [t for t in scene_filter.tokens if t in scene_loader.scene_frames_dicts]

    def features(token, d):
        # RAPV2Agent._features edits the cameras in place: a fresh AgentInput for every variant.
        agent_input = scene_loader.get_agent_input_from_token(token)
        images, _ = tr.render(rmod, scenarios[token], d)
        for name in tr.CAMS:
            cam = getattr(agent_input.cameras[-1], name)
            cam.image = images[name].astype(np.float32)
            tr.update_model_camera(cam, d)
        return agent._features(agent_input), images["cam_f0"]

    def run(f):
        imf = rp.encode(model, f)[0]
        with rp.record_sca(model) as rec:
            out = rp.decode(model, f, imf)
        return imf, rec, out

    rows = []
    for idx, token in enumerate(tokens):
        if idx % 20 == 0:
            logger.info("thread %s: %d / %d", thread_id[:8], idx, len(tokens))
        t0 = time.time()
        try:
            if token not in scenarios:
                raise RuntimeError("no stage A scenario for this token")
            scen = scenarios[token]
            with torch.no_grad():
                f_c1, img_c1 = features(token, tr.IDENT)
                imf_c1, rec_c1, out_c1 = run(f_c1)
                ref = model(f_c1, None, return_score=True)
                if not (torch.allclose(out_c1["proposals"], ref["trajectory"], rtol=0, atol=ATOL_MODEL)
                        and torch.allclose(out_c1["score"], ref["score"], rtol=0, atol=ATOL_MODEL)):
                    raise AssertionError("decode(encode(c1)) != model(c1)")
            surf_c1 = tr.surface_map(rmod, scen, tr.IDENT, (CH_F0,))[CH_F0]
        except Exception as e:
            logger.warning("c1 failed for token %s", token)
            traceback.print_exc()
            for p in deltas:
                rows.append({"token": token, "preset": p, "valid": False, "assert_fail": isinstance(e, AssertionError),
                             "error": repr(e)[:300], "t_token": time.time() - t0})
            continue

        for p, d in deltas.items():
            row: Dict[str, Any] = {"token": token, "preset": p, "valid": True, "assert_fail": False, "error": ""}
            t1 = time.time()
            try:
                with torch.no_grad():
                    f_new, img_new = features(token, d)
                    same = torch.equal(f_new["lidar2img"], f_c1["lidar2img"])
                    if same != d.is_identity_geometry():
                        raise AssertionError(f"lidar2img unchanged={same} for preset {p}")
                    imf_new, rec_new, out_new = run(f_new)
                    if p == "ident" and not (
                            torch.equal(f_new["camera_feature"], f_c1["camera_feature"])
                            and torch.allclose(out_new["proposals"], out_c1["proposals"], rtol=0, atol=ATOL_MODEL)):
                        raise AssertionError("preset ident does not reproduce c1")
                    row.update(tp.anchor_cos(model, rec_c1, imf_c1, rec_new, imf_new, SLOT_F0, f_c1, f_new))
                    surf_new = tr.surface_map(rmod, scen, d, (CH_F0,))[CH_F0]
                    pairs, counts = tr.surface_pairs(surf_c1, surf_new, stride=job["stride"])
                    row.update(counts)
                    row["surf_match_frac"] = counts["n_matched"] / max(counts["n_surface"], 1)
                    row.update(tp.surface_cos(model, imf_c1, imf_new, pairs, SLOT_F0))
                    row["surf_rgb_diff"] = tr.surface_rgb_diff(img_c1, img_new, pairs)
                    row["surf_depth"] = float(pairs["depth"].mean()) if counts["n_matched"] else float("nan")
                # ident round trip is exact up to the renderer's float32 rotation (~1e-4 px, 40 px cells)
                if p == "ident" and counts["n_surface"] and not (
                        counts["n_matched"] == counts["n_surface"] and row["surf_cos"] > 1 - 1e-4
                        and row["surf_shift_cells"] < 1e-4 and row["surf_rgb_diff"] == 0):
                    raise AssertionError(f"ident surface pairs are not the identity: {counts}, "
                                         f"cos={row['surf_cos']}, shift={row['surf_shift_cells']}")
            except Exception as e:
                logger.warning("preset %s failed for token %s", p, token)
                traceback.print_exc()
                row.update(valid=False, assert_fail=isinstance(e, AssertionError), error=repr(e)[:300])
            row["t_token"] = time.time() - t1
            rows.append(row)
    return rows


def t2_anchor(exp_root: Path, tag: str, preset: str) -> pd.Series:
    """Per-token anchor_cos_new_c1 of the latest T2 run t2probe_<tag>_<preset>."""
    run_dir = exp_root / f"t2probe_{tag}_{preset}"
    csvs = sorted(run_dir.glob("*/*.csv"), key=lambda c: (int(c.parent.name) if c.parent.name.isdigit() else -1, c.name))
    if not csvs:
        raise SystemExit(f"no T2 CSV under {run_dir}")
    df = pd.read_csv(csvs[-1], index_col=0)
    df = df[(df["token"] != "average") & df["valid"].astype(bool)]
    return df.set_index("token")["anchor_cos_new_c1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True, help="yaml with _target_ RAPV2Agent (not the rig agent)")
    ap.add_argument("--metric-cache", type=Path, required=True, help="defines the token set (as T2)")
    ap.add_argument("--scenario-dir", type=Path, required=True, help="build_t2_scenarios.py output")
    ap.add_argument("--presets", nargs="+", default=list(PRESETS), help="t2_render.PRESETS names")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--stride", type=int, default=20, help="surface sample grid on the 1920 x 1080 crop (px)")
    ap.add_argument("--t2-tag", default="", help="cross-check anchor_cos against t2probe_<tag>_<preset> runs")
    ap.add_argument("--t2-atol", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-log", type=int, default=0, help="smoke test: first N tokens of each log")
    ap.add_argument("--max-logs", type=int, default=0, help="smoke test: first N logs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import navsim
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    from nuplan.planning.utils.multithreading.worker_utils import worker_map
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader
    from rap_eval_agents import t2_render as tr

    nav = str(Path(navsim.__file__).resolve())
    if not nav.startswith(str(args.rap_root.resolve())):
        raise SystemExit(f"navsim resolves to {nav}, expected RAP's v1 copy under {args.rap_root}")
    for p in args.presets:  # unknown names / non-T2-base presets fail here, before any GPU work
        d = tr.get_delta(p)
        if d.calib != "updated" or d.view_mask or d.drop or d.scene != "full":
            raise SystemExit(f"preset {p} is not a plain rig change (updated calibration, full scene, no mask / drop)")
    exp_root = Path(os.environ.get("NAVSIM_EXP_ROOT", "/gs/bs/tga-RLA/qdeng/navsim_workspace/exp"))
    t2 = {p: t2_anchor(exp_root, args.t2_tag, p) for p in args.presets} if args.t2_tag else {}
    logger.info("navsim=%s presets=%s stride=%d t2_tag=%s selfcheck=%.2fpx", nav, args.presets, args.stride,
                args.t2_tag or "-", tr.selfcheck_offset(tr.renderer_module(args.rap_root)))

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_split, scene_filter = load_split(args.rap_root, args.split)
    navsim_log_path = data_root / "navsim_logs" / data_split
    sensor_blobs_path = data_root / "sensor_blobs" / data_split
    scene_loader = SceneLoader(
        data_path=navsim_log_path, sensor_blobs_path=None,
        scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    per_log = {log: [t for t in toks if t in metric_cache_paths]
               for log, toks in scene_loader.get_tokens_list_per_log().items()}
    per_log = {log: toks for log, toks in per_log.items() if toks}
    if args.per_log:
        per_log = {log: sorted(toks)[: args.per_log] for log, toks in per_log.items()}
    if args.max_logs:
        per_log = dict(sorted(per_log.items())[: args.max_logs])
    missing = [log for log in per_log if not (args.scenario_dir / f"{log}.pkl").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} logs have no stage A scenarios in {args.scenario_dir}, e.g. {missing[:3]}")
    logger.info("tokens=%d logs=%d", sum(len(t) for t in per_log.values()), len(per_log))

    job = dict(
        agent_config=str(Path(args.agent_config).resolve()),
        rap_root=str(args.rap_root),
        presets=list(args.presets),
        stride=args.stride,
        scenario_dir=str(args.scenario_dir),
        scene_filter=scene_filter,
        navsim_log_path=str(navsim_log_path),
        sensor_blobs_path=str(sensor_blobs_path),
    )
    data_points = [dict(job, log_file=log, tokens=toks) for log, toks in per_log.items()]
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.workers)
    df = pd.DataFrame(worker_map(worker, surface_logs, data_points))

    if t2:  # same tokens, same forward: the recomputed anchor cosine must equal T2's
        df["anchor_cos_t2"] = [t2[p].get(t, np.nan) for t, p in zip(df["token"], df["preset"])]
        diff = (df["anchor_cos"] - df["anchor_cos_t2"]).abs()
        bad = df["valid"].astype(bool) & (diff > args.t2_atol)
        df.loc[bad, ["valid", "assert_fail"]] = [False, True]
        df.loc[bad, "error"] = "anchor_cos differs from the T2 run"
        logger.info("T2 cross-check: compared=%d max|diff|=%.2e over tolerance=%d",
                    int(diff.notna().sum()), float(diff.max()), int(bad.sum()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}.csv"
    df.to_csv(out)
    ok = df[df["valid"].astype(bool)]
    cols = ["surf_cos", "surf_samepix_cos", "anchor_cos", "anchor_cos_drawn", "surf_match_frac", "surf_rgb_diff"]
    for p in args.presets:
        g = ok[ok["preset"] == p]
        logger.info("preset=%-7s valid=%d %s", p, len(g), " ".join(f"{c}={g[c].mean():.4f}" for c in cols if c in g))
    n_assert = int(df["assert_fail"].astype(bool).sum())
    logger.info("Finished: rows=%d valid=%d assert_fail=%d t_token=%.2fs csv=%s",
                len(df), len(ok), n_assert, float(df["t_token"].mean()), out)
    if n_assert:
        raise SystemExit(f"{n_assert} rows failed a consistency check; see 'error' column in {out}")


if __name__ == "__main__":
    main()
