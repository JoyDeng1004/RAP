"""T2 factor experiment, one rig preset, NAVSIM v1, official RAP weights, forward only.

Per token (same token set / pool as run_t1_probe_v1.py):
  real     real images, original calibration (the T1 ``raw`` input)
  c1       raster, identity rig, official convention (t2_render), in the preset's scene (map_* presets:
           boxes and lights removed in c1 and new alike): the reference of every Delta
  new      raster under the preset's rig change; calibration updated or stale per the preset
  fix_all  c1 proposals teacher-forced in every round, all rounds read the ``new`` images
           (fixed candidates: separates generation from selection)
Statistics: PDMS of the scorer pick, oracle@64 and mean64 for real / c1 / new, plus the PDMS
sub-metrics of the pick (pick_<metric>_<variant> in the CSV) and of all 64 candidates (sub_<metric>_<variant>
in the npz: no_collision, drivable_area, progress (pairwise EP), ttc, comfortable, driving_direction); query cosine and
slot-matched proposal ADE per refiner round (new vs c1, c1 vs real); F0 SCA attention mass per round
and waypoint (rig_probe.sca_stats); round 1 / layer 0 flip statistics between c1 and new
(t2_probe.r1l0_flip); feature cosine at the same 3D reference points (t2_probe.anchor_cos);
raster coverage and view-mask fractions (t2_render).

Checks per token (failure -> row invalid with assert_fail=True; run exits non-zero after the CSV):
  - decode(encode(real)) == model(real) (proposals and scores)
  - c1 has the real input's lidar2img; stale presets keep it, updated non-identity presets change it
  - round 1 / layer 0 attention weights are identical under c1 and new (rig-independent queries)
  - fix_all returns exactly the c1 proposals; presets equal to their reference (ident, map_ident) reproduce c1
  map_scaled presets (H_scale) also record equiv_f0_*: t2_render.image_equivalence of their F0 input and the
  F0 input of the equivalent moved rig (t2_render.equivalent_of)
  - recomputed PDMS of the real / new picks == official pdm_score (< 1e-6)

    python run_t2_probe_v1.py --rap-root $RAP --agent-config rap_v2_official_agent.yaml \
        --metric-cache <subset cache> --scenario-dir <stage A dir> --preset h_m0.7 --output-dir <dir>
"""
import argparse
import logging
import lzma
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

logger = logging.getLogger("run_t2_probe_v1")

ATOL_MODEL = 1e-5
SLOT_F0 = 1  # RAP camera order b0, f0, l0, r0
CAND_VARIANTS = ("real", "c1", "new")
SCORE_VARIANTS = ("real", "c1", "new", "fix_all")


def score_logs_t2(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import torch
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from oracle64_v1 import check_idx, score_candidates
    from rap_eval_agents import rig_probe as rp
    from rap_eval_agents import t2_probe as tp
    from rap_eval_agents import t2_render as tr
    from rap_eval_agents.rap_v2_agent import RAPV2Agent

    job = args[0]
    thread_id = str(uuid.uuid4())
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())

    agent = instantiate(OmegaConf.load(job["agent_config"]))
    if type(agent) is not RAPV2Agent:
        raise SystemExit(f"use the plain RAPV2Agent config (the probe renders the rig itself), got {type(agent)}")
    agent.initialize()
    model = agent._model

    rmod = tr.renderer_module(Path(job["rap_root"]))
    tr.selfcheck_offset(rmod)
    delta = tr.get_delta(job["preset"])
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
    paths = job["metric_cache_paths"]
    tokens = [t for t in scene_filter.tokens if t in paths and t in scene_loader.scene_frames_dicts]

    def features(token, d):
        # RAPV2Agent._features edits the cameras in place: a fresh AgentInput for every variant.
        agent_input = scene_loader.get_agent_input_from_token(token)
        info: Dict[str, dict] = {}
        if d is not None:
            images, info = tr.render(rmod, scenarios[token], d)
            for name in tr.CAMS:
                cam = getattr(agent_input.cameras[-1], name)
                cam.image = images[name].astype(np.float32)
                tr.update_model_camera(cam, d)
        return agent._features(agent_input), info

    def cands(out):
        return (out["proposals"][0, :, :8].float().cpu().numpy(), out["score"][0].float().cpu().numpy())

    rows = []
    for idx, token in enumerate(tokens):
        if idx % 20 == 0:
            logger.info("thread %s: %d / %d", thread_id[:8], idx, len(tokens))
        row: Dict[str, Any] = {"token": token, "valid": True, "assert_fail": False, "error": ""}
        t0 = time.time()
        try:
            if token not in scenarios:
                raise RuntimeError("no stage A scenario for this token")
            with lzma.open(paths[token], "rb") as f:
                metric_cache = pickle.load(f)
            with torch.no_grad():
                f_real, _ = features(token, None)
                f_c1, info_c1 = features(token, tr.reference_of(delta))  # identity rig, same scene
                f_new, info_new = features(token, delta)
                if not torch.equal(f_c1["lidar2img"], f_real["lidar2img"]):
                    raise AssertionError("c1 calibration != real calibration (official raster convention)")
                same = torch.equal(f_new["lidar2img"], f_c1["lidar2img"])
                if same != (delta.calib == "stale" or delta.is_identity_geometry()):
                    raise AssertionError(f"lidar2img unchanged={same} does not match preset calib={delta.calib}")

                outs, recs, imfs = {}, {}, {}
                for k, f in (("real", f_real), ("c1", f_c1), ("new", f_new)):
                    imfs[k] = rp.encode(model, f)[0]
                    with rp.record_sca(model) as rec:
                        outs[k] = rp.decode(model, f, imfs[k])
                    recs[k] = rec
                ref = model(f_real, None, return_score=True)
                if not (torch.allclose(outs["real"]["proposals"], ref["trajectory"], rtol=0, atol=ATOL_MODEL)
                        and torch.allclose(outs["real"]["score"], ref["score"], rtol=0, atol=ATOL_MODEL)):
                    raise AssertionError("decode(encode(real)) != model(real)")
                outs["fix_all"] = rp.decode(model, f_new, imfs["new"], force=outs["c1"]["props"])
                if not torch.equal(outs["fix_all"]["proposals"], outs["c1"]["proposals"]):
                    raise AssertionError("fix_all changed the c1 proposals")
                if delta == tr.reference_of(delta) and not (
                        torch.equal(f_new["camera_feature"], f_c1["camera_feature"])
                        and torch.allclose(outs["new"]["proposals"], outs["c1"]["proposals"], rtol=0, atol=ATOL_MODEL)
                        and torch.allclose(outs["new"]["score"], outs["c1"]["score"], rtol=0, atol=ATOL_MODEL)):
                    raise AssertionError("identity-geometry preset does not reproduce c1")

                n_poses = outs["c1"]["proposals"].shape[2]
                for k in ("real", "c1", "new"):
                    row[f"_sca_{k}"] = rp.sca_stats(recs[k], {"real": f_real, "c1": f_c1, "new": f_new}[k],
                                                     n_poses, SLOT_F0).numpy()
                flip = tp.r1l0_flip(recs["c1"], recs["new"], f_new, n_poses, SLOT_F0)
                row["_flip_in_pose"] = flip.pop("_flip_in_pose")
                row.update({f"r1l0_{k}": v for k, v in flip.items()})
                for name, (a, b) in {"new_c1": ("c1", "new"), "c1_real": ("real", "c1")}.items():
                    raster = {"feats_a": f_c1, "feats_b": f_new} if name == "new_c1" else {}
                    ac = tp.anchor_cos(model, recs[a], imfs[a], recs[b], imfs[b], SLOT_F0, n_poses=n_poses, **raster)
                    for kk, vv in ac.items():  # anchor_cos(_p0..p3, _drawn, _bg), anchor_n, _pose_pt (npz)
                        row[f"{kk}_{name}"] = vv
                    o_a, o_b = outs[a], outs[b]
                    q_a = o_a["bev_in"][1:] + [o_a["bev_out"]]
                    q_b = o_b["bev_in"][1:] + [o_b["bev_out"]]
                    for r, (qa, qb) in enumerate(zip(q_a, q_b), start=1):
                        row[f"qcos_r{r}_{name}"] = rp.query_cos(qb, qa)
                    for r, (pa, pb) in enumerate(zip(o_a["props"], o_b["props"]), start=1):
                        row[f"pade_r{r}_{name}"] = rp.proposal_ade(pb, pa)
                    row[f"cover_{name}"] = rp.pool_cover(o_a["proposals"], o_b["proposals"])
                if delta.scene == "map_scaled":  # H_scale: how close the scaled map is to the moved rig
                    eq_imgs, _ = tr.render(rmod, scenarios[token], tr.equivalent_of(delta))
                    new_imgs, _ = tr.render(rmod, scenarios[token], delta)
                    for kk, vv in tr.image_equivalence(new_imgs["cam_f0"], eq_imgs["cam_f0"]).items():
                        row[f"equiv_f0_{kk}"] = vv
                for tag, info in (("c1", info_c1), ("new", info_new)):
                    for ch, v in info.items():
                        for kk, vv in v.items():
                            row[f"img_{tag}_{ch.lower()}_{kk}"] = vv

            arr = {k: cands(o) for k, o in outs.items()}
            scored = {k: score_candidates(metric_cache, arr[k][0], simulator, scorer, return_subscores=True)
                      for k in CAND_VARIANTS}
            s64 = {k: v[0] for k, v in scored.items()}
            for k in SCORE_VARIANTS:
                c, sc = arr[k]
                pool = s64[k] if k in s64 else s64["c1"]  # fix_all re-ranks the c1 pool
                sel = int(sc.argmax())
                row[f"pdms_{k}"] = float(pool[sel])
                row[f"idx_{k}"] = sel
                row[f"_score_{k}"] = sc
                if k in s64:
                    row[f"_cand_{k}"] = c
                    row[f"oracle_{k}"] = float(pool.max())
                    row[f"mean64_{k}"] = float(pool.mean())
                    row[f"_s64_{k}"] = pool
                    for name, v in scored[k][1].items():  # sub-metrics of every candidate and of the pick
                        row[f"_sub_{name}_{k}"] = v
                        row[f"pick_{name}_{k}"] = float(v[sel])
            for k in ("real", "new"):
                check_idx(metric_cache, arr[k][0], s64[k], row[f"idx_{k}"], simulator, scorer, k)
        except Exception as e:
            logger.warning("probe failed for token %s", token)
            traceback.print_exc()
            row.update(valid=False, assert_fail=isinstance(e, AssertionError), error=repr(e)[:300])
        row["t_token"] = time.time() - t0
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True, help="yaml with _target_ RAPV2Agent (not the rig agent)")
    ap.add_argument("--metric-cache", type=Path, required=True)
    ap.add_argument("--scenario-dir", type=Path, required=True, help="build_t2_scenarios.py output")
    ap.add_argument("--preset", required=True, help="t2_render.PRESETS name")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="navtest")
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
    delta = tr.get_delta(args.preset)  # unknown names fail here, before any GPU work
    logger.info("navsim=%s preset=%s delta=%s selfcheck=%.2fpx", nav, args.preset, delta,
                tr.selfcheck_offset(tr.renderer_module(args.rap_root)))

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
        preset=args.preset,
        scenario_dir=str(args.scenario_dir),
        scene_filter=scene_filter,
        navsim_log_path=str(navsim_log_path),
        sensor_blobs_path=str(sensor_blobs_path),
        metric_cache_paths=metric_cache_paths,
    )
    data_points = [dict(job, log_file=log, tokens=toks) for log, toks in per_log.items()]

    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.workers)
    rows = worker_map(worker, score_logs_t2, data_points)

    arrays: Dict[str, List[np.ndarray]] = {}
    arr_tokens = []
    for r in rows:
        if r["valid"]:
            arr_tokens.append(r["token"])
        for k in [k for k in r if k.startswith("_")]:
            v = r.pop(k)
            if r["valid"]:
                arrays.setdefault(k[1:], []).append(v)

    df = pd.DataFrame(rows)
    n_ok = int(df["valid"].sum())
    n_assert = int(df["assert_fail"].sum())
    ok = df[df["valid"]]
    d_new = float((ok["pdms_new"] - ok["pdms_c1"]).mean()) if n_ok else float("nan")
    d_c1 = float((ok["pdms_c1"] - ok["pdms_real"]).mean()) if n_ok else float("nan")
    average = df.drop(columns=["token", "valid", "assert_fail", "error"] + [c for c in df if c.startswith("idx_")],
                      errors="ignore").mean(skipna=True, numeric_only=True)
    average["token"] = "average"
    average["valid"] = bool(df["valid"].all())
    average["assert_fail"] = n_assert > 0
    df.loc[len(df)] = average

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out = args.output_dir / f"{stamp}.csv"
    df.to_csv(out)
    if arr_tokens:
        np.savez(args.output_dir / f"{stamp}_arrays.npz", token=np.array(arr_tokens),
                 **{k: np.stack(v) for k, v in arrays.items() if len(v) == len(arr_tokens)})
    keys = ("pdms_", "oracle_", "mean64_", "r1l0_flip_in", "r1l0_sink", "anchor_cos_", "qcos_r1_new", "qcos_r4_new")
    msg = " ".join(f"{c}={average[c]:.4f}" for c in df.columns if c.startswith(keys) and c in average)
    logger.info("Finished: preset=%s valid=%d failed=%d assert_fail=%d | dPDMS new-c1=%+.4f c1-real=%+.4f | %s "
                "| t_token=%.2fs csv=%s", args.preset, n_ok, len(df) - 1 - n_ok, n_assert, d_new, d_c1, msg,
                average["t_token"], out)
    if n_assert:
        raise SystemExit(f"{n_assert} tokens failed a probe consistency check; see 'error' column in {out}")


if __name__ == "__main__":
    main()
