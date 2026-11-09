"""T2 feature-swap intervention, one rig preset, NAVSIM v1, official RAP weights, forward only.

Question: if the encoder represented every surface exactly as under the reference rig (the target of a
same-surface feature alignment loss, entry point 1), how much of the rig-change ΔPDMS would come back?
The decoder keeps the changed rig's calibration, so whatever remains is due to where the fixed 3D
reference points land (geometry), not to the encoder.

Per token:
  c1           raster, identity rig, official convention (t2_render), as T2
  new          raster under the preset's rig change, updated calibration, as T2
  swap_f0      new, but every F0 feature cell whose centre shows a surface also shown under c1 takes the
               c1 feature of that surface (t2_probe.warp_swap on t2_render.surface_pairs)
  swap_all     the same for F0, L0, R0 (all rendered cameras; B0 is black in both)
  swap_all_bg  control, not a bound: swap_all, and the remaining content cells take the c1 cell at the
               same position without warping, i.e. c1 content read with the moved calibration (like T2's
               stale presets); the smoke run showed it far worse than new
PDMS of the scorer pick, oracle@64 and mean64 for every variant (oracle64_v1.score_candidates).

Checks per token (failure -> row invalid with assert_fail=True; run exits non-zero after the CSV):
  - decode(encode(c1)) == model(c1); updated non-identity presets change lidar2img, ident does not
  - ident: every swap reproduces c1 (proposals and scores)
  - recomputed PDMS of the picks == official pdm_score (check_idx)
  - with --t2-tag: pdms / oracle / mean64 of c1 and new equal the T2 run of the same preset

    python run_t2_swap_v1.py --rap-root $RAP --agent-config rap_v2_official_agent.yaml \
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

logger = logging.getLogger("run_t2_swap_v1")

ATOL_MODEL = 1e-5
ATOL_SWAP_IDENT = 1e-4  # ident swaps re-sample c1 at its own cell centres (float32 grid_sample)
SWAPS = ("swap_f0", "swap_all", "swap_all_bg")
VARIANTS = ("c1", "new") + SWAPS
SLOTS = {"cam_f0": 1, "cam_l0": 2, "cam_r0": 3}  # RAP camera order b0, f0, l0, r0
T2_COLS = ("pdms_c1", "pdms_new", "oracle_c1", "oracle_new", "mean64_c1", "mean64_new")


def swap_logs(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        images, _ = tr.render(rmod, scenarios[token], d)
        for name in tr.CAMS:
            cam = getattr(agent_input.cameras[-1], name)
            cam.image = images[name].astype(np.float32)
            tr.update_model_camera(cam, d)
        return agent._features(agent_input)

    rows = []
    for idx, token in enumerate(tokens):
        if idx % 20 == 0:
            logger.info("thread %s: %d / %d", thread_id[:8], idx, len(tokens))
        row: Dict[str, Any] = {"token": token, "valid": True, "assert_fail": False, "error": ""}
        t0 = time.time()
        try:
            if token not in scenarios:
                raise RuntimeError("no stage A scenario for this token")
            scen = scenarios[token]
            with lzma.open(paths[token], "rb") as f:
                metric_cache = pickle.load(f)
            surf_c1 = tr.surface_map(rmod, scen, tr.IDENT)
            surf_new = tr.surface_map(rmod, scen, delta)
            pairs = {slot: tr.surface_pairs(surf_new[tr.CHANNELS[name]], surf_c1[tr.CHANNELS[name]],
                                            stride=tp.CELL_PX)[0] for name, slot in SLOTS.items()}
            with torch.no_grad():
                f_c1, f_new = features(token, tr.IDENT), features(token, delta)
                same = torch.equal(f_new["lidar2img"], f_c1["lidar2img"])
                if same != delta.is_identity_geometry():
                    raise AssertionError(f"lidar2img unchanged={same} for preset {job['preset']}")
                imf_c1, imf_new = rp.encode(model, f_c1)[0], rp.encode(model, f_new)[0]
                outs = {"c1": rp.decode(model, f_c1, imf_c1), "new": rp.decode(model, f_new, imf_new)}
                ref = model(f_c1, None, return_score=True)
                if not (torch.allclose(outs["c1"]["proposals"], ref["trajectory"], rtol=0, atol=ATOL_MODEL)
                        and torch.allclose(outs["c1"]["score"], ref["score"], rtol=0, atol=ATOL_MODEL)):
                    raise AssertionError("decode(encode(c1)) != model(c1)")
                for name, sel, fill in (("swap_f0", (1,), False), ("swap_all", (1, 2, 3), False),
                                        ("swap_all_bg", (1, 2, 3), True)):
                    imf_s, frac = tp.warp_swap(imf_new, imf_c1, {s: pairs[s] for s in sel}, fill_rest=fill)
                    outs[name] = rp.decode(model, f_new, imf_s)
                    if name != "swap_all_bg":
                        for s, v in frac.items():
                            row[f"swap_frac_s{s}"] = v
                if job["preset"] == "ident":
                    for name in SWAPS:
                        if not (torch.allclose(outs[name]["proposals"], outs["c1"]["proposals"], rtol=0, atol=ATOL_SWAP_IDENT)
                                and torch.allclose(outs[name]["score"], outs["c1"]["score"], rtol=0, atol=ATOL_SWAP_IDENT)):
                            raise AssertionError(f"preset ident: {name} does not reproduce c1")
                for name in SWAPS:
                    row[f"pade_{name}_new"] = rp.proposal_ade(outs[name]["proposals"], outs["new"]["proposals"])
                    row[f"pade_{name}_c1"] = rp.proposal_ade(outs[name]["proposals"], outs["c1"]["proposals"])
                row["pade_new_c1"] = rp.proposal_ade(outs["new"]["proposals"], outs["c1"]["proposals"])

            for k in VARIANTS:
                cands = outs[k]["proposals"][0, :, :8].float().cpu().numpy()
                sc = outs[k]["score"][0].float().cpu().numpy()
                pool = score_candidates(metric_cache, cands, simulator, scorer)
                sel = int(sc.argmax())
                row[f"pdms_{k}"] = float(pool[sel])
                row[f"oracle_{k}"] = float(pool.max())
                row[f"mean64_{k}"] = float(pool.mean())
                row[f"idx_{k}"] = sel
                if k != "c1":
                    check_idx(metric_cache, cands, pool, sel, simulator, scorer, k)
        except Exception as e:
            logger.warning("swap failed for token %s", token)
            traceback.print_exc()
            row.update(valid=False, assert_fail=isinstance(e, AssertionError), error=repr(e)[:300])
        row["t_token"] = time.time() - t0
        rows.append(row)
    return rows


def t2_scores(exp_root: Path, tag: str, preset: str) -> pd.DataFrame:
    """Per-token T2_COLS of the latest T2 run t2probe_<tag>_<preset>."""
    run_dir = exp_root / f"t2probe_{tag}_{preset}"
    csvs = sorted(run_dir.glob("*/*.csv"), key=lambda c: (int(c.parent.name) if c.parent.name.isdigit() else -1, c.name))
    if not csvs:
        raise SystemExit(f"no T2 CSV under {run_dir}")
    df = pd.read_csv(csvs[-1], index_col=0)
    df = df[(df["token"] != "average") & df["valid"].astype(bool)]
    return df.set_index("token")[list(T2_COLS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True, help="yaml with _target_ RAPV2Agent (not the rig agent)")
    ap.add_argument("--metric-cache", type=Path, required=True)
    ap.add_argument("--scenario-dir", type=Path, required=True, help="build_t2_scenarios.py output")
    ap.add_argument("--preset", required=True, help="t2_render.PRESETS name (plain rig change)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--t2-tag", default="", help="cross-check c1 / new scores against t2probe_<tag>_<preset>")
    ap.add_argument("--t2-atol", type=float, default=1e-6)
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
    if delta.calib != "updated" or delta.view_mask or delta.drop or getattr(delta, "scene", "full") != "full":
        raise SystemExit(f"preset {args.preset} is not a plain rig change (updated calibration, full scene, no mask / drop)")
    exp_root = Path(os.environ.get("NAVSIM_EXP_ROOT", "/gs/bs/tga-RLA/qdeng/navsim_workspace/exp"))
    t2 = t2_scores(exp_root, args.t2_tag, args.preset) if args.t2_tag else None
    logger.info("navsim=%s preset=%s delta=%s t2_tag=%s selfcheck=%.2fpx", nav, args.preset, delta,
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
        preset=args.preset,
        scenario_dir=str(args.scenario_dir),
        scene_filter=scene_filter,
        navsim_log_path=str(navsim_log_path),
        sensor_blobs_path=str(sensor_blobs_path),
        metric_cache_paths=metric_cache_paths,
    )
    data_points = [dict(job, log_file=log, tokens=toks) for log, toks in per_log.items()]
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.workers)
    df = pd.DataFrame(worker_map(worker, swap_logs, data_points))

    if t2 is not None:  # same tokens, same forward, same scorer: c1 / new must equal T2's
        valid = df["valid"].astype(bool)
        worst = 0.0
        bad = pd.Series(False, index=df.index)
        for c in T2_COLS:
            ref = df["token"].map(t2[c])
            diff = (df[c] - ref).abs()
            worst = max(worst, float(diff[valid].max()) if valid.any() else 0.0)
            bad |= valid & ((diff > args.t2_atol) | ref.isna())
        df.loc[bad, ["valid", "assert_fail"]] = [False, True]
        df.loc[bad, "error"] = "c1 / new scores differ from the T2 run"
        logger.info("T2 cross-check: rows=%d max|diff|=%.2e failed=%d", int(valid.sum()), worst, int(bad.sum()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}.csv"
    df.to_csv(out)
    ok = df[df["valid"].astype(bool)]
    n_assert = int(df["assert_fail"].astype(bool).sum())
    if len(ok):
        d_new = (ok["pdms_new"] - ok["pdms_c1"]).mean()
        msg = " ".join(f"d{k}={(ok[f'pdms_{k}'] - ok['pdms_c1']).mean():+.4f}" for k in VARIANTS[1:])
        rescue = " ".join(f"rescue_{k}={(ok[f'pdms_{k}'] - ok['pdms_new']).mean() / -d_new:.2f}" if d_new < 0 else ""
                          for k in SWAPS)
        logger.info("preset=%s valid=%d %s %s swap_frac_f0=%.3f", args.preset, len(ok), msg, rescue,
                    ok["swap_frac_s1"].mean())
    logger.info("Finished: rows=%d valid=%d assert_fail=%d t_token=%.2fs csv=%s",
                len(df), len(ok), n_assert, float(df["t_token"].mean()), out)
    if n_assert:
        raise SystemExit(f"{n_assert} rows failed a consistency check; see 'error' column in {out}")


if __name__ == "__main__":
    main()
