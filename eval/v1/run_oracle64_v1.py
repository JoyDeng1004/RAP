"""NAVSIM v1 oracle@64 for RAPV2Agent, same token set / process pool as run_pdm_score_v1.py.

Per token: PDM-Closed + all 64 proposals are simulated and scored in one batch,
EP is re-normalised pairwise (oracle64_v1.score_candidates), and the scorer and
medoid picks are re-scored with the official pdm_score as an exact check.

Checks per token (a failure marks the row invalid with assert_fail=True; the run
exits non-zero after writing the CSV):
  - compute_candidates()[selected idx] == agent.compute_trajectory() (audited path),
    each on its own freshly loaded AgentInput: RAPV2Agent._features mutates the
    cameras in place, so one AgentInput cannot be featurised twice
  - recomputed score of the scorer / medoid pick == official pdm_score (< 1e-6)

Outputs <ts>.csv (per-token rows + average row) and <ts>_arrays.npz
(token, score64, logit64, ade64; N x 64, valid rows only).

    python run_oracle64_v1.py --rap-root $RAP --agent-config <agent.yaml> \
        --metric-cache $RAP/metric_cache_v1_navtest --output-dir <dir> --workers 4
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

logger = logging.getLogger("run_oracle64_v1")

GUARD_ATOL = 1e-5
ARRAY_KEYS = ("_s64", "_logit64", "_ade64")
INDEX_COLS = ["idx_oracle", "idx_scorer", "idx_medoid"]


def score_logs_oracle(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Worker: build the agent once, then oracle-score every token of the given logs."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from oracle64_v1 import ade, check_idx, gt_future_poses, score_candidates
    from rap_eval_agents.rap_v2_candidates import compute_candidates

    job = args[0]
    thread_id = str(uuid.uuid4())
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())
    assert scorer._config.progress_distance_threshold == 5.0, scorer._config.progress_distance_threshold

    agent = instantiate(OmegaConf.load(job["agent_config"]))
    agent.initialize()
    selector = agent._selector

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

    rows = []
    for idx, token in enumerate(tokens):
        if idx % 20 == 0:
            logger.info("thread %s: %d / %d", thread_id[:8], idx, len(tokens))
        row: Dict[str, Any] = {"token": token, "valid": True, "assert_fail": False, "error": ""}
        t0 = time.time()
        try:
            with lzma.open(paths[token], "rb") as f:
                metric_cache = pickle.load(f)
            agent_input = scene_loader.get_agent_input_from_token(token)
            cands, logits, i_scorer, i_medoid = compute_candidates(agent, agent_input)
            i_sel = i_scorer if selector == "scorer" else i_medoid
            trajectory = agent.compute_trajectory(scene_loader.get_agent_input_from_token(token))
            if not np.allclose(trajectory.poses, cands[i_sel], rtol=0.0, atol=GUARD_ATOL):
                raise AssertionError(f"candidate path != agent.compute_trajectory (selector={selector}, idx={i_sel})")

            s = score_candidates(metric_cache, cands, simulator, scorer)
            s_scorer = check_idx(metric_cache, cands, s, i_scorer, simulator, scorer, "scorer")
            s_medoid = check_idx(metric_cache, cands, s, i_medoid, simulator, scorer, "medoid")

            gt = gt_future_poses(scene_loader.scene_frames_dicts[token], scene_filter.num_history_frames, cands.shape[1])
            d = ade(cands, gt)
            i_oracle = int(s.argmax())
            row.update(
                oracle64=float(s[i_oracle]), mean64=float(s.mean()),
                score_scorer=s_scorer, score_medoid=s_medoid, frac_zero64=float((s == 0.0).mean()),
                minADE64=float(d.min()), ade_scorer=float(d[i_scorer]), ade_medoid=float(d[i_medoid]),
                ade_oracle=float(d[i_oracle]),
                idx_oracle=i_oracle, idx_scorer=i_scorer, idx_medoid=i_medoid,
                _s64=s, _logit64=logits, _ade64=d,
            )
        except Exception as e:
            logger.warning("oracle failed for token %s", token)
            traceback.print_exc()
            row.update(valid=False, assert_fail=isinstance(e, AssertionError), error=repr(e)[:300])
        row["t_token"] = time.time() - t0
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True, help="yaml with _target_ for RAPV2Agent")
    ap.add_argument("--metric-cache", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-log", type=int, default=0, help="smoke test: first N tokens of each log")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import navsim
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    from nuplan.planning.utils.multithreading.worker_utils import worker_map
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader

    nav = str(Path(navsim.__file__).resolve())
    if not nav.startswith(str(args.rap_root.resolve())):
        raise SystemExit(f"navsim resolves to {nav}, expected RAP's v1 copy under {args.rap_root}")
    logger.info("navsim=%s", nav)

    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_split, scene_filter = load_split(args.rap_root, args.split)
    navsim_log_path = data_root / "navsim_logs" / data_split
    sensor_blobs_path = data_root / "sensor_blobs" / data_split

    scene_loader = SceneLoader(
        data_path=navsim_log_path, sensor_blobs_path=None,
        scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_paths = MetricCacheLoader(args.metric_cache).metric_cache_paths
    per_log = scene_loader.get_tokens_list_per_log()
    if args.per_log:
        per_log = {log: sorted(toks)[: args.per_log] for log, toks in per_log.items()}
    all_tokens = {t for toks in per_log.values() for t in toks}
    missing = all_tokens - set(metric_cache_paths)
    logger.info("tokens=%d logs=%d missing_metric_cache=%d unused_metric_cache=%d",
                len(all_tokens), len(per_log), len(missing), len(set(metric_cache_paths) - all_tokens))

    job = dict(
        agent_config=str(Path(args.agent_config).resolve()),
        scene_filter=scene_filter,
        navsim_log_path=str(navsim_log_path),
        sensor_blobs_path=str(sensor_blobs_path),
        metric_cache_paths=metric_cache_paths,
    )
    data_points = [dict(job, log_file=log, tokens=toks) for log, toks in per_log.items()]

    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.workers)
    rows = worker_map(worker, score_logs_oracle, data_points)

    arr_tokens, arrays = [], {k: [] for k in ARRAY_KEYS}
    for r in rows:
        if "_s64" in r:
            arr_tokens.append(r["token"])
            for k in ARRAY_KEYS:
                arrays[k].append(r.pop(k))

    df = pd.DataFrame(rows)
    n_ok = int(df["valid"].sum())
    n_assert = int(df["assert_fail"].sum())
    average = df.drop(columns=["token", "valid", "assert_fail", "error", *INDEX_COLS], errors="ignore").mean(skipna=True)
    average["token"] = "average"
    average["valid"] = bool(df["valid"].all())
    average["assert_fail"] = n_assert > 0
    df.loc[len(df)] = average

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out = args.output_dir / f"{stamp}.csv"
    df.to_csv(out)
    if arr_tokens:
        np.savez(
            args.output_dir / f"{stamp}_arrays.npz",
            token=np.array(arr_tokens),
            score64=np.stack(arrays["_s64"]),
            logit64=np.stack(arrays["_logit64"]),
            ade64=np.stack(arrays["_ade64"]),
        )
    logger.info(
        "Finished: valid=%d failed=%d assert_fail=%d | oracle64=%.4f mean64=%.4f scorer=%.4f medoid=%.4f "
        "minADE64=%.3f ade_scorer=%.3f | t_token=%.2fs csv=%s",
        n_ok, len(df) - 1 - n_ok, n_assert, average.get("oracle64", np.nan), average.get("mean64", np.nan),
        average.get("score_scorer", np.nan), average.get("score_medoid", np.nan),
        average.get("minADE64", np.nan), average.get("ade_scorer", np.nan), average["t_token"], out,
    )
    if n_assert:
        raise SystemExit(f"{n_assert} tokens failed an oracle64 consistency check; see 'error' column in {out}")


if __name__ == "__main__":
    main()
