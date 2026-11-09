"""NAVSIM v1 PDMS for one agent config, on a process pool instead of Ray.

RAP's navsim/planning/script/run_pdm_score.py hard-codes RayDistributed with one
GPU per task, which fails on TSUBAME (psutil.AccessDenied). This keeps its
scoring path unchanged -- per-log tasks, SceneLoader ∩ MetricCacheLoader tokens,
navsim.evaluate.pdm_score with the default PDMSimulator/PDMScorer (== RAP's
default_scoring_parameters.yaml), average row = column mean over valid rows --
and only swaps the worker for nuplan's SingleMachineParallelExecutor.

Run with RAP's own navsim (v1) importable and the v2 devkit NOT on PYTHONPATH.

    python run_pdm_score_v1.py --rap-root $RAP --agent-config <agent.yaml> \
        --metric-cache $RAP/metric_cache_v1_navtest --output-dir <dir> --workers 4
"""
import argparse
import logging
import lzma
import os
import pickle
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger("run_pdm_score_v1")


def load_split(rap_root: Path, split: str):
    split_dir = rap_root / "navsim/planning/script/config/common/train_test_split"
    split_cfg = yaml.safe_load(open(split_dir / f"{split}.yaml"))
    filter_name = split_cfg["defaults"][0]["scene_filter"]
    raw = yaml.safe_load(open(split_dir / "scene_filter" / f"{filter_name}.yaml"))
    from navsim.common.dataclasses import SceneFilter

    return split_cfg["data_split"], SceneFilter(**{k: v for k, v in raw.items() if not k.startswith("_")})


def score_logs(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Worker: build the agent once, then score every token of the given logs."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import SceneLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    job = args[0]
    thread_id = str(uuid.uuid4())
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig())

    agent = instantiate(OmegaConf.load(job["agent_config"]))
    agent.initialize()

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
        if idx % 100 == 0:
            logger.info("thread %s: %d / %d", thread_id[:8], idx, len(tokens))
        row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            with lzma.open(paths[token], "rb") as f:
                metric_cache = pickle.load(f)
            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene_loader.get_scene_from_token(token))
            else:
                trajectory = agent.compute_trajectory(agent_input)
            if not isinstance(trajectory, Trajectory):
                raise TypeError(f"agent returned {type(trajectory).__module__}.{type(trajectory).__name__}")
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            row.update(asdict(result))
        except Exception:
            logger.warning("agent failed for token %s", token)
            traceback.print_exc()
            row["valid"] = False
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rap-root", type=Path, required=True)
    ap.add_argument("--agent-config", required=True, help="yaml with _target_ for the agent")
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
    # Every data point references the same job objects; pickle memoizes them once per chunk.
    data_points = [dict(job, log_file=log, tokens=toks) for log, toks in per_log.items()]

    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=args.workers)
    rows = worker_map(worker, score_logs, data_points)

    df = pd.DataFrame(rows)
    n_ok = int(df["valid"].sum())
    average = df.drop(columns=["token", "valid"]).mean(skipna=True)
    average["token"] = "average"
    average["valid"] = bool(df["valid"].all())
    df.loc[len(df)] = average

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{datetime.now().strftime('%Y.%m.%d.%H.%M.%S')}.csv"
    df.to_csv(out)
    logger.info("Finished: valid=%d failed=%d average score=%.4f csv=%s",
                n_ok, len(df) - 1 - n_ok, average["score"], out)


if __name__ == "__main__":
    main()
