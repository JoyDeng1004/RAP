"""NAVSIM v1 oracle@K: score PDM-Closed + K candidates in one simulate/score pass.

Only EP couples proposals (batch-max normalisation in PDMScorer._aggregate_scores),
so each candidate's stand-alone v1 PDMS is recovered by re-normalising EP pairwise
against PDM-Closed (index 0). All other sub-metrics are per-proposal.
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2

from navsim.common.dataclasses import Scene, Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

ATOL = 1e-6


def score_candidates(
    metric_cache: MetricCache,
    cands: npt.NDArray,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    return_subscores: bool = False,
) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]]:
    """
    :param cands: (K, 8, 3) local poses, same array the agent feeds into Trajectory().
    :param return_subscores: also return each sub-metric as {lower-case enum name: (K,)}, with
        progress being the pairwise-normalised EP that enters the score.
    :return: (K,) v1 PDMS of each candidate as if scored alone against PDM-Closed.
    """
    ego = metric_cache.ego_state
    sampling = simulator.proposal_sampling
    pdm_states = get_trajectory_as_array(metric_cache.trajectory, sampling, ego.time_point)
    cand_states = [
        get_trajectory_as_array(transform_trajectory(Trajectory(c), ego), sampling, ego.time_point) for c in cands
    ]
    states = np.stack([pdm_states, *cand_states], axis=0)

    simulated = simulator.simulate_proposals(states, ego)
    scorer.score_proposals(
        simulated,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    multi = scorer._multi_metrics.prod(axis=0)
    raw = scorer._progress_raw * multi
    weighted = scorer._weighted_metrics.copy()

    # pairwise EP vs PDM-Closed, mirroring _aggregate_scores with a 2-proposal batch
    pair_max = np.maximum(raw[0], raw)
    ep = (multi != 0.0).astype(np.float64)
    above = pair_max > scorer._config.progress_distance_threshold
    ep[above] = raw[above] / pair_max[above]
    weighted[WeightedMetricIndex.PROGRESS] = ep

    w = scorer._config.weighted_metrics_array
    ws = (weighted * w[..., None]).sum(axis=0)
    ws /= w.sum()
    pdms = (multi * ws)[1:]
    if not return_subscores:
        return pdms
    sub = {m.name.lower(): scorer._multi_metrics[m][1:].astype(np.float64) for m in MultiMetricIndex}
    sub.update({m.name.lower(): weighted[m][1:].astype(np.float64) for m in WeightedMetricIndex})
    return pdms, sub


def check_idx(
    metric_cache: MetricCache,
    cands: npt.NDArray,
    scores: npt.NDArray[np.float64],
    idx: int,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    tag: str,
) -> float:
    """Official pdm_score of cands[idx]; raises if the batched recomputation disagrees."""
    ref = float(pdm_score(metric_cache, Trajectory(cands[idx]), simulator.proposal_sampling, simulator, scorer).score)
    if abs(ref - scores[idx]) > ATOL:
        raise AssertionError(f"oracle64 {tag} mismatch idx={idx}: recomputed={scores[idx]!r} official={ref!r}")
    return ref


def gt_future_poses(scene_frames: List[Dict], num_history_frames: int, num_poses: int) -> npt.NDArray[np.float64]:
    """Scene.get_future_trajectory(num_poses).poses without building the map api or loading sensors."""
    start = num_history_frames - 1
    poses = [Scene._build_ego_status(scene_frames[i]).ego_pose for i in range(start, start + num_poses + 1)]
    return convert_absolute_to_relative_se2_array(StateSE2(*poses[0]), np.array(poses[1:], dtype=np.float64))


def ade(cands: npt.NDArray, gt_poses: npt.NDArray) -> npt.NDArray[np.float64]:
    """(K,) mean L2 over the xy waypoints."""
    return np.linalg.norm(cands[:, :, :2].astype(np.float64) - gt_poses[None, :, :2], axis=-1).mean(axis=-1)
