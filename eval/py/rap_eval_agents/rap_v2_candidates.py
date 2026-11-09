"""All 64 RAP proposals for one AgentInput, mirroring RAPV2Agent.compute_trajectory.

Kept outside rap_v2_agent.py so the audited agent file is untouched; the oracle
runner cross-checks this path against agent.compute_trajectory on every token.
"""
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch

from navsim.common.dataclasses import AgentInput


def compute_candidates(agent, agent_input: AgentInput) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], int, int]:
    """
    :return: proposals (64, 8, 3) float32 local poses, scorer scores (64,),
        index picked by selector="scorer", index picked by selector="medoid".
    """
    if agent._model is None:
        agent.initialize()
    with torch.no_grad():
        pred = agent._model(agent._features(agent_input), None, return_score=True)
    proposals = pred["trajectory"][0, :, :8].float().cpu().numpy()  # (64, 8, 3): first 4 s
    scores = pred["score"][0].float().cpu().numpy()
    idx_scorer = int(pred["score"][0].argmax())
    xy = proposals[..., :2]
    idx_medoid = int(np.linalg.norm(xy[:, None] - xy[None], axis=-1).mean(-1).sum(1).argmin())
    return proposals, scores, idx_scorer, idx_medoid
