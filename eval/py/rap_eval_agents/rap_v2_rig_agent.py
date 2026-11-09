"""RAPV2Agent with an Exp-0 / T1 camera-rig change applied to every input.

Kept outside rap_v2_agent.py so the audited agent file is untouched. The change
is chosen by name (rig_transform.PRESETS); ``raw`` leaves the input untouched
and must reproduce RAPV2Agent's CSVs token for token.
"""
import logging

from navsim.common.dataclasses import AgentInput

from .rap_v2_agent import RAPV2Agent
from .rig_transform import apply_rig, get_spec

logger = logging.getLogger(__name__)


def _as_bool(x) -> bool:
    # Values come from ${oc.env:...}, i.e. strings such as "0" / "1".
    return str(x).strip().lower() in ("1", "true", "yes")


class RAPV2RigAgent(RAPV2Agent):
    def __init__(self, *args, rig_preset: str = "raw", undistort="0", **kwargs):
        super().__init__(*args, **kwargs)
        self._rig_preset = str(rig_preset)
        self._rig = get_spec(self._rig_preset, _as_bool(undistort))  # unknown names fail here, before any GPU work
        self._logged = False

    def name(self) -> str:
        return f"RAPV2RigAgent_{self._selector}_{self._rig_preset}"

    def _features(self, agent_input: AgentInput):
        # Like the parent, this edits the cameras in place: never feed one AgentInput twice.
        stats = apply_rig(agent_input.cameras[-1], self._rig)
        if not self._logged:
            logger.info("rig preset=%s spec=%s invalid_frac=%s", self._rig_preset, self._rig,
                        {k: round(v["invalid_frac"], 4) for k, v in stats.items()})
            self._logged = True
        return super()._features(agent_input)
