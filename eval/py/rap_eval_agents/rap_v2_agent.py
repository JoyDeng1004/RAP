"""RAP (vita-epfl/RAP) as an inference-only NAVSIM agent (v2 devkit, or RAP's own v1 navsim).

RAP's model code has no absolute ``navsim`` imports, so it is imported from the
RAP checkout as the top-level package ``rap_dino``; ``navsim`` stays bound to
the v2 devkit. Image preprocessing reuses RAP's real-image branch unchanged.
"""
import inspect
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

logger = logging.getLogger(__name__)

CAMS = ("cam_b0", "cam_f0", "cam_l0", "cam_r0")  # order of RAP's LoadMultiViewImageFromFiles


BACKBONE = "_backbone.img_backbone."
# Training-only module (domain adversarial loss); forward skips it in eval mode.
TRAIN_ONLY = ("domain_classifier.",)


def load_heads_only(model: torch.nn.Module, state: dict) -> None:
    """Load everything but the frozen DINOv3 backbone, which timm already holds.

    Official RAP checkpoints name the backbone in HF transformers style; its
    tensors equal the pretrained weights (checked by inspect_rap_ckpt.py).
    """
    heads = {k: v for k, v in state.items() if not k.startswith(BACKBONE)}
    missing, unexpected = model.load_state_dict(heads, strict=False)
    bad = [k for k in missing if not k.startswith(BACKBONE) and not k.startswith(TRAIN_ONLY)]
    if bad or unexpected:
        raise RuntimeError(f"head keys do not match: missing={bad[:10]} unexpected={list(unexpected)[:10]}")
    skipped = [k for k in missing if k.startswith(TRAIN_ONLY)]
    logger.info("loaded %d head tensors; kept pretrained backbone; not in ckpt (training-only): %d",
                len(heads), len(skipped))


class RAPV2Agent(AbstractAgent):
    def __init__(
        self,
        rap_root: str,
        checkpoint_path: str,
        dino_ckpt_path: str,
        model_time_horizon: float = 5.0,  # training value; E1 predicts 10 poses
        selector: str = "scorer",  # "scorer" = official argmax, "medoid" = scorer-free
        load_backbone: bool = True,  # False: keep timm's pretrained DINOv3 (official RAP ckpts use HF names)
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        if "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters:
            super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=False)
        else:  # RAP's v1 navsim: AbstractAgent(requires_scene) only
            super().__init__(requires_scene=False)
            self._trajectory_sampling = trajectory_sampling
        assert selector in ("scorer", "medoid"), selector
        self._rap_root = rap_root
        self._checkpoint_path = checkpoint_path
        self._dino_ckpt_path = dino_ckpt_path
        self._model_time_horizon = model_time_horizon
        self._selector = selector
        self._load_backbone = load_backbone
        self._model = None

    def name(self) -> str:
        return f"RAPV2Agent_{self._selector}"

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig(
            cam_f0=[3], cam_l0=[3], cam_l1=False, cam_l2=False,
            cam_r0=[3], cam_r1=False, cam_r2=False, cam_b0=[3], lidar_pc=False,
        )

    def initialize(self) -> None:
        # Called inside each scoring worker; the parent process never touches CUDA.
        if not torch.cuda.is_available():
            raise RuntimeError("RAPV2Agent needs a GPU; submit to a GPU resource type")
        t0 = time.time()
        agents_dir = str(Path(self._rap_root) / "navsim" / "agents")
        if agents_dir not in sys.path:
            sys.path.append(agents_dir)  # append: must not shadow the v2 devkit
        from rap_dino.navsim_config import RAPConfig
        from rap_dino.rap_model import RAPModel
        from rap_dino.bevformer import bev_feature_build

        self._bfb = bev_feature_build
        config = RAPConfig(
            trajectory_sampling=TrajectorySampling(time_horizon=self._model_time_horizon, interval_length=0.5),
            dino_ckpt_path=self._dino_ckpt_path,
            distill_feature=True,
            pdm_scorer=True,
        )
        model = RAPModel(config)
        state = torch.load(self._checkpoint_path, map_location="cpu")["state_dict"]
        prefix = "agent._rap_model."
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if self._load_backbone:
            model.load_state_dict(state, strict=True)  # a shape error here means a wrong model_time_horizon
        else:
            load_heads_only(model, state)
        del state
        self._device = torch.device("cuda")
        self._model = model.eval().to(self._device)
        import rap_dino
        logger.info(
            "RAPV2Agent ready pid=%d selector=%s load_backbone=%s ckpt=%s rap_dino=%s gpu_mem=%.1fGB load=%.0fs",
            os.getpid(), self._selector, self._load_backbone, self._checkpoint_path, list(rap_dino.__path__),
            torch.cuda.memory_allocated() / 2**30, time.time() - t0,
        )

    def _features(self, agent_input: AgentInput):
        # v2 Camera has no real_valid / rendered_* fields; RAP's real branch reads only real_valid.
        # RAP's v1 loader substitutes a black image and real_valid=False when a file is missing.
        for cams in agent_input.cameras:
            for name in CAMS:
                cam = getattr(cams, name)
                if getattr(cam, "real_valid", None) is False:
                    raise RuntimeError(f"{name}: real camera image missing")
                cam.real_valid = cam.image is not None
        bfb = self._bfb
        r = bfb.LoadMultiViewImageFromFiles(agent_input, synthetic=False)
        r = bfb.NormalizeMultiviewImage(r)
        r = bfb.RandomScaleImageMultiViewImage(r)  # deterministic resize to 432x768, rescales lidar2img
        r = bfb.PadMultiViewImage(r)
        if len(r["img"]) != 4:
            raise RuntimeError(f"expected 4 cameras, got {len(r['img'])}")
        ego = [
            np.concatenate([s.ego_pose, s.ego_velocity, s.ego_acceleration, s.driving_command]).astype(np.float32)
            for s in agent_input.ego_statuses if s is not None
        ]
        feats = {
            "camera_feature": torch.tensor(np.ascontiguousarray(np.stack([im.transpose(2, 0, 1) for im in r["img"]]))),
            "img_shape": torch.FloatTensor(np.stack(r["img_shape"])),
            "lidar2img": torch.FloatTensor(np.stack(r["lidar2img"])),
            "ego_status": torch.from_numpy(np.stack(ego)),
        }
        return {k: v.unsqueeze(0).to(self._device) for k, v in feats.items()}

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        if self._model is None:
            self.initialize()
        with torch.no_grad():
            pred = self._model(self._features(agent_input), None, return_score=True)
        proposals = pred["trajectory"][0, :, :8].float().cpu().numpy()  # (64, 8, 3): first 4 s
        if self._selector == "scorer":
            idx = int(pred["score"][0].argmax())
        else:
            xy = proposals[..., :2]
            idx = int(np.linalg.norm(xy[:, None] - xy[None], axis=-1).mean(-1).sum(1).argmin())
        return Trajectory(proposals[idx], self._trajectory_sampling)
