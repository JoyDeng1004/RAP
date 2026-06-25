"""Dataset-agnostic intermediate representation, all in the EGO frame.

Adapters (nuScenes / NAVSIM) do *all* dataset-specific work -- global->ego
transform, field reordering, category normalization -- and emit a ``SceneMeta``.
``canonical_bev.build_canonical_bev`` only ever sees ego-frame geometry, so the
rasterizer never needs to know which dataset it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class SceneMeta:
    # agents: (N, 7) = x, y, z, length, width, height, yaw  (ego frame)
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 7), np.float32))
    # normalized category per box: one of {"vehicle", "pedestrian", ...}
    names: List[str] = field(default_factory=list)
    # (N, 2) ego-frame vx, vy -- reserved, v1 may leave as zeros
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))

    # map_polygons[key] -> list of (P, 2) ego-frame polygon rings
    map_polygons: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    # map_polylines[key] -> list of (P, 2) ego-frame polylines
    map_polylines: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    # ego footprint (length, width) in meters
    ego_size: tuple = (4.6, 2.0)

    # free-form provenance for debugging / titles
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 7) \
            if len(self.boxes) else np.zeros((0, 7), np.float32)
        if len(self.velocity) != len(self.boxes):
            self.velocity = np.zeros((len(self.boxes), 2), np.float32)
