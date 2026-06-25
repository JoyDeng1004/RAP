"""Config + channel registry for the canonical BEV raster.

The channel set is a *registry* (a list of :class:`ChannelSpec`), NOT hard-coded
into the rasterizer.  To add velocity / yaw / multi-frame / route channels later,
append a ``ChannelSpec`` here and (if it is a new ``kind``) handle it in
``canonical_bev.build_canonical_bev`` -- the core stays the same.

Coordinate convention (pinned by tests/canonical_bev/test_coord.py, do NOT change):
    ego frame: +x = forward, +y = left, ego at raster center.
    col = size/2 + (y / bev_range) * (size/2)
    row = size/2 - (x / bev_range) * (size/2)
    => forward is "up" in the image; (0,0) -> (col=64, row=64) at size=128.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# --- canonical category names (adapters normalize raw labels into these) ---
_VEHICLE_TYPES: List[str] = ["vehicle"]
_PEDESTRIAN_TYPES: List[str] = ["pedestrian"]


@dataclass(frozen=True)
class ChannelSpec:
    """One output channel of the canonical BEV raster.

    kind:
      - "box":          paint rotated agent footprints whose normalized name is
                        in ``types``.
      - "map_polygon":  fillPoly over ``SceneMeta.map_polygons[source]``.
      - "map_polyline": polylines over ``SceneMeta.map_polylines[source]``.
      - "ego":          paint the ego footprint at the raster center.
    """

    name: str
    kind: str
    source: str  # map-layer key (map_*) or unused for box/ego
    color: Tuple[int, int, int]  # RGB for the composite visualization
    types: Sequence[str] = ()  # accepted normalized names, for kind == "box"


# Default v1 registry: 6 semantic-occupancy channels, all eyeball-verifiable.
DEFAULT_CHANNELS: List[ChannelSpec] = [
    ChannelSpec("drivable_area", "map_polygon", "drivable", (90, 90, 90)),
    ChannelSpec("lane_centerline", "map_polyline", "centerline", (255, 221, 0)),
    ChannelSpec("walkway", "map_polygon", "walkway", (0, 170, 170)),
    ChannelSpec("vehicle", "box", "", (40, 120, 255), types=_VEHICLE_TYPES),
    ChannelSpec("pedestrian", "box", "", (255, 60, 60), types=_PEDESTRIAN_TYPES),
    ChannelSpec("ego", "ego", "", (0, 220, 0)),
]


@dataclass
class CanonBEVConfig:
    canon_raster_size: int = 128          # symmetric H = W
    canon_bev_range: float = 32.0         # half-range in meters (matches point_cloud_range x_max)
    channels: List[ChannelSpec] = field(default_factory=lambda: list(DEFAULT_CHANNELS))
    polyline_thickness: int = 2
    default_ego_size: Tuple[float, float] = (4.6, 2.0)  # (length, width) meters

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    def channel_index(self, name: str) -> int:
        for i, ch in enumerate(self.channels):
            if ch.name == name:
                return i
        raise KeyError(f"no channel named {name!r}; have {[c.name for c in self.channels]}")


# --- module-level constants kept for the pinned unit test (test_coord.py) ---
_DEFAULT_CFG = CanonBEVConfig()
CANON_RASTER_CH: int = _DEFAULT_CFG.num_channels
CH_VEHICLE: int = _DEFAULT_CFG.channel_index("vehicle")
CH_PEDESTRIAN: int = _DEFAULT_CFG.channel_index("pedestrian")
CH_EGO: int = _DEFAULT_CFG.channel_index("ego")
