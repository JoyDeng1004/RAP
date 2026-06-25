"""Standalone canonical-BEV toolkit (Stage 0).

Builds an ego-centric, camera-agnostic multi-channel BEV raster purely from
dataset metadata (3D boxes + map + ego), for nuScenes and NAVSIM, and
visualizes it.  Independent of the RAP training pipeline.

See ``plan_v1_0624.md`` (docs/) for the full Sekikawa camera-agnostic idea; this
package is only the metadata->BEV front-end + visualization.
"""
