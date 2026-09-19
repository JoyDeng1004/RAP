# Generate 3D Rasterization Data

## process.sh

The script `process.sh` generates OpenScene v1.1 metadata from nuPlan `.db` files.  
It wraps around the `create_openscene_metadata*.py` scripts and provides a unified entry point for data processing.

### Requirements
- [NAVSim](https://github.com/autonomousvision/navsim)  
- [nuplan-devkit](https://github.com/motional/nuplan-devkit)
- [ScenarioNet](https://github.com/metadriverse/scenarionet)

### Modes
By editing the `process.sh`, you can choose different processing modes:
- **Default (`create_openscene_metadata.py`)**: Generate OpenScene metadata and rasterized multi-camera views.  
- **Perturbed (`create_openscene_metadata_perturbed.py`)**: Generate recovery-oriented trajectory perturbations (requires nuPlan logs with real camera inputs).  
- **Augmented (`create_openscene_metadata_aug.py`)**: Generate cross-agent synthesis data.  

### Usage
```bash
bash process.sh
```

## Cluster launchers

`process.sh` is the single-machine entry point. On TSUBAME / UGE use the
rasterization launchers instead. They are split by **source dataset**, and each
one's second argument names that dataset's variant axis:

| Script | Source | 2nd argument | Metadata script |
| --- | --- | --- | --- |
| `rasterize_nuplan.sh` | nuPlan `.db` | stage: `ego` / `perturbed` / `aug` | `create_openscene_metadata*.py` |
| `rasterize_nuscenes.sh` | nuScenes JSON | rig: `navsim` / `native` / `hybrid` | `create_nuscenes_metadata.py` |

The nuPlan pipeline always renders the NAVSIM canonical rig and offers no rig
choice; the rig axis exists only on the nuScenes side. Each script has a
matching `.qsub` array-job wrapper.

### rasterize_nuscenes.sh

```bash
bash process_data/rasterize_nuscenes.sh verify   native            # one frame, self-check
bash process_data/rasterize_nuscenes.sh plan     native 8          # shard plan
bash process_data/rasterize_nuscenes.sh pilot    native 8          # 8 scenes, for timing
bash process_data/rasterize_nuscenes.sh run      native 1 1        # full render, 850 scenes
bash process_data/rasterize_nuscenes.sh progress native            # checkpoint-based progress
bash process_data/rasterize_nuscenes.sh audit    native            # raster coverage, exit 1 on gaps
```

Each rig writes to its own `data_split` under one `dataset_nuscenes` root, so
several rigs coexist: `navsim` keeps the unsuffixed `nuscenes_trainval`, the
others get `nuscenes_trainval_<rig>`. The `_index/` directory is rig-independent
and shared; `build-index` runs once and the launcher refuses to rebuild it.

Run `audit` before training. A missing raster makes `dataclasses.py:87` fall
back to a 1080x1920 zero array, which collides with the native rig's 900x1600
crop at collate time, and the resulting error points nowhere near the cause.

