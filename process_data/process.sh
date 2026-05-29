export OPENBLAS_NUM_THREADS=1

# Please install https://github.com/autonomousvision/navsim.
# This is used for generating High-Level Driving Commands, used in E2E AD.
export NAVSIM_DEVKIT_ROOT=/gs/bs/tga-RLA/qdeng/navsim_workspace/navsim
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}

split=mini
# Please download all the nuplan data from https://www.nuscenes.org/nuplan.
export NUPLAN_PATH=/gs/bs/tga-RLA/qdeng/nuplan_dataset
export NUPLAN_DB_PATH=${NUPLAN_PATH}/data/cache/${split}
export NUPLAN_SENSOR_PATH=/gs/bs/tga-RLA/qdeng/nuplan_dataset/nuplan-v1.1/sensor_blobs
export NUPLAN_MAPS_ROOT=/gs/bs/tga-RLA/qdeng/nuplan_dataset/maps

# mv checkpoint.txt checkpoint_normal.txt

OUT_DIR='/gs/bs/tga-RLA/qdeng/RAP/dataset_norm/navsim_logs/mini'
# 1. Generate OpenScene metadata and 3D rasterized multi-camera views 
#    for all nuPlan logs (~1200h).
python -u create_openscene_metadata.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${NUPLAN_SENSOR_PATH} \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num 32 \
  --start-index 0 \
  --end-index 14561

mv checkpoint.txt checkpoint_normal.txt

OUT_DIR='/gs/bs/tga-RLA/qdeng/RAP/dataset_perturbed/navsim_logs/mini'
# 2. Generate recovery-oriented trajectory perturbations 
python -u create_openscene_metadata_perturbed.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${NUPLAN_SENSOR_PATH} \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num 32 \
  --start-index 0 \
  --end-index 14561

mv checkpoint.txt checkpoint_normal.txt

OUT_DIR='/gs/bs/tga-RLA/qdeng/RAP/dataset_aug/navsim_logs/mini'
# 3. Generate cross-agent view synthesis 
python -u create_openscene_metadata_aug.py \
  --nuplan-root-path ${NUPLAN_PATH} \
  --nuplan-db-path ${NUPLAN_DB_PATH} \
  --nuplan-sensor-path ${NUPLAN_SENSOR_PATH} \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root ${NUPLAN_MAPS_ROOT} \
  --out-dir ${OUT_DIR} \
  --split ${split} \
  --thread-num 32 \
  --start-index 0 \
  --end-index 14561