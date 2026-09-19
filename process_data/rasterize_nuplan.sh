#!/bin/bash
# RAP rasterization launcher for nuPlan sources (TSUBAME / UGE).
# The nuScenes counterpart is rasterize_nuscenes.sh. This script always renders
# the NAVSIM canonical rig, so its second argument is a stage, not a rig.
#
# Usage:
#   ./rasterize_nuplan.sh plan <stage> [shard-count]
#   ./rasterize_nuplan.sh run <stage> <shard-index> <shard-count>
#   ./rasterize_nuplan.sh pilot <stage> [log-count]
#   ./rasterize_nuplan.sh progress <stage>
# Stages: ego, perturbed, aug.
# Overrides: REPO, SPLIT, NUPLAN_PATH, NUPLAN_DB_PATH, NUPLAN_MAPS_ROOT,
# NUPLAN_MAP_VERSION, REAL_SENSOR_PATH, DATA_ROOT, DATA_ROOT_PERTURBED,
# DATA_ROOT_AUG, WORK_ROOT, THREADS, PYTHON_BIN.

set -euo pipefail

REPO="${REPO:-/gs/bs/tga-RLA/qdeng/RAP}"
SPLIT="${SPLIT:-trainval}"

NUPLAN_PATH="${NUPLAN_PATH:-/gs/bs/tga-RLA/qdeng/nuplan_dataset/nuplan-v1.1}"
NUPLAN_DB_PATH="${NUPLAN_DB_PATH:-$NUPLAN_PATH/splits/$SPLIT}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/gs/bs/tga-RLA/qdeng/RAP/datasets/navsim/maps}"
NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

# Used by helpers/nuplan_cameras_utils.py to check source-camera availability.
# Keep this separate from --nuplan-sensor-path, which determines raster output.
REAL_SENSOR_PATH="${REAL_SENSOR_PATH:-/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset/sensor_blobs/$SPLIT}"

DATA_ROOT="${DATA_ROOT:-/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset}"
DATA_ROOT_PERTURBED="${DATA_ROOT_PERTURBED:-/gs/bs/tga-RLA/qdeng/RAP/dataset_perturbed}"
DATA_ROOT_AUG="${DATA_ROOT_AUG:-/gs/bs/tga-RLA/qdeng/RAP/dataset_aug}"

WORK_ROOT="${WORK_ROOT:-/gs/bs/tga-RLA/qdeng/RAP/exp/rasterize}"
THREADS="${THREADS:-32}"
PYTHON_BIN="${PYTHON_BIN:-/gs/bs/tga-RLA/qdeng/anaconda3/envs/rap/bin/python}"

log() { printf '[%s] %s\n' "$(date +'%m-%d %H:%M:%S')" "$*" >&2; }

# The upstream "purturbed" filename is intentional.
stage_script() {
  case "$1" in
    ego)       echo "create_openscene_metadata.py" ;;
    perturbed) echo "create_openscene_metadata_purturbed.py" ;;
    aug)       echo "create_openscene_metadata_aug.py" ;;
    *) log "未知 stage: $1（可选 ego|perturbed|aug）"; return 1 ;;
  esac
}
stage_root() {
  case "$1" in
    ego)       echo "$DATA_ROOT" ;;
    perturbed) echo "$DATA_ROOT_PERTURBED" ;;
    aug)       echo "$DATA_ROOT_AUG" ;;
  esac
}

# The perturbed pipeline filters DBs internally, so its range limit is an upper bound.
db_total() {
  find "$NUPLAN_DB_PATH" -maxdepth 1 -type f -name '*.db' | wc -l | tr -d ' '
}

preflight() {
  local stage="$1" root; root="$(stage_root "$stage")"
  local sensor_path="$root/sensor_blobs/$SPLIT"

  [ -d "$NUPLAN_DB_PATH" ] || { log "FAIL nuPlan DB 目录不存在: $NUPLAN_DB_PATH"; return 1; }

  # The upstream loader treats every file in this directory as a database.
  local n_all n_db
  n_all=$(find "$NUPLAN_DB_PATH" -maxdepth 1 -type f | wc -l | tr -d ' ')
  n_db=$(find "$NUPLAN_DB_PATH" -maxdepth 1 -type f -name '*.db' | wc -l | tr -d ' ')
  if [ "$n_all" -ne "$n_db" ]; then
    log "FAIL DB 目录混有 $((n_all - n_db)) 个非 .db 文件，会被当成假 log:"
    find "$NUPLAN_DB_PATH" -maxdepth 1 -type f ! -name '*.db' \
      | sed 's|.*/|    |' | head -20 >&2
    log "     请把它们移出 $NUPLAN_DB_PATH 后重试"
    return 1
  fi
  [ -d "$NUPLAN_MAPS_ROOT" ] || { log "FAIL maps 不存在: $NUPLAN_MAPS_ROOT"; return 1; }
  # A missing source sensor root makes every camera_exists value false.
  [ -d "$REAL_SENSOR_PATH" ] \
    || { log "FAIL 真实 sensor 根不存在: $REAL_SENSOR_PATH（camera_exists 会全 False）"; return 1; }
  [ -f "$REPO/process_data/$(stage_script "$stage")" ] \
    || { log "FAIL 预处理脚本不存在: $(stage_script "$stage")"; return 1; }

  # Raster output derives from this exact, single "sensor_blobs" segment.
  case "$sensor_path" in
    *sensor_blobs*) : ;;
    *) log "FAIL NUPLAN_SENSOR_PATH 必须含 'sensor_blobs' 字面量: $sensor_path"; return 1 ;;
  esac
  local n; n=$(awk -v s="$sensor_path" 'BEGIN{n=0;i=1;while((p=index(substr(s,i),"sensor_blobs"))>0){n++;i+=p+11}print n}')
  [ "$n" -eq 1 ] || { log "FAIL 'sensor_blobs' 在路径里出现 $n 次（需恰好 1 次）: $sensor_path"; return 1; }

  echo "$sensor_path"
}

cmd_plan() {
  local stage="${1:?用法: plan <stage> <分片数>}" nshard="${2:-40}"
  local total; total=$(db_total)
  local per=$(( (total + nshard - 1) / nshard ))
  local root; root="$(stage_root "$stage")"
  echo "stage            : $stage  ($(stage_script "$stage"))"
  echo "nuPlan DB        : $NUPLAN_DB_PATH  ($total 个 .db)"
  echo "OUT_DIR (pkl)    : $root/navsim_logs/$SPLIT"
  echo "NUPLAN_SENSOR_PATH: $root/sensor_blobs/$SPLIT"
  echo "渲染图将落在     : $root/rendered_sensor_blobs$([ "$stage" = aug ] && echo _augmented)/$SPLIT"
  echo "分片             : $nshard 片 x 约 $per 个 log，每片 $THREADS 进程"
  echo
  local i
  for i in $(seq 1 "$nshard"); do
    local s=$(( (i - 1) * per )) e=$(( i * per ))
    [ "$e" -gt "$total" ] && e=$total
    [ "$s" -ge "$total" ] && break
    printf '  task %2d: --start-index %-6s --end-index %-6s\n' "$i" "$s" "$e"
  done
}

run_stage() {
  local stage="$1" start="$2" end="$3" tag="$4"
  local script root sensor_path out_dir workdir
  script="$(stage_script "$stage")"
  root="$(stage_root "$stage")"
  sensor_path="$(preflight "$stage")" || return 1
  out_dir="$root/navsim_logs/$SPLIT"
  workdir="$WORK_ROOT/$stage/$tag"

  mkdir -p "$out_dir" "$workdir"
  cd "$workdir" # Isolate checkpoint.txt for each stage and shard.

  export NUPLAN_DATA_ROOT="$NUPLAN_PATH"
  export NUPLAN_MAPS_ROOT NUPLAN_MAP_VERSION
  # Required at import time by helpers/nuplan_cameras_utils.py.
  export NUPLAN_DB_PATH
  export NUPLAN_SENSOR_PATH="$REAL_SENSOR_PATH"
  export NAVSIM_DEVKIT_ROOT="$REPO"
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export OPENBLAS_NUM_THREADS=1 # Avoid BLAS oversubscription inside the worker pool.

  log "stage=$stage shard=$tag range=[$start,$end) threads=$THREADS"
  log "  OUT_DIR       = $out_dir"
  log "  env 真实sensor = $NUPLAN_SENSOR_PATH （存在性检查 -> camera_exists）"
  log "  CLI 输出模板   = $sensor_path"
  log "  渲染根        = ${sensor_path/sensor_blobs/rendered_sensor_blobs}"
  log "  workdir       = $workdir（checkpoint.txt 在这里）"

  "$PYTHON_BIN" -u "$REPO/process_data/$script" \
    --nuplan-root-path   "$NUPLAN_PATH" \
    --nuplan-db-path     "$NUPLAN_DB_PATH" \
    --nuplan-sensor-path "$sensor_path" \
    --nuplan-map-version "$NUPLAN_MAP_VERSION" \
    --nuplan-map-root    "$NUPLAN_MAPS_ROOT" \
    --out-dir            "$out_dir" \
    --split              "$SPLIT" \
    --thread-num         "$THREADS" \
    --start-index        "$start" \
    --end-index          "$end"
}

cmd_run() {
  local stage="${1:?用法: run <stage> <第几片> <总片数>}"
  local idx="${2:?}" nshard="${3:?}"
  local total; total=$(db_total)
  local per=$(( (total + nshard - 1) / nshard ))
  local s=$(( (idx - 1) * per )) e=$(( idx * per ))
  [ "$e" -gt "$total" ] && e=$total
  if [ "$s" -ge "$total" ]; then log "分片 $idx 无任务（total=$total）"; return 0; fi
  run_stage "$stage" "$s" "$e" "shard_$(printf '%03d' "$idx")_of_$nshard"
}

cmd_pilot() {
  local stage="${1:-ego}" n="${2:-20}"
  log "试渲染 $stage 的前 $n 个 log —— 目的是量单 log 耗时与体积,不是出数据"
  run_stage "$stage" 0 "$n" "pilot_$n"
}

cmd_progress() {
  local stage="${1:-ego}" root; root="$(stage_root "$stage")"
  local out_dir="$root/navsim_logs/$SPLIT"
  local raster_root="$root/rendered_sensor_blobs$([ "$stage" = aug ] && echo _augmented)/$SPLIT"
  local total; total=$(db_total)

  local done_logs pkl
  done_logs=$(cat "$WORK_ROOT/$stage"/*/checkpoint.txt 2>/dev/null | sort -u | grep -c . || true)
  pkl=$(find "$out_dir" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l | tr -d ' ')

  echo "stage        : $stage"
  echo "db 总数      : $total"
  echo "已完成 log   : $done_logs / $total  ($(( done_logs * 100 / (total > 0 ? total : 1) ))%)  <- 唯一可信的进度指标"
  # Existing files and in-place overwrites make this count unsuitable for progress.
  echo "目录内 pkl   : $pkl   （$out_dir；含原有文件，不是进度）"
  echo "渲染根       : $raster_root"
  echo "渲染图占用   : $(du -sh "$raster_root" 2>/dev/null | cut -f1 || echo '(尚无)')"
  if [ "$done_logs" -gt 0 ]; then
    local kb bytes
    kb=$(du -sk "$raster_root" 2>/dev/null | cut -f1 || echo 0)
    bytes=$(( kb * 1024 ))
    if [ "$bytes" -gt 0 ]; then
      local per_log=$(( bytes / done_logs ))
      echo "单 log 均摊  : $(numfmt --to=iec --suffix=B "$per_log" 2>/dev/null || echo "$per_log B")"
      echo "全量外推     : $(numfmt --to=iec --suffix=B $(( per_log * total )) 2>/dev/null || echo $(( per_log * total )))"
    fi
  fi
  echo
  echo "各分片状态："
  local d
  for d in "$WORK_ROOT/$stage"/*/; do
    [ -d "$d" ] || continue
    printf '  %-28s %s log\n' "$(basename "$d")" \
      "$(grep -c . "$d/checkpoint.txt" 2>/dev/null || echo 0)"
  done
}

case "${1:-}" in
  plan)     shift; cmd_plan "$@" ;;
  run)      shift; cmd_run "$@" ;;
  pilot)    shift; cmd_pilot "$@" ;;
  progress) shift; cmd_progress "$@" ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
