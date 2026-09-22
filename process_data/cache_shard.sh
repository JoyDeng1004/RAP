#!/bin/bash
# Build RAP feature caches with log-level sharding on TSUBAME/UGE.
#
# Run disjoint log sets in separate processes because dataset caching is
# single-process. Cache paths include log_name, so shards do not overlap.
#
# Usage:
#   ./cache_shard.sh plan     <dataset> [shard-count]
#   ./cache_shard.sh run      <dataset> <shard-index> <shard-count>
#   ./cache_shard.sh pilot    <dataset> [log-count]
#   ./cache_shard.sh progress <dataset>
#   ./cache_shard.sh audit    <dataset> [shard-index] [shard-count]
#
# Datasets:
#   nusc_navsimrig   nuScenes content with NAVSIM rig raster (C)
#   nusc_nativerig   nuScenes content with native rig raster (A)
#   navsim_ego       NAVSIM content with NAVSIM rig raster (B/D)
#   navsim_perturbed optional RAP recovery perturbations
#   navsim_aug       optional RAP cross-agent views
#
# Overrides: REPO, PYTHON_BIN, DS_ROOT_NUSC, DS_ROOT_NAVSIM, DS_ROOT_PERTURBED,
# DS_ROOT_AUG, CACHE_ROOT, TIME_HORIZON, NAVSIM_EXP_ROOT.

set -euo pipefail

REPO="${REPO:-/gs/bs/tga-RLA/qdeng/RAP}"
PYTHON_BIN="${PYTHON_BIN:-/gs/bs/tga-RLA/qdeng/anaconda3/envs/rap/bin/python}"

DS_ROOT_NUSC="${DS_ROOT_NUSC:-$REPO/dataset_nuscenes}"
DS_ROOT_NAVSIM="${DS_ROOT_NAVSIM:-/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset}"
DS_ROOT_PERTURBED="${DS_ROOT_PERTURBED:-$REPO/dataset_perturbed}"
DS_ROOT_AUG="${DS_ROOT_AUG:-$REPO/dataset_aug}"

CACHE_ROOT="${CACHE_ROOT:-$REPO/cache}"
# Must match training because it determines trajectory length in rap_target.gz.
TIME_HORIZON="${TIME_HORIZON:-5}"
SHARD_MODE="${SHARD_MODE:-stride}"

SCRIPT="$REPO/navsim/planning/script/run_dataset_caching.py"
FILTER_DIR="${FILTER_DIR:-$REPO/navsim/planning/script/config/common/train_test_split/scene_filter}"

log() { printf '[%s] %s\n' "$(date +'%m-%d %H:%M:%S')" "$*" >&2; }

# Resolve each dataset to its root, split, filter, and cache name.
resolve() {
  case "$1" in
    nusc_navsimrig)
      DS_ROOT="$DS_ROOT_NUSC"; SPLIT_CFG="nuscenes_trainval"
      FILTER="nuscenes_all";   DATA_SPLIT="nuscenes_trainval"
      CACHE_NAME="rap_nusc_navsimrig" ;;
    nusc_nativerig)
      DS_ROOT="$DS_ROOT_NUSC"; SPLIT_CFG="nuscenes_trainval_native"
      FILTER="nuscenes_all";   DATA_SPLIT="nuscenes_trainval_native"
      CACHE_NAME="rap_nusc_nativerig" ;;
    navsim_ego)
      DS_ROOT="$DS_ROOT_NAVSIM"; SPLIT_CFG="navtrain"
      FILTER="navall";           DATA_SPLIT="trainval"
      CACHE_NAME="rap_ego" ;;
    navsim_perturbed)
      DS_ROOT="$DS_ROOT_PERTURBED"; SPLIT_CFG="navtrain"
      FILTER="navall";              DATA_SPLIT="trainval"
      CACHE_NAME="rap_perturbed" ;;
    navsim_aug)
      DS_ROOT="$DS_ROOT_AUG"; SPLIT_CFG="navtrain"
      FILTER="navall";        DATA_SPLIT="trainval"
      CACHE_NAME="rap_aug" ;;
    *)
      log "未知 dataset: $1"
      log "可选: nusc_navsimrig | nusc_nativerig | navsim_ego | navsim_perturbed | navsim_aug"
      return 1 ;;
  esac
  LOGS_DIR="$DS_ROOT/navsim_logs/$DATA_SPLIT"
  FILTER_YAML="$FILTER_DIR/$FILTER.yaml"
  CACHE_PATH="${CACHE_PATH_OVERRIDE:-$CACHE_ROOT/$CACHE_NAME}"
}

preflight() {
  [ -f "$SCRIPT" ]      || { log "FAIL 缺 $SCRIPT"; return 1; }
  [ -d "$LOGS_DIR" ]    || { log "FAIL navsim_logs 不存在: $LOGS_DIR"; return 1; }
  [ -f "$FILTER_YAML" ] || { log "FAIL scene_filter 不存在: $FILTER_YAML"; return 1; }
  # dataclasses.py derives the rendered path from the real sensor path.
  [ -d "$DS_ROOT/rendered_sensor_blobs/$DATA_SPLIT" ] || {
    log "FAIL 渲染图根不存在: $DS_ROOT/rendered_sensor_blobs/$DATA_SPLIT"
    log "     没有它，dataclasses.py:87 会 fallback 成全零 1080x1920 数组，静默产出废 cache"
    return 1
  }
  [ -d "$DS_ROOT/sensor_blobs/$DATA_SPLIT" ] || {
    log "WARN 真实图根不存在: $DS_ROOT/sensor_blobs/$DATA_SPLIT"
    log "     camera_valid 会全 False —— 只有故意要关掉 alignment 时才该这样"
  }
}

universe() {
  "$PYTHON_BIN" "$REPO/tools/log_universe.py" \
    --logs-dir "$LOGS_DIR" --scene-filter "$FILTER_YAML" "$@"
}

run_caching() {
  local tag="$1"; shift              # Pass remaining options to log_universe.
  local log_list count
  count="$(universe "$@" --format count)"
  [ "$count" -gt 0 ] || { log "分片为空，跳过"; return 0; }
  log_list="$(universe "$@" --format hydra)"

  local workdir="$REPO/exp/cache/$CACHE_NAME/$tag"
  mkdir -p "$workdir" "$CACHE_PATH"

  log "dataset=$DATASET tag=$tag logs=$count"
  log "  OPENSCENE_DATA_ROOT = $DS_ROOT"
  log "  train_test_split    = $SPLIT_CFG (data_split=$DATA_SPLIT)"
  log "  cache_path          = $CACHE_PATH"

  export OPENSCENE_DATA_ROOT="$DS_ROOT"
  export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO/exp}"
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"
  export OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=1          # Prevent BLAS from spawning extra threads.

  # cache_data skips model setup; force_cache_computation resumes cached shards.
  local -a cmd=(
    "$PYTHON_BIN" -u "$SCRIPT"
    agent=rap_agent
    dataset=navsim_dataset
    agent.config.cache_data=True
    "agent.config.trajectory_sampling.time_horizon=$TIME_HORIZON"
    "train_test_split=$SPLIT_CFG"
    train_test_split.scene_filter.has_route=false
    "train_test_split.scene_filter.log_names=$log_list"
    "experiment_name=cache_${CACHE_NAME}_${tag}"
    "cache_path=$CACHE_PATH"
    force_cache_computation=false
  )

  if [ -n "${DRY_RUN:-}" ]; then
    log "DRY_RUN —— 只打印命令，不执行"
    printf '%q ' "${cmd[@]}"; echo
    return 0
  fi
  "${cmd[@]}" 2>&1 | tee -a "$workdir/cache.log"
}

cmd_plan() {
  local nshard="${1:-40}"
  preflight || return 1
  local total; total="$(universe --format count)"
  echo "dataset          : $DATASET"
  echo "navsim_logs      : $LOGS_DIR"
  echo "scene_filter     : $FILTER_YAML"
  echo "log 全集         : $total"
  echo "cache_path       : $CACHE_PATH"
  echo "分片             : $nshard 片（$SHARD_MODE），每片约 $(( (total + nshard - 1) / nshard )) 个 log"
  echo
  echo "逐片命令："
  local i
  for i in $(seq 1 "$nshard"); do
    local n; n="$(universe --quiet --shard "$i" --num-shards "$nshard" --shard-mode "$SHARD_MODE" --format count)"
    [ "$n" -gt 0 ] || continue
    printf '  ./cache_shard.sh run %s %d %d    # %s logs\n' "$DATASET" "$i" "$nshard" "$n"
  done
  echo
  echo "体积先验：每样本约 25-30MB（4 路 real + 4 路 rendered 的 float32，"
  echo "gzip level=1 对归一化浮点几乎压不动）。先跑 pilot 量一次真实值再铺开。"
}

cmd_run() {
  local idx="${1:?用法: run <dataset> <第几片> <总片数>}" nshard="${2:?}"
  preflight || return 1
  run_caching "shard_$(printf '%03d' "$idx")_of_$nshard" \
    --shard "$idx" --num-shards "$nshard" --shard-mode "$SHARD_MODE"
}

# Use pilot runs only to measure time and storage per log.
cmd_pilot() {
  local n="${1:-5}"
  preflight || return 1
  local total; total="$(universe --format count)"
  local nshard=$(( (total + n - 1) / n ))
  [ "$nshard" -ge 1 ] || nshard=1
  log "试缓存 $DATASET 的 1/$nshard 片（约 $n 个 log）—— 量耗时与体积用"
  local before; before="$(du -sk "$CACHE_PATH" 2>/dev/null | cut -f1 || echo 0)"
  local t0; t0="$(date +%s)"
  run_caching "pilot_$n" --shard 1 --num-shards "$nshard" --shard-mode "$SHARD_MODE"
  local t1; t1="$(date +%s)"
  local after; after="$(du -sk "$CACHE_PATH" 2>/dev/null | cut -f1 || echo 0)"
  local tokens; tokens="$(count_tokens)"
  echo
  echo "耗时     : $(( t1 - t0 )) s"
  echo "新增体积 : $(numfmt --to=iec --suffix=B $(( (after - before) * 1024 )) 2>/dev/null || echo "$(( after - before )) KB")"
  echo "已缓存 token : $tokens"
  if [ "$tokens" -gt 0 ] && [ "$after" -gt "$before" ]; then
    local per=$(( (after - before) * 1024 / tokens ))
    echo "单 token : $(numfmt --to=iec --suffix=B "$per" 2>/dev/null || echo "$per B")"
    echo "全量外推 : 用 plan 里的 log 全集 × 每 log 的平均 token 数自行乘"
  fi
}

count_tokens() {
  # Match CacheOnlyDataset's definition of a complete token.
  find "$CACHE_PATH" -mindepth 2 -maxdepth 2 -type d 2>/dev/null \
    | while read -r d; do
        [ -f "$d/rap_feature.gz" ] && [ -f "$d/rap_target.gz" ] && echo 1
      done | wc -l | tr -d ' '
}

cmd_progress() {
  preflight || return 1
  local total_logs cached_logs tokens
  total_logs="$(universe --format count)"
  cached_logs="$(find "$CACHE_PATH" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
  tokens="$(count_tokens)"
  echo "dataset       : $DATASET"
  echo "cache_path    : $CACHE_PATH"
  echo "log 全集      : $total_logs"
  echo "cache 内 log  : $cached_logs   （最后一个 log 可能只缓存了一半，别当完成数）"
  echo "完整 token    : $tokens        <- 唯一可信的进度指标"
  echo "占用          : $(du -sh "$CACHE_PATH" 2>/dev/null | cut -f1 || echo '(尚无)')"
}

# Report partial tokens that CacheOnlyDataset would silently skip.
cmd_audit() {
  local idx="${1:-}" nshard="${2:-}"
  preflight || return 1
  local list_args=()
  [ -n "$idx" ] && list_args=(--shard "$idx" --num-shards "${nshard:?audit 给了片号就得给总片数}"
                              --shard-mode "$SHARD_MODE")

  "$PYTHON_BIN" - "$CACHE_PATH" <<EOF
import sys
from pathlib import Path

cache = Path(sys.argv[1])
wanted = set("""$(universe "${list_args[@]+"${list_args[@]}"}" --format lines)""".split())

missing_log, partial, empty, ok = [], [], [], 0
for name in sorted(wanted):
    d = cache / name
    if not d.is_dir():
        missing_log.append(name)
        continue
    toks = [t for t in d.iterdir() if t.is_dir()]
    if not toks:
        empty.append(name)
        continue
    for t in toks:
        f, g = t / "rap_feature.gz", t / "rap_target.gz"
        if f.is_file() and g.is_file():
            ok += 1
        else:
            partial.append(str(t.relative_to(cache)))

print(f"目标 log      : {len(wanted)}")
print(f"完整 token    : {ok}")
print(f"未缓存 log    : {len(missing_log)}")
print(f"空 log 目录   : {len(empty)}")
print(f"半截 token    : {len(partial)}   <- 会被 CacheOnlyDataset 静默跳过")
for n in missing_log[:5]:
    print("  MISSING-LOG", n)
for n in partial[:5]:
    print("  PARTIAL    ", n)
if missing_log or partial or empty:
    print("\nFAIL 缓存不完整 —— 半截 token 直接 rm -rf 掉再重跑对应分片")
    sys.exit(1)
print("\nOK 缓存完整")
EOF
}

DATASET="${2:-}"
[ -n "$DATASET" ] || { sed -n '2,30p' "$0"; exit 1; }
resolve "$DATASET" || exit 1

case "${1:-}" in
  plan)     shift 2; cmd_plan "$@" ;;
  run)      shift 2; cmd_run "$@" ;;
  pilot)    shift 2; cmd_pilot "$@" ;;
  progress) shift 2; cmd_progress "$@" ;;
  audit)    shift 2; cmd_audit "$@" ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
