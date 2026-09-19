#!/bin/bash
# RAP rasterization launcher for nuScenes sources (TSUBAME / UGE).
# The nuPlan counterpart is rasterize_nuplan.sh, where the second argument is a
# stage and the rig is always NAVSIM. Here the second argument is the rig.
#
# Usage:
#   ./rasterize_nuscenes.sh plan <rig> [shard-count]
#   ./rasterize_nuscenes.sh run <rig> <shard-index> <shard-count>
#   ./rasterize_nuscenes.sh pilot <rig> [scene-count]
#   ./rasterize_nuscenes.sh verify <rig> [scene-name] [frame-index]
#   ./rasterize_nuscenes.sh progress <rig>
#   ./rasterize_nuscenes.sh audit <rig>
# Rigs: navsim (NAVSIM canonical), native (nuScenes calibrated_sensor), hybrid.
# Overrides: REPO, NUSC_ROOT, DS_ROOT, INDEX_DIR, WORK_ROOT, THREADS, VERSION,
# SPLIT, VIEWPOINT_SHIFT, PYTHON_BIN.

set -euo pipefail

REPO="${REPO:-/gs/bs/tga-RLA/qdeng/RAP}"
SCRIPT="$REPO/process_data/create_nuscenes_metadata.py"

NUSC_ROOT="${NUSC_ROOT:-/gs/bs/tga-RLA/qdeng/data/nuscenes}"
VERSION="${VERSION:-v1.0-trainval}"

# Every rig writes under this one root, separated by data_split.
DS_ROOT="${DS_ROOT:-$REPO/dataset_nuscenes}"
INDEX_DIR="${INDEX_DIR:-$DS_ROOT/_index}"

WORK_ROOT="${WORK_ROOT:-$REPO/exp/rasterize}"
THREADS="${THREADS:-32}"
# RAP's -2m/+0.8m virtual viewpoint. Keep identical across rigs, or a rig
# comparison also varies the viewpoint.
VIEWPOINT_SHIFT="${VIEWPOINT_SHIFT:-true}"
PYTHON_BIN="${PYTHON_BIN:-/gs/bs/tga-RLA/qdeng/anaconda3/envs/rap/bin/python}"

log() { printf '[%s] %s\n' "$(date +'%m-%d %H:%M:%S')" "$*" >&2; }

check_rig() {
  case "$1" in
    navsim|native|hybrid) : ;;
    *) log "未知 rig: $1（可选 navsim|native|hybrid）"; return 1 ;;
  esac
}

# The navsim rig predates this launcher and owns the unsuffixed split name;
# renaming it would orphan the existing pkl and raster directories.
rig_split() {
  if [ -n "${SPLIT:-}" ]; then echo "$SPLIT"; return; fi
  case "$1" in
    navsim) echo "nuscenes_trainval" ;;
    *)      echo "nuscenes_trainval_$1" ;;
  esac
}

scene_total() {
  "$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['num_scenes'])" \
    "$INDEX_DIR/manifest.json"
}

# Echoes the sensor path on success.
preflight() {
  local rig="$1" split; split="$(rig_split "$rig")"
  local sensor_path="$DS_ROOT/sensor_blobs/$split"

  [ -f "$SCRIPT" ] || { log "FAIL 预处理脚本不存在: $SCRIPT"; return 1; }
  [ -d "$NUSC_ROOT/$VERSION" ] \
    || { log "FAIL nuScenes 元数据目录不存在: $NUSC_ROOT/$VERSION"; return 1; }

  # The index is rig-independent and expensive; never rebuild it from here.
  [ -f "$INDEX_DIR/manifest.json" ] || {
    log "FAIL 索引不存在: $INDEX_DIR/manifest.json"
    log "     先跑一次（高内存，与 rig 无关，所有 rig 共用）："
    log "     $PYTHON_BIN $SCRIPT build-index --nuscenes-root $NUSC_ROOT \\"
    log "         --version $VERSION --out-dir $DS_ROOT --index-dir $INDEX_DIR"
    return 1
  }
  [ -d "$INDEX_DIR/_maps" ] || { log "FAIL 地图包不存在: $INDEX_DIR/_maps"; return 1; }

  # sensor_blobs/<split> only supplies real images to the training loader; the
  # renderer reads nothing from it. A symlink per split, never a copy.
  [ -e "$sensor_path" ] || {
    log "FAIL 真实图根不存在: $sensor_path"
    log "     建一个 symlink（不复制数据）: ln -sfn $NUSC_ROOT $sensor_path"
    return 1
  }

  # Raster output derives from this exact, single "sensor_blobs" segment.
  case "$sensor_path" in
    *sensor_blobs*) : ;;
    *) log "FAIL sensor-path 必须含 'sensor_blobs' 字面量: $sensor_path"; return 1 ;;
  esac
  local n; n=$(awk -v s="$sensor_path" 'BEGIN{n=0;i=1;while((p=index(substr(s,i),"sensor_blobs"))>0){n++;i+=p+11}print n}')
  [ "$n" -eq 1 ] || { log "FAIL 'sensor_blobs' 在路径里出现 $n 次（需恰好 1 次）: $sensor_path"; return 1; }

  echo "$sensor_path"
}

# Populates COMMON_ARGS, so no caller can drift on --index-dir or the rig.
COMMON_ARGS=()
set_common_args() {
  COMMON_ARGS=(
    --nuscenes-root "$NUSC_ROOT"
    --version "$VERSION"
    --index-dir "$INDEX_DIR"
    --camera-mode "$1"
    --apply-rap-viewpoint-shift "$VIEWPOINT_SHIFT"
  )
}

run_render() {
  local rig="$1" shard="$2" nshard="$3" tag="$4"; shift 4
  local split sensor_path out_dir workdir
  split="$(rig_split "$rig")"
  sensor_path="$(preflight "$rig")" || return 1
  out_dir="$DS_ROOT/navsim_logs/$split"
  workdir="$WORK_ROOT/nuscenes_$rig/$tag"

  mkdir -p "$out_dir" "$workdir"

  log "rig=$rig split=$split shard=$shard/$nshard threads=$THREADS"
  log "  OUT_DIR  = $out_dir"
  log "  真实图根 = $sensor_path"
  log "  渲染根   = ${sensor_path/sensor_blobs/rendered_sensor_blobs}"
  log "  workdir  = $workdir（checkpoint.txt 在这里）"

  export OPENBLAS_NUM_THREADS=1 # Avoid BLAS oversubscription inside the worker pool.
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"

  # Run from the workdir so a stray relative checkpoint stays shard-local.
  cd "$workdir"
  set_common_args "$rig"
  "$PYTHON_BIN" -u "$SCRIPT" render \
    "${COMMON_ARGS[@]}" \
    --out-dir "$out_dir" \
    --sensor-path "$sensor_path" \
    --shard "$shard" \
    --num-shards "$nshard" \
    --thread-num "$THREADS" \
    --checkpoint "$workdir/checkpoint.txt" \
    "$@"
}

cmd_plan() {
  local rig="${1:?用法: plan <rig> [分片数]}" nshard="${2:-1}"
  check_rig "$rig"
  local sensor_path; sensor_path="$(preflight "$rig")" || return 1
  echo "rig          : $rig"
  echo "data_split   : $(rig_split "$rig")"
  echo "索引（共用） : $INDEX_DIR"
  echo
  # The python plan subcommand owns the shard arithmetic.
  set_common_args "$rig"
  "$PYTHON_BIN" -u "$SCRIPT" plan \
    "${COMMON_ARGS[@]}" \
    --out-dir "$DS_ROOT/navsim_logs/$(rig_split "$rig")" \
    --sensor-path "$sensor_path" \
    --num-shards "$nshard" \
    --thread-num "$THREADS"
}

cmd_run() {
  local rig="${1:?用法: run <rig> <第几片> <总片数>}"
  local idx="${2:?}" nshard="${3:?}"
  check_rig "$rig"
  run_render "$rig" "$idx" "$nshard" "shard_$(printf '%03d' "$idx")_of_$nshard"
}

cmd_pilot() {
  local rig="${1:?用法: pilot <rig> [scene 数]}" n="${2:-8}"
  check_rig "$rig"
  log "试渲染 $rig 的前 $n 个 scene —— 目的是量单 scene 耗时与体积,不是出数据"
  run_render "$rig" 1 1 "pilot_$n" --limit-scenes "$n"
}

cmd_verify() {
  local rig="${1:?用法: verify <rig> [scene 名] [帧号]}"
  local scene="${2:-scene-0001}" frame="${3:-0}"
  check_rig "$rig"
  preflight "$rig" >/dev/null || return 1
  local out_dir="$WORK_ROOT/nuscenes_$rig/verify"
  mkdir -p "$out_dir"
  set_common_args "$rig"
  "$PYTHON_BIN" -u "$SCRIPT" verify \
    "${COMMON_ARGS[@]}" \
    --out-dir "$out_dir" --scene-name "$scene" --frame-index "$frame"
  echo
  log "四路的 nonzero_fraction 若完全相同，说明 camera_models 被同一组参数覆盖了 —— 别继续渲"
  log "拼联络图: $PYTHON_BIN $REPO/tools/contact_sheet.py $out_dir --mode $rig"
}

cmd_progress() {
  local rig="${1:?用法: progress <rig>}"
  check_rig "$rig"
  local split; split="$(rig_split "$rig")"
  local out_dir="$DS_ROOT/navsim_logs/$split"
  local raster_root="$DS_ROOT/rendered_sensor_blobs/$split"
  local total; total=$(scene_total)

  local done_scenes pkl
  done_scenes=$(cat "$WORK_ROOT/nuscenes_$rig"/*/checkpoint.txt 2>/dev/null | sort -u | grep -c . || true)
  pkl=$(find "$out_dir" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l | tr -d ' ')

  echo "rig          : $rig   (data_split=$split)"
  echo "scene 总数   : $total"
  echo "已完成 scene : $done_scenes / $total  ($(( done_scenes * 100 / (total > 0 ? total : 1) ))%)  <- 唯一可信的进度指标"
  # Existing files and in-place overwrites make this count unsuitable for progress.
  echo "目录内 pkl   : $pkl   （$out_dir；含原有文件，不是进度）"
  echo "渲染根       : $raster_root"
  echo "渲染图占用   : $(du -sh "$raster_root" 2>/dev/null | cut -f1 || echo '(尚无)')"
  echo
  echo "各分片状态："
  local d
  for d in "$WORK_ROOT/nuscenes_$rig"/*/; do
    [ -d "$d" ] || continue
    printf '  %-28s %s scene\n' "$(basename "$d")" \
      "$(grep -c . "$d/checkpoint.txt" 2>/dev/null || echo 0)"
  done
}

# A missing raster makes dataclasses.py:87 fall back to a 1080x1920 zero array,
# which collides with the native rig's 900x1600 crop at collate time.
cmd_audit() {
  local rig="${1:?用法: audit <rig>}"
  check_rig "$rig"
  local split; split="$(rig_split "$rig")"
  "$PYTHON_BIN" - "$DS_ROOT/navsim_logs/$split" \
                  "$DS_ROOT/rendered_sensor_blobs/$split" \
                  "$(scene_total)" <<'EOF'
import pickle, sys
from pathlib import Path

logs, root, total = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
pkls = sorted(logs.glob("*.pkl"))
want, frames = set(), 0
for p in pkls:
    for info in pickle.loads(p.read_bytes()):
        frames += 1
        for cam in info["cams"].values():
            want.add(cam["data_path"])
have = {str(f.relative_to(root)) for f in root.rglob("*.jpg")} if root.exists() else set()
missing, extra = want - have, have - want

print(f"pkl          : {len(pkls)} / {total}")
print(f"frames       : {frames}")
print(f"metadata 引用: {len(want)}")
print(f"磁盘实有     : {len(have)}")
print(f"missing      : {len(missing)}")
print(f"extra        : {len(extra)}")
for s in sorted(missing)[:5]:
    print("  MISSING", s)
if missing or len(pkls) != total:
    print("\nFAIL 覆盖不全 —— 别拿这套数据训练，先补渲缺的 scene")
    sys.exit(1)
print("\nOK 覆盖完整")
EOF
}

case "${1:-}" in
  plan)     shift; cmd_plan "$@" ;;
  run)      shift; cmd_run "$@" ;;
  pilot)    shift; cmd_pilot "$@" ;;
  verify)   shift; cmd_verify "$@" ;;
  progress) shift; cmd_progress "$@" ;;
  audit)    shift; cmd_audit "$@" ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
