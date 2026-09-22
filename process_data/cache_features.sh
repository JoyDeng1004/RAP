#!/bin/bash
# Build RAP feature caches on TSUBAME/UGE.
#
# Converts rasterized logs and sensor images into rap_feature.gz and
# rap_target.gz files for cached training.
#
# Usage:
#   ./cache_features.sh plan     <target> [shard-count]
#   ./cache_features.sh run      <target> <shard-index> <shard-count>
#   ./cache_features.sh pilot    <target> [log-count]
#   ./cache_features.sh progress <target> [shard-count]
#   ./cache_features.sh merge    <target>
#   ./cache_features.sh verify   <target>
# Targets:
#   C = nuScenes logs + NAVSIM canonical rig raster  (task input C)
#   A = nuScenes logs + nuScenes native rig raster   (alignment pair A)
#   D = nuPlan/navtrain logs + NAVSIM rig raster     (alignment B + task D)
# Overrides: REPO, NUSC_DS_ROOT, NAVSIM_DS_ROOT, CACHE_ROOT, NAVSIM_EXP_ROOT,
#            NUPLAN_MAPS_ROOT, TIME_HORIZON, HAS_ROUTE, PYTHON_BIN.

set -euo pipefail

REPO="${REPO:-/gs/bs/tga-RLA/qdeng/RAP}"
cd "$REPO"

NUSC_DS_ROOT="${NUSC_DS_ROOT:-$REPO/dataset_nuscenes}"
NAVSIM_DS_ROOT="${NAVSIM_DS_ROOT:-/gs/bs/tga-RLA/qdeng/navsim_workspace/dataset}"
CACHE_ROOT="${CACHE_ROOT:-$REPO/cache}"

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO/exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$REPO/datasets/navsim/maps}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

# Must match training because it determines num_poses in rap_target.gz.
TIME_HORIZON="${TIME_HORIZON:-5}"
# Keep routes disabled because nuScenes roadblock_ids are sparse.
HAS_ROUTE="${HAS_ROUTE:-false}"

PYTHON_BIN="${PYTHON_BIN:-/gs/bs/tga-RLA/qdeng/anaconda3/envs/rap/bin/python}"

# Planning estimates from the 2026-09-20 two-scene nuScenes pilot.
MB_PER_SAMPLE=12.4
SEC_PER_SAMPLE=0.80

# Increase only with a larger cpu_* memory request.
MAX_LOGS_PER_SHARD="${MAX_LOGS_PER_SHARD:-60}"

log() { printf '[%s] %s\n' "$(date +'%m-%d %H:%M:%S')" "$*" >&2; }

check_target() {
  case "$1" in
    C|A|D) : ;;
    *) log "未知 target: $1（可选 C|A|D）"; return 1 ;;
  esac
}

# data_split selects the log and sensor directories.
target_split() {
  case "$1" in
    C) echo "nuscenes_trainval" ;;
    A) echo "nuscenes_trainval_native" ;;
    D) echo "trainval" ;;
  esac
}
target_root() {
  case "$1" in
    C|A) echo "$NUSC_DS_ROOT" ;;
    D)   echo "$NAVSIM_DS_ROOT" ;;
  esac
}
# Hydra train_test_split config name.
target_test_split() {
  case "$1" in
    C) echo "nuscenes_trainval" ;;
    A) echo "nuscenes_trainval_native" ;;
    D) echo "navtrain" ;;
  esac
}
target_filter_yaml() {
  local base="navsim/planning/script/config/common/train_test_split/scene_filter"
  case "$1" in
    C|A) echo "$base/nuscenes_all.yaml" ;;
    D)   echo "$base/navall.yaml" ;;
  esac
}
target_cache_name() {
  case "$1" in
    C) echo "rap_nusc_navsimrig" ;;
    A) echo "rap_nusc_nativerig" ;;
    D) echo "rap_ego" ;;
  esac
}

parts_dir() { echo "$CACHE_ROOT/$(target_cache_name "$1").parts"; }
final_dir() { echo "$CACHE_ROOT/$(target_cache_name "$1")"; }
# Store the samples-per-log rate measured by pilot.
rate_file() { echo "$CACHE_ROOT/.rate_$1"; }

# Read the canonical log set from scene_filter.
filter_log_names() {
  awk '
    /^log_names:/ { inlist = 1; next }
    inlist && /^[[:space:]]*-[[:space:]]+/ { sub(/^[[:space:]]*-[[:space:]]+/, ""); gsub(/["'\'']/, ""); print; next }
    inlist && NF && !/^[[:space:]]*-/ { exit }
  ' "$(target_filter_yaml "$1")"
}

# Keep only logs with rendered metadata.
available_log_names() {
  local target="$1" root split
  root="$(target_root "$target")"; split="$(target_split "$target")"
  comm -12 \
    <(filter_log_names "$target" | sort -u) \
    <(find "$root/navsim_logs/$split" -maxdepth 1 -name '*.pkl' -printf '%f\n' 2>/dev/null \
        | sed 's/\.pkl$//' | sort -u)
}

# Balance variable-length logs across shards with round-robin assignment.
shard_log_names() {
  local target="$1" idx="$2" nshard="$3"
  available_log_names "$target" | awk -v n="$nshard" -v i="$idx" 'NR % n == i % n'
}

preflight() {
  local target="$1" root split
  root="$(target_root "$target")"; split="$(target_split "$target")"

  [ -x "$PYTHON_BIN" ] || { log "FAIL python 不存在: $PYTHON_BIN"; return 1; }
  [ -d "$root/navsim_logs/$split" ] \
    || { log "FAIL 元数据目录不存在: $root/navsim_logs/$split（先跑 rasterize_*.sh）"; return 1; }
  [ -e "$root/sensor_blobs/$split" ] \
    || { log "FAIL 真实图根不存在: $root/sensor_blobs/$split"; return 1; }
  [ -e "$root/rendered_sensor_blobs/$split" ] \
    || { log "FAIL 渲染图根不存在: $root/rendered_sensor_blobs/$split"; return 1; }
  # Only target D requires nuPlan maps.
  if [ "$target" = "D" ] && [ ! -d "$NUPLAN_MAPS_ROOT" ]; then
    log "FAIL nuPlan maps 不存在: $NUPLAN_MAPS_ROOT"; return 1
  fi
  return 0
}

# Read log names from stdin and cache them for the selected target.
run_caching() {
  local target="$1" cache_path="$2" exp_name="$3"
  local logs n_logs
  logs="$(sed "s/^/'/;s/$/'/" | paste -sd, -)"
  [ -n "$logs" ] || { log "FAIL 该分片没有 log，检查 plan 输出"; return 1; }
  n_logs=$(awk -F, '{print NF}' <<<"$logs")

  export OPENSCENE_DATA_ROOT="$(target_root "$target")"
  export PYTHONPATH="$REPO:${PYTHONPATH:-}"
  # Prevent BLAS from spawning extra threads.
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export TF_ENABLE_ONEDNN_OPTS=0

  mkdir -p "$cache_path"

  log "target=$target split=$(target_split "$target") logs=$n_logs"
  log "  OPENSCENE_DATA_ROOT = $OPENSCENE_DATA_ROOT"
  log "  cache_path          = $cache_path"

  # Use the sequential worker because Ray cannot access TSUBAME's protected /proc.
  "$PYTHON_BIN" -u navsim/planning/script/run_dataset_caching.py \
    agent=rap_agent \
    dataset=navsim_dataset \
    worker=sequential \
    agent.config.cache_data=True \
    "agent.config.trajectory_sampling.time_horizon=$TIME_HORIZON" \
    "train_test_split=$(target_test_split "$target")" \
    "train_test_split.scene_filter.has_route=$HAS_ROUTE" \
    "train_test_split.scene_filter.log_names=[$logs]" \
    "experiment_name=$exp_name" \
    "cache_path=$cache_path"
}

cmd_plan() {
  local target="${1:?用法: plan <target> [分片数]}" nshard="${2:-8}"
  check_target "$target"
  preflight "$target" || return 1

  local declared avail missing
  declared=$(filter_log_names "$target" | sort -u | wc -l)
  avail=$(available_log_names "$target" | wc -l)
  missing=$(( declared - avail ))

  echo "target        : $target  ($(target_cache_name "$target"))"
  echo "data_split    : $(target_split "$target")"
  echo "OPENSCENE_DATA_ROOT : $(target_root "$target")"
  echo "scene_filter  : $(target_filter_yaml "$target")"
  echo "  声明的 log  : $declared"
  echo "  已渲染 log  : $avail"
  echo "  缺 pkl 的   : $missing   （缺的会被跳过，不影响其它分片）"
  echo
  echo "分片输出      : $(parts_dir "$target")/shard_XXX_of_$nshard"
  echo "合并目标      : $(final_dir "$target")"
  echo "time_horizon  : $TIME_HORIZON   <- 必须与训练时一致"
  echo "has_route     : $HAS_ROUTE"
  echo

  local i n max_n=0
  echo "每片 log 数（round-robin）："
  for i in $(seq 1 "$nshard"); do
    n=$(shard_log_names "$target" "$i" "$nshard" | wc -l)
    [ "$n" -gt "$max_n" ] && max_n=$n
    printf '  shard %2d/%s : %s log\n' "$i" "$nshard" "$n"
  done
  echo

  # Peak memory scales with logs per shard; cpu_4 failed at 606 nuPlan logs.
  if [ "$max_n" -gt "$MAX_LOGS_PER_SHARD" ]; then
    local want=$(( (avail + MAX_LOGS_PER_SHARD - 1) / MAX_LOGS_PER_SHARD ))
    echo "!! 每片 $max_n 个 log 超过建议上限 $MAX_LOGS_PER_SHARD —— 内存峰值随每片 log 数线性增长，"
    echo "   nuPlan log 在 cpu_4 上 606/片 会被 OOM kill（分片目录建出来了但一个样本都没写）。"
    echo "   建议改用 $want 片： -t 1-$want ... shard $target $want   （并考虑 -l cpu_16=1）"
    echo
  fi
  # Use pilot measurements when available.
  local rf; rf="$(rate_file "$target")"
  if [ -f "$rf" ]; then
    local spl; spl=$(cat "$rf")
    "$PYTHON_BIN" - "$avail" "$spl" "$nshard" "$MB_PER_SAMPLE" "$SEC_PER_SAMPLE" <<'EOF'
import sys
logs, spl, nshard, mb, sec = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
n = logs * spl
print(f"样本数估计    : {n:,.0f}  ({spl:.1f} samples/log, 由 pilot 实测)")
print(f"体积估计      : {n*mb/1024:,.0f} GB   ({mb} MB/sample)")
print(f"单片耗时估计  : {n*sec/nshard/3600:,.1f} h  ({sec} s/sample, {nshard} 片)")
EOF
  else
    echo "样本数估计    : 未知 —— 先跑 'pilot $target 2' 测出 samples/log"
  fi
  echo
  echo "提交： qsub -g <group> -t 1-$nshard process_data/cache_features.qsub shard $target $nshard"
}

cmd_pilot() {
  local target="${1:?用法: pilot <target> [log 数]}" n="${2:-2}"
  check_target "$target"
  preflight "$target" || return 1

  local cache_path="$CACHE_ROOT/pilot_$target"
  rm -rf "$cache_path"
  log "试缓存 $target 的前 $n 个 log —— 目的是量 samples/log、MB/sample 与 s/sample"

  # Materialize the list to avoid SIGPIPE failures under pipefail.
  local list; list="$(mktemp)"
  available_log_names "$target" > "$list"
  local t0 t1
  t0=$(date +%s)
  head -n "$n" "$list" | run_caching "$target" "$cache_path" "pilot_$target"
  t1=$(date +%s)
  rm -f "$list"

  local samples kb
  samples=$(find "$cache_path" -name 'rap_feature.gz' | wc -l)
  kb=$(du -sk "$cache_path" | cut -f1)
  [ "$samples" -gt 0 ] || { log "FAIL 没产出任何 rap_feature.gz"; return 1; }

  "$PYTHON_BIN" - "$samples" "$kb" "$((t1-t0))" "$n" "$(rate_file "$target")" <<'EOF'
import sys
samples, kb, secs, logs, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
print(f"samples      : {samples}  ({samples/logs:.1f} per log)")
print(f"MB/sample    : {kb/1024/samples:.2f}")
print(f"s/sample     : {secs/samples:.2f}")
import pathlib; p = pathlib.Path(out); p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(f"{samples/logs:.4f}\n")
print(f"\nsamples/log 已写入 {out}，plan 会用它做外推")
EOF
}

cmd_run() {
  local target="${1:?用法: run <target> <第几片> <总片数>}"
  local idx="${2:?}" nshard="${3:?}"
  check_target "$target"
  preflight "$target" || return 1

  local tag cache_path
  tag="shard_$(printf '%03d' "$idx")_of_$nshard"
  cache_path="$(parts_dir "$target")/$tag"

  # Requeued shards resume from their own output directory.
  shard_log_names "$target" "$idx" "$nshard" \
    | run_caching "$target" "$cache_path" "$(target_cache_name "$target")_$tag"

  log "分片完成: $cache_path"
}

cmd_progress() {
  local target="${1:?用法: progress <target> [分片数]}" nshard="${2:-8}"
  check_target "$target"
  local parts; parts="$(parts_dir "$target")"
  local fin; fin="$(final_dir "$target")"

  local declared avail
  declared=$(filter_log_names "$target" | sort -u | wc -l)
  avail=$(available_log_names "$target" | wc -l)

  echo "target       : $target  ($(target_cache_name "$target"))"
  echo "声明的 log   : $declared   （$(target_filter_yaml "$target")）"
  echo "已渲染 log   : $avail   <- caching 的上限；远小于声明数就是渲染没铺完"
  echo
  local d done_logs done_samples
  echo "各分片状态："
  for d in "$parts"/shard_*/; do
    [ -d "$d" ] || continue
    done_logs=$(find "$d" -mindepth 1 -maxdepth 1 -type d | wc -l)
    done_samples=$(find "$d" -name 'rap_feature.gz' | wc -l)
    printf '  %-24s %5s log  %7s samples  %6s\n' \
      "$(basename "$d")" "$done_logs" "$done_samples" "$(du -sh "$d" 2>/dev/null | cut -f1)"
  done
  local parts_samples fin_logs fin_samples
  parts_samples=$(find "$parts" -name 'rap_feature.gz' 2>/dev/null | wc -l)
  echo
  echo "分片合计     : $parts_samples samples  $(du -sh "$parts" 2>/dev/null | cut -f1)"
  if [ -d "$fin" ]; then
    fin_logs=$(find "$fin" -mindepth 1 -maxdepth 1 -type d | wc -l)
    fin_samples=$(find "$fin" -name 'rap_feature.gz' | wc -l)
    echo "已合并       : $fin_logs log  $fin_samples samples  $(du -sh "$fin" | cut -f1)"
    # Empty shard directories are expected after merge moves their logs.
    if [ "$parts_samples" -eq 0 ] && [ "$fin_samples" -gt 0 ]; then
      echo
      echo "注：分片显示 0 是 merge 之后的正常状态（log 目录已被移入 $fin）。"
      echo "    判断覆盖是否完整请看上面的「已渲染 log」vs「已合并 log」，或跑 verify。"
    fi
  else
    echo "已合并       : (尚未 merge)"
  fi
}

# Merge disjoint shard outputs by moving log directories.
cmd_merge() {
  local target="${1:?用法: merge <target>}"
  check_target "$target"
  local parts fin d logname
  parts="$(parts_dir "$target")"; fin="$(final_dir "$target")"
  [ -d "$parts" ] || { log "FAIL 没有分片目录: $parts"; return 1; }
  mkdir -p "$fin"

  local moved=0 merged=0
  for d in "$parts"/shard_*/*/; do
    [ -d "$d" ] || continue
    logname="$(basename "$d")"
    if [ -d "$fin/$logname" ]; then
      # Merge defensively if a log appears in multiple shards.
      mv "$d"/* "$fin/$logname"/ && rmdir "$d"
      merged=$((merged + 1))
    else
      mv "$d" "$fin/$logname"
      moved=$((moved + 1))
    fi
  done

  log "合并完成: 移动 $moved 个 log 目录，合并 $merged 个已存在的"
  log "  最终 cache: $fin  ($(find "$fin" -name 'rap_feature.gz' | wc -l) samples, $(du -sh "$fin" | cut -f1))"
  log "  分片残留 : $(du -sh "$parts" 2>/dev/null | cut -f1)（确认无误后可以 rm -rf）"
}

# Detect partial tokens that CacheOnlyDataset would silently skip.
cmd_verify() {
  local target="${1:?用法: verify <target>}"
  check_target "$target"
  local fin; fin="$(final_dir "$target")"
  [ -d "$fin" ] || { log "FAIL cache 不存在: $fin（先 merge）"; return 1; }

  # List missing logs to distinguish rendering gaps from failed shards.
  local expected; expected="$(mktemp)"
  available_log_names "$target" > "$expected"
  filter_log_names "$target" | sort -u | wc -l > "$expected.declared"

  # Preserve the verifier status while allowing temporary-file cleanup.
  local status=0
  "$PYTHON_BIN" - "$fin" "$expected" <<'EOF' || status=$?
import sys
from pathlib import Path

root, expected_path = Path(sys.argv[1]), Path(sys.argv[2])
expected = {l.strip() for l in expected_path.read_text().splitlines() if l.strip()}
declared = int(Path(str(expected_path) + ".declared").read_text().strip())

logs = {d.name for d in root.iterdir() if d.is_dir()}
total = ok = 0
broken = []
for name in sorted(logs):
    for tok in (root / name).iterdir():
        if not tok.is_dir():
            continue
        total += 1
        if (tok / "rap_feature.gz").is_file() and (tok / "rap_target.gz").is_file():
            ok += 1
        else:
            broken.append(tok)

missing, extra = sorted(expected - logs), sorted(logs - expected)
print(f"scene_filter 声明 : {declared}")
print(f"已渲染 (有 pkl)   : {len(expected)}")
print(f"cache 里的 log    : {len(logs)}")
print(f"token 目录        : {total}")
print(f"完整 token        : {ok}")
print(f"缺 .gz 的 token   : {len(broken)}")
print(f"该有却没有的 log  : {len(missing)}")
print(f"多出来的 log      : {len(extra)}   （旧残留或跑错 split）")
for b in broken[:10]:
    print("  BROKEN", b)
for m in missing[:10]:
    print("  MISSING", m)
for e in extra[:10]:
    print("  EXTRA  ", e)

if declared > len(expected):
    print(f"\n注意：scene_filter 声明 {declared} 个 log，但只有 {len(expected)} 个有 pkl。")
    print("     caching 最多只能覆盖到后者 —— 若这不是预期，问题在 rasterize 阶段，不在 caching。")

if broken or missing or extra:
    print("\nFAIL 覆盖不全 —— 先看上面 MISSING/EXTRA/BROKEN 定位是哪一步短了")
    sys.exit(1)
print("\nOK 覆盖完整（相对于已渲染的 log）")
EOF
  rm -f "$expected" "$expected.declared"
  return $status
}

case "${1:-}" in
  plan)     shift; cmd_plan "$@" ;;
  run)      shift; cmd_run "$@" ;;
  pilot)    shift; cmd_pilot "$@" ;;
  progress) shift; cmd_progress "$@" ;;
  merge)    shift; cmd_merge "$@" ;;
  verify)   shift; cmd_verify "$@" ;;
  *) sed -n '2,24p' "$0"; exit 1 ;;
esac
