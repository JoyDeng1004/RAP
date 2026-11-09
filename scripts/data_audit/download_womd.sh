#!/bin/bash
# Waymo Open Motion Dataset（WOMD）scenario 格式下载器（TSUBAME / UGE）。
#
# 只下 RAP raster 用得到的 uncompressed/scenario/：每条 Scenario proto 含
# 地图、全部 agent 轨迹（带 3D 尺寸）、红绿灯状态和 sdc_track_index。
# 不下 tf_example/（模型输入格式，冗余）和 lidar_and_camera/（只有前 1s，用不上）。
#
# 桶：gs://waymo_open_dataset_motion_v_1_3_1/（v1.3.1，2025-10，公开 sdc_paths）
#     不是匿名可读：需先在 https://waymo.com/open 用 Google 账号注册并接受
#     许可，再在 TSUBAME 上 `gcloud auth login --no-launch-browser`。
#
# 用法：
#   ./download_womd.sh check                     # 环境、登录账号、桶可访问性
#   ./download_womd.sh manifest [split ...]      # 列出远端 shard 和尺寸，写清单
#   ./download_womd.sh size                      # 按清单汇总各 split 大小
#   ./download_womd.sh fetch <split> <task> <ntasks>  # 下载清单中第 task 份
#   ./download_womd.sh verify [split ...]        # 本地尺寸逐个对清单
#   ./download_womd.sh status                    # 进度总览
#
# 环境变量：
#   WOMD_ROOT      默认 /gs/bs/tga-RLA/qdeng/womd_dataset/v_1_3_1
#   WOMD_BUCKET    默认 gs://waymo_open_dataset_motion_v_1_3_1
#   WOMD_SPLITS    默认 "training_20s validation"
#   GCLOUD         默认 gcloud（自测时可换成桩程序）
#
# 为什么默认是 training_20s 而不是 training：training 是从同一批 20s 片段切出
# 的 9.1s 窗口，内容是 training_20s 的子集；RAP 需要「当前帧 + 4s 未来」，
# 20s 片段能切出更多不重叠的样本。testing 的未来轨迹不公开，不下。

set -euo pipefail

WOMD_ROOT="${WOMD_ROOT:-/gs/bs/tga-RLA/qdeng/womd_dataset/v_1_3_1}"
WOMD_BUCKET="${WOMD_BUCKET:-gs://waymo_open_dataset_motion_v_1_3_1}"
WOMD_SPLITS="${WOMD_SPLITS:-training_20s validation}"
GCLOUD="${GCLOUD:-gcloud}"

REMOTE_PREFIX="$WOMD_BUCKET/uncompressed/scenario"
DATA_DIR="$WOMD_ROOT/scenario"
MANIFEST_DIR="$WOMD_ROOT/_manifest"

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 B"; }
log()   { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

file_size() {
  [ -e "$1" ] || { echo 0; return 0; }
  stat -c %s "$1" 2>/dev/null && return 0   # GNU coreutils (Linux/TSUBAME)
  stat -f %z "$1" 2>/dev/null && return 0   # BSD stat (macOS,便于本地测试)
  echo ""
}

manifest_of() { echo "$MANIFEST_DIR/$1.tsv"; }

cmd_check() {
  echo "WOMD_ROOT   = $WOMD_ROOT"
  echo "WOMD_BUCKET = $WOMD_BUCKET"
  echo "WOMD_SPLITS = $WOMD_SPLITS"
  echo "GCLOUD      = $(command -v "$GCLOUD" || echo '未找到')"
  command -v "$GCLOUD" >/dev/null 2>&1 || { echo "=> 先安装 gcloud CLI"; return 1; }
  echo "账号        = $("$GCLOUD" auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
  echo
  echo "远端 $REMOTE_PREFIX/ :"
  if ! "$GCLOUD" storage ls "$REMOTE_PREFIX/"; then
    echo
    echo "=> 无法列出桶。常见原因：该 Google 账号没在 waymo.com/open 注册/接受许可，"
    echo "   或 gcloud 登录的不是那个账号，或版本号不对（改 WOMD_BUCKET 试 v_1_2_0）。"
    return 1
  fi
  echo
  echo "目标盘剩余："
  df -h "$(dirname "$WOMD_ROOT")" 2>/dev/null || df -h /gs/bs 2>/dev/null || true
}

# 清单 = 远端每个 shard 的 <bytes>\t<url>。它是后续一切校验的依据：
# 官方不发 checksum 清单，尺寸精确匹配是唯一可用的完整性判据（同 nuPlan 下载器）。
cmd_manifest() {
  local splits=("$@")
  [ ${#splits[@]} -gt 0 ] || read -r -a splits <<< "$WOMD_SPLITS"
  mkdir -p "$MANIFEST_DIR"
  for split in "${splits[@]}"; do
    local out tmp
    out="$(manifest_of "$split")"; tmp="$out.tmp"
    log "列出 $REMOTE_PREFIX/$split/"
    "$GCLOUD" storage ls -l "$REMOTE_PREFIX/$split/" \
      | awk '$NF ~ /^gs:\/\// && $NF !~ /\/$/ {print $1 "\t" $NF}' \
      | sort -k2 > "$tmp"
    local n; n=$(wc -l < "$tmp" | tr -d ' ')
    [ "$n" -gt 0 ] || { rm -f "$tmp"; log "FAIL $split 在远端为空或不存在"; return 1; }
    mv "$tmp" "$out"
    log "$split: $n 个 shard，合计 $(human "$(awk '{s+=$1} END {printf "%d", s}' "$out")") -> $out"
  done
}

cmd_size() {
  local total=0
  for split in $WOMD_SPLITS; do
    local m; m="$(manifest_of "$split")"
    [ -f "$m" ] || { echo "$split: 无清单，先跑 manifest"; continue; }
    local b; b=$(awk '{s+=$1} END {printf "%d", s}' "$m")
    total=$((total + b))
    printf '%-14s %5d shards  %10s\n' "$split" "$(wc -l < "$m" | tr -d ' ')" "$(human "$b")"
  done
  printf '%-14s %18s\n' TOTAL "$(human "$total")"
}

# 第 task 份 = 清单中行号 % ntasks == task-1 的 shard（轮转切分，各份大小接近）。
# 每个 task 只写自己那批文件，互不重叠，无需加锁。已存在且尺寸匹配的跳过，
# 所以超时后重投同一个 -t 序号即可续上。
cmd_fetch() {
  local split="${1:?用法: fetch <split> <task> <ntasks>}"
  local task="${2:?}" ntasks="${3:?}"
  local m; m="$(manifest_of "$split")"
  [ -f "$m" ] || { log "无清单 $m，先跑 manifest"; return 1; }

  local dest="$DATA_DIR/$split"
  mkdir -p "$dest"
  local todo=() n_skip=0
  while IFS=$'\t' read -r want url; do
    local local_path="$dest/${url##*/}" have
    have=$(file_size "$local_path")
    if [ "$have" = "$want" ]; then
      n_skip=$((n_skip + 1))
    else
      # 尺寸不符的残留（含 stat 失败）一律删掉重下，绝不拼接。
      [ -e "$local_path" ] && rm -f "$local_path"
      todo+=("$url")
    fi
  done < <(awk -v t="$task" -v n="$ntasks" '(NR - 1) % n == t - 1' "$m")

  log "$split task $task/$ntasks：跳过 $n_skip 个已完成，待下载 ${#todo[@]} 个"
  if [ ${#todo[@]} -gt 0 ]; then
    printf '%s\n' "${todo[@]}" | "$GCLOUD" storage cp -I "$dest/"
  fi

  local bad=0
  while IFS=$'\t' read -r want url; do
    local have; have=$(file_size "$dest/${url##*/}")
    if [ "$have" != "$want" ]; then
      log "FAIL 尺寸不符 ${url##*/}: 实得 ${have:-?}，应为 $want"; bad=1
    fi
  done < <(awk -v t="$task" -v n="$ntasks" '(NR - 1) % n == t - 1' "$m")
  [ "$bad" -eq 0 ] && log "$split task $task/$ntasks 全部尺寸校验通过"
  return $bad
}

cmd_verify() {
  local splits=("$@")
  [ ${#splits[@]} -gt 0 ] || read -r -a splits <<< "$WOMD_SPLITS"
  local rc=0
  for split in "${splits[@]}"; do
    local m; m="$(manifest_of "$split")"
    [ -f "$m" ] || { echo "$split: 无清单"; rc=1; continue; }
    local ok=0 missing=0 partial=0
    while IFS=$'\t' read -r want url; do
      local have; have=$(file_size "$DATA_DIR/$split/${url##*/}")
      if [ "$have" = "$want" ]; then ok=$((ok + 1))
      elif [ "$have" = "0" ]; then missing=$((missing + 1))
      else partial=$((partial + 1)); fi
    done < "$m"
    local n=$((ok + missing + partial))
    printf '%-14s OK %5d / %-5d  缺失 %5d  尺寸不符 %d\n' "$split" "$ok" "$n" "$missing" "$partial"
    [ "$ok" -eq "$n" ] || rc=1
  done
  return $rc
}

cmd_status() {
  echo "数据目录 : $DATA_DIR"
  echo "占用     : $(du -sh "$DATA_DIR" 2>/dev/null | cut -f1 || echo 0)"
  echo
  cmd_size
  echo
  cmd_verify || true
}

case "${1:-}" in
  check)    cmd_check ;;
  manifest) shift; cmd_manifest "$@" ;;
  size)     cmd_size ;;
  fetch)    shift; cmd_fetch "$@" ;;
  verify)   shift; cmd_verify "$@" ;;
  status)   cmd_status ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
