#!/bin/bash
# nuPlan v1.1 train+val DB 归档下载器（TSUBAME / UGE）。
#
# 背景：nuPlan 的 trainval DB 不是散装 .db 分发的，是按城市打包的 zip，所以
# `aws s3 sync s3://.../splits/trainval/` 没有东西可同步 —— 那个前缀不存在。
# 本脚本下载 13 个归档，校验后把其中的 *.db 平铺展开到 splits/trainval/。
#
# 桶：s3://motional-nuplan/public/nuplan-v1.1/（AWS Open Data，匿名可读，
#     region ap-northeast-1 —— 与 TSUBAME 同区）
# 来源依据：RAP Issue #3 作者答复「You still need to download nuplan logs,
#           it is called nuplan train split」+ https://www.nuscenes.org/nuplan
#
# 用法：
#   ./download_nuplan_trainval.sh list                 # 清单与本地状态
#   ./download_nuplan_trainval.sh fetch 4              # 下载并展开第 4 个归档
#   ./download_nuplan_trainval.sh fetch all            # 串行全部（不推荐，用 array job）
#   ./download_nuplan_trainval.sh verify               # 只校验，不下载
#   ./download_nuplan_trainval.sh status               # 进度总览
#
# 环境变量：
#   NUPLAN_ROOT    默认 /gs/bs/tga-RLA/qdeng/nuplan_dataset/nuplan-v1.1
#   STAGING_DIR    zip 暂存区，默认 $NUPLAN_ROOT/_staging
#   KEEP_ZIP       1=展开后保留 zip（默认 0，展开成功即删，控制峰值占用）
#
# 容量：压缩态合计 1,037.7 GB；展开后更大。KEEP_ZIP=0 时峰值 ≈ 最大归档
# (163.5 GB) + 其展开结果，而不是 1TB+1TB。开跑前先确认配额。

set -euo pipefail

NUPLAN_ROOT="${NUPLAN_ROOT:-/gs/bs/tga-RLA/qdeng/nuplan_dataset/nuplan-v1.1}"
TARGET_DIR="$NUPLAN_ROOT/splits/trainval"
STAGING_DIR="${STAGING_DIR:-$NUPLAN_ROOT/_staging}"
KEEP_ZIP="${KEEP_ZIP:-0}"

BUCKET="motional-nuplan"
PREFIX="public/nuplan-v1.1"
REGION="ap-northeast-1"
# 可覆盖：镜像源，或本地 file:// 前缀（自测时用，避免误触真实下载）。
HTTP_BASE="${HTTP_BASE:-https://${BUCKET}.s3.${REGION}.amazonaws.com/${PREFIX}}"

# name<TAB>expected_bytes —— 2026-09-15 匿名 ListObjectsV2 实测值。
# 尺寸是唯一可用的完整性判据：官方没有发布 checksum 清单，所以本脚本
# 用「尺寸精确匹配 + unzip -t 全成员 CRC 校验」两道，而不是假装有 md5。
ARCHIVES=(
  "nuplan-v1.1_train_boston.zip	38161149300"
  "nuplan-v1.1_train_pittsburgh.zip	30620248893"
  "nuplan-v1.1_train_singapore.zip	34959594178"
  "nuplan-v1.1_train_vegas_1.zip	154406370335"
  "nuplan-v1.1_train_vegas_2.zip	152025130331"
  "nuplan-v1.1_train_vegas_3.zip	151596886608"
  "nuplan-v1.1_train_vegas_4.zip	143595811311"
  "nuplan-v1.1_train_vegas_5.zip	136371675016"
  "nuplan-v1.1_train_vegas_6.zip	175541219902"
  "nuplan-v1.1_val.zip	96957279835"
)

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 B"; }
log()   { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# 文件不存在返回 0；stat 本身失败（权限、FS 异常、非 GNU stat）返回空串。
# 这个区分是必要的：下面的续传逻辑把"大小未知"当作危险状态处理，绝不能把它
# 和"大小为 0"混为一谈 —— 后者可以安全续传，前者续传会拼出坏文件。
file_size() {
  [ -e "$1" ] || { echo 0; return 0; }
  stat -c %s "$1" 2>/dev/null && return 0   # GNU coreutils (Linux/TSUBAME)
  stat -f %z "$1" 2>/dev/null && return 0   # BSD stat (macOS,便于本地测试)
  echo ""                                    # 拿不到大小
}

# 展开后的 .db 落在 TARGET_DIR 下（平铺）。用一个 .done 标记表示该归档已完整
# 展开过 —— 不靠 .db 计数，因为不同归档的 db 混在同一个目录里数不清。
#
# 标记【绝不能】放在 TARGET_DIR 里：RAP 的 create_openscene_metadata.py:537
# 构造 db 列表时不过滤后缀，直接 `f[:-3]` 盲切目录下的每个文件 ——
# `.nuplan-v1.1_train_vegas_2.done` 会被切成 `.nuplan-v1.1_train_vegas_2.d`，
# 再拼 `.db` 去读，整个渲染作业当场崩。所以放同级的 _manifest/。
MANIFEST_DIR="${MANIFEST_DIR:-$NUPLAN_ROOT/_manifest}"
done_marker() { echo "$MANIFEST_DIR/${1%.zip}.done"; }

cmd_list() {
  printf '%-40s %12s  %s\n' ARCHIVE SIZE STATUS
  local idx=0 total=0
  for row in "${ARCHIVES[@]}"; do
    idx=$((idx + 1))
    local name="${row%%	*}" want="${row##*	}"
    total=$((total + want))
    local status="missing"
    if [ -f "$(done_marker "$name")" ]; then
      status="extracted"
    elif [ -f "$STAGING_DIR/$name" ]; then
      local have; have=$(file_size "$STAGING_DIR/$name")
      if [ "$have" -eq "$want" ]; then status="downloaded"
      else status="partial $(( have * 100 / want ))%"; fi
    fi
    printf '%2d. %-36s %12s  %s\n' "$idx" "$name" "$(human "$want")" "$status"
  done
  printf '%-40s %12s\n' "TOTAL" "$(human "$total")"
}

# 下载单个归档。aws cli 走多段并行；没有 aws 就退回 curl -C -（可断点续传，
# 对这种单个百 GB 文件其实更稳）。两条路都用匿名访问。
download_one() {
  local name="$1" want="$2" dest="$STAGING_DIR/$name"
  mkdir -p "$STAGING_DIR"

  # 先判定本地残留处于哪种状态。curl -C - 是"盲目追加"，只有在确知本地是
  # 同一文件的真前缀时才安全，所以这里必须先把其余情况排干净。
  local have; have=$(file_size "$dest")
  if [ -z "$have" ]; then
    log "无法确定本地文件大小，按不可信处理，删除重下: $dest"
    rm -f "$dest"; have=0
  elif [ "$have" -eq "$want" ]; then
    log "已存在且尺寸匹配，跳过下载: $name"
    return 0
  elif [ "$have" -gt "$want" ]; then
    # 比目标还大 —— 不可能是前缀，续传只会越拼越坏。
    log "本地文件大于目标（$have > $want），不是有效前缀，删除重下: $name"
    rm -f "$dest"; have=0
  fi

  if command -v aws >/dev/null 2>&1; then
    log "aws s3 cp $name ($(human "$want"))"
    # aws s3 cp 不支持断点续传：任何残留都先删掉再整下，避免拼接出坏文件。
    [ -f "$dest" ] && rm -f "$dest"
    aws s3 cp --no-sign-request --region "$REGION" \
      "s3://${BUCKET}/${PREFIX}/${name}" "$dest"
  else
    [ "$have" -gt 0 ] && log "从 $(human "$have") 处续传: $name" \
                      || log "未找到 aws cli，用 curl 下载: $name"
    curl -fL --retry 10 --retry-delay 15 --retry-all-errors \
      -C - -o "$dest" "${HTTP_BASE}/${name}"
  fi

  have=$(file_size "$dest")
  if [ "$have" != "$want" ]; then
    log "FAIL 尺寸不符 $name: 实得 $have，应为 $want"
    return 1
  fi
  log "尺寸校验通过: $name"
}

# 展开。RAP 的 create_openscene_metadata.py 对 NUPLAN_DB_PATH 做的是
# os.listdir 取平铺文件（process_data/create_openscene_metadata.py:537），
# 所以这里用 -j 丢弃归档内部目录层级，只取 *.db，与归档内部布局无关。
extract_one() {
  local name="$1" src="$STAGING_DIR/$name"
  local marker; marker="$(done_marker "$name")"
  [ -f "$marker" ] && { log "已展开，跳过: $name"; return 0; }

  log "CRC 校验（unzip -t，全成员）: $name"
  unzip -tq "$src"

  mkdir -p "$TARGET_DIR" "$MANIFEST_DIR"
  local before after
  before=$(find "$TARGET_DIR" -maxdepth 1 -name '*.db' | wc -l | tr -d " ")
  log "展开 *.db -> $TARGET_DIR"
  unzip -o -j -q "$src" '*.db' -d "$TARGET_DIR"
  after=$(find "$TARGET_DIR" -maxdepth 1 -name '*.db' | wc -l | tr -d " ")

  printf '%s\textracted_at=%s\tdb_added=%s\n' \
    "$name" "$(date -Iseconds)" "$((after - before))" > "$marker"
  log "完成 $name：新增 $((after - before)) 个 .db（累计 $after）"

  if [ "$KEEP_ZIP" != "1" ]; then
    rm -f "$src"
    log "已删除暂存 zip（KEEP_ZIP=1 可保留）: $name"
  fi
}

cmd_fetch() {
  local selector="${1:?用法: fetch <序号 1-${#ARCHIVES[@]}|all>}"
  local indices=()
  if [ "$selector" = "all" ]; then
    for i in $(seq 1 ${#ARCHIVES[@]}); do indices+=("$i"); done
  else
    indices=("$selector")
  fi
  for i in "${indices[@]}"; do
    local row="${ARCHIVES[$((i - 1))]}"
    [ -n "$row" ] || { log "序号越界: $i"; return 1; }
    local name="${row%%	*}" want="${row##*	}"
    log "=== [$i/${#ARCHIVES[@]}] $name ==="
    # .done 必须在下载之前查：KEEP_ZIP=0 时展开后 zip 就删了，若等到
    # extract_one 才发现已完成，重投这个 array task 会白下一遍上百 GB。
    if [ -f "$(done_marker "$name")" ]; then
      log "已完成（$(done_marker "$name")），跳过: $name"
      continue
    fi
    download_one "$name" "$want"
    extract_one "$name"
  done
}

cmd_verify() {
  local bad=0
  for row in "${ARCHIVES[@]}"; do
    local name="${row%%	*}" want="${row##*	}"
    local marker; marker="$(done_marker "$name")"
    if [ -f "$marker" ]; then
      printf 'OK       %-36s %s\n' "$name" "$(tr '\t' ' ' < "$marker" | cut -d' ' -f2-)"
    elif [ -f "$STAGING_DIR/$name" ]; then
      local have; have=$(file_size "$STAGING_DIR/$name")
      if [ "$have" -eq "$want" ]; then
        printf 'ZIP-ONLY %-36s 尺寸符合，待展开\n' "$name"
      else
        printf 'PARTIAL  %-36s %s / %s\n' "$name" "$(human "$have")" "$(human "$want")"; bad=1
      fi
    else
      printf 'MISSING  %-36s\n' "$name"; bad=1
    fi
  done
  return $bad
}

cmd_status() {
  local n; n=$(find "$TARGET_DIR" -maxdepth 1 -name '*.db' 2>/dev/null | wc -l | tr -d " ")
  echo "目标目录 : $TARGET_DIR"
  echo ".db 文件 : $n"
  echo "占用     : $(du -sh "$TARGET_DIR" 2>/dev/null | cut -f1)"
  echo "暂存区   : $(du -sh "$STAGING_DIR" 2>/dev/null | cut -f1 || echo 0)"
  echo
  cmd_verify || true
  echo
  echo "覆盖率检查（需要 navsim_logs 才能算,见下文 coverage 子命令）"
}

# 真正重要的那个数：下到的 .db 覆盖了 RAP 需要 rasterize 的多少个 log。
#
# 参照系必须是 navall.yaml 的 14,539 个 log，不是 navsim_logs/trainval 里那
# 1,320 个 pkl。后者是 OpenScene 对 nuPlan 的下采样再分发（少一个数量级），
# 而 RAP 的 process.sh 是**重新生成** navsim_logs/trainval —— 它要覆盖完整的
# nuPlan trainval（作者所说的 1200h），navall 那 14,539 就是那次全量跑的产物。
# 拿 1,320 当分母会在远未下够时就报"全覆盖"。
NAVALL_YAML_DEFAULT=/gs/bs/tga-RLA/qdeng/RAP/navsim/planning/script/config/common/train_test_split/scene_filter/navall.yaml

cmd_coverage() {
  local navall="${1:-$NAVALL_YAML_DEFAULT}"
  [ -f "$navall" ] || { echo "navall.yaml 不存在: $navall"; return 1; }

  local need have hit
  # navall.yaml 的 log_names 段：取 tokens: 之前的所有列表项，去掉引号。
  need=$(sed -n '/^log_names:/,/^tokens:/p' "$navall" \
         | sed -n "s/^[[:space:]]*-[[:space:]]*['\"]\{0,1\}\([^'\"]*\)['\"]\{0,1\}[[:space:]]*$/\1/p" \
         | sort -u)
  have=$(find "$TARGET_DIR" -maxdepth 1 -name '*.db' | sed 's|.*/||; s|\.db$||' | sort -u)
  hit=$(comm -12 <(echo "$need") <(echo "$have") | wc -l | tr -d " ")

  local n_need n_have
  n_need=$(echo "$need" | grep -c . || true)
  n_have=$(echo "$have" | grep -c . || true)

  echo "navall 需要 rasterize 的 log : $n_need"
  echo "已下到的 nuPlan .db          : $n_have"
  echo "交集（可 rasterize）         : $hit  ($(( hit * 100 / (n_need > 0 ? n_need : 1) ))%)"
  if [ "$hit" -eq "$n_need" ]; then
    echo "=> 全覆盖，可以开始 rasterization"
  else
    echo "=> 还差 $(( n_need - hit )) 个 log，继续下剩余归档"
  fi
}

case "${1:-}" in
  list)     cmd_list ;;
  fetch)    shift; cmd_fetch "$@" ;;
  verify)   cmd_verify ;;
  status)   cmd_status ;;
  coverage) shift; cmd_coverage "$@" ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
