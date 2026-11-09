#!/bin/bash
# K0 real-vs-raster diagnosis, step 1: run_t2_probe_v1.py with the identity preset on D-K checkpoints.
# With PRESET=ident, "c1" is the official-convention raster of the same token under the real calibration,
# so each run gives real vs raster PDMS / oracle@64 / mean64, the 64 candidate scores of both inputs and the
# query cosine per refiner round, on the T2 navtest subset (same tokens and stage A scenarios as T2).
#
#   bash submit_k0_probe.sh <tag> [model ...] [--dry-run]
#
# Models (default: all but official; official's ident run already exists as t2probe_full_ident):
#   D2-K0-scene-mix_e6 D2-K0-scene-mix_last D2-K1-mix_e13 D2-K10-mix D1-K100_e16 D1-K10 D1-K1_e16 official
# Results: $NAVSIM_EXP_ROOT/k0probe_<tag>_<model>/<job>/; logs: $RAP/qsub_outputs/k0p_<tag>_<model>.o<job>.
# Summary: scripts/jobs/eval/summarize_k0_probe.qsub.
set -euo pipefail

RAP=/gs/bs/tga-RLA/qdeng/RAP
EVAL=/gs/bs/tga-RLA/qdeng/rap_eval
CACHE=/gs/bs/tga-RLA/qdeng/navsim_workspace/exp/metric_cache_v1_navtest_t1sub
SCEN_DIR="${SCEN_DIR:-/gs/bs/tga-RLA/qdeng/navsim_workspace/exp/t2_scenarios_navtest_t1sub}"
H_RT="${H_RT:-4:00:00}"
OURS="$RAP/eval/hydra/rap_v2_agent.yaml"
OFFICIAL="$RAP/eval/hydra/rap_v2_official_agent.yaml"

TAG="${1:?usage: submit_k0_probe.sh <tag> [model ...] [--dry-run]}"
shift
DRY=""
MODELS=()
for a in "$@"; do
  if [ "$a" = --dry-run ]; then DRY=1; else MODELS+=("$a"); fi
done
[ "${#MODELS[@]}" -gt 0 ] || MODELS=(D2-K0-scene-mix_e6 D2-K0-scene-mix_last D2-K1-mix_e13 D2-K10-mix D1-K100_e16 D1-K10 D1-K1_e16)

# model -> "checkpoint glob|agent config" (eval aliases made by submit_dk_eval.sh / the E2a-E3 evals)
declare -A SPEC=(
  [D2-K0-scene-mix_e6]="$EVAL/ckpts/D2-K0-scene-mix_s0/epoch6-step1498.ckpt|$OURS"
  [D2-K0-scene-mix_last]="$EVAL/ckpts/D2-K0-scene-mix_s0/last-epoch*-step4280.ckpt|$OURS"
  [D2-K1-mix_e13]="$EVAL/ckpts/D2-K1-mix_s0/epoch13-step3038.ckpt|$OURS"
  [D2-K10-mix]="$EVAL/ckpts/E3_mix_p010_s0/epoch16-step3638.ckpt|$OURS"
  [D1-K100_e16]="$EVAL/ckpts/E2a_navsim_p100_s0/epoch16-step5661.ckpt|$OURS"
  [D1-K10]="$EVAL/ckpts/E2a_navsim_p010_s0/epoch18-step646.ckpt|$OURS"
  [D1-K1_e16]="$EVAL/ckpts/D1-K1_s0/epoch16-step612.ckpt|$OURS"
  [official]="$EVAL/ckpts/official/RAP_DINO_navsimv1.ckpt|$OFFICIAL"
)

[ -f "$CACHE/metadata/metric_cache_v1_navtest_t1sub_metadata_node_0.csv" ] || { echo "缺少 metric cache: $CACHE" >&2; exit 1; }
[ -d "$SCEN_DIR" ] || { echo "缺少 stage A 目录: $SCEN_DIR" >&2; exit 1; }

# Resolve everything before submitting anything.
declare -A CKPT CONF
for m in "${MODELS[@]}"; do
  [ -n "${SPEC[$m]:-}" ] || { echo "未知模型: $m（可选: ${!SPEC[*]}）" >&2; exit 1; }
  pat="${SPEC[$m]%%|*}"; conf="${SPEC[$m]#*|}"
  mapfile -t hits < <(compgen -G "$pat" || true)
  [ "${#hits[@]}" -eq 1 ] || { echo "$m: $pat 匹配到 ${#hits[@]} 个文件" >&2; exit 1; }
  [ -f "$conf" ] || { echo "缺少 agent 配置: $conf" >&2; exit 1; }
  CKPT[$m]="${hits[0]}"; CONF[$m]="$conf"
  echo "[k0-probe] $m: $(readlink -f "${hits[0]}") ($(basename "$conf"))"
done

n=0
for m in "${MODELS[@]}"; do
  cmd=(qsub -g tga-RLA -l gpu_1=1 -l h_rt="$H_RT" -N "k0p_${TAG}_${m}" -o "$RAP/qsub_outputs/"
       -v "RAP_CKPT=${CKPT[$m]},AGENT_CONFIG=${CONF[$m]},PRESET=ident,CACHE_PATH=$CACHE,SCEN_DIR=$SCEN_DIR,EXPERIMENT=k0probe_${TAG}_${m},PER_LOG=${PER_LOG:-0},MAX_LOGS=${MAX_LOGS:-0}"
       "$RAP/scripts/jobs/eval/eval_t2_probe_v1.qsub")
  echo "+ ${cmd[*]}"
  [ -n "$DRY" ] || "${cmd[@]}"
  n=$((n + 1))
done
echo "${DRY:+[dry-run] }$n K0 probe jobs (tag=$TAG, per_log=${PER_LOG:-0}, max_logs=${MAX_LOGS:-0}); logs: $RAP/qsub_outputs/k0p_${TAG}_*.o<job>"
