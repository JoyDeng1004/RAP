#!/bin/bash
# T2 factor experiment: one probe job per rig preset (eval_t2_probe_v1.qsub).
#
#   bash submit_t2_probe.sh <tag> [preset ...]
#
# Environment:
#   SCEN_DIR   stage A output (build_t2_scenarios.qsub); default: the full navtest t1sub set
#   PER_LOG / MAX_LOGS   smoke test, must match what stage A built
#   HOLD_JID   job id (or name pattern) to wait for, e.g. the stage A job
# Results: $NAVSIM_EXP_ROOT/t2probe_<tag>_<preset>/<job>/; logs: $RAP/qsub_outputs/t2p_<tag>_<preset>.o<job>.
set -euo pipefail

RAP=/gs/bs/tga-RLA/qdeng/RAP
TAG="${1:?usage: submit_t2_probe.sh <tag> [preset ...]}"
CKPT=/gs/bs/tga-RLA/qdeng/rap_eval/ckpts/official/RAP_DINO_navsimv1.ckpt
CACHE=/gs/bs/tga-RLA/qdeng/navsim_workspace/exp/metric_cache_v1_navtest_t1sub
SCEN_DIR="${SCEN_DIR:-/gs/bs/tga-RLA/qdeng/navsim_workspace/exp/t2_scenarios_navtest_t1sub}"
H_RT="${H_RT:-4:00:00}"

if [ $# -gt 1 ]; then
  PRESETS="${*:2}"
else
  PRESETS="ident mask_ident drop_f0 h_p1.0 h_m0.7 d_p1.0 d_m1.0 l_p0.5 l_m0.5 f_0.8 f_1.25"
  PRESETS="$PRESETS stale_h_p1.0 stale_h_m0.7 stale_d_p1.0 stale_d_m1.0 stale_l_p0.5 stale_l_m0.5 stale_f_0.8 stale_f_1.25"
  PRESETS="$PRESETS mask_h_m0.7 mask_h_p1.0 mask_d_m1.0 mask_f_0.8"
fi
for f in "$CKPT" "$CACHE/metadata/metric_cache_v1_navtest_t1sub_metadata_node_0.csv"; do
  [ -f "$f" ] || { echo "缺少文件: $f" >&2; exit 1; }
done
HOLD=()
[ -n "${HOLD_JID:-}" ] && HOLD=(-hold_jid "$HOLD_JID")
if [ -z "${HOLD_JID:-}" ] && [ ! -d "$SCEN_DIR" ]; then
  echo "缺少 stage A 目录: $SCEN_DIR（先跑 build_t2_scenarios.qsub，或设 HOLD_JID）" >&2; exit 1
fi

n=0
for p in $PRESETS; do
  qsub -g tga-RLA -l gpu_1=1 -l h_rt="$H_RT" -N "t2p_${TAG}_${p//./p}" -o "$RAP/qsub_outputs/" "${HOLD[@]}" \
    -v "RAP_CKPT=$CKPT,PRESET=$p,CACHE_PATH=$CACHE,SCEN_DIR=$SCEN_DIR,EXPERIMENT=t2probe_${TAG}_${p},PER_LOG=${PER_LOG:-0},MAX_LOGS=${MAX_LOGS:-0}" \
    "$RAP/scripts/jobs/eval/eval_t2_probe_v1.qsub"
  n=$((n + 1))
done
echo "submitted $n T2 probe jobs (tag=$TAG, scen=$SCEN_DIR, per_log=${PER_LOG:-0}, max_logs=${MAX_LOGS:-0}); logs: $RAP/qsub_outputs/t2p_${TAG}_*.o<job>"
