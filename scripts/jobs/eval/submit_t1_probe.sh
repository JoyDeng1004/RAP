#!/bin/bash
# Exp-0 / T1 step 8: one probe job per rig preset (8a fixed candidates for every preset; 8b / 8c
# backbone equivariance and feature translation for pure translations), then one summary job.
#
#   bash submit_t1_probe.sh [tag] [preset ...]
#
# PER_LOG=2 in the environment -> smoke test (first 2 tokens per log), e.g.
#   PER_LOG=2 bash submit_t1_probe.sh smoke ppy_m7r pitch_up_m10
# Results: $NAVSIM_EXP_ROOT/t1probe_<tag>_<preset>/<job>/; summary eval/t1_results/t1_probe_<tag>.md.
# Logs: $RAP/qsub_outputs/t1p_<tag>_<preset>.o<job> (synced back by mutagen); run from any directory.
set -euo pipefail

RAP=/gs/bs/tga-RLA/qdeng/RAP
TAG="${1:-official}"
CKPT=/gs/bs/tga-RLA/qdeng/rap_eval/ckpts/official/RAP_DINO_navsimv1.ckpt
CACHE=/gs/bs/tga-RLA/qdeng/navsim_workspace/exp/metric_cache_v1_navtest_t1sub
H_RT="${H_RT:-3:00:00}"

if [ $# -gt 1 ]; then
  PRESETS="${*:2}"
else
  # pure translations (8a + 8b + 8c): whole feature rows, the T1 vertical shifts, horizontal pp shifts
  PRESETS="ppy_m7r ppy_m3r ppy_m1r ppy_p1r ppy_p3r ppy_m272 ppy_m135 ppy_p135 pp_p5 pp_m5"
  # 8a only
  PRESETS="$PRESETS pitch_up_m10 pitch_up_m5 pitch_up_p5 focal_0.8 focal_1.25 drop_black_f0"
fi
for f in "$CKPT" "$CACHE/metadata/metric_cache_v1_navtest_t1sub_metadata_node_0.csv"; do
  [ -f "$f" ] || { echo "缺少文件: $f" >&2; exit 1; }
done

n=0
for p in $PRESETS; do
  qsub -g tga-RLA -l gpu_1=1 -l h_rt="$H_RT" -N "t1p_${TAG}_${p//./p}" -o "$RAP/qsub_outputs/" \
    -v "RAP_CKPT=$CKPT,PRESET=$p,CACHE_PATH=$CACHE,EXPERIMENT=t1probe_${TAG}_${p},PER_LOG=${PER_LOG:-0}" \
    "$RAP/scripts/jobs/eval/eval_t1_probe_v1.qsub"
  n=$((n + 1))
done
qsub -g tga-RLA -l cpu_4=1 -l h_rt=0:15:00 -N "sum_t1p_${TAG}" -o "$RAP/qsub_outputs/" -hold_jid "t1p_${TAG}_*" \
  -v "TAG=$TAG" "$RAP/scripts/jobs/eval/summarize_t1_probe.qsub"
echo "submitted $n probe jobs + 1 summary job (tag=$TAG, per_log=${PER_LOG:-0}); logs: $RAP/qsub_outputs/{t1p,sum_t1p}_${TAG}*.o<job>"
