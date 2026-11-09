#!/bin/bash
# T2 feature-swap intervention: one job per rig preset (eval_t2_swap_v1.qsub), then the summary.
#
#   bash submit_t2_swap.sh <tag> [preset ...]
#
# Default presets: the two heights plus the two focal lengths (contrast where the encoder itself changes).
# Environment: PER_LOG / MAX_LOGS (smoke test), T2_TAG (default full; empty = no cross-check), H_RT.
# Results: $NAVSIM_EXP_ROOT/t2swap_<tag>_<preset>/<job>/; summary eval/t2_results/t2_swap_<tag>.md
# (summarize_t2_swap.qsub, held on the probe jobs); logs: $RAP/qsub_outputs/t2s_<tag>_<preset>.o<job>.
set -euo pipefail

RAP=/gs/bs/tga-RLA/qdeng/RAP
TAG="${1:?usage: submit_t2_swap.sh <tag> [preset ...]}"
H_RT="${H_RT:-6:00:00}"
T2_TAG="${T2_TAG-full}"
if [ $# -gt 1 ]; then PRESETS="${*:2}"; else PRESETS="h_p1.0 h_m0.7 f_0.8 f_1.25"; fi

n=0
for p in $PRESETS; do
  qsub -g tga-RLA -l gpu_1=1 -l h_rt="$H_RT" -N "t2s_${TAG}_${p//./p}" -o "$RAP/qsub_outputs/" \
    -v "PRESET=$p,EXPERIMENT=t2swap_${TAG}_${p},T2_TAG=$T2_TAG,PER_LOG=${PER_LOG:-0},MAX_LOGS=${MAX_LOGS:-0}" \
    "$RAP/scripts/jobs/eval/eval_t2_swap_v1.qsub"
  n=$((n + 1))
done
qsub -g tga-RLA -l cpu_4=1 -l h_rt=0:30:00 -N "sum_t2s_${TAG}" -o "$RAP/qsub_outputs/" -hold_jid "t2s_${TAG}_*" \
  -v "TAG=$TAG" "$RAP/scripts/jobs/eval/summarize_t2_swap.qsub"
echo "submitted $n T2 swap jobs + summary (tag=$TAG, presets=[$PRESETS], per_log=${PER_LOG:-0}, max_logs=${MAX_LOGS:-0})"
