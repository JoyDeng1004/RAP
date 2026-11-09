#!/bin/bash
# Exp-0 / T1 step 7: one eval_pdm_score_v1 job per rig preset on the 1,501-token navtest subset.
#
#   bash submit_t1_sweep.sh [tag] [ckpt] [agent_yaml] [preset ...]
#
# Defaults: tag=official, the official RAP v1 checkpoint, the official rig agent, and
# "raw" plus every entry of rig_transform.PRESETS. Results land in
# $NAVSIM_EXP_ROOT/t1_sub_<tag>_<preset>/<job>/; summarize with eval/tools/summarize_t1.py.
# Logs go to $RAP/qsub_outputs/t1_<tag>_<preset>.o<job> (synced back by mutagen); run from any directory.
set -euo pipefail

RAP=/gs/bs/tga-RLA/qdeng/RAP
PY=/gs/bs/tga-RLA/qdeng/anaconda3/envs/rap/bin/python
TAG="${1:-official}"
CKPT="${2:-/gs/bs/tga-RLA/qdeng/rap_eval/ckpts/official/RAP_DINO_navsimv1.ckpt}"
AGENT="${3:-$RAP/eval/hydra/rap_v2_official_rig_agent.yaml}"
CACHE=/gs/bs/tga-RLA/qdeng/navsim_workspace/exp/metric_cache_v1_navtest_t1sub

if [ $# -gt 3 ]; then
  PRESETS="${*:4}"
else
  PRESETS="raw $(cd "$RAP/eval" && PYTHONPATH=py "$PY" -c 'from rap_eval_agents.rig_transform import PRESETS; print(" ".join(PRESETS))')"
fi
for f in "$CKPT" "$AGENT" "$CACHE/metadata/metric_cache_v1_navtest_t1sub_metadata_node_0.csv"; do
  [ -f "$f" ] || { echo "缺少文件: $f" >&2; exit 1; }
done

n=0
for p in $PRESETS; do
  qsub -g tga-RLA -l gpu_1=1 -l h_rt=1:30:00 -N "t1_${TAG}_${p//./p}" -o "$RAP/qsub_outputs/" \
    -v "RAP_CKPT=$CKPT,AGENT_CONFIG=$AGENT,EVAL_DIR=$RAP/eval,RAP_SELECTOR=scorer,RIG_PRESET=$p,CACHE_PATH=$CACHE,EXPERIMENT=t1_sub_${TAG}_${p}" \
    "$RAP/scripts/jobs/eval/eval_pdm_score_v1.qsub"
  n=$((n + 1))
done
# Summary once every preset job of this tag has finished; writes $RAP/eval/t1_results/t1_summary_<tag>.md.
# (A job script, not "-b y python ...": that form exited 0 in 0.4 s with no output, job 9015671.)
qsub -g tga-RLA -l cpu_4=1 -l h_rt=0:15:00 -N "sum_t1_${TAG}" -o "$RAP/qsub_outputs/" -hold_jid "t1_${TAG}_*" \
  -v "TAG=$TAG" "$RAP/scripts/jobs/eval/summarize_t1.qsub"
echo "submitted $n jobs + 1 summary job (tag=$TAG); logs: $RAP/qsub_outputs/{t1,sum_t1}_${TAG}*.o<job>"
