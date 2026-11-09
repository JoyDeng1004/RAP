#!/bin/bash
# Link one D-K checkpoint into $EVAL/ckpts, inspect it, and submit the same five NAVSIM eval
# jobs used for E2a-E4 (v1 scorer, v1 medoid, v2 navtest scorer, navhard two-stage scorer,
# v1 oracle@64), with logs in $RAP/qsub_outputs.
# Usage (run on TSUBAME):
#   bash scripts/jobs/eval/submit_dk_eval.sh <EXP> <EPOCH|last> <TAG> <TRAIN_JOB> [--dry-run]
#   EXP:   training experiment_name, e.g. D2-K1-mix_s0
#   EPOCH: epoch number of the kept checkpoint, or "last" for last.ckpt
#   TAG:   name used in EXPERIMENT, e.g. D2-K1-mix_e13
# Example:
#   bash scripts/jobs/eval/submit_dk_eval.sh D2-K1-mix_s0 13 D2-K1-mix_e13 9062115

set -euo pipefail

EXP="${1:?EXP}"; EPOCH="${2:?EPOCH or last}"; TAG="${3:?TAG}"; TRAIN_JOB="${4:?TRAIN_JOB}"
DRY="${5:-}"

RAP="${RAP:-/gs/bs/tga-RLA/qdeng/RAP}"
EVAL="${EVAL:-/gs/bs/tga-RLA/qdeng/rap_eval}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/gs/bs/tga-RLA/qdeng/navsim_workspace/exp}"
PY="${PY:-/gs/bs/tga-RLA/qdeng/anaconda3/envs/rap/bin/python}"
OUT="$RAP/qsub_outputs/"

fail() { echo "[dk-eval] 失败: $*" >&2; exit 1; }
case "$TAG" in *=*|*+*) fail "TAG 不能含 = 或 +: $TAG" ;; esac

CKPT_DIR="$RAP/rap/$EXP/checkpoints"
if [ "$EPOCH" = last ]; then
  SRC="$CKPT_DIR/last.ckpt"
else
  mapfile -t hits < <(ls "$CKPT_DIR"/epoch="$EPOCH"-step=*.ckpt 2>/dev/null)
  [ "${#hits[@]}" -eq 1 ] || fail "$CKPT_DIR 里 epoch=$EPOCH 的 ckpt 有 ${#hits[@]} 个"
  SRC="${hits[0]}"
fi
[ -f "$SRC" ] || fail "找不到 $SRC"

DST_DIR="$EVAL/ckpts/$EXP"
if [ "$EPOCH" = last ]; then
  # last.ckpt carries no epoch in its name; read it from the checkpoint.
  META=$(env -u PYTHONPATH "$PY" -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu'); print(c['epoch'], c['global_step'])" "$SRC")
  CKPT="$DST_DIR/last-epoch${META% *}-step${META#* }.ckpt"
else
  CKPT="$DST_DIR/$(basename "$SRC" | tr -d '=')"
fi
echo "[dk-eval] SRC=$SRC"
echo "[dk-eval] CKPT=$CKPT"

if [ "$DRY" != --dry-run ]; then
  mkdir -p "$DST_DIR"
  [ -e "$CKPT" ] && fail "别名已存在，先检查: $CKPT"
  ln -s "$SRC" "$CKPT"
  {
    echo "alias: $CKPT"
    echo "target: $(readlink -f "$CKPT")"
    echo "sha256: $(sha256sum "$SRC" | cut -d' ' -f1)"
    echo "wandb_run_id: $EXP"
    echo "eval_tag: $TAG"
    echo "train_job: $TRAIN_JOB"
    echo "hydra_dir: $(ls -td "$NAVSIM_EXP_ROOT/$EXP"/*/ | head -1)"
    echo "train_log: $RAP/logs/${EXP}_${TRAIN_JOB}.log"
    echo "---"
  } >> "$DST_DIR/SOURCE.txt"
  tail -9 "$DST_DIR/SOURCE.txt"
  env -u PYTHONPATH "$PY" "$EVAL/inspect_rap_ckpt.py" --ckpt "$CKPT" --rap-root "$RAP" \
    --dino-ckpt "$RAP/ckpts/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth" 2>/dev/null | grep -E "^\[top\] (epoch|global_step)|^\[heads\]|^\[backbone\] exact|^VERDICT"
fi

run() { echo "+ $*"; [ "$DRY" = --dry-run ] || "$@"; }
cd "$RAP"
run qsub -g tga-RLA -l gpu_1=1 -l h_rt=3:00:00 -o "$OUT" -v RAP_CKPT="$CKPT",RAP_SELECTOR=scorer,EXPERIMENT="eval_v1_navtest_${TAG}_scorer" "$EVAL/jobs/eval_pdm_score_v1.qsub" 4
run qsub -g tga-RLA -l gpu_1=1 -l h_rt=3:00:00 -o "$OUT" -v RAP_CKPT="$CKPT",RAP_SELECTOR=medoid,EXPERIMENT="eval_v1_navtest_${TAG}_medoid" "$EVAL/jobs/eval_pdm_score_v1.qsub" 4
run qsub -g tga-RLA -l gpu_1=1 -l h_rt=3:00:00 -o "$OUT" -v PYTHONPATH="$EVAL/py",RAP_CKPT="$CKPT",RAP_SELECTOR=scorer,EXPERIMENT="eval_v2_navtest_${TAG}_scorer" "$RAP/process_data/eval_pdm_score_v2.qsub" navtest 4 rap_v2_agent
run qsub -g tga-RLA -l gpu_1=1 -l h_rt=2:00:00 -o "$OUT" -v PYTHONPATH="$EVAL/py",RAP_CKPT="$CKPT",RAP_SELECTOR=scorer,EXPERIMENT="eval_v2_navhard_two_stage_${TAG}_scorer" "$RAP/process_data/eval_pdm_score_v2.qsub" navhard_two_stage 4 rap_v2_agent
run qsub -g tga-RLA -l gpu_1=1 -l h_rt=5:00:00 -o "$OUT" -v RAP_CKPT="$CKPT",RAP_SELECTOR=scorer,EXPERIMENT="eval_v1_oracle64_navtest_${TAG}" "$EVAL/jobs/eval_oracle64_v1.qsub" 4
