#!/bin/bash
# Queue several train_dk.qsub runs back to back inside one advance reservation, each held on
# the previous one (-hold_jid), so the reserved node never idles between runs.
# Usage (run on TSUBAME):
#   bash scripts/jobs/submit_dk_chain.sh <AR_ID> [--after <JOB_ID>] [--dry-run] RUN:SEED:H_RT ...
#   --after: hold the first job on an existing job too (e.g. a run that will be cut by the
#            end of an earlier AR and must resume from last.ckpt in this one).
# Example:
#   bash scripts/jobs/submit_dk_chain.sh 9346 --after 9220400 \
#       D2-K1-nusc-tgt-23k:0:6:00:00 D3-K1-nusc-womd-tgt-23k:0:15:00:00
# A job only starts if its h_rt still fits before the reservation ends, so keep the sum of
# the expected run times below the AR length. A rerun of RUN:SEED resumes from last.ckpt.

set -euo pipefail

RAP="${RAP:-/gs/bs/tga-RLA/qdeng/RAP}"
OUT="$RAP/qsub_outputs/"
AR="${1:?AR_ID}"; shift
PREV=""; DRY=""
SPECS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --after) PREV="${2:?--after needs a job id}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) SPECS+=("$1"); shift ;;
  esac
done
[ "${#SPECS[@]}" -gt 0 ] || { echo "[chain] 没有 RUN:SEED:H_RT" >&2; exit 1; }

for spec in "${SPECS[@]}"; do
  RUN="${spec%%:*}"; rest="${spec#*:}"; SEED="${rest%%:*}"; HRT="${rest#*:}"
  case "$SEED" in 0|1|2) ;; *) echo "[chain] SEED 只能是 0/1/2: $spec" >&2; exit 1 ;; esac
  [[ "$HRT" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "[chain] h_rt 须为 H:MM:SS: $spec" >&2; exit 1; }
  grep -qE "^  $RUN\)" "$RAP/scripts/jobs/train_dk.qsub" || { echo "[chain] train_dk.qsub 里没有 RUN=$RUN" >&2; exit 1; }
  cmd=(qsub -terse -g tga-RLA -ar "$AR" -l node_f=1 -l "h_rt=$HRT" -o "$OUT"
       -v "RUN=$RUN,SEED=$SEED,WANDB_MODE=online")
  [ -n "$PREV" ] && cmd+=(-hold_jid "$PREV")
  cmd+=("$RAP/scripts/jobs/train_dk.qsub")
  if [ -n "$DRY" ]; then
    echo "+ ${cmd[*]}"
    PREV="<job:${RUN}_s${SEED}>"
  else
    PREV="$("${cmd[@]}")"
    echo "[chain] ${RUN}_s${SEED}  h_rt=$HRT  job=$PREV"
  fi
done
