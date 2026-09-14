#!/bin/bash
# Resume a T5 chain from Stage 4 onward, for a chain whose Stage 0-3 (setup,
# data, baseline, monotonic) already completed successfully but whose
# evaluation/attack/aggregate stages need re-running -- e.g. because the
# Stage 4 merge hit the '_metadata' bug (fixed in stage_4_merge_shards.py)
# and everything depending on it (uat/hotflip/hotflip_transfer/aggregate)
# was therefore never able to run.
#
# Does NOT touch Stage 0-3 and does not check they succeeded -- only run
# this for a chain you've confirmed already has baseline+monotonic done.
#
# Usage (same env vars as submit_pipeline.sh):
#   EXPERIMENT_SEED=1337 ./jobs/resume_from_eval.sh
#   T5_MODEL_NAME=t5-base EXPERIMENT_SEED=2024 CURC_PARTITION=artxpro6000 CONDA_ENV=mono_s2s_cu128 ./jobs/resume_from_eval.sh
#   T5_ABLATION_MODE=sign_frozen ./jobs/resume_from_eval.sh
#
# Set SKIP_ORDER_PRESERVATION=1 if Stage 8 has already been separately
# resubmitted for this chain and you don't want a second one.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SEED="${EXPERIMENT_SEED:-42}"
MODEL="${T5_MODEL_NAME:-t5-small}"
ABLATION="${T5_ABLATION_MODE:-nonneg}"
FULL="${USE_FULL_TEST_SETS:-1}"
PARTITION="${CURC_PARTITION:-aa100}"
CONDA_ENV="${CONDA_ENV:-mono_s2s}"

case "$PARTITION" in
  aa100)
    GRES="${CURC_GRES:-gpu:a100-40gb:1}"
    QOS="${CURC_QOS:-gpu-normal}"
    ;;
  ah200)
    GRES="${CURC_GRES:-gpu:h200:1}"
    QOS="${CURC_QOS:-gpu-normal}"
    CONDA_ENV="${CONDA_ENV:-mono_s2s_cu128}"
    ;;
  artxpro6000|artxpro60)
    PARTITION="artxpro6000"
    GRES="${CURC_GRES:-gpu:rtx_pro_6000:1}"
    QOS="${CURC_QOS:-gpu-normal}"
    CONDA_ENV="${CONDA_ENV:-mono_s2s_cu128}"
    ;;
  *)
    echo "Unknown CURC_PARTITION=$PARTITION (use aa100, ah200, or artxpro6000)"
    exit 1
    ;;
esac

EXPORT="ALL,EXPERIMENT_SEED=${SEED},T5_MODEL_NAME=${MODEL},T5_ABLATION_MODE=${ABLATION},USE_FULL_TEST_SETS=${FULL},CONDA_ENV=${CONDA_ENV},SCRATCH=${SCRATCH:-/scratch/alpine/$USER},PROJECT=${PROJECT:-/projects/$USER}"

submit_gpu() {
  local script="$1"
  local dep="${2:-}"
  local extra="${3:-}"
  local export_arg="$EXPORT"
  if [ -n "$extra" ]; then
    export_arg="$EXPORT,$extra"
  fi
  if [ -n "$dep" ]; then
    sbatch --partition="$PARTITION" --qos="$QOS" --gres="$GRES" --export="$export_arg" --dependency="$dep" "$script"
  else
    sbatch --partition="$PARTITION" --qos="$QOS" --gres="$GRES" --export="$export_arg" "$script"
  fi
}

echo "Resuming from Stage 4: seed=$SEED model=$MODEL ablation=$ABLATION partition=$PARTITION gres=$GRES"

if [ "$FULL" = "1" ]; then
  J4A=$(submit_gpu jobs/job_4_evaluate.sh "" "EVAL_DATASET_FILTER=cnn_dm" | awk '{print $4}')
  J4B=$(submit_gpu jobs/job_4_evaluate.sh "" "EVAL_DATASET_FILTER=xsum" | awk '{print $4}')
  J4C=$(submit_gpu jobs/job_4_evaluate.sh "" "EVAL_DATASET_FILTER=samsum" | awk '{print $4}')
  J4=$(sbatch --partition=acpu --qos=cpu-normal --export="$EXPORT" --dependency="afterok:$J4A:$J4B:$J4C" jobs/job_4_merge.sh | awk '{print $4}')
  echo "Sharded Stage 4 eval: cnn_dm=$J4A xsum=$J4B samsum=$J4C merge=$J4"
else
  J4=$(submit_gpu jobs/job_4_evaluate.sh | awk '{print $4}')
fi

J5=$(submit_gpu jobs/job_5_uat.sh "afterok:$J4" | awk '{print $4}')
J6=$(submit_gpu jobs/job_6_hotflip.sh "afterok:$J5" | awk '{print $4}')
J6B=$(submit_gpu jobs/job_6b_hotflip_transfer.sh "afterok:$J6" | awk '{print $4}')

if [ "${SKIP_ORDER_PRESERVATION:-0}" = "1" ]; then
  J8="${EXISTING_J8:?Set EXISTING_J8=<jobid> when SKIP_ORDER_PRESERVATION=1}"
else
  J8=$(submit_gpu jobs/job_8_order_preservation.sh | awk '{print $4}')
fi

J7=$(sbatch --partition=acpu --qos=cpu-normal --export="$EXPORT" --dependency="afterok:$J6B:$J8" jobs/job_7_aggregate.sh | awk '{print $4}')

echo "Resumed: 4=$J4 5=$J5 6=$J6 6b=$J6B 8=$J8 7=$J7"
echo "Monitor: squeue -u $USER"
