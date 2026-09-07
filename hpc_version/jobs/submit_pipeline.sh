#!/bin/bash
# Submit a T5 stage chain with the current CURC partition names.
#
# Usage:
#   ./jobs/submit_pipeline.sh
#   EXPERIMENT_SEED=1337 USE_FULL_TEST_SETS=1 ./jobs/submit_pipeline.sh
#   T5_MODEL_NAME=t5-base CURC_PARTITION=artxpro6000 ./jobs/submit_pipeline.sh
#   T5_ABLATION_MODE=sign_frozen ./jobs/submit_pipeline.sh
#
# Verify live names first: sinfo -o "%P %G %l"

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

submit_cpu() {
  local script="$1"
  sbatch --partition=acpu --qos=cpu-normal --export="$EXPORT" "$script"
}

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

echo "Submitting T5 chain seed=$SEED model=$MODEL ablation=$ABLATION partition=$PARTITION gres=$GRES"

J0=$(submit_cpu jobs/job_0_setup.sh | awk '{print $4}')
J1=$(sbatch --partition=acpu --qos=cpu-normal --export="$EXPORT" --dependency="afterok:$J0" jobs/job_1_data.sh | awk '{print $4}')
J2=$(submit_gpu jobs/job_2_baseline.sh "afterok:$J1" | awk '{print $4}')
J3=$(submit_gpu jobs/job_3_monotonic.sh "afterok:$J2" | awk '{print $4}')

if [ "$FULL" = "1" ]; then
  # Full test sets push a single 3-datasets x 3-models eval job past CURC's
  # 24h gpu-normal QoS ceiling (~40h+ observed). Shard by dataset into
  # parallel jobs, then merge back into the canonical evaluation_results.json
  # that Stage 5/6/7 expect (see stage_4_merge_shards.py).
  J4A=$(submit_gpu jobs/job_4_evaluate.sh "afterok:$J3" "EVAL_DATASET_FILTER=cnn_dm" | awk '{print $4}')
  J4B=$(submit_gpu jobs/job_4_evaluate.sh "afterok:$J3" "EVAL_DATASET_FILTER=xsum" | awk '{print $4}')
  J4C=$(submit_gpu jobs/job_4_evaluate.sh "afterok:$J3" "EVAL_DATASET_FILTER=samsum" | awk '{print $4}')
  J4=$(sbatch --partition=acpu --qos=cpu-normal --export="$EXPORT" --dependency="afterok:$J4A:$J4B:$J4C" jobs/job_4_merge.sh | awk '{print $4}')
  echo "Sharded Stage 4 eval: cnn_dm=$J4A xsum=$J4B samsum=$J4C merge=$J4"
else
  J4=$(submit_gpu jobs/job_4_evaluate.sh "afterok:$J3" | awk '{print $4}')
fi

J5=$(submit_gpu jobs/job_5_uat.sh "afterok:$J4" | awk '{print $4}')
J6=$(submit_gpu jobs/job_6_hotflip.sh "afterok:$J5" | awk '{print $4}')
J6B=$(submit_gpu jobs/job_6b_hotflip_transfer.sh "afterok:$J6" | awk '{print $4}')
J8=$(submit_gpu jobs/job_8_order_preservation.sh "afterok:$J3" | awk '{print $4}')
J7=$(sbatch --partition=acpu --qos=cpu-normal --export="$EXPORT" --dependency="afterok:$J6B:$J8" jobs/job_7_aggregate.sh | awk '{print $4}')

echo "Submitted: 0=$J0 1=$J1 2=$J2 3=$J3 4=$J4 5=$J5 6=$J6 6b=$J6B 8=$J8 7=$J7"
echo "Monitor: squeue -u $USER"
