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
  if [ -n "$dep" ]; then
    sbatch --partition="$PARTITION" --qos="$QOS" --gres="$GRES" --export="$EXPORT" --dependency="$dep" "$script"
  else
    sbatch --partition="$PARTITION" --qos="$QOS" --gres="$GRES" --export="$EXPORT" "$script"
  fi
}

echo "Submitting T5 chain seed=$SEED model=$MODEL ablation=$ABLATION partition=$PARTITION gres=$GRES"

J0=$(submit_cpu jobs/job_0_setup.sh | awk '{print $4}')
J1=$(sbatch --partition=acpu --qos=cpu-normal --export="$EXPORT" --dependency="afterok:$J0" jobs/job_1_data.sh | awk '{print $4}')
J2=$(submit_gpu jobs/job_2_baseline.sh "afterok:$J1" | awk '{print $4}')
J3=$(submit_gpu jobs/job_3_monotonic.sh "afterok:$J2" | awk '{print $4}')
J4=$(submit_gpu jobs/job_4_evaluate.sh "afterok:$J3" | awk '{print $4}')
J5=$(submit_gpu jobs/job_5_uat.sh "afterok:$J4" | awk '{print $4}')
J6=$(submit_gpu jobs/job_6_hotflip.sh "afterok:$J5" | awk '{print $4}')
J6B=$(submit_gpu jobs/job_6b_hotflip_transfer.sh "afterok:$J6" | awk '{print $4}')
J8=$(submit_gpu jobs/job_8_order_preservation.sh "afterok:$J3" | awk '{print $4}')
J7=$(sbatch --partition=acpu --qos=cpu-normal --export="$EXPORT" --dependency="afterok:$J6B:$J8" jobs/job_7_aggregate.sh | awk '{print $4}')

echo "Submitted: 0=$J0 1=$J1 2=$J2 3=$J3 4=$J4 5=$J5 6=$J6 6b=$J6B 8=$J8 7=$J7"
echo "Monitor: squeue -u $USER"
