#!/bin/bash
# Submit a foundation-model stage chain. Override partition for H200 / RTX Pro.
#
#   EXPERIMENT_SEED=42 MONOTONIC_VARIANT=mlp_both ./jobs/submit_pipeline.sh
#   PYTHIA_MODEL_NAME=EleutherAI/pythia-2.8b CURC_PARTITION=ah200 ./jobs/submit_pipeline.sh
#   FOUNDATION_MODEL_NAME=meta-llama/Llama-3.2-1B MONOTONIC_VARIANT=gated_updown ./jobs/submit_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

SEED="${EXPERIMENT_SEED:-42}"
MODEL="${FOUNDATION_MODEL_NAME:-${PYTHIA_MODEL_NAME:-EleutherAI/pythia-1.4b}}"
VARIANT="${MONOTONIC_VARIANT:-mlp_both}"
PARTITION="${CURC_PARTITION:-aa100}"
CONDA_ENV="${CONDA_ENV:-mono_s2s}"

if [[ "$MODEL" == meta-llama/* ]] && [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: $MODEL is a gated HuggingFace repo."
  echo "Accept the license on huggingface.co and export HF_TOKEN=hf_..."
  exit 1
fi

case "$PARTITION" in
  aa100)
    GRES="${CURC_GRES:-gpu:a100_80gb:1}"
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
    echo "Unknown CURC_PARTITION=$PARTITION"
    exit 1
    ;;
esac

EXPORT="ALL,EXPERIMENT_SEED=${SEED},FOUNDATION_MODEL_NAME=${MODEL},PYTHIA_MODEL_NAME=${MODEL},MONOTONIC_VARIANT=${VARIANT},CONDA_ENV=${CONDA_ENV},HF_TOKEN=${HF_TOKEN:-},SCRATCH=${SCRATCH:-/scratch/alpine/$USER},PROJECT=${PROJECT:-/projects/$USER}"

submit_gpu() {
  local script="$1"
  local dep="${2:-}"
  if [ -n "$dep" ]; then
    sbatch --partition="$PARTITION" --qos="$QOS" --gres="$GRES" --export="$EXPORT" --dependency="$dep" "$script"
  else
    sbatch --partition="$PARTITION" --qos="$QOS" --gres="$GRES" --export="$EXPORT" "$script"
  fi
}

echo "Submitting foundation chain seed=$SEED model=$MODEL variant=$VARIANT partition=$PARTITION"

J0=$(submit_gpu jobs/job_0_setup.sh | awk '{print $4}')
J1=$(submit_gpu jobs/job_1_monotonicity.sh "afterok:$J0" | awk '{print $4}')
J2=$(submit_gpu jobs/job_2_baseline.sh "afterok:$J1" | awk '{print $4}')
J3=$(submit_gpu jobs/job_3_monotonic.sh "afterok:$J2" | awk '{print $4}')
J4=$(submit_gpu jobs/job_4_evaluate.sh "afterok:$J3" | awk '{print $4}')
J5=$(submit_gpu jobs/job_5_uat.sh "afterok:$J4" | awk '{print $4}')
J5B=$(submit_gpu jobs/job_5b_uat_transfer.sh "afterok:$J5" | awk '{print $4}')
J6=$(submit_gpu jobs/job_6_hotflip.sh "afterok:$J4" | awk '{print $4}')
J6B=$(submit_gpu jobs/job_6b_hotflip_transfer.sh "afterok:$J6" | awk '{print $4}')
J11=$(submit_gpu jobs/job_11_order_preservation.sh "afterok:$J3" | awk '{print $4}')
J7=$(submit_gpu jobs/job_7_aggregate.sh "afterok:$J5B:$J6B:$J11" | awk '{print $4}')

echo "Submitted: 0=$J0 1=$J1 2=$J2 3=$J3 4=$J4 5=$J5 5b=$J5B 6=$J6 6b=$J6B 11=$J11 7=$J7"
