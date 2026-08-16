#!/bin/bash
#SBATCH --job-name=foundation_order_preservation
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=77G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/%u/mono-s2s/foundation_llm_experiments/logs/job_11_order_preservation_%j.out
#SBATCH --error=/projects/%u/mono-s2s/foundation_llm_experiments/logs/job_11_order_preservation_%j.err

echo "=========================================="
echo "SLURM Job: Stage 11 - Order Preservation (Pythia)"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

CONDA_BASE="/projects/$USER/miniconda3"
CONDA_ENV="${CONDA_ENV:-mono_s2s}"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV" || exit 1

export PYTHONHASHSEED=${EXPERIMENT_SEED:-42}
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export FOUNDATION_MODEL_NAME="${FOUNDATION_MODEL_NAME:-${PYTHIA_MODEL_NAME:-EleutherAI/pythia-1.4b}}"
export PYTHIA_MODEL_NAME="${PYTHIA_MODEL_NAME:-$FOUNDATION_MODEL_NAME}"
export MONOTONIC_VARIANT="${MONOTONIC_VARIANT:-mlp_both}"
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}
export HF_HOME="$SCRATCH/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface_cache/transformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false

cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || exit 1

echo "Seed=$EXPERIMENT_SEED model=$FOUNDATION_MODEL_NAME variant=$MONOTONIC_VARIANT"
python stage_11_order_preservation.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Stage 11 completed successfully at $(date)"
else
    echo "Stage 11 failed (exit $EXIT_CODE) at $(date)"
fi
exit $EXIT_CODE
