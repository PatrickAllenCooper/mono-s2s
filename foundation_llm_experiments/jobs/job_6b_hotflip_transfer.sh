#!/bin/bash
#SBATCH --job-name=foundation_hotflip_transfer
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=77G
#SBATCH --time=10:00:00
#SBATCH --output=/projects/%u/mono-s2s/foundation_llm_experiments/logs/job_6b_hotflip_transfer_%j.out
#SBATCH --error=/projects/%u/mono-s2s/foundation_llm_experiments/logs/job_6b_hotflip_transfer_%j.err

# Stage 6b: HotFlip substitution transfer + gradient-free controls
# (re-attacks both models, cross-evaluates, adds a random-substitution
# control and an optional query-based gradient-free attack). Set
# OVERRIDE_RUN_QUERY_ATTACK=0 to skip the most expensive control if
# time-constrained.

echo "=========================================="
echo "SLURM Job: Stage 6b - HotFlip Transfer + Controls"
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
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}
export HF_HOME="$SCRATCH/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface_cache/transformers"
export MONOTONIC_VARIANT="${MONOTONIC_VARIANT:-mlp_both}"
export OVERRIDE_RUN_QUERY_ATTACK="${OVERRIDE_RUN_QUERY_ATTACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Early-exit if this stage already completed (continuation job safety check).
_WORK_DIR="${LAMBDA_SEED_WORK:-/scratch/alpine/${USER}/foundation_llm_work_seed${EXPERIMENT_SEED}}"
if [ -f "${_WORK_DIR}/stage_6b_hotflip_transfer_complete.flag" ]; then
    echo "Stage 6b already complete for seed ${EXPERIMENT_SEED}. Nothing to do."
    exit 0
fi

cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || exit 1

echo ""
echo "Running HotFlip transfer + controls (variant=$MONOTONIC_VARIANT, query_attack=$OVERRIDE_RUN_QUERY_ATTACK)..."
python stage_6b_hotflip_transfer.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 6b: COMPLETED SUCCESSFULLY"
    echo "HotFlip transfer + control results saved"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 6b: FAILED (exit code: $EXIT_CODE)"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
