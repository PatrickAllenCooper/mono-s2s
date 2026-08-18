#!/bin/bash
#SBATCH --job-name=foundation_monotonic
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=77G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/%u/mono-s2s/foundation_llm_experiments/logs/job_1_monotonicity_%j.out
#SBATCH --error=/projects/%u/mono-s2s/foundation_llm_experiments/logs/job_1_monotonicity_%j.err

# Stage 1: Apply Monotonicity Constraints

echo "=========================================="
echo "SLURM Job: Stage 1 - Apply Monotonicity"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

# Activate conda environment (installed to /projects)
CONDA_ENV="${CONDA_ENV:-mono_s2s}"
source "${SLURM_SUBMIT_DIR}/jobs/activate_conda.sh"

# Set environment variables for determinism
export PYTHONHASHSEED=${EXPERIMENT_SEED:-42}
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export FOUNDATION_MODEL_NAME="${FOUNDATION_MODEL_NAME:-${PYTHIA_MODEL_NAME:-EleutherAI/pythia-1.4b}}"
export PYTHIA_MODEL_NAME="${PYTHIA_MODEL_NAME:-$FOUNDATION_MODEL_NAME}"

# Set up paths (with fallbacks)
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

# Redirect HuggingFace cache to scratch
export HF_HOME="$SCRATCH/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface_cache/transformers"
export MONOTONIC_VARIANT="${MONOTONIC_VARIANT:-mlp_both}"

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Navigate to scripts directory
cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || {
    echo "ERROR: Cannot find scripts directory"
    exit 1
}

# Run monotonicity application script
echo ""
echo "Applying monotonicity constraints ($MONOTONIC_VARIANT) to Pythia-1.4B..."
python stage_1_apply_monotonicity.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 1: COMPLETED SUCCESSFULLY"
    echo "Monotonicity constraints applied and verified"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 1: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs above for errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
