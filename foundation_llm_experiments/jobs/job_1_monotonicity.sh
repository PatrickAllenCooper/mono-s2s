#!/bin/bash
#SBATCH --job-name=foundation_monotonic
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/job_1_monotonicity_%j.out
#SBATCH --error=logs/job_1_monotonicity_%j.err

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
CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s || {
    echo "ERROR: Failed to activate conda environment 'mono_s2s'"
    exit 1
}

# Set environment variables for determinism
export PYTHONHASHSEED=${EXPERIMENT_SEED:-42}
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}

# Set up paths (with fallbacks)
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

# Redirect HuggingFace cache to scratch
export HF_HOME="$SCRATCH/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface_cache/transformers"

# Navigate to scripts directory
cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || {
    echo "ERROR: Cannot find scripts directory"
    exit 1
}

# Run monotonicity application script
echo ""
echo "Applying monotonicity constraints to Pythia-1.4B FFN layers..."
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
