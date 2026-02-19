#!/bin/bash
#SBATCH --job-name=foundation_monotonic
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=77G
#SBATCH --time=23:50:00
#SBATCH --output=logs/job_3_monotonic_%j.out
#SBATCH --error=logs/job_3_monotonic_%j.err
#SBATCH --signal=SIGUSR1@600

# MAX TIME: 23:50:00 (24h with 10min buffer for final checkpoint save)
# Original estimate was 32h, so this will need ~2 auto-resubmissions
# SLURM will send SIGUSR1 signal 10 minutes before timeout
# Monitor script will automatically resubmit and resume from checkpoint
#
# Stage 3: Monotonic Recovery Training (W≥0 FFN Constraints)
# Trained with IDENTICAL settings as baseline for fair comparison

echo "=========================================="
echo "SLURM Job: Stage 3 - Monotonic Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Load modules (try to load CUDA if available)
module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

# Activate conda environment (installed to /projects)
CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s || {
    echo "ERROR: Failed to activate conda environment 'mono_s2s'"
    exit 1
}

# Reduce CUDA memory fragmentation (helps when eval follows a full training epoch)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set environment variables for determinism
export PYTHONHASHSEED=${EXPERIMENT_SEED:-42}
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}

# Set up paths (with fallbacks)
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

# Redirect HuggingFace cache to scratch (not home directory)
export HF_HOME="$SCRATCH/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface_cache/transformers"

# Print GPU info for debugging
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Navigate to scripts directory
cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || {
    echo "ERROR: Cannot find scripts directory"
    exit 1
}

echo "Training MONOTONIC Pythia-1.4B (W≥0 constraints on FFN)..."
echo "This model has softplus-parametrized weights in feed-forward layers"
echo "Training with extended warmup (15% vs 10%) for stability under constraints"
echo ""

python stage_3_train_monotonic.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 3: COMPLETED SUCCESSFULLY"
    echo "Monotonic model trained and saved"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 3: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs above for training errors"
    echo "If partial completion: Checkpoints saved, can resume by resubmitting"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
