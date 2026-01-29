#!/bin/bash
#SBATCH --job-name=foundation_baseline
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=78G
#SBATCH --time=23:50:00
#SBATCH --output=logs/job_2_baseline_%j.out
#SBATCH --error=logs/job_2_baseline_%j.err
#SBATCH --signal=SIGUSR1@600

# MAX TIME: 23:50:00 (24h with 10min buffer for final checkpoint save)
# SLURM will send SIGUSR1 signal 10 minutes before timeout
# This gives us time to save checkpoint and exit gracefully

# Stage 2: Baseline Recovery Training (Pythia-1.4B on Pile)
# This is the FAIR BASELINE - trained with same data/hyperparameters as monotonic

echo "=========================================="
echo "SLURM Job: Stage 2 - Baseline Training"
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
    echo "Please ensure conda environment exists:"
    echo "  conda create -n mono_s2s python=3.10"
    echo "  conda activate mono_s2s"
    echo "  pip install -r requirements.txt"
    exit 1
}

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

echo "Training BASELINE Pythia-1.4B (unconstrained)..."
echo "This model uses standard Pythia with NO monotonic constraints"
echo "Training with IDENTICAL settings as monotonic model for fair comparison"
echo ""

python stage_2_train_baseline.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 2: COMPLETED SUCCESSFULLY"
    echo "Baseline model trained and saved"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 2: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs above for training errors"
    echo "If partial completion: Checkpoints saved, can resume by resubmitting"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
