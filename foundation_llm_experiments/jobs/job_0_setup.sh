#!/bin/bash
#SBATCH --job-name=foundation_setup
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=77G
#SBATCH --time=01:00:00
#SBATCH --output=logs/job_0_setup_%j.out
#SBATCH --error=logs/job_0_setup_%j.err

# Stage 0: Setup - Download Pythia-1.4B and prepare environment

echo "=========================================="
echo "SLURM Job: Stage 0 - Setup"
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
    echo "Please create it first using:"
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

# Print GPU info
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

# Run setup script
echo "Running setup (download Pythia-1.4B and prepare environment)..."
python stage_0_setup.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 0: COMPLETED SUCCESSFULLY"
    echo "Pythia-1.4B downloaded and environment verified"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 0: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs above for setup errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
