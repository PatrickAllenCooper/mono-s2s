#!/bin/bash
#SBATCH --job-name=mono_s2s_setup
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/job_0_setup_%j.out
#SBATCH --error=logs/job_0_setup_%j.err

# Stage 0: Setup and Environment Verification
# This job sets up the environment and downloads models

echo "=========================================="
echo "SLURM Job: Stage 0 - Setup"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Load modules (Alpine compute nodes may have additional modules)
# Note: Login nodes may not show all modules - they become available on compute nodes
module purge 2>/dev/null || true

# Try to load CUDA if available (ignore errors)
module load cuda 2>/dev/null || true

# Activate conda environment (installed to /projects, not $HOME)
# This gives us Python 3.10+ (system Python 3.6.8 is too old)
CONDA_BASE="/projects/$USER/miniconda3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate mono_s2s
else
    echo "Warning: Conda not found at $CONDA_BASE"
fi

# Set environment variables for determinism (BEFORE running Python)
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

# Set experiment seed (can be overridden)
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}

# Set HPC paths (customize)
export SCRATCH=${SCRATCH:-/scratch/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(python --version)"
echo "  CUDA: $CUDA_VERSION"
echo "  Seed: $EXPERIMENT_SEED"
echo "  Scratch: $SCRATCH"
echo ""

# Navigate to script directory
cd $(dirname $0)/../scripts

# Run setup stage
echo "Running stage 0: setup..."
python stage_0_setup.py

# Check exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 0: COMPLETED SUCCESSFULLY"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 0: FAILED (exit code: $EXIT_CODE)"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

