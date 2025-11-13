#!/bin/bash
#SBATCH --job-name=mono_s2s_data
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/job_1_data_%j.out
#SBATCH --error=logs/job_1_data_%j.err

# Stage 1: Data Preparation
# Downloads and prepares all datasets

echo "=========================================="
echo "SLURM Job: Stage 1 - Data Preparation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module purge
module load anaconda  # Use anaconda on Alpine
module load cuda

# Activate conda environment
conda activate mono_s2s 2>/dev/null || true

# Set environment variables
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export SCRATCH=${SCRATCH:-/scratch/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

# Navigate and run
cd $(dirname $0)/../scripts
python stage_1_prepare_data.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 1: COMPLETED SUCCESSFULLY"
    echo "Datasets prepared and cached"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 1: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs for details"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

