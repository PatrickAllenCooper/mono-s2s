#!/bin/bash
#SBATCH --job-name=mono_s2s_hotflip
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/job_6_hotflip_%j.out
#SBATCH --error=logs/job_6_hotflip_%j.err

# Stage 6: HotFlip Gradient-Based Attacks
# Performs gradient-based token flipping attacks on all models

echo "=========================================="
echo "SLURM Job: Stage 6 - HotFlip Attacks"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module purge
module load python/3.10.0
module load cuda/11.8

# Set environment variables for determinism
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export SCRATCH=${SCRATCH:-/scratch/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Navigate and run
cd $(dirname $0)/../scripts

echo "Running HotFlip gradient-based attacks..."
echo "Flipping tokens to maximize loss using embedding gradients"
echo "Computing vulnerability statistics and success rates"
echo ""

python stage_6_hotflip_attacks.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 6: COMPLETED SUCCESSFULLY"
    echo "HotFlip attacks complete with statistics"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 6: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs for attack errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

