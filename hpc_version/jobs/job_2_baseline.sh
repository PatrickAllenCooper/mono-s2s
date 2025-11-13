#!/bin/bash
#SBATCH --job-name=mono_s2s_baseline
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_2_baseline_%j.out
#SBATCH --error=logs/job_2_baseline_%j.err

# Stage 2: Train Baseline Model (Unconstrained T5)
# This is the FAIR BASELINE - trained with same data/hyperparameters as monotonic

echo "=========================================="
echo "SLURM Job: Stage 2 - Train Baseline Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Load modules (try to load CUDA if available)
module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate mono_s2s 2>/dev/null || true

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

echo "Training BASELINE model (unconstrained T5)..."
echo "This model uses standard T5 with NO monotonic constraints"
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
    echo "Check logs for training errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

