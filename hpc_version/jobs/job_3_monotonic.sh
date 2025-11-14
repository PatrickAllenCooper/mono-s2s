#!/bin/bash
#SBATCH --job-name=mono_s2s_monotonic
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_3_monotonic_%j.out
#SBATCH --error=logs/job_3_monotonic_%j.err

# Stage 3: Train Monotonic Model (W≥0 FFN Constraints)
# Trained with IDENTICAL settings as baseline for fair comparison

echo "=========================================="
echo "SLURM Job: Stage 3 - Train Monotonic Model"
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
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s

# Set environment variables for determinism
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/pl/active/$USER}

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Navigate and run
cd $SLURM_SUBMIT_DIR/scripts

echo "Training MONOTONIC model (W≥0 FFN constraints)..."
echo "Using softplus reparametrization: W = softplus(V)"
echo "Training with IDENTICAL settings as baseline for fair comparison"
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
    echo "Check logs for training errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

