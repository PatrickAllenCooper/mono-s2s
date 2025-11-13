#!/bin/bash
#SBATCH --job-name=mono_s2s_uat
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/job_5_uat_%j.out
#SBATCH --error=logs/job_5_uat_%j.err

# Stage 5: UAT Attacks with Transfer Matrix
# Learns universal triggers and evaluates cross-model transferability

echo "=========================================="
echo "SLURM Job: Stage 5 - UAT Attacks"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module purge
module load anaconda  # Use anaconda on Alpine
module load cuda

# Activate conda environment
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

echo "Running UAT (Universal Adversarial Trigger) attacks..."
echo "Learning model-specific triggers with multiple restarts"
echo "Evaluating on held-out test set"
echo "Computing transfer attack matrix"
echo ""

python stage_5_uat_attacks.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 5: COMPLETED SUCCESSFULLY"
    echo "UAT attacks complete with transfer matrix"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 5: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs for attack errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

