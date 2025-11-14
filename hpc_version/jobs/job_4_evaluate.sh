#!/bin/bash
#SBATCH --job-name=mono_s2s_evaluate
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_4_evaluate_%j.out
#SBATCH --error=logs/job_4_evaluate_%j.err

# Stage 4: Comprehensive Evaluation
# Evaluates all three models on all test datasets with bootstrap CIs

echo "=========================================="
echo "SLURM Job: Stage 4 - Comprehensive Evaluation"
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

echo "Evaluating ALL THREE models on all test datasets..."
echo "Models: Standard T5, Baseline T5, Monotonic T5"
echo "Datasets: CNN/DailyMail, XSUM, SAMSum"
echo "Includes: Bootstrap 95% CIs, length statistics, brevity penalty"
echo ""

python stage_4_evaluate.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 4: COMPLETED SUCCESSFULLY"
    echo "All models evaluated on all test datasets"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 4: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs for evaluation errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

