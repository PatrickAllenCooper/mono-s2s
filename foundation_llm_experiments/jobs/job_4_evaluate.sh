#!/bin/bash
#SBATCH --job-name=foundation_eval
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/job_4_evaluate_%j.out
#SBATCH --error=logs/job_4_evaluate_%j.err

# Stage 4: Evaluation on LLM Benchmarks

echo "=========================================="
echo "SLURM Job: Stage 4 - Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo "=========================================="

module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s || {
    echo "ERROR: Failed to activate conda environment 'mono_s2s'"
    exit 1
}

export PYTHONHASHSEED=${EXPERIMENT_SEED:-42}
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}
export HF_HOME="$SCRATCH/huggingface_cache"

cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || exit 1

echo ""
echo "Evaluating models on Pile test set..."
python stage_4_evaluate.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 4: COMPLETED SUCCESSFULLY"
    echo "Evaluation results saved"
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
