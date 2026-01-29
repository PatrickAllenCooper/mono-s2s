#!/bin/bash
#SBATCH --job-name=foundation_hotflip
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=78G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_6_hotflip_%j.out
#SBATCH --error=logs/job_6_hotflip_%j.err

# Stage 6: HotFlip Gradient-Based Attacks

echo "=========================================="
echo "SLURM Job: Stage 6 - HotFlip Attacks"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s || exit 1

export PYTHONHASHSEED=${EXPERIMENT_SEED:-42}
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}
export HF_HOME="$SCRATCH/huggingface_cache"

cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || exit 1

echo ""
echo "Running HotFlip attacks (up to 10 flips per example)..."
python stage_6_hotflip_attacks.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 6: COMPLETED SUCCESSFULLY"
    echo "HotFlip attack results saved"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 6: FAILED (exit code: $EXIT_CODE)"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
