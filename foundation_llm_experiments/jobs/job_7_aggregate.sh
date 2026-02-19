#!/bin/bash
#SBATCH --job-name=foundation_aggregate
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/job_7_aggregate_%j.out
#SBATCH --error=logs/job_7_aggregate_%j.err

# Stage 7: Aggregate Results

echo "=========================================="
echo "SLURM Job: Stage 7 - Aggregate Results"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

module purge 2>/dev/null || true

CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s || exit 1

export PYTHONHASHSEED=${EXPERIMENT_SEED:-42}
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

cd $SLURM_SUBMIT_DIR || cd "$(dirname "$0")/.."
cd scripts || exit 1

echo ""
echo "Aggregating all results..."
python stage_7_aggregate.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 7: COMPLETED SUCCESSFULLY"
    echo "All results aggregated and final summary created"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 7: FAILED (exit code: $EXIT_CODE)"
    echo "Check that all previous stages completed"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
