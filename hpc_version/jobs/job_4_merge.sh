#!/bin/bash
#SBATCH --job-name=mono_s2s_eval_merge
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/job_4_merge_%j.out
#SBATCH --error=logs/job_4_merge_%j.err

# Stage 4 (merge): combine per-dataset evaluation shards (see
# EVAL_DATASET_FILTER in job_4_evaluate.sh / stage_4_evaluate.py) into the
# single evaluation_results.json that Stage 5/6/7 expect.

echo "=========================================="
echo "SLURM Job: Stage 4 (merge) - Combine evaluation shards"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "=========================================="

module purge 2>/dev/null || true

CONDA_ENV="${CONDA_ENV:-mono_s2s}"
source "${SLURM_SUBMIT_DIR}/jobs/activate_conda.sh"

export PYTHONHASHSEED=42
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export T5_ABLATION_MODE=${T5_ABLATION_MODE:-nonneg}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}
export EVAL_SHARDS="${EVAL_SHARDS:-cnn_dm,xsum,samsum}"

cd $SLURM_SUBMIT_DIR/scripts
python stage_4_merge_shards.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 4 (merge): COMPLETED SUCCESSFULLY"
    echo "Ended: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Stage 4 (merge): FAILED (exit code: $EXIT_CODE)"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
