#!/bin/bash
#SBATCH --job-name=mono_s2s_aggregate
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/job_7_aggregate_%j.out
#SBATCH --error=logs/job_7_aggregate_%j.err

# Stage 7: Aggregate Results and Final Analysis
# Combines all results and creates final comparison tables

echo "=========================================="
echo "SLURM Job: Stage 7 - Aggregate Results"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

# Load modules (no GPU needed for aggregation)
module purge 2>/dev/null || true

# Activate conda environment (installed to /projects)
CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s

# Set environment variables
export PYTHONHASHSEED=42
export EXPERIMENT_SEED=${EXPERIMENT_SEED:-42}
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

# Navigate and run
cd $SLURM_SUBMIT_DIR/scripts

echo "Aggregating all results..."
echo "Creating comparison tables"
echo "Generating human-readable summary"
echo "Copying to permanent storage"
echo ""

python stage_7_aggregate.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 7: COMPLETED SUCCESSFULLY"
    echo "All results aggregated and saved"
    echo "Ended: $(date)"
    echo "=========================================="
    echo ""
    echo "✅ FULL EXPERIMENT PIPELINE COMPLETE!"
    echo ""
    echo "Results saved to:"
    echo "  Scratch:  $SCRATCH/mono_s2s_results/"
    echo "  Project:  $PROJECT/mono_s2s_final_results/"
    echo ""
    echo "Key files:"
    echo "  - final_results.json (comprehensive results)"
    echo "  - experiment_summary.txt (human-readable)"
    echo "  - evaluation_results.json (ROUGE with CIs)"
    echo "  - uat_results.json (UAT + transfer matrix)"
    echo "  - hotflip_results.json (gradient attacks)"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Stage 7: FAILED (exit code: $EXIT_CODE)"
    echo "Check logs for aggregation errors"
    echo "Ended: $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

