#!/bin/bash
################################################################################
# Master Submission Script for Foundation LLM Monotonicity Experiments
#
# Submits all experimental stages with proper dependencies.
# Adapted from main project's run_all.sh for decoder-only models.
################################################################################

set -euo pipefail

echo "======================================================================"
echo "FOUNDATION LLM MONOTONICITY EXPERIMENTS"
echo "Master Submission Script"
echo "======================================================================"
echo ""
echo "This will submit SLURM jobs for all experimental stages:"
echo "  Stage 0: Setup (download Pythia-1.4B)"
echo "  Stage 1: Apply monotonicity constraints"
echo "  Stage 2: Baseline recovery training"
echo "  Stage 3: Monotonic recovery training"
echo "  Stage 4: Evaluation (perplexity, benchmarks)"
echo "  Stage 5: UAT attacks"
echo "  Stage 6: HotFlip attacks"
echo "  Stage 7: Aggregate results"
echo ""
echo "Expected total time: ~60-70 hours per seed"
echo ""

# Check for required directories
if [ ! -d "jobs" ]; then
    echo "ERROR: jobs/ directory not found"
    echo "Please run this script from foundation_llm_experiments/ directory"
    exit 1
fi

if [ ! -d "scripts" ]; then
    echo "ERROR: scripts/ directory not found"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Get seed from environment or use default
SEED=${EXPERIMENT_SEED:-42}
echo "Using random seed: $SEED"
echo ""

# Confirm submission
read -p "Submit all jobs? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs..."
echo ""

# Stage 0: Setup
# Extract only numeric job ID (handle SLURM warning messages)
JOB0=$(sbatch --parsable jobs/job_0_setup.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
echo "Stage 0 (Setup): Job ID $JOB0"

# Stage 1: Apply monotonicity (depends on setup)
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 jobs/job_1_monotonicity.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
echo "Stage 1 (Apply Monotonicity): Job ID $JOB1 (depends on $JOB0)"

# Stage 2: Baseline training (depends on setup, parallel with stage 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB0 jobs/job_2_baseline.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
echo "Stage 2 (Baseline Training): Job ID $JOB2 (depends on $JOB0)"

# Stage 3: Monotonic training (depends on stage 1)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/job_3_monotonic.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
echo "Stage 3 (Monotonic Training): Job ID $JOB3 (depends on $JOB1)"

# Stage 4: Evaluation (depends on both training stages)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2,afterok:$JOB3 jobs/job_4_evaluate.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
echo "Stage 4 (Evaluation): Job ID $JOB4 (depends on $JOB2, $JOB3)"

# Stage 5: UAT attacks (depends on both training stages)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB2,afterok:$JOB3 jobs/job_5_uat.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
echo "Stage 5 (UAT Attacks): Job ID $JOB5 (depends on $JOB2, $JOB3)"

# Stage 6: HotFlip attacks (depends on both training stages)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB2,afterok:$JOB3 jobs/job_6_hotflip.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
echo "Stage 6 (HotFlip Attacks): Job ID $JOB6 (depends on $JOB2, $JOB3)"

# Stage 7: Aggregate (depends on evaluation and attacks)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB4,afterok:$JOB5,afterok:$JOB6 jobs/job_7_aggregate.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
echo "Stage 7 (Aggregate): Job ID $JOB7 (depends on $JOB4, $JOB5, $JOB6)"

echo ""
echo "======================================================================"
echo "All jobs submitted successfully!"
echo "======================================================================"
echo ""
echo "Job Summary:"
echo "  Stage 0: $JOB0"
echo "  Stage 1: $JOB1"
echo "  Stage 2: $JOB2"
echo "  Stage 3: $JOB3"
echo "  Stage 4: $JOB4"
echo "  Stage 5: $JOB5"
echo "  Stage 6: $JOB6"
echo "  Stage 7: $JOB7"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER | grep foundation"
echo "  tail -f logs/job_0_setup_${JOB0}.out"
echo ""
echo "Check results:"
echo "  cat \$SCRATCH/foundation_llm_work/experiment_summary.txt"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB0 $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6 $JOB7"
echo ""
echo "======================================================================"
