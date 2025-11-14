#!/bin/bash
#
# Master Orchestration Script for Mono-S2S HPC Experiments
#
# This script submits all stages as SLURM jobs with proper dependencies.
# Each stage waits for the previous to complete successfully before starting.
#
# Usage:
#   ./run_all.sh [seed]
#
# Example:
#   ./run_all.sh 42      # Run with seed 42
#   ./run_all.sh         # Run with default seed (42)
#

set -e  # Exit on error

# Configuration
SEED=${1:-42}
export EXPERIMENT_SEED=$SEED

# Set SCRATCH and PROJECT for Alpine (needed for flag file checking)
export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export PROJECT=${PROJECT:-/projects/$USER}

echo "=========================================="
echo "Mono-S2S HPC Experiment Orchestrator"
echo "=========================================="
echo "Seed: $SEED"
echo "Started: $(date)"
echo ""

# Create logs directory
mkdir -p logs

# Function to check if job completed successfully
check_job_status() {
    local job_id=$1
    local job_name=$2
    
    echo "Waiting for $job_name (Job ID: $job_id) to complete..."
    
    # Wait for job to finish
    while squeue -j $job_id 2>/dev/null | grep -q $job_id; do
        sleep 30
    done
    
    # Check if completion flag exists
    local flag_file="${SCRATCH}/mono_s2s_work/${job_name}_complete.flag"
    if [ -f "$flag_file" ]; then
        echo "✓ $job_name completed successfully"
        return 0
    else
        echo "❌ $job_name FAILED - check logs/job_*_${job_id}.err"
        return 1
    fi
}

# Submit jobs with dependencies
echo "Submitting SLURM jobs with dependencies..."
echo ""

# Stage 0: Setup
echo "Stage 0: Setup and environment verification..."
JOB0=$(sbatch --parsable jobs/job_0_setup.sh)
echo "  Job ID: $JOB0"

check_job_status $JOB0 "stage_0_setup" || {
    echo "❌ Setup failed. Aborting."
    exit 1
}

# Stage 1: Data Preparation
echo ""
echo "Stage 1: Data preparation..."
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 jobs/job_1_data.sh)
echo "  Job ID: $JOB1"
echo "  Depends on: $JOB0"

check_job_status $JOB1 "stage_1_data_prep" || {
    echo "❌ Data preparation failed. Aborting."
    exit 1
}

# Stage 2: Train Baseline (depends on data)
echo ""
echo "Stage 2: Train baseline model (unconstrained)..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/job_2_baseline.sh)
echo "  Job ID: $JOB2"
echo "  Depends on: $JOB1"
echo "  ⏱  Expected time: 4-12 hours"

# Stage 3: Train Monotonic (depends on data, can run parallel with baseline)
echo ""
echo "Stage 3: Train monotonic model (W≥0 constraints)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/job_3_monotonic.sh)
echo "  Job ID: $JOB3"
echo "  Depends on: $JOB1"
echo "  ⏱  Expected time: 4-12 hours"
echo "  ℹ️  Runs in PARALLEL with baseline training"

# Wait for both training jobs
echo ""
echo "⏳ Waiting for training jobs to complete..."
echo "  This may take 4-12 hours depending on configuration"
echo "  Monitor progress with: squeue -u $USER"
echo ""

check_job_status $JOB2 "stage_2_train_baseline" || {
    echo "❌ Baseline training failed. Aborting."
    exit 1
}

check_job_status $JOB3 "stage_3_train_monotonic" || {
    echo "❌ Monotonic training failed. Aborting."
    exit 1
}

# Stage 4: Comprehensive Evaluation (depends on both models)
echo ""
echo "Stage 4: Comprehensive evaluation (all 3 models, all 3 test sets)..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 jobs/job_4_evaluate.sh)
echo "  Job ID: $JOB4"
echo "  Depends on: $JOB2, $JOB3"
echo "  ⏱  Expected time: 2-4 hours"

check_job_status $JOB4 "stage_4_evaluate" || {
    echo "❌ Evaluation failed. Aborting."
    exit 1
}

# Stage 5: UAT Attacks (depends on evaluation)
echo ""
echo "Stage 5: UAT attacks with transfer matrix..."
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 jobs/job_5_uat.sh)
echo "  Job ID: $JOB5"
echo "  Depends on: $JOB4"
echo "  ⏱  Expected time: 2-3 hours"

# Stage 6: HotFlip Attacks (can run parallel with UAT)
echo ""
echo "Stage 6: HotFlip attacks..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB4 jobs/job_6_hotflip.sh)
echo "  Job ID: $JOB6"
echo "  Depends on: $JOB4"
echo "  ⏱  Expected time: 1-2 hours"
echo "  ℹ️  Runs in PARALLEL with UAT attacks"

# Wait for attack jobs
check_job_status $JOB5 "stage_5_uat" || {
    echo "❌ UAT attacks failed. Aborting."
    exit 1
}

check_job_status $JOB6 "stage_6_hotflip" || {
    echo "❌ HotFlip attacks failed. Aborting."
    exit 1
}

# Stage 7: Aggregate Results (depends on all attacks)
echo ""
echo "Stage 7: Aggregate results and final analysis..."
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5:$JOB6 jobs/job_7_aggregate.sh)
echo "  Job ID: $JOB7"
echo "  Depends on: $JOB5, $JOB6"
echo "  ⏱  Expected time: 5-15 minutes"

check_job_status $JOB7 "stage_7_aggregate" || {
    echo "❌ Result aggregation failed. Aborting."
    exit 1
}

# All done!
echo ""
echo "=========================================="
echo "✅ ALL STAGES COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Ended: $(date)"
echo ""
echo "Results saved to:"
echo "  Work dir: ${SCRATCH}/mono_s2s_results/"
echo "  Final results: ${PROJECT}/mono_s2s_final_results/"
echo ""
echo "Key files:"
echo "  - experiment_metadata.json (complete configuration)"
echo "  - evaluation_results.json (primary comparison with CIs)"
echo "  - transfer_matrix.json (cross-model attack results)"
echo "  - final_results.json (aggregated analysis)"
echo ""
echo "Job IDs for reference:"
echo "  Setup: $JOB0"
echo "  Data: $JOB1"
echo "  Baseline Training: $JOB2"
echo "  Monotonic Training: $JOB3"
echo "  Evaluation: $JOB4"
echo "  UAT Attacks: $JOB5"
echo "  HotFlip Attacks: $JOB6"
echo "  Aggregation: $JOB7"
echo ""
echo "=========================================="

exit 0

