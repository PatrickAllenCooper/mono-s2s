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
    
    # Give filesystem a moment to sync (helps with NFS delays)
    sleep 5
    
    # Check if completion flag exists
    local flag_file="${SCRATCH}/mono_s2s_work/${job_name}_complete.flag"
    if [ -f "$flag_file" ]; then
        echo "[SUCCESS] $job_name completed successfully"
        cat "$flag_file"
        return 0
    fi
    
    # Flag missing - check if training actually completed by verifying checkpoints
    # This handles the case where training finished but flag wasn't created
    
    if [[ "$job_name" == "stage_3_train_monotonic" ]]; then
        local checkpoint_dir="${SCRATCH}/mono_s2s_work/checkpoints/monotonic_checkpoints"
        local expected_epochs=7
        
        if [ -d "$checkpoint_dir" ]; then
            local epoch_count=$(ls "$checkpoint_dir"/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
            local has_best=$([ -f "$checkpoint_dir/best_model.pt" ] && echo "yes" || echo "no")
            
            echo "[DIAGNOSTIC] Checking monotonic training state..."
            echo "  Epoch checkpoints found: $epoch_count/$expected_epochs"
            echo "  Best model saved: $has_best"
            
            if [ $epoch_count -ge $expected_epochs ] && [ "$has_best" = "yes" ]; then
                echo "[AUTO-FIX] Training completed but flag missing. Creating flag..."
                cat > "$flag_file" <<EOF
Completed at: $(date '+%Y-%m-%d %H:%M:%S')
Seed: ${EXPERIMENT_SEED:-42}
Note: Auto-created by run_all.sh after verifying $epoch_count checkpoints
EOF
                echo "[SUCCESS] $job_name completed (flag auto-created)"
                return 0
            fi
        fi
    fi
    
    if [[ "$job_name" == "stage_2_train_baseline" ]]; then
        local checkpoint_dir="${SCRATCH}/mono_s2s_work/checkpoints/baseline_checkpoints"
        local expected_epochs=5
        
        if [ -d "$checkpoint_dir" ]; then
            local epoch_count=$(ls "$checkpoint_dir"/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
            local has_best=$([ -f "$checkpoint_dir/best_model.pt" ] && echo "yes" || echo "no")
            
            echo "[DIAGNOSTIC] Checking baseline training state..."
            echo "  Epoch checkpoints found: $epoch_count/$expected_epochs"
            echo "  Best model saved: $has_best"
            
            if [ $epoch_count -ge $expected_epochs ] && [ "$has_best" = "yes" ]; then
                echo "[AUTO-FIX] Training completed but flag missing. Creating flag..."
                cat > "$flag_file" <<EOF
Completed at: $(date '+%Y-%m-%d %H:%M:%S')
Seed: ${EXPERIMENT_SEED:-42}
Note: Auto-created by run_all.sh after verifying $epoch_count checkpoints
EOF
                echo "[SUCCESS] $job_name completed (flag auto-created)"
                return 0
            fi
        fi
    fi
    
    # Genuinely failed
    echo "[FAILED] $job_name FAILED"
    echo "  Expected flag: $flag_file"
    echo "  Check job logs:"
    echo "    Output: logs/job_*_${job_id}.out"
    echo "    Errors: logs/job_*_${job_id}.err"
    
    if ls logs/job_*_${job_id}.out 1>/dev/null 2>&1; then
        echo ""
        echo "  Last 20 lines of job output:"
        tail -n 20 logs/job_*_${job_id}.out 2>/dev/null | sed 's/^/    /'
    fi
    
    return 1
}

# Submit jobs with dependencies
echo "Submitting SLURM jobs with dependencies..."
echo ""

# Stage 0: Setup
echo "Stage 0: Setup and environment verification..."
if [ -f "${SCRATCH}/mono_s2s_work/stage_0_setup_complete.flag" ]; then
    echo "  [SKIP] Already complete, skipping..."
    JOB0="completed"
else
    JOB0=$(sbatch --parsable jobs/job_0_setup.sh)
    JOB0=$(echo "$JOB0" | cut -d';' -f1)
    echo "  Job ID: $JOB0"
    check_job_status $JOB0 "stage_0_setup" || {
        echo "[FAILED] Setup failed. Aborting."
        exit 1
    }
fi

# Stage 1: Data Preparation
echo ""
echo "Stage 1: Data preparation..."
if [ -f "${SCRATCH}/mono_s2s_work/stage_1_data_prep_complete.flag" ]; then
    echo "  [SKIP] Already complete, skipping..."
    JOB1="completed"
else
    if [ "$JOB0" = "completed" ]; then
        JOB1=$(sbatch --parsable jobs/job_1_data.sh)
    else
        JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 jobs/job_1_data.sh)
    fi
    JOB1=$(echo "$JOB1" | cut -d';' -f1)
    echo "  Job ID: $JOB1"
    check_job_status $JOB1 "stage_1_data_prep" || {
        echo "[FAILED] Data preparation failed. Aborting."
        exit 1
    }
fi

# Stage 2: Train Baseline (depends on data)
echo ""
echo "Stage 2: Train baseline model (unconstrained)..."
if [ -f "${SCRATCH}/mono_s2s_work/stage_2_train_baseline_complete.flag" ]; then
    echo "  [SKIP] Already complete, skipping..."
    JOB2="completed"
else
    if [ "$JOB1" = "completed" ]; then
        JOB2=$(sbatch --parsable jobs/job_2_baseline.sh)
    else
        JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 jobs/job_2_baseline.sh)
    fi
    JOB2=$(echo "$JOB2" | cut -d';' -f1)
    echo "  Job ID: $JOB2"
    echo "  [TIME] Expected time: 4-12 hours"
fi

# Stage 3: Train Monotonic (depends on data, can run parallel with baseline)
echo ""
echo "Stage 3: Train monotonic model (W≥0 constraints)..."
if [ -f "${SCRATCH}/mono_s2s_work/stage_3_train_monotonic_complete.flag" ]; then
    echo "  [SKIP] Already complete, skipping..."
    JOB3="completed"
else
    # Number of epochs (must match MONOTONIC_NUM_EPOCHS in configs/experiment_config.py)
    MONOTONIC_EPOCHS=7
    
    # Determine initial dependency
    if [ "$JOB1" = "completed" ]; then
        LAST_ID=""
    else
        LAST_ID=$JOB1
    fi

    echo "  Submitting chain of $MONOTONIC_EPOCHS jobs (one per epoch) to avoid 24h timeout..."
    
    for (( i=1; i<=MONOTONIC_EPOCHS; i++ ))
    do
        if [ -z "$LAST_ID" ]; then
             # First job, no dependency
             JOB_ID=$(sbatch --parsable jobs/job_3_monotonic.sh --max_epochs_per_run 1)
        else
             # Dependent on previous job (either data prep or previous epoch)
             JOB_ID=$(sbatch --parsable --dependency=afterok:$LAST_ID jobs/job_3_monotonic.sh --max_epochs_per_run 1)
        fi
        
        # Strip cluster name if present (e.g., 12345;cluster)
        JOB_ID=$(echo "$JOB_ID" | cut -d';' -f1)
        LAST_ID=$JOB_ID
        echo "    Epoch $i: Job $JOB_ID"
    done
    
    JOB3=$LAST_ID
    echo "  Final Job ID: $JOB3"
    echo "  [TIME] Expected time: 4-12 hours (total for chain)"
    echo "  [INFO] Runs in PARALLEL with baseline training"
fi

# Wait for both training jobs (only if they were submitted)
if [ "$JOB2" != "completed" ] || [ "$JOB3" != "completed" ]; then
    echo ""
    echo "[WAIT] Waiting for training jobs to complete..."
    echo "  This may take 4-12 hours depending on configuration"
    echo "  Monitor progress with: squeue -u $USER"
    echo ""
fi

if [ "$JOB2" != "completed" ]; then
    check_job_status $JOB2 "stage_2_train_baseline" || {
        echo "[FAILED] Baseline training failed. Aborting."
        exit 1
    }
fi

if [ "$JOB3" != "completed" ]; then
    check_job_status $JOB3 "stage_3_train_monotonic" || {
        echo "[FAILED] Monotonic training failed. Aborting."
        exit 1
    }
fi

# Stage 4: Comprehensive Evaluation (depends on both models)
echo ""
echo "Stage 4: Comprehensive evaluation (all 3 models, all 3 test sets)..."
if [ -f "${SCRATCH}/mono_s2s_work/stage_4_evaluate_complete.flag" ]; then
    echo "  [SKIP] Already complete, skipping..."
    JOB4="completed"
else
    # Build dependency list (only on jobs that actually ran)
    DEPS=""
    [ "$JOB2" != "completed" ] && DEPS="$JOB2"
    [ "$JOB3" != "completed" ] && DEPS="$DEPS:$JOB3"
    DEPS=$(echo $DEPS | sed 's/^://') # Remove leading colon
    
    # Try to submit with dependencies if needed
    if [ -n "$DEPS" ]; then
        echo "  [INFO] Attempting to submit with dependencies: $DEPS"
        if JOB4=$(sbatch --parsable --dependency=afterok:$DEPS jobs/job_4_evaluate.sh 2>&1); then
            JOB4=$(echo "$JOB4" | cut -d';' -f1)
            echo "  Job ID: $JOB4"
        else
            # Dependency failed - check if prerequisites are actually complete
            echo "  [WARNING] Dependency submission failed (jobs may have already completed)"
            echo "  [INFO] Checking if training stages completed via flags..."
            
            PREREQ_MET=true
            [ ! -f "${SCRATCH}/mono_s2s_work/stage_2_train_baseline_complete.flag" ] && PREREQ_MET=false
            [ ! -f "${SCRATCH}/mono_s2s_work/stage_3_train_monotonic_complete.flag" ] && PREREQ_MET=false
            
            if [ "$PREREQ_MET" = true ]; then
                echo "  [AUTO-RECOVER] Prerequisites complete, submitting without dependencies..."
                JOB4=$(sbatch --parsable jobs/job_4_evaluate.sh 2>&1)
                JOB4=$(echo "$JOB4" | cut -d';' -f1)
                echo "  Job ID: $JOB4"
            else
                echo "  [FAILED] Prerequisites not met. Cannot continue."
                exit 1
            fi
        fi
    else
        JOB4=$(sbatch --parsable jobs/job_4_evaluate.sh 2>&1)
        JOB4=$(echo "$JOB4" | cut -d';' -f1)
        echo "  Job ID: $JOB4"
    fi
    
    # Check config for expected time
    if grep -q "USE_FULL_TEST_SETS = True" configs/experiment_config.py; then
        echo "  [TIME] Expected time: 12-16 hours (full test sets)"
    else
        echo "  [TIME] Expected time: 1-2 hours (quick test: 200 samples)"
    fi
    
    check_job_status $JOB4 "stage_4_evaluate" || {
        echo "[FAILED] Evaluation failed. Aborting."
        exit 1
    }
fi

# Stage 5: UAT Attacks (depends on evaluation)
echo ""
echo "Stage 5: UAT attacks with transfer matrix..."
if [ -f "${SCRATCH}/mono_s2s_work/stage_5_uat_complete.flag" ]; then
    echo "  [SKIP] Already complete, skipping..."
    JOB5="completed"
else
    if [ "$JOB4" != "completed" ]; then
        if JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 jobs/job_5_uat.sh 2>&1); then
            JOB5=$(echo "$JOB5" | cut -d';' -f1)
        else
            echo "  [WARNING] Dependency submission failed, checking prerequisites..."
            if [ -f "${SCRATCH}/mono_s2s_work/stage_4_evaluate_complete.flag" ]; then
                echo "  [AUTO-RECOVER] Stage 4 complete, submitting without dependency..."
                JOB5=$(sbatch --parsable jobs/job_5_uat.sh 2>&1)
                JOB5=$(echo "$JOB5" | cut -d';' -f1)
            else
                echo "  [FAILED] Stage 4 not complete. Cannot continue."
                exit 1
            fi
        fi
    else
        JOB5=$(sbatch --parsable jobs/job_5_uat.sh 2>&1)
        JOB5=$(echo "$JOB5" | cut -d';' -f1)
    fi
    echo "  Job ID: $JOB5"
    echo "  [TIME] Expected time: 2-3 hours"
fi

# Stage 6: HotFlip Attacks (can run parallel with UAT)
echo ""
echo "Stage 6: HotFlip attacks..."
if [ -f "${SCRATCH}/mono_s2s_work/stage_6_hotflip_complete.flag" ]; then
    echo "  [SKIP] Already complete, skipping..."
    JOB6="completed"
else
    if [ "$JOB4" != "completed" ]; then
        if JOB6=$(sbatch --parsable --dependency=afterok:$JOB4 jobs/job_6_hotflip.sh 2>&1); then
            JOB6=$(echo "$JOB6" | cut -d';' -f1)
        else
            echo "  [WARNING] Dependency submission failed, checking prerequisites..."
            if [ -f "${SCRATCH}/mono_s2s_work/stage_4_evaluate_complete.flag" ]; then
                echo "  [AUTO-RECOVER] Stage 4 complete, submitting without dependency..."
                JOB6=$(sbatch --parsable jobs/job_6_hotflip.sh 2>&1)
                JOB6=$(echo "$JOB6" | cut -d';' -f1)
            else
                echo "  [FAILED] Stage 4 not complete. Cannot continue."
                exit 1
            fi
        fi
    else
        JOB6=$(sbatch --parsable jobs/job_6_hotflip.sh 2>&1)
        JOB6=$(echo "$JOB6" | cut -d';' -f1)
    fi
    echo "  Job ID: $JOB6"
    echo "  [TIME] Expected time: 1-2 hours"
    echo "  [INFO] Runs in PARALLEL with UAT attacks"
fi

# Wait for attack jobs (only if they were submitted)
if [ "$JOB5" != "completed" ] && [ "$JOB6" != "completed" ]; then
    check_job_status $JOB5 "stage_5_uat" || {
        echo "[FAILED] UAT attacks failed. Aborting."
        exit 1
    }
    check_job_status $JOB6 "stage_6_hotflip" || {
        echo "[FAILED] HotFlip attacks failed. Aborting."
        exit 1
    }
fi

# Stage 7: Aggregate Results (depends on all attacks)
echo ""
echo "Stage 7: Aggregate results and final analysis..."
if [ -f "${SCRATCH}/mono_s2s_work/stage_7_aggregate_complete.flag" ]; then
    echo "  [SKIP] Already complete, skipping..."
    JOB7="completed"
else
    # Build dependency list
    DEPS=""
    [ "$JOB5" != "completed" ] && DEPS="$JOB5"
    [ "$JOB6" != "completed" ] && DEPS="$DEPS:$JOB6"
    DEPS=$(echo $DEPS | sed 's/^://') # Remove leading colon
    
    if [ -n "$DEPS" ]; then
        echo "  [INFO] Attempting to submit with dependencies: $DEPS"
        if JOB7=$(sbatch --parsable --dependency=afterok:$DEPS jobs/job_7_aggregate.sh 2>&1); then
            JOB7=$(echo "$JOB7" | cut -d';' -f1)
        else
            echo "  [WARNING] Dependency submission failed, checking prerequisites..."
            PREREQ_MET=true
            [ ! -f "${SCRATCH}/mono_s2s_work/stage_5_uat_complete.flag" ] && PREREQ_MET=false
            [ ! -f "${SCRATCH}/mono_s2s_work/stage_6_hotflip_complete.flag" ] && PREREQ_MET=false
            
            if [ "$PREREQ_MET" = true ]; then
                echo "  [AUTO-RECOVER] Prerequisites complete, submitting without dependencies..."
                JOB7=$(sbatch --parsable jobs/job_7_aggregate.sh 2>&1)
                JOB7=$(echo "$JOB7" | cut -d';' -f1)
            else
                echo "  [FAILED] Prerequisites not met. Cannot continue."
                exit 1
            fi
        fi
    else
        JOB7=$(sbatch --parsable jobs/job_7_aggregate.sh 2>&1)
        JOB7=$(echo "$JOB7" | cut -d';' -f1)
    fi
    echo "  Job ID: $JOB7"
    echo "  [TIME] Expected time: 5-15 minutes"
    
    check_job_status $JOB7 "stage_7_aggregate" || {
        echo "[FAILED] Result aggregation failed. Aborting."
        exit 1
    }
fi

# All done!
echo ""
echo "=========================================="
echo "[SUCCESS] ALL STAGES COMPLETED SUCCESSFULLY!"
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

