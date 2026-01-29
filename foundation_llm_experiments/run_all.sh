#!/bin/bash
################################################################################
# Master Submission Script for Foundation LLM Monotonicity Experiments
#
# Automatically handles ALL setup and job submission:
# - Checks for conda/environment (runs bootstrap if needed)
# - Submits all experimental stages with proper dependencies
#
# USAGE: Just run this script - it handles everything!
#   ./run_all.sh
################################################################################

set -euo pipefail

echo "======================================================================"
echo "FOUNDATION LLM MONOTONICITY EXPERIMENTS"
echo "Master Submission Script"
echo "======================================================================"
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

# ======================================================================
# AUTOMATIC ENVIRONMENT SETUP
# ======================================================================

echo "Checking environment setup..."
echo ""

# Define expected conda location and environment name
CONDA_BASE="/projects/$USER/miniconda3"
ENV_NAME="mono_s2s"
SETUP_NEEDED=false

# Check if conda is installed
if [ ! -f "$CONDA_BASE/bin/conda" ]; then
    echo "⚠️  Conda not found at $CONDA_BASE"
    SETUP_NEEDED=true
else
    # Initialize conda for this script
    source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
    
    # Check if environment exists
    if ! conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
        echo "⚠️  Conda environment '$ENV_NAME' not found"
        SETUP_NEEDED=true
    else
        # Check if PyTorch is installed
        conda activate "$ENV_NAME" 2>/dev/null || true
        if ! python -c "import torch" 2>/dev/null; then
            echo "⚠️  PyTorch not installed in environment '$ENV_NAME'"
            SETUP_NEEDED=true
        else
            echo "✓ Environment '$ENV_NAME' is ready"
            echo "  Conda: $CONDA_BASE"
            echo "  Python: $(which python)"
            echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')"
        fi
    fi
fi

# Run bootstrap if needed
if [ "$SETUP_NEEDED" = true ]; then
    echo ""
    echo "======================================================================"
    echo "FIRST-TIME SETUP REQUIRED"
    echo "======================================================================"
    echo ""
    echo "The bootstrap script will automatically:"
    echo "  - Install Miniconda to /projects/$USER (avoids home quota issues)"
    echo "  - Create conda environment with Python 3.10"
    echo "  - Install PyTorch with CUDA 11.8"
    echo "  - Install all dependencies"
    echo "  - Configure HuggingFace cache"
    echo ""
    echo "This will take ~5-10 minutes."
    echo ""
    read -p "Run automatic setup now? (y/N) " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Setup cancelled. You can run it manually later:"
        echo "  bash bootstrap_curc.sh"
        exit 0
    fi
    
    echo ""
    echo "Running bootstrap script..."
    echo ""
    
    if [ -f "bootstrap_curc.sh" ]; then
        bash bootstrap_curc.sh
        
        # Re-initialize conda after bootstrap
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
        
        echo ""
        echo "✓ Setup complete!"
        echo ""
    else
        echo "ERROR: bootstrap_curc.sh not found"
        echo "Please ensure you're in the foundation_llm_experiments directory"
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "SUBMITTING EXPERIMENTAL STAGES"
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

# Save job IDs for monitoring script
echo "$JOB0 $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6 $JOB7" > .job_ids
echo "✓ Job IDs saved to .job_ids"
echo ""

# Ask about automatic monitoring and resubmission
read -p "Start automatic job monitoring with auto-resubmit on timeout? (Y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "Starting job monitor in background..."
    echo "This will automatically resubmit jobs if they timeout."
    echo ""
    
    # Make monitor script executable
    chmod +x monitor_and_resubmit.sh 2>/dev/null || true
    
    # Start monitor in background
    nohup ./monitor_and_resubmit.sh $JOB0 $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6 $JOB7 > logs/monitor_${JOB0}.out 2>&1 &
    MONITOR_PID=$!
    
    echo "✓ Monitor started (PID: $MONITOR_PID)"
    echo "  Monitor log: logs/monitor_${JOB0}.out"
    echo "  To stop monitoring: kill $MONITOR_PID"
    echo ""
else
    echo ""
    echo "Skipping automatic monitoring."
    echo "You can start it manually later:"
    echo "  ./monitor_and_resubmit.sh \$(cat .job_ids)"
    echo ""
fi

echo "======================================================================"
echo "MONITORING COMMANDS"
echo "======================================================================"
echo ""
echo "Check job status:"
echo "  squeue -u \$USER | grep foundation"
echo ""
echo "Watch specific job log:"
echo "  tail -f logs/job_2_baseline_${JOB2}.out"
echo ""
echo "Check monitor log (if monitoring enabled):"
echo "  tail -f logs/monitor_${JOB0}.out"
echo ""
echo "Check results:"
echo "  cat \$SCRATCH/foundation_llm_work/experiment_summary.txt"
echo ""
echo "Cancel all jobs:"
echo "  scancel $JOB0 $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6 $JOB7"
echo ""
echo "Manual resubmit if needed (checkpoint resume automatic):"
echo "  sbatch jobs/job_2_baseline.sh"
echo ""
echo "======================================================================"
