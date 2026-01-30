#!/bin/bash
################################################################################
# Diagnostic Script for SLURM Job Failures
################################################################################

echo "======================================================================"
echo "SLURM JOB FAILURE DIAGNOSTICS"
echo "======================================================================"
echo ""

# Get job IDs from command line or from .job_ids file
if [ $# -gt 0 ]; then
    JOBS="$@"
else
    if [ -f .job_ids ]; then
        JOBS=$(cat .job_ids)
    else
        echo "ERROR: No job IDs provided and .job_ids file not found"
        echo "Usage: $0 <job_id1> <job_id2> ..."
        exit 1
    fi
fi

echo "Checking jobs: $JOBS"
echo ""

# Extract just the first 3 jobs (the ones that likely failed)
JOB_ARRAY=($JOBS)
FAILED_JOBS="${JOB_ARRAY[0]} ${JOB_ARRAY[1]} ${JOB_ARRAY[2]}"

echo "======================================================================"
echo "1. CHECKING JOB STATUS (sacct)"
echo "======================================================================"
echo ""
sacct -j ${FAILED_JOBS// /,} --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,MaxRSS,ReqMem,AllocNodes,Reason -P
echo ""

echo "======================================================================"
echo "2. CHECKING LOG FILES"
echo "======================================================================"
echo ""

for job_id in ${JOB_ARRAY[@]}; do
    echo "--- Job $job_id ---"
    
    # Find log files for this job
    LOG_FILES=$(ls logs/*_${job_id}.{out,err} 2>/dev/null)
    
    if [ -z "$LOG_FILES" ]; then
        echo "  No log files found for job $job_id"
    else
        for log_file in $LOG_FILES; do
            if [ -f "$log_file" ]; then
                echo "  Found: $log_file"
                echo "  Size: $(du -h "$log_file" | cut -f1)"
                echo "  Last 20 lines:"
                tail -20 "$log_file" | sed 's/^/    /'
                echo ""
            fi
        done
    fi
    echo ""
done

echo "======================================================================"
echo "3. CHECKING ENVIRONMENT"
echo "======================================================================"
echo ""

# Check conda installation
CONDA_BASE="/projects/$USER/miniconda3"
if [ -f "$CONDA_BASE/bin/conda" ]; then
    echo "✓ Conda found at $CONDA_BASE"
    
    # Check if environment exists
    source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null
    if conda env list 2>/dev/null | grep -q "^mono_s2s "; then
        echo "✓ Environment 'mono_s2s' exists"
        
        # Activate and check PyTorch
        conda activate mono_s2s 2>/dev/null
        if python -c "import torch" 2>/dev/null; then
            echo "✓ PyTorch installed: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
        else
            echo "✗ PyTorch not found in mono_s2s environment"
        fi
    else
        echo "✗ Environment 'mono_s2s' not found"
    fi
else
    echo "✗ Conda not found at $CONDA_BASE"
fi
echo ""

# Check HuggingFace cache directory
SCRATCH_DIR=${SCRATCH:-/scratch/alpine/$USER}
HF_CACHE="$SCRATCH_DIR/huggingface_cache"
if [ -d "$HF_CACHE" ]; then
    echo "✓ HuggingFace cache exists: $HF_CACHE"
    echo "  Size: $(du -sh "$HF_CACHE" 2>/dev/null | cut -f1)"
else
    echo "✗ HuggingFace cache not found: $HF_CACHE"
fi
echo ""

# Check working directory
WORK_DIR="$SCRATCH_DIR/foundation_llm_work"
if [ -d "$WORK_DIR" ]; then
    echo "✓ Working directory exists: $WORK_DIR"
    echo "  Contents:"
    ls -lh "$WORK_DIR" 2>/dev/null | sed 's/^/    /'
else
    echo "✓ Working directory will be created: $WORK_DIR"
fi
echo ""

echo "======================================================================"
echo "4. CHECKING PARTITION LIMITS"
echo "======================================================================"
echo ""

# Show partition info
scontrol show partition aa100 2>/dev/null | grep -E "(PartitionName|MaxMemPerNode|MaxTime|State)" || echo "Could not get partition info"
echo ""

echo "======================================================================"
echo "5. CURRENT QUEUE STATUS"
echo "======================================================================"
echo ""
squeue -u $USER
echo ""

echo "======================================================================"
echo "DIAGNOSTIC SUMMARY"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Review the log files above for error messages"
echo "  2. Check if conda environment needs to be set up"
echo "  3. Verify partition limits match job requirements"
echo "  4. If jobs are stuck, cancel them:"
echo "     scancel $(echo $JOBS | tr ' ' ' ')"
echo ""
echo "To resubmit after fixing issues:"
echo "  ./run_all.sh"
echo ""
