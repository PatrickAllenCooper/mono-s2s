#!/bin/bash
################################################################################
# Job Monitoring and Auto-Resubmission Script
#
# Monitors submitted SLURM jobs and automatically resubmits them if they:
# - Timeout (hit time limit)
# - Fail with certain recoverable errors
# - Get preempted
#
# Maintains checkpoint-based resume across resubmissions.
#
# USAGE:
#   ./monitor_and_resubmit.sh JOB_ID1 JOB_ID2 JOB_ID3 ...
#
# Or after run_all.sh which saves job IDs:
#   ./monitor_and_resubmit.sh $(cat .job_ids)
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_RESUBMISSIONS=5
RESUBMISSION_DELAY=60  # seconds
CHECK_INTERVAL=300     # Check every 5 minutes
SCRATCH="${SCRATCH:-/scratch/alpine/$USER}"
WORK_DIR="$SCRATCH/foundation_llm_work"

# Track resubmissions
declare -A RESUBMISSION_COUNT

# Job name to script mapping
declare -A JOB_SCRIPT_MAP=(
    ["foundation_setup"]="jobs/job_0_setup.sh"
    ["foundation_monotonicity"]="jobs/job_1_monotonicity.sh"
    ["foundation_baseline"]="jobs/job_2_baseline.sh"
    ["foundation_monotonic"]="jobs/job_3_monotonic.sh"
    ["foundation_eval"]="jobs/job_4_evaluate.sh"
    ["foundation_uat"]="jobs/job_5_uat.sh"
    ["foundation_hotflip"]="jobs/job_6_hotflip.sh"
    ["foundation_aggregate"]="jobs/job_7_aggregate.sh"
)

echo "======================================================================"
echo "FOUNDATION LLM EXPERIMENTS - JOB MONITOR"
echo "======================================================================"
echo ""
echo "Monitoring jobs with automatic resubmission on timeout"
echo "Max resubmissions per job: $MAX_RESUBMISSIONS"
echo "Check interval: $CHECK_INTERVAL seconds"
echo ""

# Parse job IDs from arguments
if [ $# -eq 0 ]; then
    echo "ERROR: No job IDs provided"
    echo "USAGE: $0 JOB_ID1 JOB_ID2 ..."
    exit 1
fi

MONITORED_JOBS=("$@")
echo "Monitoring ${#MONITORED_JOBS[@]} jobs: ${MONITORED_JOBS[*]}"
echo ""

# Initialize resubmission counters
for job in "${MONITORED_JOBS[@]}"; do
    RESUBMISSION_COUNT[$job]=0
done

# Save PID for user to kill if needed
MONITOR_PID=$$
echo "Monitor PID: $MONITOR_PID (use 'kill $MONITOR_PID' to stop monitoring)"
echo "Log file: logs/monitor_$(date +%Y%m%d_%H%M%S).log"
echo ""

LOG_FILE="logs/monitor_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Function to log with timestamp
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

# Function to get job status
get_job_status() {
    local job_id=$1
    # Query SLURM for job status
    squeue -j "$job_id" -h -o "%T" 2>/dev/null || echo "NOT_FOUND"
}

# Function to get job name
get_job_name() {
    local job_id=$1
    squeue -j "$job_id" -h -o "%j" 2>/dev/null || sacct -j "$job_id" -n -o JobName%50 | head -1 | tr -d ' '
}

# Function to check if checkpoint exists for a job
check_checkpoint_exists() {
    local job_name=$1
    
    # Check for various checkpoint patterns
    if [[ "$job_name" == *"baseline"* ]]; then
        ls "$WORK_DIR"/checkpoints/baseline_checkpoints/checkpoint_*.pt 2>/dev/null | head -1
    elif [[ "$job_name" == *"monotonic"* ]]; then
        ls "$WORK_DIR"/checkpoints/monotonic_checkpoints/checkpoint_*.pt 2>/dev/null | head -1
    else
        # Generic checkpoint check
        ls "$WORK_DIR"/checkpoints/checkpoint_*.pt 2>/dev/null | head -1
    fi
}

# Function to resubmit a job
resubmit_job() {
    local orig_job_id=$1
    local job_name=$2
    local reason=$3
    
    # Check resubmission count
    local count=${RESUBMISSION_COUNT[$orig_job_id]}
    if [ "$count" -ge "$MAX_RESUBMISSIONS" ]; then
        log "${RED}✗ Job $orig_job_id ($job_name) hit max resubmissions ($MAX_RESUBMISSIONS)${NC}"
        log "${RED}  Manual intervention required.${NC}"
        return 1
    fi
    
    # Find the job script
    local job_script=""
    for name_pattern in "${!JOB_SCRIPT_MAP[@]}"; do
        if [[ "$job_name" == *"$name_pattern"* ]]; then
            job_script="${JOB_SCRIPT_MAP[$name_pattern]}"
            break
        fi
    done
    
    if [ -z "$job_script" ] || [ ! -f "$job_script" ]; then
        log "${RED}✗ Cannot find job script for $job_name${NC}"
        return 1
    fi
    
    # Check if checkpoint exists (for training jobs)
    local checkpoint_msg=""
    if [[ "$job_name" == *"baseline"* ]] || [[ "$job_name" == *"monotonic"* ]]; then
        local checkpoint=$(check_checkpoint_exists "$job_name")
        if [ -n "$checkpoint" ]; then
            checkpoint_msg=" (will resume from checkpoint)"
        else
            checkpoint_msg=" (no checkpoint found, will start from beginning)"
        fi
    fi
    
    log "${YELLOW}↻ Resubmitting job $orig_job_id ($job_name)${NC}"
    log "${YELLOW}  Reason: $reason${NC}"
    log "${YELLOW}  Resubmission #$((count + 1))/$MAX_RESUBMISSIONS${checkpoint_msg}${NC}"
    
    # Wait before resubmitting
    sleep "$RESUBMISSION_DELAY"
    
    # Resubmit
    local new_job_id=$(sbatch --parsable "$job_script" 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    
    if [ -n "$new_job_id" ]; then
        log "${GREEN}✓ Resubmitted as job $new_job_id${NC}"
        
        # Update tracking
        RESUBMISSION_COUNT[$new_job_id]=$((count + 1))
        
        # Add to monitored jobs (remove old, add new)
        MONITORED_JOBS=("${MONITORED_JOBS[@]/$orig_job_id}")
        MONITORED_JOBS+=("$new_job_id")
        
        return 0
    else
        log "${RED}✗ Failed to resubmit job${NC}"
        return 1
    fi
}

# Function to check job and decide if resubmission needed
check_and_resubmit() {
    local job_id=$1
    local status=$(get_job_status "$job_id")
    local job_name=$(get_job_name "$job_id")
    
    case "$status" in
        "NOT_FOUND")
            # Job finished - check exit code via sacct
            local exit_code=$(sacct -j "$job_id" -n -o ExitCode | head -1 | cut -d: -f1 | tr -d ' ')
            local state=$(sacct -j "$job_id" -n -o State | head -1 | tr -d ' ')
            
            case "$state" in
                "COMPLETED")
                    log "${GREEN}✓ Job $job_id ($job_name) completed successfully${NC}"
                    return 2  # Signal: remove from monitoring, job complete
                    ;;
                "TIMEOUT"|"CANCELLED+")
                    log "${YELLOW}⚠ Job $job_id ($job_name) timed out${NC}"
                    resubmit_job "$job_id" "$job_name" "TIMEOUT"
                    return $?
                    ;;
                "FAILED"|"NODE_FAIL")
                    log "${YELLOW}⚠ Job $job_id ($job_name) failed (state: $state)${NC}"
                    resubmit_job "$job_id" "$job_name" "$state"
                    return $?
                    ;;
                "PREEMPTED")
                    log "${YELLOW}⚠ Job $job_id ($job_name) was preempted${NC}"
                    resubmit_job "$job_id" "$job_name" "PREEMPTED"
                    return $?
                    ;;
                "OUT_OF_MEMORY")
                    log "${RED}✗ Job $job_id ($job_name) ran out of memory${NC}"
                    log "${RED}  This requires manual intervention (increase memory in job script)${NC}"
                    return 2  # Don't resubmit OOM errors automatically
                    ;;
                *)
                    log "${RED}✗ Job $job_id ($job_name) ended with state: $state (exit: $exit_code)${NC}"
                    return 2  # Don't resubmit unknown failures
                    ;;
            esac
            ;;
        "RUNNING"|"PENDING"|"REQUEUED"|"CONFIGURING")
            # Job is still active, keep monitoring
            return 0
            ;;
        *)
            log "${BLUE}ℹ Job $job_id ($job_name) status: $status${NC}"
            return 0
            ;;
    esac
}

# Main monitoring loop
log "Starting monitoring loop..."
ACTIVE_JOBS=("${MONITORED_JOBS[@]}")

while [ ${#ACTIVE_JOBS[@]} -gt 0 ]; do
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log "Checking ${#ACTIVE_JOBS[@]} active jobs..."
    
    NEW_ACTIVE_JOBS=()
    
    for job in "${ACTIVE_JOBS[@]}"; do
        if [ -z "$job" ]; then continue; fi
        
        check_and_resubmit "$job"
        result=$?
        
        if [ $result -eq 0 ]; then
            # Still active, keep monitoring
            NEW_ACTIVE_JOBS+=("$job")
        elif [ $result -eq 2 ]; then
            # Completed or failed permanently, remove from monitoring
            :
        fi
        # result 1 means resubmitted, new job added to MONITORED_JOBS
    done
    
    # Update active jobs list (includes newly resubmitted jobs)
    ACTIVE_JOBS=()
    for job in "${MONITORED_JOBS[@]}"; do
        if [ -z "$job" ]; then continue; fi
        local status=$(get_job_status "$job")
        if [[ "$status" != "NOT_FOUND" ]] || [[ "$status" == "RUNNING" ]] || [[ "$status" == "PENDING" ]]; then
            ACTIVE_JOBS+=("$job")
        fi
    done
    
    if [ ${#ACTIVE_JOBS[@]} -eq 0 ]; then
        log "${GREEN}All jobs completed!${NC}"
        break
    fi
    
    log "Next check in $CHECK_INTERVAL seconds..."
    sleep "$CHECK_INTERVAL"
done

log ""
log "======================================================================"
log "MONITORING COMPLETE"
log "======================================================================"
log ""
log "Final status summary:"
for job in "${MONITORED_JOBS[@]}"; do
    if [ -z "$job" ]; then continue; fi
    local job_name=$(get_job_name "$job")
    local state=$(sacct -j "$job" -n -o State | head -1 | tr -d ' ')
    local count=${RESUBMISSION_COUNT[$job]:-0}
    log "  Job $job ($job_name): $state (resubmitted $count times)"
done

log ""
log "Check results in: $SCRATCH/foundation_llm_results/"
log "Log saved to: $LOG_FILE"
