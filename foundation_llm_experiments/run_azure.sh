#!/bin/bash
################################################################################
# Azure Spot VM Runner (2x A100 80GB)
# Foundation LLM Monotonicity Experiments
#
# KEY ADVANTAGE: Runs baseline (GPU 0) and monotonic (GPU 1) in PARALLEL,
# cutting total training time roughly in half vs sequential execution.
#
# SPOT DEALLOCATION HANDLING:
#   - Checkpoints saved every 500 steps AND every 30 minutes
#   - On reboot after spot recovery, crontab auto-restarts interrupted training
#   - Manual restart: just run this script again (skips completed stages)
#
# USAGE:
#   ./run_azure.sh                    # Default seed 42
#   EXPERIMENT_SEED=1337 ./run_azure.sh
#
# EXPECTED RUNTIME (2x A100 80GB PCIe, single seed):
#   Stage 0+1:  ~10 min (sequential, GPU 0)
#   Stage 2+3:  ~12 hours (PARALLEL: baseline GPU 0, monotonic GPU 1)
#   Stage 4:    ~1 hour (sequential)
#   Stage 5+6:  ~3 hours (PARALLEL: UAT GPU 0, HotFlip GPU 1)
#   Stage 7:    ~5 min
#   Total:      ~16-18 hours (vs ~55 hours sequential)
#
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SEED="${EXPERIMENT_SEED:-42}"
export EXPERIMENT_SEED="$SEED"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ============================================================================
# ENVIRONMENT
# ============================================================================

if [ -f "${SCRIPT_DIR}/.azure_env" ]; then
    source "${SCRIPT_DIR}/.azure_env"
else
    echo "ERROR: .azure_env not found. Run bootstrap_azure.sh first."
    exit 1
fi

CONDA_BASE="${HOME}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mono_s2s 2>/dev/null || {
    echo "ERROR: conda environment 'mono_s2s' not found. Run bootstrap_azure.sh."
    exit 1
}

# Seed-namespaced paths
WORK_DIR="${AZURE_WORK}_seed${SEED}"
RESULTS_DIR="${AZURE_RESULTS}_seed${SEED}"
export LAMBDA_SEED_WORK="$WORK_DIR"
export LAMBDA_SEED_RESULTS="$RESULTS_DIR"

mkdir -p "${WORK_DIR}/checkpoints/baseline_checkpoints"
mkdir -p "${WORK_DIR}/checkpoints/monotonic_checkpoints"
mkdir -p "${WORK_DIR}/stage_logs"
mkdir -p "${RESULTS_DIR}"

# ============================================================================
# HELPERS
# ============================================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log()     { echo -e "[$(date '+%H:%M:%S')] $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] $*${NC}"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] $*${NC}" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}== $* ==${NC}"; }

flag() { echo "${WORK_DIR}/stage_${1}_complete.flag"; }

stage_done() { [ -f "$(flag "$1")" ]; }

run_on_gpu() {
    local gpu="$1"
    local label="$2"
    local script="$3"
    local logfile="$4"

    log "[$label] Starting on GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu EXPERIMENT_SEED=$SEED \
        python "$script" > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        success "[$label] COMPLETED"
    else
        error "[$label] FAILED (exit $rc). Log: $logfile"
    fi
    return $rc
}

# ============================================================================
# PRE-FLIGHT
# ============================================================================

header "FOUNDATION LLM EXPERIMENTS - AZURE 2xA100 PARALLEL"

log "Seed:       $SEED"
log "Work dir:   $WORK_DIR"
log "Results:    $RESULTS_DIR"
log "Python:     $(which python)"

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
log "GPUs:       $GPU_COUNT"

for i in $(seq 0 $((GPU_COUNT-1))); do
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name($i))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties($i).total_memory/1e9:.0f}GB')")
    log "  GPU $i:    $GPU_NAME ($GPU_MEM)"
done
echo ""

if [ "$GPU_COUNT" -lt 2 ]; then
    warn "Only $GPU_COUNT GPU(s) detected. Will run sequentially instead of parallel."
fi

# ============================================================================
# STAGE 0: SETUP
# ============================================================================

if stage_done "0_setup"; then
    success "[Stage 0] Already complete, skipping"
else
    header "STAGE 0: SETUP"
    cd scripts
    run_on_gpu 0 "Stage 0" stage_0_setup.py "$LOG_DIR/stage0_seed${SEED}.log"
    cd "$SCRIPT_DIR"
fi

# ============================================================================
# STAGE 1: APPLY MONOTONICITY
# ============================================================================

if stage_done "1_apply_monotonicity"; then
    success "[Stage 1] Already complete, skipping"
else
    header "STAGE 1: APPLY MONOTONICITY"
    cd scripts
    run_on_gpu 0 "Stage 1" stage_1_apply_monotonicity.py "$LOG_DIR/stage1_seed${SEED}.log"
    cd "$SCRIPT_DIR"
fi

# ============================================================================
# STAGES 2+3: PARALLEL TRAINING (baseline GPU 0, monotonic GPU 1)
# ============================================================================

BASELINE_DONE=false
MONOTONIC_DONE=false

stage_done "2_train_baseline" && BASELINE_DONE=true
stage_done "3_train_monotonic" && MONOTONIC_DONE=true

if $BASELINE_DONE && $MONOTONIC_DONE; then
    success "[Stage 2+3] Both training stages already complete"
else
    header "STAGES 2+3: PARALLEL TRAINING"

    if $BASELINE_DONE; then
        success "[Stage 2] Baseline already complete"
    else
        log "[Stage 2] Starting BASELINE on GPU 0..."
    fi

    if $MONOTONIC_DONE; then
        success "[Stage 3] Monotonic already complete"
    else
        log "[Stage 3] Starting MONOTONIC on GPU 1..."
    fi

    cd scripts

    BASELINE_PID=""
    MONOTONIC_PID=""

    if ! $BASELINE_DONE; then
        CUDA_VISIBLE_DEVICES=0 EXPERIMENT_SEED=$SEED \
            python stage_2_train_baseline.py \
            > "$LOG_DIR/stage2_gpu0_seed${SEED}.log" 2>&1 &
        BASELINE_PID=$!
        log "[Stage 2] Baseline PID: $BASELINE_PID (GPU 0)"
    fi

    if ! $MONOTONIC_DONE; then
        CUDA_VISIBLE_DEVICES=1 EXPERIMENT_SEED=$SEED \
            python stage_3_train_monotonic.py \
            > "$LOG_DIR/stage3_gpu1_seed${SEED}.log" 2>&1 &
        MONOTONIC_PID=$!
        log "[Stage 3] Monotonic PID: $MONOTONIC_PID (GPU 1)"
    fi

    log ""
    log "Both training jobs running in parallel."
    log "Monitor with:"
    log "  tail -f $LOG_DIR/stage2_gpu0_seed${SEED}.log"
    log "  tail -f $LOG_DIR/stage3_gpu1_seed${SEED}.log"
    log "  watch -n 10 nvidia-smi"
    log ""
    log "Waiting for training to complete..."

    TRAIN_FAILED=false

    if [ -n "$BASELINE_PID" ]; then
        wait $BASELINE_PID
        BASELINE_RC=$?
        if [ $BASELINE_RC -eq 0 ]; then
            success "[Stage 2] Baseline training COMPLETED"
        else
            error "[Stage 2] Baseline training FAILED (exit $BASELINE_RC)"
            error "  Log: $LOG_DIR/stage2_gpu0_seed${SEED}.log"
            TRAIN_FAILED=true
        fi
    fi

    if [ -n "$MONOTONIC_PID" ]; then
        wait $MONOTONIC_PID
        MONOTONIC_RC=$?
        if [ $MONOTONIC_RC -eq 0 ]; then
            success "[Stage 3] Monotonic training COMPLETED"
        else
            error "[Stage 3] Monotonic training FAILED (exit $MONOTONIC_RC)"
            error "  Log: $LOG_DIR/stage3_gpu1_seed${SEED}.log"
            TRAIN_FAILED=true
        fi
    fi

    cd "$SCRIPT_DIR"

    if $TRAIN_FAILED; then
        error "One or both training stages failed. Check logs above."
        error "Checkpoints are saved - rerun this script to resume."
        exit 1
    fi
fi

# ============================================================================
# STAGE 4: EVALUATION
# ============================================================================

if stage_done "4_evaluate"; then
    success "[Stage 4] Already complete, skipping"
else
    header "STAGE 4: EVALUATION"
    cd scripts
    run_on_gpu 0 "Stage 4" stage_4_evaluate.py "$LOG_DIR/stage4_seed${SEED}.log"
    cd "$SCRIPT_DIR"
fi

# ============================================================================
# STAGES 5+6: PARALLEL ATTACKS (UAT GPU 0, HotFlip GPU 1)
# ============================================================================

UAT_DONE=false
HOTFLIP_DONE=false

stage_done "5_uat" && UAT_DONE=true
stage_done "6_hotflip" && HOTFLIP_DONE=true

if $UAT_DONE && $HOTFLIP_DONE; then
    success "[Stage 5+6] Both attack stages already complete"
else
    header "STAGES 5+6: PARALLEL ATTACKS"

    cd scripts

    UAT_PID=""
    HOTFLIP_PID=""

    if ! $UAT_DONE; then
        log "[Stage 5] Starting UAT attacks on GPU 0..."
        CUDA_VISIBLE_DEVICES=0 EXPERIMENT_SEED=$SEED \
            python stage_5_uat_attacks.py \
            > "$LOG_DIR/stage5_gpu0_seed${SEED}.log" 2>&1 &
        UAT_PID=$!
    fi

    if ! $HOTFLIP_DONE; then
        log "[Stage 6] Starting HotFlip attacks on GPU 1..."
        CUDA_VISIBLE_DEVICES=1 EXPERIMENT_SEED=$SEED \
            python stage_6_hotflip_attacks.py \
            > "$LOG_DIR/stage6_gpu1_seed${SEED}.log" 2>&1 &
        HOTFLIP_PID=$!
    fi

    log "Attack jobs running in parallel. Waiting..."

    ATTACK_FAILED=false

    if [ -n "$UAT_PID" ]; then
        wait $UAT_PID
        if [ $? -eq 0 ]; then
            success "[Stage 5] UAT attacks COMPLETED"
        else
            error "[Stage 5] UAT attacks FAILED. Log: $LOG_DIR/stage5_gpu0_seed${SEED}.log"
            ATTACK_FAILED=true
        fi
    fi

    if [ -n "$HOTFLIP_PID" ]; then
        wait $HOTFLIP_PID
        if [ $? -eq 0 ]; then
            success "[Stage 6] HotFlip attacks COMPLETED"
        else
            error "[Stage 6] HotFlip attacks FAILED. Log: $LOG_DIR/stage6_gpu1_seed${SEED}.log"
            ATTACK_FAILED=true
        fi
    fi

    cd "$SCRIPT_DIR"

    if $ATTACK_FAILED; then
        warn "One or both attack stages failed. Continuing to aggregation."
    fi
fi

# ============================================================================
# STAGE 7: AGGREGATE
# ============================================================================

if stage_done "7_aggregate"; then
    success "[Stage 7] Already complete, skipping"
else
    header "STAGE 7: AGGREGATE"
    cd scripts
    run_on_gpu 0 "Stage 7" stage_7_aggregate.py "$LOG_DIR/stage7_seed${SEED}.log" || true
    cd "$SCRIPT_DIR"
fi

# ============================================================================
# SUMMARY
# ============================================================================

header "EXPERIMENT COMPLETE - SEED $SEED"

echo ""
log "Completion flags:"
for stage in 0_setup 1_apply_monotonicity 2_train_baseline 3_train_monotonic 4_evaluate 5_uat 6_hotflip 7_aggregate; do
    if [ -f "$(flag "$stage")" ]; then
        success "  Stage $stage: DONE"
    else
        warn "  Stage $stage: INCOMPLETE"
    fi
done

echo ""

EVAL_FILE="${RESULTS_DIR}/evaluation_results.json"
if [ -f "$EVAL_FILE" ]; then
    log "Results:"
    python -c "
import json
with open('${EVAL_FILE}') as f:
    r = json.load(f)
pile = r.get('pile_test', {})
b = pile.get('baseline_pythia', {}).get('perplexity', 'N/A')
m = pile.get('monotonic_pythia', {}).get('perplexity', 'N/A')
print(f'  Baseline perplexity:  {b:.2f}' if isinstance(b, float) else f'  Baseline:  {b}')
print(f'  Monotonic perplexity: {m:.2f}' if isinstance(m, float) else f'  Monotonic: {m}')
" 2>/dev/null || true
fi

echo ""
log "Results directory: $RESULTS_DIR"
log ""
log "To copy results to local machine:"
log "  scp -r azureuser@<vm-ip>:${RESULTS_DIR} ./azure_results_seed${SEED}"
echo ""
