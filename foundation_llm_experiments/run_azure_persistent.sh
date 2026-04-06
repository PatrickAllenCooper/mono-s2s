#!/bin/bash
################################################################################
# Azure Spot VM Runner with Persistent Checkpoint Storage
#
# Uses TWO storage tiers:
#   /persist  - Managed data disk (SURVIVES deallocation)
#              Stores: checkpoints, results, completion flags
#   /data     - NVMe local SSD (WIPED on deallocation, but fast)
#              Stores: model cache, dataset cache, training scratch
#
# On deallocation recovery:
#   1. NVMe is reformatted and remounted
#   2. Checkpoints are restored from /persist
#   3. Training resumes from the latest checkpoint automatically
#
# FIRST TIME SETUP:
#   1. Attach a managed data disk in Azure Portal (256GB+)
#   2. Run: bash setup_persistent_disk.sh
#   3. Run: bash bootstrap_azure.sh
#   4. tmux new-session -s sprint
#   5. ./run_azure_persistent.sh
#
# AFTER SPOT DEALLOCATION:
#   1. Start the VM again
#   2. SSH in
#   3. ./recover_and_resume.sh
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
# STORAGE CONFIGURATION
# ============================================================================

PERSIST="/persist"       # Managed disk - survives deallocation
FAST="/data"             # NVMe - wiped on deallocation, but fast

PERSIST_WORK="${PERSIST}/foundation_llm_work_seed${SEED}"
PERSIST_RESULTS="${PERSIST}/foundation_llm_results_seed${SEED}"
FAST_CACHE="${FAST}/foundation_llm_cache"

# ============================================================================
# VERIFY STORAGE
# ============================================================================

if ! mountpoint -q "$PERSIST" 2>/dev/null; then
    echo "ERROR: $PERSIST is not mounted."
    echo ""
    echo "If you haven't set up the persistent disk yet:"
    echo "  1. Attach a managed data disk in Azure Portal"
    echo "  2. Run: bash setup_persistent_disk.sh"
    echo ""
    echo "If the VM was just deallocated and restarted:"
    echo "  Run: bash recover_and_resume.sh"
    exit 1
fi

if ! mountpoint -q "$FAST" 2>/dev/null; then
    echo "NVMe not mounted at $FAST. Mounting..."
    sudo mkfs.ext4 -F /dev/nvme0n1 2>/dev/null || true
    sudo mkdir -p "$FAST"
    sudo mount /dev/nvme0n1 "$FAST" 2>/dev/null || true
    sudo chown $USER:$USER "$FAST"
fi

echo "Storage verified:"
echo "  Persistent: $(df -h $PERSIST | awk 'NR==2{print $4}') free on $PERSIST"
echo "  Fast cache: $(df -h $FAST | awk 'NR==2{print $4}') free on $FAST"

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

mkdir -p "${PERSIST_WORK}/checkpoints/baseline_checkpoints"
mkdir -p "${PERSIST_WORK}/checkpoints/monotonic_checkpoints"
mkdir -p "${PERSIST_WORK}/stage_logs"
mkdir -p "${PERSIST_RESULTS}"
mkdir -p "${FAST_CACHE}/huggingface"

# ============================================================================
# ENVIRONMENT
# ============================================================================

export AZURE_WORK="${PERSIST_WORK}"
export AZURE_RESULTS="${PERSIST_RESULTS}"
export AZURE_CACHE="${FAST_CACHE}"
export LAMBDA_SEED_WORK="${PERSIST_WORK}"
export LAMBDA_SEED_RESULTS="${PERSIST_RESULTS}"
export LAMBDA_CACHE="${FAST_CACHE}"
export HF_HOME="${FAST_CACHE}/huggingface"
export HF_DATASETS_CACHE="${FAST_CACHE}/huggingface/datasets"
export TRANSFORMERS_CACHE="${FAST_CACHE}/huggingface/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:16:8

CONDA_BASE="${HOME}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mono_s2s 2>/dev/null || {
    echo "ERROR: conda env not found. Run: bash bootstrap_azure.sh"
    exit 1
}

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

flag() { echo "${PERSIST_WORK}/stage_${1}_complete.flag"; }
stage_done() { [ -f "$(flag "$1")" ]; }

# ============================================================================
# PRE-FLIGHT
# ============================================================================

header "AZURE 2xA100 - PERSISTENT CHECKPOINT MODE"

log "Seed:         $SEED"
log "Persist dir:  $PERSIST_WORK  (survives deallocation)"
log "Cache dir:    $FAST_CACHE  (fast NVMe, ephemeral)"
log "Results dir:  $PERSIST_RESULTS"
echo ""

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
log "GPUs: $GPU_COUNT"
for i in $(seq 0 $((GPU_COUNT-1))); do
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name($i))")
    log "  GPU $i: $GPU_NAME"
done

# Show checkpoint status
echo ""
log "Checkpoint status:"
for stage in 0_setup 1_apply_monotonicity 2_train_baseline 3_train_monotonic 4_evaluate 5_uat 6_hotflip; do
    if stage_done "$stage"; then
        success "  $stage: COMPLETE"
    else
        # Check for partial checkpoints
        case "$stage" in
            2_train_baseline)
                CKPTS=$(find ${PERSIST_WORK}/checkpoints/baseline_checkpoints/ -name "checkpoint_epoch_*.pt" 2>/dev/null | wc -l || echo 0)
                [ "$CKPTS" -gt 0 ] && warn "  $stage: $CKPTS epoch(s) checkpointed" || log "  $stage: not started"
                ;;
            3_train_monotonic)
                CKPTS=$(find ${PERSIST_WORK}/checkpoints/monotonic_checkpoints/ -name "checkpoint_epoch_*.pt" 2>/dev/null | wc -l || echo 0)
                [ "$CKPTS" -gt 0 ] && warn "  $stage: $CKPTS epoch(s) checkpointed" || log "  $stage: not started"
                ;;
            *)
                log "  $stage: not started"
                ;;
        esac
    fi
done
echo ""

# ============================================================================
# STAGE 0: SETUP
# ============================================================================

if stage_done "0_setup"; then
    success "[Stage 0] Already complete"
else
    header "STAGE 0: SETUP"
    cd scripts
    CUDA_VISIBLE_DEVICES=0 python stage_0_setup.py 2>&1 | tee "$LOG_DIR/stage0_seed${SEED}.log"
    cd "$SCRIPT_DIR"
fi

# ============================================================================
# STAGE 1: APPLY MONOTONICITY
# ============================================================================

if stage_done "1_apply_monotonicity"; then
    success "[Stage 1] Already complete"
else
    header "STAGE 1: APPLY MONOTONICITY"
    cd scripts
    CUDA_VISIBLE_DEVICES=0 python stage_1_apply_monotonicity.py 2>&1 | tee "$LOG_DIR/stage1_seed${SEED}.log"
    cd "$SCRIPT_DIR"
fi

# ============================================================================
# STAGES 2+3: PARALLEL TRAINING
# ============================================================================

BASELINE_DONE=false
MONOTONIC_DONE=false
stage_done "2_train_baseline" && BASELINE_DONE=true
stage_done "3_train_monotonic" && MONOTONIC_DONE=true

if $BASELINE_DONE && $MONOTONIC_DONE; then
    success "[Stage 2+3] Both complete"
else
    header "STAGES 2+3: PARALLEL TRAINING (GPU 0 + GPU 1)"

    BASELINE_PID=""
    MONOTONIC_PID=""

    cd scripts

    if ! $BASELINE_DONE; then
        log "Starting BASELINE on GPU 0..."
        CUDA_VISIBLE_DEVICES=0 python stage_2_train_baseline.py \
            2>&1 | tee "$LOG_DIR/stage2_gpu0_seed${SEED}.log" &
        BASELINE_PID=$!
        log "  Baseline PID: $BASELINE_PID"
    fi

    if ! $MONOTONIC_DONE; then
        log "Starting MONOTONIC on GPU 1..."
        CUDA_VISIBLE_DEVICES=1 python stage_3_train_monotonic.py \
            2>&1 | tee "$LOG_DIR/stage3_gpu1_seed${SEED}.log" &
        MONOTONIC_PID=$!
        log "  Monotonic PID: $MONOTONIC_PID"
    fi

    log ""
    log "Training in parallel. Checkpoints save to $PERSIST (persistent)."
    log "If deallocated, rerun this script to resume from latest checkpoint."
    log ""

    FAIL=false

    if [ -n "$BASELINE_PID" ]; then
        wait $BASELINE_PID || { error "Baseline FAILED"; FAIL=true; }
        $FAIL || success "Baseline training COMPLETE"
    fi

    if [ -n "$MONOTONIC_PID" ]; then
        wait $MONOTONIC_PID || { error "Monotonic FAILED"; FAIL=true; }
        $FAIL || success "Monotonic training COMPLETE"
    fi

    cd "$SCRIPT_DIR"
    $FAIL && { error "Training failed. Checkpoints saved - rerun to resume."; exit 1; }
fi

# ============================================================================
# STAGE 4: EVALUATION
# ============================================================================

if stage_done "4_evaluate"; then
    success "[Stage 4] Already complete"
else
    header "STAGE 4: EVALUATION"
    cd scripts
    CUDA_VISIBLE_DEVICES=0 python stage_4_evaluate.py 2>&1 | tee "$LOG_DIR/stage4_seed${SEED}.log"
    cd "$SCRIPT_DIR"
fi

# ============================================================================
# STAGES 5+6: PARALLEL ATTACKS
# ============================================================================

UAT_DONE=false
HOTFLIP_DONE=false
stage_done "5_uat" && UAT_DONE=true
stage_done "6_hotflip" && HOTFLIP_DONE=true

if $UAT_DONE && $HOTFLIP_DONE; then
    success "[Stage 5+6] Both complete"
else
    header "STAGES 5+6: PARALLEL ATTACKS"
    cd scripts

    UAT_PID=""
    HOTFLIP_PID=""

    if ! $UAT_DONE; then
        CUDA_VISIBLE_DEVICES=0 python stage_5_uat_attacks.py \
            2>&1 | tee "$LOG_DIR/stage5_gpu0_seed${SEED}.log" &
        UAT_PID=$!
    fi

    if ! $HOTFLIP_DONE; then
        CUDA_VISIBLE_DEVICES=1 python stage_6_hotflip_attacks.py \
            2>&1 | tee "$LOG_DIR/stage6_gpu1_seed${SEED}.log" &
        HOTFLIP_PID=$!
    fi

    [ -n "$UAT_PID" ] && { wait $UAT_PID && success "UAT COMPLETE" || warn "UAT FAILED"; }
    [ -n "$HOTFLIP_PID" ] && { wait $HOTFLIP_PID && success "HotFlip COMPLETE" || warn "HotFlip FAILED"; }

    cd "$SCRIPT_DIR"
fi

# ============================================================================
# STAGE 7: AGGREGATE
# ============================================================================

header "STAGE 7: AGGREGATE"
cd scripts
CUDA_VISIBLE_DEVICES=0 python stage_7_aggregate.py 2>&1 | tee "$LOG_DIR/stage7_seed${SEED}.log" || true
cd "$SCRIPT_DIR"

# ============================================================================
# SUMMARY
# ============================================================================

header "COMPLETE"
for stage in 0_setup 1_apply_monotonicity 2_train_baseline 3_train_monotonic 4_evaluate 5_uat 6_hotflip; do
    stage_done "$stage" && success "  $stage" || warn "  $stage: INCOMPLETE"
done

EVAL_FILE="${PERSIST_RESULTS}/evaluation_results.json"
if [ -f "$EVAL_FILE" ]; then
    echo ""
    python -c "
import json
with open('${EVAL_FILE}') as f:
    r = json.load(f)
p = r.get('pile_test', {})
b = p.get('baseline_pythia', {}).get('perplexity', 'N/A')
m = p.get('monotonic_pythia', {}).get('perplexity', 'N/A')
print(f'Baseline:  {b}')
print(f'Monotonic: {m}')
" 2>/dev/null || true
fi

echo ""
log "Results on persistent disk: $PERSIST_RESULTS"
log "These survive spot deallocation."
