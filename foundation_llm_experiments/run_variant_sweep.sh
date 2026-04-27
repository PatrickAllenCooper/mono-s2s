#!/bin/bash
################################################################################
# Monotonic Variant Sweep Orchestrator
#
# Runs Phase A (seed 42, two new variants) then Phase B (winner variant on
# seeds 1337 and 2024). Each cell (seed x variant) is fully isolated in its
# own /persist/sweep_work_* and /persist/sweep_results_* directory so the
# per-stage completion flags and atomic JSON checkpoints work independently.
#
# Usage
# -----
#   ./run_variant_sweep.sh                   # Full Phase A + B (auto-select winner)
#   ./run_variant_sweep.sh --phase-a-only    # Screen variants on seed 42 only
#   ./run_variant_sweep.sh --phase-b VARIANT # Run Phase B with VARIANT directly
#
# Fast-mode training budget per cell (override on cmdline if needed):
#   FAST_TRAINING_SAMPLES : rows from train split  (default 30000)
#   FAST_RECOVERY_EPOCHS  : baseline epochs         (default 3)
#   FAST_MONOTONIC_EPOCHS : monotonic epochs        (default 4)
#   FAST_SEQ_LEN          : max sequence length     (default 1024)
#   FAST_HOTFLIP_SAMPLES  : HotFlip eval sample     (default 200)
#   FAST_PILE_TEST_SIZE   : eval perplexity rows    (default 2000)
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# FAST-MODE DEFAULTS (can be overridden by environment before launch)
# ============================================================================
FAST_TRAINING_SAMPLES="${FAST_TRAINING_SAMPLES:-30000}"
FAST_RECOVERY_EPOCHS="${FAST_RECOVERY_EPOCHS:-3}"
FAST_MONOTONIC_EPOCHS="${FAST_MONOTONIC_EPOCHS:-4}"
FAST_SEQ_LEN="${FAST_SEQ_LEN:-1024}"
FAST_HOTFLIP_SAMPLES="${FAST_HOTFLIP_SAMPLES:-200}"
FAST_PILE_TEST_SIZE="${FAST_PILE_TEST_SIZE:-2000}"

# ============================================================================
# STORAGE CONFIG (same layout as run_azure_persistent.sh)
# ============================================================================
PERSIST="/persist"
FAST="/data"
FAST_CACHE="${FAST}/foundation_llm_cache"
PERSIST_DATASETS_CACHE="${PERSIST}/foundation_llm_cache/huggingface/datasets"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; BLUE='\033[0;34m'; NC='\033[0m'
log()     { echo "[$(date '+%H:%M:%S')] $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] $*${NC}"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] $*${NC}" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}== $* ==${NC}"; }

# ============================================================================
# COMMAND LINE
# ============================================================================
PHASE_A_ONLY=false
FORCE_VARIANT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase-a-only) PHASE_A_ONLY=true; shift ;;
        --phase-b)      FORCE_VARIANT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ============================================================================
# VERIFY MOUNTS AND ENVIRONMENT
# ============================================================================
if ! mountpoint -q "$PERSIST" 2>/dev/null; then
    error "$PERSIST not mounted. Run recover_and_resume.sh first."; exit 1
fi
if ! mountpoint -q "$FAST" 2>/dev/null; then
    warn "NVMe not mounted at $FAST, attempting..."
    sudo mkfs.ext4 -F /dev/nvme0n1 2>/dev/null || true
    sudo mkdir -p "$FAST"
    sudo mount /dev/nvme0n1 "$FAST" 2>/dev/null || true
    sudo chown "$USER:$USER" "$FAST"
fi
mkdir -p "${FAST_CACHE}/huggingface" "${PERSIST_DATASETS_CACHE}"

export HF_DATASETS_CACHE="${PERSIST_DATASETS_CACHE}"
export HF_HOME="${FAST_CACHE}/huggingface"
export TRANSFORMERS_CACHE="${FAST_CACHE}/huggingface/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:16:8

CONDA_BASE="${HOME}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mono_s2s 2>/dev/null || {
    error "conda env not found. Run bootstrap_azure.sh first."; exit 1
}

# ============================================================================
# HELPER: run a single (seed, variant) cell
#   run_cell SEED VARIANT CELL_LOG_PREFIX
# Executes stages 0→4 and 6, writing to isolated /persist/sweep_* dirs.
# Idempotent: per-stage completion flags inside the cell dir make it safe
# to re-invoke after a spot deallocation.
# ============================================================================
run_cell() {
    local SEED="$1"
    local VARIANT="$2"
    local CELL_TAG="seed${SEED}_${VARIANT}"
    local CELL_WORK="${PERSIST}/sweep_work_${CELL_TAG}"
    local CELL_RESULTS="${PERSIST}/sweep_results_${CELL_TAG}"

    header "CELL: seed=${SEED} variant=${VARIANT}"
    log "Work:    ${CELL_WORK}"
    log "Results: ${CELL_RESULTS}"

    export EXPERIMENT_SEED="$SEED"
    export MONOTONIC_VARIANT="$VARIANT"
    export AZURE_CACHE="${FAST_CACHE}"
    # Use LAMBDA_SEED_WORK / LAMBDA_SEED_RESULTS which Config reads directly
    # as WORK_DIR / RESULTS_DIR with no seed-suffix appended (unlike AZURE_WORK
    # which triggers Config's seed-suffix logic and produces double suffixes).
    export LAMBDA_SEED_WORK="${CELL_WORK}"
    export LAMBDA_SEED_RESULTS="${CELL_RESULTS}"
    export LAMBDA_CACHE="${FAST_CACHE}"
    unset AZURE_WORK AZURE_RESULTS

    # Fast-mode training budget
    export OVERRIDE_TRAINING_SAMPLES="$FAST_TRAINING_SAMPLES"
    export OVERRIDE_RECOVERY_EPOCHS="$FAST_RECOVERY_EPOCHS"
    export OVERRIDE_MONOTONIC_RECOVERY_EPOCHS="$FAST_MONOTONIC_EPOCHS"
    export OVERRIDE_MAX_SEQ_LENGTH="$FAST_SEQ_LEN"
    export OVERRIDE_HOTFLIP_NUM_SAMPLES="$FAST_HOTFLIP_SAMPLES"
    export OVERRIDE_QUICK_PILE_TEST_SIZE="$FAST_PILE_TEST_SIZE"

    mkdir -p "${CELL_WORK}/checkpoints/baseline_checkpoints"
    mkdir -p "${CELL_WORK}/checkpoints/monotonic_checkpoints"
    mkdir -p "${CELL_WORK}/stage_logs"
    mkdir -p "${CELL_RESULTS}"

    local LOG_CELL="${LOG_DIR}/sweep_${CELL_TAG}"
    mkdir -p "$LOG_CELL"

    flag()       { echo "${CELL_WORK}/stage_${1}_complete.flag"; }
    stage_done() { [ -f "$(flag "$1")" ]; }

    cd scripts

    # Stage 0: setup (model download, dir creation)
    if stage_done "0_setup"; then
        success "  [${CELL_TAG}] Stage 0 already done"
    else
        log "  [${CELL_TAG}] Stage 0: setup..."
        CUDA_VISIBLE_DEVICES=0 python stage_0_setup.py \
            2>&1 | tee "${LOG_CELL}/stage0.log" \
            && success "  Stage 0 done" || { error "  Stage 0 FAILED"; cd "$SCRIPT_DIR"; return 1; }
    fi

    # Stage 1: apply monotonicity init (variant-aware)
    if stage_done "1_apply_monotonicity"; then
        success "  [${CELL_TAG}] Stage 1 already done"
    else
        log "  [${CELL_TAG}] Stage 1: apply monotonicity (variant=${VARIANT})..."
        CUDA_VISIBLE_DEVICES=0 python stage_1_apply_monotonicity.py \
            2>&1 | tee "${LOG_CELL}/stage1.log" \
            && success "  Stage 1 done" || { error "  Stage 1 FAILED"; cd "$SCRIPT_DIR"; return 1; }
    fi

    # Stages 2+3: baseline and monotonic training (sequential, same GPU)
    if stage_done "2_train_baseline"; then
        success "  [${CELL_TAG}] Stage 2 already done"
    else
        log "  [${CELL_TAG}] Stage 2: train baseline (samples=${FAST_TRAINING_SAMPLES}, epochs=${FAST_RECOVERY_EPOCHS}, seqlen=${FAST_SEQ_LEN})..."
        CUDA_VISIBLE_DEVICES=0 python stage_2_train_baseline.py \
            2>&1 | tee "${LOG_CELL}/stage2.log" \
            && success "  Stage 2 done" || { error "  Stage 2 FAILED"; cd "$SCRIPT_DIR"; return 1; }
    fi

    if stage_done "3_train_monotonic"; then
        success "  [${CELL_TAG}] Stage 3 already done"
    else
        log "  [${CELL_TAG}] Stage 3: train monotonic (samples=${FAST_TRAINING_SAMPLES}, epochs=${FAST_MONOTONIC_EPOCHS})..."
        CUDA_VISIBLE_DEVICES=0 python stage_3_train_monotonic.py \
            2>&1 | tee "${LOG_CELL}/stage3.log" \
            && success "  Stage 3 done" || { error "  Stage 3 FAILED"; cd "$SCRIPT_DIR"; return 1; }
    fi

    # Stage 4: perplexity evaluation
    if stage_done "4_evaluate"; then
        success "  [${CELL_TAG}] Stage 4 already done"
    else
        log "  [${CELL_TAG}] Stage 4: evaluate (pile_test_size=${FAST_PILE_TEST_SIZE})..."
        CUDA_VISIBLE_DEVICES=0 python stage_4_evaluate.py \
            2>&1 | tee "${LOG_CELL}/stage4.log" \
            && success "  Stage 4 done" || { error "  Stage 4 FAILED"; cd "$SCRIPT_DIR"; return 1; }
    fi

    # Stage 6: HotFlip attacks (no UAT in sweep mode)
    if stage_done "6_hotflip"; then
        success "  [${CELL_TAG}] Stage 6 already done"
    else
        log "  [${CELL_TAG}] Stage 6: HotFlip attacks (samples=${FAST_HOTFLIP_SAMPLES})..."
        CUDA_VISIBLE_DEVICES=0 python stage_6_hotflip_attacks.py \
            2>&1 | tee "${LOG_CELL}/stage6.log" \
            && success "  Stage 6 done" || { error "  Stage 6 FAILED"; cd "$SCRIPT_DIR"; return 1; }
    fi

    cd "$SCRIPT_DIR"
    success "[${CELL_TAG}] CELL COMPLETE"
}

# ============================================================================
# PHASE A: screen both variants on seed 42
# ============================================================================
header "VARIANT SWEEP - PHASE A"
log "Variants: mlp_both, mlp_in_attn_out"
log "Seed:     42"
log "Budget:   ${FAST_TRAINING_SAMPLES} samples / ${FAST_RECOVERY_EPOCHS} baseline epochs / ${FAST_MONOTONIC_EPOCHS} monotonic epochs / seqlen ${FAST_SEQ_LEN}"
echo ""

run_cell 42 mlp_both
run_cell 42 mlp_in_attn_out

success "Phase A complete."

# ============================================================================
# SELECT WINNER from Phase A using sweep_aggregate.py decision logic
# ============================================================================
WINNER_FILE="${PERSIST}/sweep_winner.txt"

if [ -n "$FORCE_VARIANT" ]; then
    echo "$FORCE_VARIANT" > "$WINNER_FILE"
    warn "Winner forced to: ${FORCE_VARIANT}"
else
    log "Selecting winner from Phase A..."
    python scripts/sweep_aggregate.py \
        --seeds 42 \
        --variants mlp_both mlp_in_attn_out \
        --ppl-ceiling 2.0 \
        --pick-winner \
        --winner-file "$WINNER_FILE" \
        --results-root "${PERSIST}/sweep_results" \
        2>&1 | tee "${LOG_DIR}/phase_a_aggregate.log" || true

    if [ ! -f "$WINNER_FILE" ]; then
        warn "Auto-selection failed or both variants exceeded ppl ceiling."
        warn "Relaxing ceiling to 3.0x..."
        python scripts/sweep_aggregate.py \
            --seeds 42 \
            --variants mlp_both mlp_in_attn_out \
            --ppl-ceiling 3.0 \
            --pick-winner \
            --winner-file "$WINNER_FILE" \
            --results-root "${PERSIST}/sweep_results" \
            2>&1 | tee -a "${LOG_DIR}/phase_a_aggregate.log" || true
    fi

    if [ ! -f "$WINNER_FILE" ]; then
        warn "No winner selected (both variants fail ppl ceiling even at 3x)."
        warn "Review ${LOG_DIR}/phase_a_aggregate.log manually."
        if [ "$PHASE_A_ONLY" = false ]; then
            error "Aborting Phase B - no valid winner. Re-run with --phase-b VARIANT to force."
            exit 1
        fi
    fi
fi

if $PHASE_A_ONLY; then
    log "Phase A only - stopping here."
    log "Review logs in ${LOG_DIR}/ and re-run with:"
    log "  ./run_variant_sweep.sh --phase-b <winner_variant>"
    exit 0
fi

WINNER=$(cat "$WINNER_FILE")
success "Winner variant: ${WINNER}"

# ============================================================================
# PHASE B: rerun winner on seeds 1337 and 2024
# ============================================================================
header "VARIANT SWEEP - PHASE B (winner=${WINNER})"

run_cell 1337 "$WINNER"
run_cell 2024 "$WINNER"

success "Phase B complete."

# ============================================================================
# FINAL AGGREGATE across all 3 seeds
# ============================================================================
header "FINAL AGGREGATE"
python scripts/sweep_aggregate.py \
    --seeds 42 1337 2024 \
    --variants "$WINNER" \
    --results-root "${PERSIST}/sweep_results" \
    --output "${SCRIPT_DIR}/../paper_evidence/sweep_summary.json" \
    2>&1 | tee "${LOG_DIR}/final_aggregate.log"

success "Sweep complete. Results in paper_evidence/sweep_summary.json"
log "Results survive spot deallocation in: ${PERSIST}/sweep_results_*"
