#!/bin/bash
################################################################################
# Lambda Cloud Sequential Stage Runner
# Foundation LLM Monotonicity Experiments
#
# PURPOSE:
#   Runs all experimental stages sequentially on a Lambda Cloud GPU instance.
#   Unlike CURC Alpine (which uses SLURM job queues), Lambda Cloud gives you
#   direct GPU access, so stages run back-to-back in a single session.
#
# USAGE:
#   # Single seed (default: 42)
#   ./run_lambda.sh
#
#   # Specific seed
#   EXPERIMENT_SEED=1337 ./run_lambda.sh
#
#   # Multiple seeds sequentially
#   SEEDS="42 1337 2024" ./run_lambda.sh
#
#   # Resume from a specific stage (if previous run was interrupted)
#   START_STAGE=3 EXPERIMENT_SEED=42 ./run_lambda.sh
#
# RECOMMENDED: Run inside tmux to survive SSH disconnects:
#   tmux new-session -s mono_exp
#   ./run_lambda.sh
#   # Ctrl+B D to detach
#   # tmux attach -t mono_exp to reconnect
#
# EXPECTED RUNTIME:
#
#   GPU            Stage 2 (Baseline)  Stage 3 (Monotonic)  Total per seed
#   -------------- ------------------- -------------------- ---------------
#   A10 (24GB)     ~18-22 hours        ~40-48 hours         ~65-75 hours
#   A100 (40GB)    ~12-15 hours        ~28-34 hours         ~45-55 hours
#   A100 (80GB)    ~10-12 hours        ~22-28 hours         ~38-45 hours
#   H100 (80GB)    ~4-6 hours          ~9-12 hours          ~16-22 hours
#
# NOTE ON A10 (24GB):
#   Batch size is automatically reduced from 8 to 4 to fit in 24GB VRAM.
#   Gradient accumulation doubles from 4 to 8 to maintain effective batch=32.
#   This means ~2x more gradient accumulation steps per epoch vs A100,
#   which is why training takes roughly 2x longer.
#
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS="${SEEDS:-${EXPERIMENT_SEED:-42}}"
START_STAGE="${START_STAGE:-0}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Source lambda environment variables if available
if [ -f "${SCRIPT_DIR}/.lambda_env" ]; then
    source "${SCRIPT_DIR}/.lambda_env"
else
    # Fallback defaults for Lambda Cloud
    export LAMBDA_WORK="${HOME}/foundation_llm_work"
    export LAMBDA_RESULTS="${HOME}/foundation_llm_results"
    export LAMBDA_CACHE="${HOME}/foundation_llm_cache"
    export HF_HOME="${LAMBDA_CACHE}/huggingface"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUBLAS_WORKSPACE_CONFIG=:16:8
fi

# Activate conda environment
CONDA_BASE="${HOME}/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate mono_s2s 2>/dev/null || {
        echo "ERROR: Could not activate conda environment 'mono_s2s'"
        echo "Please run: bash bootstrap_lambda.sh"
        exit 1
    }
else
    echo "WARNING: Conda not found at $CONDA_BASE"
    echo "Assuming Python and packages are already in PATH"
fi

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
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $*${NC}"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $*${NC}" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════════════════════════${NC}"; \
            echo -e "${BOLD}${BLUE}  $*${NC}"; \
            echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════════════════════════${NC}"; }

run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    local script="$3"
    local seed="$4"
    local flag_file="$5"

    if [ "$stage_num" -lt "$START_STAGE" ]; then
        log "Skipping Stage $stage_num ($stage_name) - START_STAGE=$START_STAGE"
        return 0
    fi

    # Check if already complete
    if [ -f "$flag_file" ]; then
        success "Stage $stage_num ($stage_name): Already complete, skipping"
        return 0
    fi

    header "STAGE $stage_num: $stage_name  [seed=$seed]"

    local stage_log="${LOG_DIR}/stage_${stage_num}_seed${seed}.log"
    local start_time=$(date +%s)

    log "Script:  $script"
    log "Log:     $stage_log"
    log "Started: $(date)"
    echo ""

    # Run stage with tee so output goes to both terminal and log file
    cd "${SCRIPT_DIR}/scripts"
    EXPERIMENT_SEED="$seed" python "$script" 2>&1 | tee "$stage_log"
    local exit_code=${PIPESTATUS[0]}
    cd "$SCRIPT_DIR"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local elapsed_fmt=$(printf '%dh %02dm %02ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))

    if [ $exit_code -eq 0 ]; then
        success "Stage $stage_num ($stage_name): COMPLETED in $elapsed_fmt"
    else
        error "Stage $stage_num ($stage_name): FAILED (exit code $exit_code) after $elapsed_fmt"
        error "Check log: $stage_log"
        error "To resume from this stage: START_STAGE=$stage_num EXPERIMENT_SEED=$seed ./run_lambda.sh"
        exit $exit_code
    fi

    echo ""
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

header "FOUNDATION LLM MONOTONICITY EXPERIMENTS - LAMBDA CLOUD"

log "Working dir:  $SCRIPT_DIR"
log "Seeds:        $SEEDS"
log "Start stage:  $START_STAGE"
log "Python:       $(which python)"
log "PyTorch:      $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo ""

# GPU check
if python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')")
    GPU_MEM_INT=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024**3))")
    success "GPU: $GPU_NAME ($GPU_MEM)"

    if [ "${GPU_MEM_INT:-99}" -lt 40 ]; then
        warn "A10/small GPU detected (${GPU_MEM}). Batch size auto-reduced to 4, grad_accum=8."
        warn "Training will take ~2x longer than on A100. See runtime table in script header."
    fi
else
    error "No CUDA GPU detected - experiments require a GPU"
    exit 1
fi

# Disk space check
DISK_FREE=$(df -BG "${HOME}" | awk 'NR==2{print $4}' | tr -d 'G')
if [ "${DISK_FREE:-0}" -lt 200 ]; then
    warn "Low disk space: ${DISK_FREE}GB free. Recommend 200GB+ per seed."
fi
log "Disk free: ${DISK_FREE}GB"
echo ""

log "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv 2>/dev/null
echo ""

# ============================================================================
# CONFIRM
# ============================================================================

echo "This will run all experiment stages sequentially."
echo "Estimated total runtime: ~32-40 hours per seed."
echo ""
read -p "Start experiment for seeds: $SEEDS ? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "Cancelled."
    exit 0
fi

# ============================================================================
# MAIN LOOP: RUN ALL STAGES PER SEED
# ============================================================================

TOTAL_START=$(date +%s)

for SEED in $SEEDS; do

    header "STARTING FULL PIPELINE FOR SEED $SEED"

    # Set seed-namespaced paths
    export EXPERIMENT_SEED="$SEED"
    WORK_DIR="${LAMBDA_WORK}_seed${SEED}"
    RESULTS_DIR="${LAMBDA_RESULTS}_seed${SEED}"

    # Override config paths via env vars
    export LAMBDA_SEED_WORK="$WORK_DIR"
    export LAMBDA_SEED_RESULTS="$RESULTS_DIR"

    mkdir -p "${WORK_DIR}/checkpoints/baseline_checkpoints"
    mkdir -p "${WORK_DIR}/checkpoints/monotonic_checkpoints"
    mkdir -p "${WORK_DIR}/stage_logs"
    mkdir -p "${RESULTS_DIR}"

    log "Work dir:    $WORK_DIR"
    log "Results dir: $RESULTS_DIR"

    SEED_START=$(date +%s)

    # ------------------------------------------------------------------
    # Stage 0: Setup - Download Pythia-1.4B
    # ------------------------------------------------------------------
    run_stage 0 "Setup" \
        "stage_0_setup.py" \
        "$SEED" \
        "${WORK_DIR}/stage_0_setup_complete.flag"

    # ------------------------------------------------------------------
    # Stage 1: Apply Monotonicity Constraints
    # ------------------------------------------------------------------
    run_stage 1 "Apply Monotonicity" \
        "stage_1_apply_monotonicity.py" \
        "$SEED" \
        "${WORK_DIR}/stage_1_apply_monotonicity_complete.flag"

    # ------------------------------------------------------------------
    # Stage 2: Baseline Recovery Training
    # ------------------------------------------------------------------
    run_stage 2 "Baseline Training" \
        "stage_2_train_baseline.py" \
        "$SEED" \
        "${WORK_DIR}/stage_2_train_baseline_complete.flag"

    # ------------------------------------------------------------------
    # Stage 3: Monotonic Recovery Training
    # ------------------------------------------------------------------
    run_stage 3 "Monotonic Training" \
        "stage_3_train_monotonic.py" \
        "$SEED" \
        "${WORK_DIR}/stage_3_train_monotonic_complete.flag"

    # ------------------------------------------------------------------
    # Stage 4: Evaluation
    # ------------------------------------------------------------------
    run_stage 4 "Evaluation" \
        "stage_4_evaluate.py" \
        "$SEED" \
        "${WORK_DIR}/stage_4_evaluate_complete.flag"

    # ------------------------------------------------------------------
    # Stage 5: UAT Attacks
    # ------------------------------------------------------------------
    run_stage 5 "UAT Attacks" \
        "stage_5_uat_attacks.py" \
        "$SEED" \
        "${WORK_DIR}/stage_5_uat_complete.flag"

    # ------------------------------------------------------------------
    # Stage 6: HotFlip Attacks
    # ------------------------------------------------------------------
    run_stage 6 "HotFlip Attacks" \
        "stage_6_hotflip_attacks.py" \
        "$SEED" \
        "${WORK_DIR}/stage_6_hotflip_complete.flag"

    # ------------------------------------------------------------------
    # Stage 7: Aggregate Results
    # ------------------------------------------------------------------
    run_stage 7 "Aggregate Results" \
        "stage_7_aggregate.py" \
        "$SEED" \
        "${WORK_DIR}/stage_7_aggregate_complete.flag"

    SEED_END=$(date +%s)
    SEED_ELAPSED=$(( SEED_END - SEED_START ))
    SEED_FMT=$(printf '%dh %02dm' $((SEED_ELAPSED/3600)) $((SEED_ELAPSED%3600/60)))

    header "SEED $SEED COMPLETE"
    success "Total time for seed $SEED: $SEED_FMT"
    echo ""

    # Print results summary
    EVAL_FILE="${RESULTS_DIR}/evaluation_results.json"
    if [ -f "$EVAL_FILE" ]; then
        echo "Results summary:"
        python -c "
import json
with open('${EVAL_FILE}') as f:
    r = json.load(f)
pile = r.get('pile_test', {})
baseline = pile.get('baseline_pythia', {}).get('perplexity', 'N/A')
monotonic = pile.get('monotonic_pythia', {}).get('perplexity', 'N/A')
print(f'  Baseline perplexity:  {baseline:.2f}' if isinstance(baseline, float) else f'  Baseline perplexity:  {baseline}')
print(f'  Monotonic perplexity: {monotonic:.2f}' if isinstance(monotonic, float) else f'  Monotonic perplexity: {monotonic}')
" 2>/dev/null || true
    fi

    echo ""
    log "Results saved to: $RESULTS_DIR"
    log "To copy results to local machine:"
    log "  scp -r ubuntu@<instance-ip>:${RESULTS_DIR} ./results_seed${SEED}"
    echo ""

done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_FMT=$(printf '%dh %02dm' $((TOTAL_ELAPSED/3600)) $((TOTAL_ELAPSED%3600/60)))

header "ALL EXPERIMENTS COMPLETE"
success "Total runtime: $TOTAL_FMT"
echo ""

echo "Results are in:"
for SEED in $SEEDS; do
    echo "  ${LAMBDA_RESULTS}_seed${SEED}/"
done
echo ""

echo "To copy ALL results to your local machine, run locally:"
INSTANCE_IP="<your-instance-ip>"
for SEED in $SEEDS; do
    echo "  scp -r ubuntu@${INSTANCE_IP}:${LAMBDA_RESULTS}_seed${SEED} ./results_seed${SEED}"
done
echo ""

echo "IMPORTANT: Lambda Cloud instances are billed until terminated."
echo "Approximate cost for this run:"
echo "  A10  (~\$0.75/hr): 1 seed ~\$50-56, 3 seeds ~\$150-170"
echo "  A100 (~\$1.29/hr): 1 seed ~\$49-58, 3 seeds ~\$147-175"
echo "  H100 (~\$3.50/hr): 1 seed ~\$56-77, 3 seeds ~\$168-231"
echo ""
echo "Once you have copied your results, terminate the instance from"
echo "the Lambda Cloud dashboard to stop charges."
echo ""
