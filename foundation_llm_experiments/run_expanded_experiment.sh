#!/bin/bash
################################################################################
# Expanded Multi-Seed Experiment Runner
# Foundation LLM Monotonicity Experiments
#
# PURPOSE:
#   Runs the full monotonicity experiment pipeline across multiple random seeds
#   to produce statistically robust results for publication. Each seed is fully
#   independent: separate model checkpoints, results, and logs.
#
# WHAT THIS EXPERIMENT DOES:
#   1. Downloads Pythia-1.4B (EleutherAI's 1.4 billion parameter LLM)
#   2. Applies softplus parametrization to FFN input projections (W >= 0 constraint)
#      - Only constrains dense_h_to_4h layers (24 of 48 FFN weight matrices)
#      - Attention layers and FFN output projections remain unconstrained
#   3. Recovery-trains the monotonic model on 100K samples from The Pile
#      - 10 epochs, LR=5e-5, warmup=20%, bfloat16 + gradient checkpointing
#   4. Trains an unconstrained baseline with identical hyperparameters
#      - 5 epochs, LR=1e-5, same data and batch settings for fair comparison
#   5. Evaluates both models on held-out Pile test data
#   6. Runs Universal Adversarial Trigger (UAT) attacks on both models
#      - Optimizes a 10-token trigger prefix that maximizes perplexity increase
#   7. Runs HotFlip gradient-based attacks on both models
#   8. Aggregates all results into a single JSON summary per seed
#
# PIPELINE STAGES (SLURM jobs with dependencies):
#
#   [Stage 0: Setup]
#        |
#        +---> [Stage 1: Apply Monotonicity] ---> [Stage 3: Monotonic Training]
#        |                                                    |
#        +---> [Stage 2: Baseline Training] -----------------+
#                                                             |
#                                      +----+----+-----------+
#                                      |    |    |
#                               [Stage 4] [5] [Stage 6]
#                               Evaluate  UAT  HotFlip
#                                      |    |    |
#                                      +----+----+
#                                           |
#                                    [Stage 7: Aggregate]
#
# USAGE:
#   # Run with default seeds (42, 1337, 2024)
#   ./run_expanded_experiment.sh
#
#   # Run with custom seeds
#   SEEDS="42 1337 2024 8888" ./run_expanded_experiment.sh
#
#   # Run a single seed
#   SEEDS="42" ./run_expanded_experiment.sh
#
#   # Dry run (show what would be submitted, don't actually submit)
#   DRY_RUN=1 ./run_expanded_experiment.sh
#
# REQUIREMENTS:
#   - CURC Alpine HPC cluster account
#   - Access to aa100 partition
#   - ~250GB /scratch storage per seed (models + checkpoints)
#   - ~5GB /projects storage (conda environment)
#   - Run this script from the foundation_llm_experiments/ directory
#
# EXPECTED RUNTIME (per seed, approximate):
#   Stage 0 (Setup):                  ~5 minutes
#   Stage 1 (Apply Monotonicity):     ~5 minutes
#   Stage 2 (Baseline Training):      ~23 hours (5 epochs x 90K samples)
#   Stage 3 (Monotonic Training):     ~12 days total / ~24h per resubmission
#                                     (10 epochs, auto-resumes from checkpoints)
#   Stage 4 (Evaluation):             ~2 hours
#   Stage 5 (UAT Attacks):            ~4-5 hours
#   Stage 6 (HotFlip Attacks):        ~2 hours
#   Stage 7 (Aggregate):              ~5 minutes
#   -------------------------------------------------------------------------
#   Total per seed:                   ~14-16 days (due to 24h SLURM wall-time)
#   Total for 3 seeds (parallel):     ~14-16 days
#
# CHECKPOINT RESUME:
#   Training stages (2 and 3) checkpoint every 500 steps AND every 30 minutes.
#   When SLURM kills a job at the 24h wall-time limit, the next submission
#   automatically resumes from the latest checkpoint. No data is lost.
#
# RESULTS LOCATION:
#   /scratch/alpine/$USER/foundation_llm_work_seed{SEED}/
#     checkpoints/
#       baseline_checkpoints/best_model.pt
#       monotonic_checkpoints/best_model.pt
#   /scratch/alpine/$USER/foundation_llm_results_seed{SEED}/
#     baseline_training_history.json
#     monotonic_training_history.json
#     evaluation_results.json
#     uat_results.json
#     hotflip_results.json
#     final_summary.json
#
# MONITORING:
#   squeue -u $USER                          # See running/pending jobs
#   squeue -u $USER --start                  # See estimated start times
#   tail -f /scratch/alpine/$USER/foundation_llm_work_seed42/stage_logs/stage_3_train_monotonic.log
#
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

# Seeds to run (space-separated). Override with SEEDS env var.
SEEDS="${SEEDS:-42 1337 2024}"

# Dry run mode - set DRY_RUN=1 to preview without submitting
DRY_RUN="${DRY_RUN:-0}"

# Conda environment name and base path
CONDA_BASE="/projects/$USER/miniconda3"
ENV_NAME="mono_s2s"

# ============================================================================
# HELPERS
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}======================================================================${NC}"; \
            echo -e "${BOLD}${BLUE} $*${NC}"; \
            echo -e "${BOLD}${BLUE}======================================================================${NC}"; }

submit_job() {
    local label="$1"
    local script="$2"
    local seed="$3"
    shift 3
    local deps=("$@")

    local dep_str=""
    for dep in "${deps[@]}"; do
        if [ -n "$dep_str" ]; then
            dep_str="${dep_str},afterok:${dep}"
        else
            dep_str="afterok:${dep}"
        fi
    done

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] Would submit: sbatch${dep_str:+ --dependency=$dep_str} --export=ALL,EXPERIMENT_SEED=$seed $script"
        echo "99999999"
        return 0
    fi

    local cmd="sbatch --parsable"
    [ -n "$dep_str" ] && cmd="$cmd --dependency=$dep_str"
    cmd="$cmd --export=ALL,EXPERIMENT_SEED=$seed $script"

    local result
    result=$(eval "$cmd" 2>&1 | grep -oE '[0-9]{7,}' | tail -1)

    if [ -z "$result" ]; then
        error "Failed to submit $label for seed $seed"
        error "Command: $cmd"
        return 1
    fi

    echo "$result"
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

header "FOUNDATION LLM MONOTONICITY EXPERIMENTS - EXPANDED MULTI-SEED RUN"

# Check we're in the right directory
if [ ! -d "jobs" ] || [ ! -d "scripts" ]; then
    error "Please run this script from the foundation_llm_experiments/ directory"
    exit 1
fi

info "Working directory: $(pwd)"
info "User: $USER"
info "Seeds to run: $SEEDS"
info "Dry run: $DRY_RUN"
echo ""

# ============================================================================
# ENVIRONMENT CHECK / BOOTSTRAP
# ============================================================================

header "ENVIRONMENT CHECK"

if [ ! -f "$CONDA_BASE/bin/conda" ]; then
    warn "Conda not found at $CONDA_BASE"
    warn "Running bootstrap to install conda and dependencies..."
    echo ""

    if [ ! -f "bootstrap_curc.sh" ]; then
        error "bootstrap_curc.sh not found - cannot set up environment automatically"
        error "Please manually install miniconda to $CONDA_BASE and run:"
        error "  conda create -n $ENV_NAME python=3.10"
        error "  conda activate $ENV_NAME"
        error "  pip install -r requirements.txt"
        exit 1
    fi

    read -p "Run bootstrap_curc.sh to install environment? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        error "Environment setup required before running experiments."
        exit 1
    fi

    bash bootstrap_curc.sh

    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    success "Bootstrap complete"
else
    source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true

    if ! conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
        warn "Conda environment '$ENV_NAME' not found. Running bootstrap..."
        bash bootstrap_curc.sh
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
    else
        conda activate "$ENV_NAME" 2>/dev/null || true
        success "Conda: $CONDA_BASE"
        success "Python: $(which python 2>/dev/null || echo 'not found')"
        TORCH_VER=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')
        success "PyTorch: $TORCH_VER"

        if ! python -c "import torch" 2>/dev/null; then
            error "PyTorch is not installed in environment '$ENV_NAME'"
            error "Please run: conda activate $ENV_NAME && pip install -r requirements.txt"
            exit 1
        fi

        # Verify key dependencies
        for pkg in transformers datasets tqdm numpy; do
            if ! python -c "import $pkg" 2>/dev/null; then
                warn "Package '$pkg' not found - running pip install..."
                pip install -r requirements.txt
                break
            fi
        done
        success "All required packages found"
    fi
fi

# Check scratch space
SCRATCH_DIR="${SCRATCH:-/scratch/alpine/$USER}"
SCRATCH_AVAIL=$(df -BG "$SCRATCH_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')
NUM_SEEDS=$(echo $SEEDS | wc -w)
SCRATCH_NEEDED=$(( NUM_SEEDS * 250 ))

echo ""
info "Scratch directory: $SCRATCH_DIR"
info "Available scratch: ${SCRATCH_AVAIL}GB"
info "Estimated needed: ~${SCRATCH_NEEDED}GB (${NUM_SEEDS} seeds x ~250GB each)"

if [ "${SCRATCH_AVAIL:-0}" -lt "$SCRATCH_NEEDED" 2>/dev/null ]; then
    warn "May not have enough scratch space. Available: ${SCRATCH_AVAIL}GB, Needed: ~${SCRATCH_NEEDED}GB"
    warn "Consider running fewer seeds or cleaning up old scratch files"
fi

# Check projects quota
PROJECTS_AVAIL=$(df -BG "/projects/$USER" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')
info "Available /projects: ${PROJECTS_AVAIL}GB"

if [ "${PROJECTS_AVAIL:-99}" -lt "10" ] 2>/dev/null; then
    warn "/projects quota is low (${PROJECTS_AVAIL}GB). Conda env needs ~5GB."
    warn "Run: conda clean --all -y  to free space"
fi

# ============================================================================
# EXPERIMENT OVERVIEW
# ============================================================================

header "EXPERIMENT OVERVIEW"

echo "This run will execute the following experiment for each seed:"
echo ""
echo "  Model:         EleutherAI/pythia-1.4b (1.4 billion parameters)"
echo "  Constraint:    Softplus parametrization on FFN input projections"
echo "                 W = softplus(V) >= 0  for dense_h_to_4h layers only"
echo "                 (24 of 48 FFN weight matrices constrained)"
echo "  Training data: 100,000 samples from The Pile (monology/pile-uncopyrighted)"
echo "  Epochs:"
echo "    Baseline:    5 epochs  (LR=1e-5, warmup=10%)"
echo "    Monotonic:  10 epochs  (LR=5e-5, warmup=20%)"
echo "  Precision:     bfloat16 + gradient checkpointing"
echo "  Attacks:"
echo "    UAT:         10-token trigger, 50 iterations, 3 restarts, 100 candidates"
echo "    HotFlip:     10 flips per example, 200 test samples"
echo ""
echo "Seeds: $SEEDS"
echo ""
echo "Key hypothesis: Monotonic FFN constraints (W >= 0) should reduce"
echo "adversarial vulnerability (UAT/HotFlip effectiveness) while preserving"
echo "reasonable language modeling performance (perplexity gap < 4x baseline)."
echo ""

if [ "$DRY_RUN" = "1" ]; then
    warn "DRY RUN MODE - Jobs will be previewed but NOT submitted"
fi

# ============================================================================
# CONFIRM SUBMISSION
# ============================================================================

echo ""
read -p "Submit all jobs for seeds: $SEEDS ? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    info "Cancelled."
    exit 0
fi

# ============================================================================
# SUBMIT JOBS PER SEED
# ============================================================================

mkdir -p logs

declare -A ALL_FINAL_JOBS

for SEED in $SEEDS; do

    header "SUBMITTING SEED $SEED"

    # Stage 0: Setup
    JOB0=$(submit_job "Stage 0 (Setup)" "jobs/job_0_setup.sh" "$SEED")
    success "Stage 0 (Setup):              Job $JOB0"

    # Stage 1: Apply Monotonicity (depends on setup)
    JOB1=$(submit_job "Stage 1 (Monotonicity)" "jobs/job_1_monotonicity.sh" "$SEED" "$JOB0")
    success "Stage 1 (Apply Monotonicity): Job $JOB1  [after $JOB0]"

    # Stage 2: Baseline Training (depends on setup, runs in parallel with stage 1)
    JOB2=$(submit_job "Stage 2 (Baseline)" "jobs/job_2_baseline.sh" "$SEED" "$JOB0")
    success "Stage 2 (Baseline Training):  Job $JOB2  [after $JOB0]"

    # Stage 3: Monotonic Training (depends on stage 1)
    JOB3=$(submit_job "Stage 3 (Monotonic)" "jobs/job_3_monotonic.sh" "$SEED" "$JOB1")
    success "Stage 3 (Monotonic Training): Job $JOB3  [after $JOB1]"

    # Stage 4: Evaluation (depends on both training stages)
    JOB4=$(submit_job "Stage 4 (Evaluate)" "jobs/job_4_evaluate.sh" "$SEED" "$JOB2" "$JOB3")
    success "Stage 4 (Evaluation):         Job $JOB4  [after $JOB2, $JOB3]"

    # Stage 5: UAT attacks (depends on both training stages)
    JOB5=$(submit_job "Stage 5 (UAT)" "jobs/job_5_uat.sh" "$SEED" "$JOB2" "$JOB3")
    success "Stage 5 (UAT Attacks):        Job $JOB5  [after $JOB2, $JOB3]"

    # Stage 6: HotFlip attacks (depends on both training stages)
    JOB6=$(submit_job "Stage 6 (HotFlip)" "jobs/job_6_hotflip.sh" "$SEED" "$JOB2" "$JOB3")
    success "Stage 6 (HotFlip Attacks):    Job $JOB6  [after $JOB2, $JOB3]"

    # Stage 7: Aggregate (depends on eval + both attacks)
    JOB7=$(submit_job "Stage 7 (Aggregate)" "jobs/job_7_aggregate.sh" "$SEED" "$JOB4" "$JOB5" "$JOB6")
    success "Stage 7 (Aggregate):          Job $JOB7  [after $JOB4, $JOB5, $JOB6]"

    # Save job IDs for this seed
    echo "$JOB0 $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6 $JOB7" > ".job_ids_seed${SEED}"
    success "Job IDs saved to .job_ids_seed${SEED}"

    ALL_FINAL_JOBS[$SEED]="$JOB7"

done

# ============================================================================
# SUMMARY
# ============================================================================

header "ALL JOBS SUBMITTED"

echo "Job pipeline submitted for seeds: $SEEDS"
echo ""
echo "Current queue status:"
squeue -u $USER -o "%.10i %.9P %.30j %.8u %.2t %.10M %.10L %R" 2>/dev/null || true
echo ""

# Build scancel command for all jobs
ALL_JOB_IDS=""
for SEED in $SEEDS; do
    if [ -f ".job_ids_seed${SEED}" ]; then
        ALL_JOB_IDS="$ALL_JOB_IDS $(cat .job_ids_seed${SEED})"
    fi
done

echo ""
echo "======================================================================"
echo "MONITORING AND MANAGEMENT COMMANDS"
echo "======================================================================"
echo ""
echo "Check all job statuses:"
echo "  squeue -u \$USER"
echo ""
echo "Check estimated start times:"
echo "  squeue -u \$USER --start"
echo ""
echo "Monitor training progress (replace SEED and STAGE as needed):"
for SEED in $SEEDS; do
    echo "  tail -f \$SCRATCH/foundation_llm_work_seed${SEED}/stage_logs/stage_3_train_monotonic.log"
done
echo ""
echo "Check SLURM stderr for live training progress bars:"
echo "  tail -f logs/job_3_monotonic_<JOBID>.err"
echo ""
echo "Check results after completion:"
for SEED in $SEEDS; do
    echo "  cat \$SCRATCH/foundation_llm_results_seed${SEED}/evaluation_results.json"
done
echo ""
echo "Check /scratch quota:"
echo "  curc-quota"
echo ""
echo "Cancel ALL submitted jobs (emergency use only):"
echo "  scancel $ALL_JOB_IDS"
echo ""
echo "======================================================================"
echo "IMPORTANT NOTES FOR STAGE 3 (MONOTONIC TRAINING)"
echo "======================================================================"
echo ""
echo "Stage 3 requires ~10 days total and will timeout multiple times."
echo "The job is designed to checkpoint and RESUME AUTOMATICALLY."
echo ""
echo "When a Stage 3 job times out, you will see:"
echo "  sacct shows: TIMEOUT"
echo "  Stage 3 flag NOT created: stage_3_train_monotonic_complete.flag"
echo "  Checkpoints saved: checkpoint_epoch_N.pt"
echo ""
echo "To continue training after a timeout, simply resubmit:"
echo "  sbatch --export=ALL,EXPERIMENT_SEED=<SEED> jobs/job_3_monotonic.sh"
echo ""
echo "The job will automatically detect the latest checkpoint and resume."
echo "You can verify resumption in the log:"
echo "  grep 'Resuming from epoch' \$SCRATCH/foundation_llm_work_seed<SEED>/stage_logs/stage_3_train_monotonic.log"
echo ""
echo "Stage 3 is complete when the flag file exists:"
echo "  ls \$SCRATCH/foundation_llm_work_seed<SEED>/stage_3_train_monotonic_complete.flag"
echo ""
echo "After Stage 3 completes, manually resubmit the downstream stages:"
echo "  sbatch --export=ALL,EXPERIMENT_SEED=<SEED> jobs/job_4_evaluate.sh"
echo "  sbatch --export=ALL,EXPERIMENT_SEED=<SEED> jobs/job_5_uat.sh"
echo "  sbatch --export=ALL,EXPERIMENT_SEED=<SEED> jobs/job_6_hotflip.sh"
echo ""
echo "======================================================================"
echo "EXPECTED RESULTS (based on seed 42 pilot run)"
echo "======================================================================"
echo ""
echo "  Baseline Pythia-1.4B perplexity (Pile test):  ~6.9"
echo "  Monotonic Pythia-1.4B perplexity (Pile test): ~27.6  (4x gap)"
echo "  UAT attack: monotonic model expected to show reduced trigger"
echo "              effectiveness vs baseline"
echo "  HotFlip: similar adversarial robustness pattern expected"
echo ""
echo "Publication target: NeurIPS / ICML / ICLR (see documentation/)"
echo ""
echo "======================================================================"
