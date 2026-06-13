#!/bin/bash
#SBATCH --job-name=sweep_cell
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=77G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/%u/mono-s2s/foundation_llm_experiments/logs/sweep_cell_%j.out
#SBATCH --error=/projects/%u/mono-s2s/foundation_llm_experiments/logs/sweep_cell_%j.err
#SBATCH --signal=SIGUSR1@600

# Variant Sweep Cell Job
#
# Runs one (seed, variant) cell through stages 0-4 and 6 sequentially,
# using fast-mode training budget for rapid screening.
#
# Required env vars (passed via --export on sbatch command line):
#   EXPERIMENT_SEED     - random seed (e.g. 42)
#   MONOTONIC_VARIANT   - variant name (mlp_in, mlp_both, mlp_in_attn_out)
#
# Fast-mode training budget (overridable via OVERRIDE_* vars):
#   OVERRIDE_TRAINING_SAMPLES       default 30000
#   OVERRIDE_RECOVERY_EPOCHS        default 3
#   OVERRIDE_MONOTONIC_RECOVERY_EPOCHS  default 4
#   OVERRIDE_MAX_SEQ_LENGTH         default 1024
#   OVERRIDE_HOTFLIP_NUM_SAMPLES    default 200
#   OVERRIDE_QUICK_PILE_TEST_SIZE   default 2000
#
# Example submission:
#   sbatch --export=ALL,EXPERIMENT_SEED=42,MONOTONIC_VARIANT=mlp_both \
#          jobs/job_sweep_cell.sh

echo "======================================================================"
echo "SWEEP CELL: seed=${EXPERIMENT_SEED:-42} variant=${MONOTONIC_VARIANT:-mlp_both}"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "======================================================================"

module purge 2>/dev/null || true
module load cuda 2>/dev/null || true

CONDA_BASE="/projects/$USER/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate mono_s2s || {
    echo "ERROR: Failed to activate conda env. Run bootstrap_curc.sh."
    exit 1
}

# ============================================================================
# ENVIRONMENT
# ============================================================================

export EXPERIMENT_SEED="${EXPERIMENT_SEED:-42}"
export MONOTONIC_VARIANT="${MONOTONIC_VARIANT:-mlp_both}"
export PYTHONHASHSEED="$EXPERIMENT_SEED"
export CUBLAS_WORKSPACE_CONFIG=:16:8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SCRATCH="${SCRATCH:-/scratch/alpine/$USER}"
export PROJECT="${PROJECT:-/projects/$USER}"

export HF_HOME="$SCRATCH/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/huggingface_cache/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Isolated cell directories on SCRATCH
CELL_TAG="seed${EXPERIMENT_SEED}_${MONOTONIC_VARIANT}"
export LAMBDA_SEED_WORK="$SCRATCH/sweep_work_${CELL_TAG}"
export LAMBDA_SEED_RESULTS="$SCRATCH/sweep_results_${CELL_TAG}"
export LAMBDA_CACHE="$SCRATCH/foundation_llm_cache"
unset AZURE_WORK AZURE_RESULTS AZURE_CACHE

mkdir -p "${LAMBDA_SEED_WORK}/checkpoints/baseline_checkpoints"
mkdir -p "${LAMBDA_SEED_WORK}/checkpoints/monotonic_checkpoints"
mkdir -p "${LAMBDA_SEED_RESULTS}"

# Fast-mode training budget
export OVERRIDE_TRAINING_SAMPLES="${OVERRIDE_TRAINING_SAMPLES:-30000}"
export OVERRIDE_RECOVERY_EPOCHS="${OVERRIDE_RECOVERY_EPOCHS:-3}"
export OVERRIDE_MONOTONIC_RECOVERY_EPOCHS="${OVERRIDE_MONOTONIC_RECOVERY_EPOCHS:-4}"
export OVERRIDE_MAX_SEQ_LENGTH="${OVERRIDE_MAX_SEQ_LENGTH:-1024}"
export OVERRIDE_HOTFLIP_NUM_SAMPLES="${OVERRIDE_HOTFLIP_NUM_SAMPLES:-200}"
export OVERRIDE_QUICK_PILE_TEST_SIZE="${OVERRIDE_QUICK_PILE_TEST_SIZE:-2000}"

echo ""
echo "Cell:    $CELL_TAG"
echo "Work:    $LAMBDA_SEED_WORK"
echo "Results: $LAMBDA_SEED_RESULTS"
echo "Budget:  ${OVERRIDE_TRAINING_SAMPLES} samples / ${OVERRIDE_RECOVERY_EPOCHS} baseline epochs / ${OVERRIDE_MONOTONIC_RECOVERY_EPOCHS} monotonic epochs / seqlen ${OVERRIDE_MAX_SEQ_LENGTH}"
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
echo ""

# ============================================================================
# STAGE HELPERS
# ============================================================================

flag() { echo "${LAMBDA_SEED_WORK}/stage_${1}_complete.flag"; }
stage_done() { [ -f "$(flag "$1")" ]; }

cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(dirname "$0")/.."
cd scripts || { echo "ERROR: cannot find scripts/"; exit 1; }

# ============================================================================
# STAGES 0 -> 4 -> 6  (skip stage 5 UAT in sweep mode)
# ============================================================================

run_stage() {
    local NUM="$1" SCRIPT="$2" LABEL="$3"
    if stage_done "$NUM"; then
        echo "[Stage $NUM] $LABEL - already complete, skipping"
        return 0
    fi
    echo ""
    echo "[Stage $NUM] $LABEL ..."
    python "$SCRIPT"
    local EC=$?
    if [ $EC -ne 0 ]; then
        echo "[Stage $NUM] FAILED (exit $EC) - cell $CELL_TAG"
        exit $EC
    fi
    echo "[Stage $NUM] $LABEL - DONE"
}

run_stage "0_setup"              stage_0_setup.py               "Setup"
run_stage "1_apply_monotonicity" stage_1_apply_monotonicity.py  "Apply monotonicity ($MONOTONIC_VARIANT)"
run_stage "2_train_baseline"     stage_2_train_baseline.py      "Train baseline (${OVERRIDE_TRAINING_SAMPLES} samples, ${OVERRIDE_RECOVERY_EPOCHS} epochs)"
run_stage "3_train_monotonic"    stage_3_train_monotonic.py     "Train monotonic (${OVERRIDE_TRAINING_SAMPLES} samples, ${OVERRIDE_MONOTONIC_RECOVERY_EPOCHS} epochs)"
run_stage "4_evaluate"           stage_4_evaluate.py            "Evaluate perplexity"
run_stage "6_hotflip"            stage_6_hotflip_attacks.py     "HotFlip attacks"

echo ""
echo "======================================================================"
echo "CELL COMPLETE: $CELL_TAG"
echo "Results: $LAMBDA_SEED_RESULTS"
echo "Ended: $(date)"
echo "======================================================================"
