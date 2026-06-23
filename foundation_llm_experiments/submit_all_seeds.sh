#!/bin/bash
################################################################################
# Multi-Seed Submission Script for CURC Alpine
#
# Submits the full 8-stage pipeline for each of three seeds (42, 1337, 2024)
# with proper SLURM dependencies. All seeds run concurrently on separate GPUs.
# Each seed occupies ~75 GPU-hours on the aa100 partition.
#
# Usage:
#   cd /projects/$USER/mono-s2s/foundation_llm_experiments
#   ./submit_all_seeds.sh                         # all three seeds
#   ./submit_all_seeds.sh 42                      # single seed
#   MONOTONIC_VARIANT=mlp_in_attn_out ./submit_all_seeds.sh  # override variant
#
# Prerequisites:
#   conda activate mono_s2s  (run bootstrap_curc.sh if not done yet)
#
# After submission, monitor with:
#   squeue -u $USER --format="%.10i %.20j %.8T %.10M %.9l %R"
#   tail -f logs/job_2_baseline_<JOBID>.out
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# CONFIGURATION
# ============================================================================

SEEDS=("${@:-42 1337 2024}")
# If arguments given as separate args, use them; otherwise use defaults
if [ $# -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(42 1337 2024)
fi

# Monotonic variant: mlp_both is the Phase-A validated choice for Pythia-1.4B.
# Override with: MONOTONIC_VARIANT=mlp_in_attn_out ./submit_all_seeds.sh
VARIANT="${MONOTONIC_VARIANT:-mlp_both}"

echo "======================================================================"
echo "FOUNDATION LLM EXPERIMENTS - MULTI-SEED SUBMISSION"
echo "======================================================================"
echo ""
echo "Seeds:   ${SEEDS[*]}"
echo "Variant: $VARIANT"
echo "Partition: aa100"
echo ""
echo "Pipeline stages per seed:"
echo "  Stage 0: Setup            (~1h)"
echo "  Stage 1: Apply monotonicity (~0.5h, depends on 0)"
echo "  Stage 2: Train baseline   (~24h, depends on 0)"
echo "  Stage 3: Train monotonic  (~32h, depends on 1)"
echo "  Stage 4: Evaluate         (~2h, depends on 2+3)"
echo "  Stage 5: UAT attacks      (~6h, depends on 2+3)"
echo "  Stage 6: HotFlip attacks  (~4h, depends on 2+3)"
echo "  Stage 7: Aggregate        (~0.5h, depends on 4+5+6)"
echo ""
echo "Total wall time per seed: ~60-70h  |  GPU-hours per seed: ~75"
echo ""

# ============================================================================
# CHECKS
# ============================================================================

if [ ! -d "jobs" ] || [ ! -d "scripts" ]; then
    echo "ERROR: Run from foundation_llm_experiments/ directory"
    exit 1
fi

mkdir -p logs

# Verify conda environment
CONDA_BASE="/projects/$USER/miniconda3"
if [ ! -f "$CONDA_BASE/bin/conda" ]; then
    echo "ERROR: Conda not found at $CONDA_BASE"
    echo "Run: bash bootstrap_curc.sh"
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mono_s2s 2>/dev/null || {
    echo "ERROR: mono_s2s environment not found. Run: bash bootstrap_curc.sh"
    exit 1
}

# ============================================================================
# PROMPT
# ============================================================================

echo "This will submit $((${#SEEDS[@]} * 8)) SLURM jobs."
read -p "Proceed? (y/N) " -n 1 -r; echo ""
[[ ! $REPLY =~ ^[Yy]$ ]] && { echo "Cancelled."; exit 0; }
echo ""

# ============================================================================
# SUBMISSION FUNCTION
# ============================================================================

submit_seed() {
    local SEED="$1"
    echo "----------------------------------------------------------------------"
    echo "Submitting seed $SEED (variant=$VARIANT)..."
    echo "----------------------------------------------------------------------"

    local SBATCH_ARGS="--export=ALL,EXPERIMENT_SEED=$SEED,MONOTONIC_VARIANT=$VARIANT,OVERRIDE_UAT_MAX_SAMPLES=200"

    # Stage 0: setup (no dependency)
    local J0
    J0=$(sbatch --parsable $SBATCH_ARGS \
        --job-name="foundation_s0_seed${SEED}" \
        jobs/job_0_setup.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    echo "  Stage 0 (Setup):            Job $J0"

    # Stage 1: apply monotonicity (depends on 0)
    local J1
    J1=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterok:$J0 \
        --job-name="foundation_s1_seed${SEED}" \
        jobs/job_1_monotonicity.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    echo "  Stage 1 (Apply Monoton.):   Job $J1  [after $J0]"

    # Stage 2: baseline training (depends on 0, parallel with 1)
    local J2
    J2=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterok:$J0 \
        --job-name="foundation_s2_seed${SEED}" \
        jobs/job_2_baseline.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    echo "  Stage 2 (Baseline Train):   Job $J2  [after $J0]"

    # Stage 3: monotonic training (depends on 1)
    local J3
    J3=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterok:$J1 \
        --job-name="foundation_s3_seed${SEED}" \
        jobs/job_3_monotonic.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    echo "  Stage 3 (Monotonic Train):  Job $J3  [after $J1]"

    # Stage 4: evaluate (depends on 2 and 3)
    local J4
    J4=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterok:$J2,afterok:$J3 \
        --job-name="foundation_s4_seed${SEED}" \
        jobs/job_4_evaluate.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    echo "  Stage 4 (Evaluate):         Job $J4  [after $J2,$J3]"

    # Stage 5: UAT attacks (depends on 2 and 3, parallel with 6)
    local J5
    J5=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterok:$J2,afterok:$J3 \
        --job-name="foundation_s5_seed${SEED}" \
        jobs/job_5_uat.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    echo "  Stage 5 (UAT Attacks):      Job $J5  [after $J2,$J3]"

    # Stage 6: HotFlip attacks (depends on 2 and 3, parallel with 5)
    local J6
    J6=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterok:$J2,afterok:$J3 \
        --job-name="foundation_s6_seed${SEED}" \
        jobs/job_6_hotflip.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    echo "  Stage 6 (HotFlip Attacks):  Job $J6  [after $J2,$J3]"

    # Stage 7: aggregate (depends on 4, 5, 6)
    local J7
    J7=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterok:$J4,afterok:$J5,afterok:$J6 \
        --job-name="foundation_s7_seed${SEED}" \
        jobs/job_7_aggregate.sh 2>&1 | grep -oE '[0-9]{7,}' | tail -1)
    echo "  Stage 7 (Aggregate):        Job $J7  [after $J4,$J5,$J6]"

    # Save job IDs for this seed
    echo "$J0 $J1 $J2 $J3 $J4 $J5 $J6 $J7" > ".job_ids_seed${SEED}"
    echo ""
    echo "  Seed $SEED job IDs saved to .job_ids_seed${SEED}"
    echo ""
}

# ============================================================================
# SUBMIT ALL SEEDS
# ============================================================================

ALL_JOBS=()

for SEED in "${SEEDS[@]}"; do
    submit_seed "$SEED"
    if [ -f ".job_ids_seed${SEED}" ]; then
        read -ra SEED_JOBS < ".job_ids_seed${SEED}"
        ALL_JOBS+=("${SEED_JOBS[@]}")
    fi
done

echo "======================================================================"
echo "All seeds submitted."
echo "======================================================================"
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER --format=\"%.10i %.25j %.8T %.10M %.9l %R\""
echo ""
echo "Watch a specific log (replace JOBID):"
echo "  tail -f logs/job_2_baseline_JOBID.out"
echo ""
echo "Check stage completion flags:"
for SEED in "${SEEDS[@]}"; do
    echo "  ls \$SCRATCH/foundation_llm_work_seed${SEED}/stage_*complete.flag"
done
echo ""
echo "Cancel everything:"
echo "  scancel ${ALL_JOBS[*]}"
echo ""

# Save all job IDs
printf '%s\n' "${ALL_JOBS[@]}" | tr '\n' ' ' > .job_ids_all
echo "All job IDs saved to .job_ids_all"
echo ""

# Offer to start the monitor script
read -p "Start background monitor (auto-resubmit on timeout)? (Y/n) " -n 1 -r; echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    chmod +x monitor_and_resubmit.sh
    nohup ./monitor_and_resubmit.sh ${ALL_JOBS[*]} \
        > "logs/monitor_all_seeds.out" 2>&1 &
    echo "Monitor started (PID $!)"
    echo "Log: logs/monitor_all_seeds.out"
fi
