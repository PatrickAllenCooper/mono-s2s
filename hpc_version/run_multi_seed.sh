#!/bin/bash
#
# Multi-Seed Orchestration Script for Mono-S2S HPC Experiments
#
# This script runs the full pipeline for multiple random seeds to capture
# training variance, then aggregates results across seeds.
#
# Usage:
#   ./run_multi_seed.sh           # Run with all 5 default seeds
#   ./run_multi_seed.sh 3         # Run with first 3 seeds only
#
# Seeds: 42, 1337, 2024, 8888, 12345 (defined in experiment_config.py)
#

set -e  # Exit on error

# CRITICAL: Set SCRATCH and PROJECT environment variables
# On CURC Alpine, these are set automatically on compute nodes but NOT on login nodes
# We need them on the login node for checking completion flags
if [ -z "$SCRATCH" ]; then
    export SCRATCH="/scratch/alpine/$USER"
    echo "[INFO] SCRATCH not set, using default: $SCRATCH"
fi

if [ -z "$PROJECT" ]; then
    export PROJECT="/projects/$USER"
    echo "[INFO] PROJECT not set, using default: $PROJECT"
fi

# Verify directories exist
if [ ! -d "$SCRATCH" ]; then
    echo "[ERROR] SCRATCH directory does not exist: $SCRATCH"
    echo "        Please check your Alpine filesystem access"
    exit 1
fi

# Configuration
ALL_SEEDS=(42 1337 2024 8888 12345)
NUM_SEEDS=${1:-${#ALL_SEEDS[@]}}  # Default: all seeds

# Validate
if [ $NUM_SEEDS -gt ${#ALL_SEEDS[@]} ]; then
    echo "ERROR: Requested $NUM_SEEDS seeds but only ${#ALL_SEEDS[@]} available"
    exit 1
fi

# Select seeds to run
SEEDS=("${ALL_SEEDS[@]:0:$NUM_SEEDS}")

echo "=========================================="
echo "Mono-S2S Multi-Seed Experiment Orchestrator"
echo "=========================================="
echo "Running ${#SEEDS[@]} seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo ""

# Track results
SUCCESSFUL_SEEDS=()
FAILED_SEEDS=()

# Run each seed
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "STARTING SEED: $SEED"
    echo "=========================================="
    echo ""
    
    # Run the full pipeline for this seed
    if ./run_all.sh $SEED; then
        echo ""
        echo "[SUCCESS] Seed $SEED completed successfully"
        SUCCESSFUL_SEEDS+=($SEED)
    else
        echo ""
        echo "[FAILED] Seed $SEED failed"
        FAILED_SEEDS+=($SEED)
        # Continue with other seeds even if one fails
    fi
done

echo ""
echo "=========================================="
echo "MULTI-SEED EXECUTION COMPLETE"
echo "=========================================="
echo "Successful seeds: ${SUCCESSFUL_SEEDS[*]:-none}"
echo "Failed seeds: ${FAILED_SEEDS[*]:-none}"
echo ""

# Check if we have enough seeds for statistical analysis
MIN_SEEDS=3
if [ ${#SUCCESSFUL_SEEDS[@]} -lt $MIN_SEEDS ]; then
    echo "[WARNING] Only ${#SUCCESSFUL_SEEDS[@]} seeds completed successfully."
    echo "          Need at least $MIN_SEEDS for cross-seed statistical tests."
    echo ""
fi

# Aggregate results across seeds
if [ ${#SUCCESSFUL_SEEDS[@]} -ge 2 ]; then
    echo "Aggregating results across seeds..."
    echo ""
    
    # Create aggregation directory (SCRATCH and PROJECT already set above)
    AGGREGATE_DIR="${PROJECT}/mono_s2s_multi_seed_results"
    mkdir -p "$AGGREGATE_DIR"
    
    # Run aggregation script
    cd scripts
    python aggregate_multi_seed.py --seeds "${SUCCESSFUL_SEEDS[@]}" --output "$AGGREGATE_DIR"
    cd ..
    
    echo ""
    echo "Multi-seed results saved to: $AGGREGATE_DIR"
else
    echo "[SKIP] Not enough successful seeds to aggregate"
fi

echo ""
echo "=========================================="
echo "Ended: $(date)"
echo "=========================================="

# Exit with error if any seeds failed
if [ ${#FAILED_SEEDS[@]} -gt 0 ]; then
    exit 1
fi

exit 0
