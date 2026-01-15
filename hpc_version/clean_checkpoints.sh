#!/bin/bash
#
# Clean ONLY Model Checkpoints
#
# This script removes all model checkpoints (both baseline and monotonic)
# while preserving datasets and other cached data. Useful for retraining
# models without re-downloading datasets.
#
# Usage:
#   ./clean_checkpoints.sh              # Interactive mode
#   ./clean_checkpoints.sh --force      # No confirmation
#   ./clean_checkpoints.sh --baseline   # Only baseline checkpoints
#   ./clean_checkpoints.sh --monotonic  # Only monotonic checkpoints
#

set -e

# Set paths
SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
WORK_DIR="${SCRATCH}/mono_s2s_work"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
BASELINE_DIR="${CHECKPOINT_DIR}/baseline_checkpoints"
MONOTONIC_DIR="${CHECKPOINT_DIR}/monotonic_checkpoints"

# Parse arguments
FORCE=false
CLEAN_BASELINE=true
CLEAN_MONOTONIC=true

for arg in "$@"; do
    case $arg in
        --force|-f)
            FORCE=true
            shift
            ;;
        --baseline|-b)
            CLEAN_MONOTONIC=false
            shift
            ;;
        --monotonic|-m)
            CLEAN_BASELINE=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Remove model checkpoints to retrain models from scratch."
            echo "Preserves datasets and cached data."
            echo ""
            echo "Options:"
            echo "  --force, -f        Skip confirmation prompt"
            echo "  --baseline, -b     Only delete baseline checkpoints"
            echo "  --monotonic, -m    Only delete monotonic checkpoints"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Delete all checkpoints (both models)"
            echo "  $0 --baseline       # Delete only baseline checkpoints"
            echo "  $0 --monotonic -f   # Delete only monotonic (no confirmation)"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "CLEAN MODEL CHECKPOINTS"
echo "=========================================="
echo ""

# Check what exists
BASELINE_EXISTS=false
MONOTONIC_EXISTS=false
BASELINE_COUNT=0
MONOTONIC_COUNT=0

if [ -d "$BASELINE_DIR" ]; then
    BASELINE_EXISTS=true
    BASELINE_COUNT=$(ls "$BASELINE_DIR"/*.pt 2>/dev/null | wc -l)
fi

if [ -d "$MONOTONIC_DIR" ]; then
    MONOTONIC_EXISTS=true
    MONOTONIC_COUNT=$(ls "$MONOTONIC_DIR"/*.pt 2>/dev/null | wc -l)
fi

# Show what will be deleted
echo "This will DELETE:"
echo ""

if [ "$CLEAN_BASELINE" = true ]; then
    if [ "$BASELINE_EXISTS" = true ]; then
        echo "  ✓ Baseline checkpoints: $BASELINE_COUNT files"
        echo "    $BASELINE_DIR"
        du -sh "$BASELINE_DIR" 2>/dev/null | sed 's/^/    Size: /'
    else
        echo "  - Baseline checkpoints: (none exist)"
    fi
else
    echo "  - Baseline checkpoints: (keeping - not selected)"
fi

echo ""

if [ "$CLEAN_MONOTONIC" = true ]; then
    if [ "$MONOTONIC_EXISTS" = true ]; then
        echo "  ✓ Monotonic checkpoints: $MONOTONIC_COUNT files"
        echo "    $MONOTONIC_DIR"
        du -sh "$MONOTONIC_DIR" 2>/dev/null | sed 's/^/    Size: /'
    else
        echo "  - Monotonic checkpoints: (none exist)"
    fi
else
    echo "  - Monotonic checkpoints: (keeping - not selected)"
fi

echo ""

# Check if anything to delete
if [ "$CLEAN_BASELINE" = true ] && [ "$BASELINE_EXISTS" = false ] && \
   [ "$CLEAN_MONOTONIC" = true ] && [ "$MONOTONIC_EXISTS" = false ]; then
    echo "[INFO] No checkpoints found. Nothing to clean."
    exit 0
fi

# Confirmation
if [ "$FORCE" = false ]; then
    echo "=========================================="
    echo "[WARNING] This will delete training progress!"
    echo "You will need to retrain from scratch."
    echo "=========================================="
    echo ""
    read -p "Are you sure you want to delete these checkpoints? (yes/no): " confirmation
    
    if [ "$confirmation" != "yes" ]; then
        echo ""
        echo "[CANCELLED] No checkpoints were deleted."
        exit 0
    fi
    echo ""
fi

echo "=========================================="
echo "CLEANING CHECKPOINTS..."
echo "=========================================="
echo ""

# Delete baseline checkpoints
if [ "$CLEAN_BASELINE" = true ]; then
    echo "[1/3] Removing baseline checkpoints..."
    if [ "$BASELINE_EXISTS" = true ]; then
        rm -rf "$BASELINE_DIR"
        rm -f "${WORK_DIR}/stage_2_train_baseline_complete.flag" 2>/dev/null || true
        echo "  [SUCCESS] Baseline checkpoints deleted"
        echo "  [SUCCESS] Removed baseline completion flag"
    else
        echo "  [SKIP] No baseline checkpoints to delete"
    fi
fi

# Delete monotonic checkpoints
echo ""
if [ "$CLEAN_MONOTONIC" = true ]; then
    echo "[2/3] Removing monotonic checkpoints..."
    if [ "$MONOTONIC_EXISTS" = true ]; then
        rm -rf "$MONOTONIC_DIR"
        rm -f "${WORK_DIR}/stage_3_train_monotonic_complete.flag" 2>/dev/null || true
        echo "  [SUCCESS] Monotonic checkpoints deleted"
        echo "  [SUCCESS] Removed monotonic completion flag"
    else
        echo "  [SKIP] No monotonic checkpoints to delete"
    fi
fi

# Clean up empty checkpoint directory
echo ""
echo "[3/3] Cleaning up empty directories..."
if [ -d "$CHECKPOINT_DIR" ]; then
    if [ -z "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
        rmdir "$CHECKPOINT_DIR"
        echo "  [SUCCESS] Removed empty checkpoint directory"
    else
        echo "  [INFO] Checkpoint directory still contains files (not removed)"
    fi
else
    echo "  [SKIP] No checkpoint directory"
fi

echo ""
echo "=========================================="
echo "[SUCCESS] CHECKPOINTS CLEANED!"
echo "=========================================="
echo ""
echo "Verification:"
if [ "$CLEAN_BASELINE" = true ]; then
    echo "  Baseline:   $([ -d "$BASELINE_DIR" ] && echo "⚠️  Still exists!" || echo "✓ Deleted")"
fi
if [ "$CLEAN_MONOTONIC" = true ]; then
    echo "  Monotonic:  $([ -d "$MONOTONIC_DIR" ] && echo "⚠️  Still exists!" || echo "✓ Deleted")"
fi
echo ""
echo "Preserved (not deleted):"
echo "  - Cached datasets (stage 1)"
echo "  - Evaluation results (stage 4+)"
echo "  - HuggingFace cache"
echo ""
echo "To retrain models:"
if [ "$CLEAN_BASELINE" = true ] && [ "$CLEAN_MONOTONIC" = true ]; then
    echo "  ./run_all.sh        # Rerun full pipeline"
    echo "  # OR submit individual stages:"
    echo "  sbatch jobs/job_2_baseline.sh"
    echo "  sbatch jobs/job_3_monotonic.sh"
elif [ "$CLEAN_BASELINE" = true ]; then
    echo "  sbatch jobs/job_2_baseline.sh"
elif [ "$CLEAN_MONOTONIC" = true ]; then
    echo "  sbatch jobs/job_3_monotonic.sh"
fi
echo ""

exit 0
