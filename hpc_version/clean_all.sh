#!/bin/bash
#
# Clean All Experimental Results
#
# This script removes all experimental data, results, and checkpoints,
# allowing you to rerun the entire pipeline from scratch.
#
# Usage:
#   ./clean_all.sh              # Interactive mode (asks for confirmation)
#   ./clean_all.sh --force      # Force mode (no confirmation)
#   ./clean_all.sh --keep-cache # Keep HuggingFace cache (don't re-download models)
#

set -e  # Exit on error

# Set paths
SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
PROJECT=${PROJECT:-/projects/$USER}
WORK_DIR="${SCRATCH}/mono_s2s_work"
RESULTS_DIR="${SCRATCH}/mono_s2s_results"
FINAL_RESULTS_DIR="${PROJECT}/mono_s2s_final_results"
HF_CACHE="${SCRATCH}/hf_cache"

# Parse arguments
FORCE=false
KEEP_CACHE=false

for arg in "$@"; do
    case $arg in
        --force|-f)
            FORCE=true
            shift
            ;;
        --keep-cache|-k)
            KEEP_CACHE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Clean all experimental results and data to rerun from scratch."
            echo ""
            echo "Options:"
            echo "  --force, -f        Skip confirmation prompt"
            echo "  --keep-cache, -k   Keep HuggingFace cache (don't re-download models/datasets)"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "What gets deleted:"
            echo "  - All completion flags"
            echo "  - All cached datasets"
            echo "  - All model checkpoints"
            echo "  - All evaluation results"
            echo "  - All experiment outputs"
            echo "  - HuggingFace cache (unless --keep-cache is used)"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "CLEAN ALL EXPERIMENTAL RESULTS"
echo "=========================================="
echo ""
echo "This will DELETE the following:"
echo ""
echo "  1. All stage flags (all stages)"
echo "     ${WORK_DIR}/**/*.flag"
echo ""
echo "  2. Cached datasets"
echo "     ${WORK_DIR}/data_cache/"
echo ""
echo "  3. Model checkpoints"
echo "     ${WORK_DIR}/checkpoints/"
echo ""
echo "  4. Evaluation results"
echo "     ${RESULTS_DIR}/"
echo ""
echo "  5. Final results (persistent)"
echo "     ${FINAL_RESULTS_DIR}/"
echo ""

if [ "$KEEP_CACHE" = false ]; then
    echo "  6. HuggingFace cache (models & datasets)"
    echo "     ${HF_CACHE}/"
    echo "     [NOTE: Use --keep-cache to preserve this]"
    echo ""
else
    echo "  6. HuggingFace cache"
    echo "     [SKIP] Keeping cache (--keep-cache flag used)"
    echo ""
fi

echo "  7. SLURM logs"
echo "     logs/job_*.out, logs/job_*.err"
echo ""

# Check if directories exist
FOUND_DATA=false
if [ -d "$WORK_DIR" ] || [ -d "$RESULTS_DIR" ] || [ -d "$FINAL_RESULTS_DIR" ] || [ -d "$HF_CACHE" ]; then
    FOUND_DATA=true
fi

if [ "$FOUND_DATA" = false ]; then
    echo "[INFO] No experimental data found. Nothing to clean."
    exit 0
fi

# Show disk usage before
echo "Current disk usage:"
if [ -d "$WORK_DIR" ]; then
    echo "  Work dir: $(du -sh $WORK_DIR 2>/dev/null | cut -f1)"
fi
if [ -d "$RESULTS_DIR" ]; then
    echo "  Results dir: $(du -sh $RESULTS_DIR 2>/dev/null | cut -f1)"
fi
if [ -d "$HF_CACHE" ] && [ "$KEEP_CACHE" = false ]; then
    echo "  HF cache: $(du -sh $HF_CACHE 2>/dev/null | cut -f1)"
fi
echo ""

# Confirmation
if [ "$FORCE" = false ]; then
    echo "=========================================="
    echo "[WARNING] This action cannot be undone!"
    echo "=========================================="
    echo ""
    read -p "Are you sure you want to delete all experimental data? (yes/no): " confirmation
    
    if [ "$confirmation" != "yes" ]; then
        echo ""
        echo "[CANCELLED] No files were deleted."
        exit 0
    fi
    echo ""
fi

echo "=========================================="
echo "CLEANING..."
echo "=========================================="
echo ""

# 1. Remove all flags (completion markers)
echo "[1/7] Removing all stage flags..."
if [ -d "$WORK_DIR" ]; then
    # Remove any flag files (not just *_complete.flag) anywhere under WORK_DIR.
    # This prevents `run_all.sh` from skipping stages due to stale markers.
    shopt -s nullglob globstar
    rm -f "${WORK_DIR}"/**/*.flag 2>/dev/null || true
    shopt -u nullglob globstar
    echo "  [SUCCESS] Stage flags removed"
else
    echo "  [SKIP] Work directory does not exist"
fi

# 2. Remove cached datasets
echo ""
echo "[2/7] Removing cached datasets..."
if [ -d "${WORK_DIR}/data_cache" ]; then
    rm -rf ${WORK_DIR}/data_cache
    echo "  [SUCCESS] Data cache removed"
else
    echo "  [SKIP] Data cache does not exist"
fi

# 3. Remove model checkpoints
echo ""
echo "[3/7] Removing model checkpoints..."
if [ -d "${WORK_DIR}/checkpoints" ]; then
    rm -rf ${WORK_DIR}/checkpoints
    echo "  [SUCCESS] Checkpoints removed"
else
    echo "  [SKIP] Checkpoints do not exist"
fi

# 4. Remove evaluation results
echo ""
echo "[4/7] Removing evaluation results..."
if [ -d "$RESULTS_DIR" ]; then
    rm -rf ${RESULTS_DIR}
    echo "  [SUCCESS] Results removed"
else
    echo "  [SKIP] Results directory does not exist"
fi

# 5. Remove final results
echo ""
echo "[5/7] Removing final results..."
if [ -d "$FINAL_RESULTS_DIR" ]; then
    rm -rf ${FINAL_RESULTS_DIR}
    echo "  [SUCCESS] Final results removed"
else
    echo "  [SKIP] Final results directory does not exist"
fi

# 6. Remove HuggingFace cache (optional)
echo ""
echo "[6/7] Removing HuggingFace cache..."
if [ "$KEEP_CACHE" = false ]; then
    if [ -d "$HF_CACHE" ]; then
        rm -rf ${HF_CACHE}
        echo "  [SUCCESS] HuggingFace cache removed"
    else
        echo "  [SKIP] HuggingFace cache does not exist"
    fi
else
    echo "  [SKIP] Keeping HuggingFace cache (--keep-cache flag)"
fi

# 7. Remove SLURM logs
echo ""
echo "[7/7] Removing SLURM logs..."
if [ -d "logs" ]; then
    rm -f logs/job_*.out logs/job_*.err 2>/dev/null || true
    echo "  [SUCCESS] SLURM logs removed"
else
    echo "  [SKIP] Logs directory does not exist"
fi

# Clean up empty work directory
if [ -d "$WORK_DIR" ]; then
    if [ -z "$(ls -A $WORK_DIR 2>/dev/null)" ]; then
        rmdir $WORK_DIR
        echo ""
        echo "[INFO] Removed empty work directory"
    fi
fi

echo ""
echo "=========================================="
echo "[SUCCESS] ALL DATA CLEANED!"
echo "=========================================="
echo ""
echo "You can now rerun the experiment from scratch:"
echo "  ./run_all.sh"
echo ""
echo "This will:"
echo "  - Download datasets (unless --keep-cache was used)"
echo "  - Train both models from scratch"
echo "  - Generate all evaluations and results"
echo ""

exit 0

