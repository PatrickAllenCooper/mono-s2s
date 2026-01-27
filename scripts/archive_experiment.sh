#!/bin/bash
################################################################################
# Archive Experiment Results to Project Directory
#
# Archives results and checkpoints from SCRATCH to PROJECT for long-term storage.
# Run this after experiment completes to preserve results.
#
# Usage:
#   bash scripts/archive_experiment.sh 42
#   bash scripts/archive_experiment.sh 42 --checkpoints  # Include checkpoints
################################################################################

SEED=${1:-42}
INCLUDE_CHECKPOINTS=${2}

echo "======================================================================"
echo "  ARCHIVING EXPERIMENT RESULTS"
echo "======================================================================"
echo ""
echo "Seed: $SEED"
echo "Include checkpoints: ${INCLUDE_CHECKPOINTS:-no}"
echo ""

# Directories
SCRATCH_WORK="${SCRATCH}/mono_s2s_work"
SCRATCH_RESULTS="${SCRATCH}/mono_s2s_results"
PROJECT_ARCHIVE="${PROJECT}/mono_s2s_final_results/seed_${SEED}"

# Check source exists
if [ ! -d "$SCRATCH_RESULTS" ]; then
    echo "ERROR: Results directory not found: $SCRATCH_RESULTS"
    echo "Make sure experiment has completed"
    exit 1
fi

# Create archive directory
mkdir -p "$PROJECT_ARCHIVE"

echo "Archiving results from SCRATCH to PROJECT..."
echo "  Source: $SCRATCH_RESULTS"
echo "  Destination: $PROJECT_ARCHIVE"
echo ""

# Copy result files (small JSON/CSV/TXT files)
echo "Copying result files..."
for file in setup_complete.json \
            data_statistics.json \
            baseline_training_history.json \
            monotonic_training_history.json \
            evaluation_results.json \
            uat_results.json \
            hotflip_results.json \
            final_results.json \
            experiment_summary.txt \
            learned_triggers.csv; do
    
    if [ -f "$SCRATCH_RESULTS/$file" ]; then
        cp "$SCRATCH_RESULTS/$file" "$PROJECT_ARCHIVE/"
        echo "  ✓ $file"
    else
        echo "  ⚠️  Missing: $file"
    fi
done

# Copy stage logs
echo ""
echo "Copying stage logs..."
if [ -d "$SCRATCH_WORK/stage_logs" ]; then
    cp -r "$SCRATCH_WORK/stage_logs" "$PROJECT_ARCHIVE/"
    echo "  ✓ stage_logs/"
fi

# Create metadata
echo ""
echo "Creating archive metadata..."
cat > "$PROJECT_ARCHIVE/archive_metadata.json" <<EOF
{
  "archived_date": "$(date -Iseconds)",
  "seed": $SEED,
  "source_scratch": "$SCRATCH_RESULTS",
  "archived_to": "$PROJECT_ARCHIVE",
  "checkpoints_included": ${INCLUDE_CHECKPOINTS:-false},
  "archived_by": "$USER",
  "hostname": "$(hostname)"
}
EOF
echo "  ✓ archive_metadata.json"

# Optionally archive checkpoints (large!)
if [ "$INCLUDE_CHECKPOINTS" == "--checkpoints" ]; then
    echo ""
    echo "Archiving checkpoints (this may take a while)..."
    
    CHECKPOINT_ARCHIVE="$PROJECT_ARCHIVE/checkpoints_seed${SEED}.tar.gz"
    
    if [ -d "$SCRATCH_WORK/checkpoints" ]; then
        tar -czf "$CHECKPOINT_ARCHIVE" \
            -C "$SCRATCH_WORK" \
            checkpoints/baseline_checkpoints/best_model.pt \
            checkpoints/monotonic_checkpoints/best_model.pt \
            2>/dev/null
        
        if [ -f "$CHECKPOINT_ARCHIVE" ]; then
            SIZE=$(du -h "$CHECKPOINT_ARCHIVE" | cut -f1)
            echo "  ✓ Checkpoints archived: $SIZE"
            echo "  Location: $CHECKPOINT_ARCHIVE"
        else
            echo "  ⚠️  Failed to create checkpoint archive"
        fi
    else
        echo "  ⚠️  No checkpoints found to archive"
    fi
fi

# Summary
echo ""
echo "======================================================================"
echo "  ✓ ARCHIVE COMPLETE"
echo "======================================================================"
echo ""
echo "Results archived to: $PROJECT_ARCHIVE"
echo ""
echo "Next steps:"
echo "  1. Organize for git:"
echo "     python scripts/organize_results.py --source $PROJECT_ARCHIVE \\"
echo "                                        --dest experiment_results/t5_experiments/seed_${SEED} \\"
echo "                                        --seed $SEED \\"
echo "                                        --experiment-type t5_summarization"
echo ""
echo "  2. Add to version control:"
echo "     git add experiment_results/t5_experiments/seed_${SEED}"
echo "     git commit -m 'Add T5 experiment results for seed $SEED'"
echo ""
echo "  3. Clean up SCRATCH (optional, after verification):"
echo "     rm -rf $SCRATCH_WORK"
echo "     rm -rf $SCRATCH_RESULTS"
echo ""
echo "======================================================================"
