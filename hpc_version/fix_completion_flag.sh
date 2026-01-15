#!/bin/bash
#
# Manual Completion Flag Fix
#
# Use this script if training actually completed successfully but the
# completion flag wasn't created due to a bug or early termination.
#
# ONLY run this after verifying that training actually finished!
#

set -e

export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
export SEED=${1:-42}

WORK_DIR="$SCRATCH/mono_s2s_work"
CHECKPOINT_DIR="$WORK_DIR/checkpoints/monotonic_checkpoints"

echo "=========================================="
echo "Manual Completion Flag Creator"
echo "=========================================="
echo ""
echo "⚠️  WARNING: Only use this if you've verified that training"
echo "   actually completed successfully!"
echo ""

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ ERROR: Checkpoint directory doesn't exist: $CHECKPOINT_DIR"
    echo "   Training hasn't run yet."
    exit 1
fi

# Count checkpoints
EPOCH_COUNT=$(ls "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
echo "Found $EPOCH_COUNT epoch checkpoints"

# Expected epochs (from config)
EXPECTED_EPOCHS=7

if [ $EPOCH_COUNT -lt $EXPECTED_EPOCHS ]; then
    echo ""
    echo "❌ ERROR: Only $EPOCH_COUNT/$EXPECTED_EPOCHS epochs found."
    echo "   Training appears to be incomplete."
    echo "   Do NOT create the completion flag!"
    exit 1
fi

echo "✓ Found $EPOCH_COUNT/$EXPECTED_EPOCHS epochs - training appears complete"
echo ""

# Check if best model exists
if [ ! -f "$CHECKPOINT_DIR/best_model.pt" ]; then
    echo "⚠️  WARNING: best_model.pt not found"
    echo "   This might indicate a problem."
    read -p "Continue anyway? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# Ask for confirmation
echo ""
echo "This will create: $WORK_DIR/stage_3_train_monotonic_complete.flag"
echo ""
read -p "Create completion flag? (yes/no): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
    # Create the flag
    cat > "$WORK_DIR/stage_3_train_monotonic_complete.flag" <<EOF
Completed at: $(date '+%Y-%m-%d %H:%M:%S')
Seed: $SEED
Note: Manually created after verification
EOF
    
    echo ""
    echo "✓ Completion flag created successfully!"
    echo ""
    echo "You can now continue with the remaining stages:"
    echo "  cd $(dirname $0)"
    echo "  sbatch jobs/job_4_evaluate.sh"
    echo ""
    echo "Or let run_all.sh detect the flag and skip stage 3."
else
    echo "Aborted."
    exit 1
fi
