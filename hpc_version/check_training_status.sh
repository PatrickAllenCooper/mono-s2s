#!/bin/bash
#
# Diagnostic Script: Check Monotonic Training Status
# 
# This script checks the actual state of monotonic training to diagnose
# why the completion flag wasn't created.
#

set -e

export SCRATCH=${SCRATCH:-/scratch/alpine/$USER}
WORK_DIR="$SCRATCH/mono_s2s_work"
CHECKPOINT_DIR="$WORK_DIR/checkpoints/monotonic_checkpoints"
FLAG_FILE="$WORK_DIR/stage_3_train_monotonic_complete.flag"

echo "=========================================="
echo "Monotonic Training Status Diagnostic"
echo "=========================================="
echo ""

# Check if work directory exists
if [ ! -d "$WORK_DIR" ]; then
    echo "❌ Work directory does not exist: $WORK_DIR"
    exit 1
fi

echo "✓ Work directory exists: $WORK_DIR"
echo ""

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory does not exist: $CHECKPOINT_DIR"
    echo "   Training may not have started."
    exit 1
fi

echo "✓ Checkpoint directory exists: $CHECKPOINT_DIR"
echo ""

# List all checkpoints
echo "Checkpoints found:"
ls -lh "$CHECKPOINT_DIR" 2>/dev/null || echo "  (none)"
echo ""

# Count epoch checkpoints
EPOCH_CHECKPOINTS=$(ls "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
echo "Number of epoch checkpoints: $EPOCH_CHECKPOINTS"

if [ $EPOCH_CHECKPOINTS -gt 0 ]; then
    echo "Latest checkpoints:"
    ls -lht "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt | head -3
    echo ""
    
    # Get the highest epoch number
    LATEST_EPOCH=$(ls "$CHECKPOINT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | \
                   sed 's/.*checkpoint_epoch_\([0-9]*\)\.pt/\1/' | \
                   sort -n | tail -1)
    echo "Latest epoch trained: $LATEST_EPOCH"
    echo ""
fi

# Check if best model exists
if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
    echo "✓ Best model saved"
    ls -lh "$CHECKPOINT_DIR/best_model.pt"
else
    echo "❌ No best model found"
fi
echo ""

# Check training history
HISTORY_FILE="$WORK_DIR/../mono_s2s_results/monotonic_training_history.json"
if [ -f "$HISTORY_FILE" ]; then
    echo "✓ Training history exists"
    echo "  Location: $HISTORY_FILE"
    
    # Try to extract info using Python (if available)
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
try:
    with open('$HISTORY_FILE', 'r') as f:
        data = json.load(f)
    train_losses = data.get('train_losses', [])
    val_losses = data.get('val_losses', [])
    best_val = data.get('best_val_loss', 'N/A')
    print(f'  Epochs completed: {len(train_losses)}')
    print(f'  Best validation loss: {best_val}')
    if train_losses:
        print(f'  Final train loss: {train_losses[-1]:.4f}')
    if val_losses:
        print(f'  Final val loss: {val_losses[-1]:.4f}')
except Exception as e:
    print(f'  Error reading history: {e}')
" 2>/dev/null || echo "  (Could not parse JSON)"
    fi
else
    echo "❌ No training history found"
fi
echo ""

# Check completion flag
if [ -f "$FLAG_FILE" ]; then
    echo "✓ Completion flag EXISTS"
    cat "$FLAG_FILE"
else
    echo "❌ Completion flag MISSING: $FLAG_FILE"
    echo ""
    echo "This is why run_all.sh reported failure."
    echo ""
    
    # Suggest manual fix if training actually completed
    if [ "$EPOCH_CHECKPOINTS" -ge 7 ]; then
        echo "ANALYSIS: Found $EPOCH_CHECKPOINTS epoch checkpoints (expected 7)."
        echo "Training appears to have completed successfully!"
        echo ""
        echo "To manually create the completion flag and continue:"
        echo "  touch $FLAG_FILE"
        echo "  echo \"Completed at: \$(date)\" >> $FLAG_FILE"
        echo "  echo \"Seed: 42\" >> $FLAG_FILE"
        echo ""
        echo "Then you can continue with the remaining stages."
    else
        echo "ANALYSIS: Only found $EPOCH_CHECKPOINTS epoch checkpoints (expected 7)."
        echo "Training may be incomplete."
    fi
fi

echo ""
echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
