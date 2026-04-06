#!/bin/bash
################################################################################
# Recovery Script for Azure Spot VM After Deallocation
#
# Run this after the VM restarts from a spot deallocation.
# It remounts drives and resumes training from the latest checkpoint.
################################################################################

set -euo pipefail

echo "======================================================================"
echo "SPOT DEALLOCATION RECOVERY"
echo "======================================================================"
echo ""

# Step 1: Check persistent disk
echo "Checking persistent disk..."
if mountpoint -q /persist 2>/dev/null; then
    echo "  /persist already mounted"
else
    echo "  Mounting /persist..."
    sudo mount /persist 2>/dev/null || sudo mount -a
    if mountpoint -q /persist 2>/dev/null; then
        echo "  /persist mounted successfully"
    else
        echo "  ERROR: Could not mount /persist. Check fstab or reattach disk in Azure Portal."
        exit 1
    fi
fi
echo "  Persistent storage: $(df -h /persist | awk 'NR==2{print $4}') free"

# Step 2: Remount NVMe (ephemeral - needs reformat after deallocation)
echo ""
echo "Setting up NVMe fast storage..."
NVME_DEV=""
for dev in /dev/nvme0n1 /dev/nvme1n1; do
    if [ -b "$dev" ]; then
        NVME_DEV="$dev"
        break
    fi
done

if [ -n "$NVME_DEV" ]; then
    if mountpoint -q /data 2>/dev/null; then
        echo "  /data already mounted"
    else
        echo "  Formatting $NVME_DEV (ephemeral - expected after deallocation)..."
        sudo mkfs.ext4 -F "$NVME_DEV" 2>/dev/null
        sudo mkdir -p /data
        sudo mount "$NVME_DEV" /data
        sudo chown $USER:$USER /data
    fi
    echo "  NVMe storage: $(df -h /data | awk 'NR==2{print $4}') free"
else
    echo "  No NVMe drive found. Using /persist for everything."
    sudo mkdir -p /data
    sudo mount --bind /persist /data 2>/dev/null || ln -sf /persist /data
fi

# Step 3: Show checkpoint status
echo ""
echo "Checkpoint status on persistent disk:"
for seed_dir in /persist/foundation_llm_work_seed*/; do
    if [ -d "$seed_dir" ]; then
        SEED=$(basename "$seed_dir" | sed 's/foundation_llm_work_seed//')
        echo ""
        echo "  Seed $SEED:"
        for stage in 0_setup 1_apply_monotonicity 2_train_baseline 3_train_monotonic 4_evaluate 5_uat 6_hotflip; do
            FLAG="${seed_dir}/stage_${stage}_complete.flag"
            if [ -f "$FLAG" ]; then
                echo "    $stage: COMPLETE"
            else
                case "$stage" in
                    2_train_baseline)
                        CKPTS=$(ls ${seed_dir}/checkpoints/baseline_checkpoints/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
                        [ "$CKPTS" -gt 0 ] && echo "    $stage: $CKPTS epoch(s) saved (will resume)" || echo "    $stage: not started"
                        ;;
                    3_train_monotonic)
                        CKPTS=$(ls ${seed_dir}/checkpoints/monotonic_checkpoints/checkpoint_epoch_*.pt 2>/dev/null | wc -l)
                        [ "$CKPTS" -gt 0 ] && echo "    $stage: $CKPTS epoch(s) saved (will resume)" || echo "    $stage: not started"
                        ;;
                    *)
                        echo "    $stage: not started"
                        ;;
                esac
            fi
        done
    fi
done

# Step 4: Activate conda
echo ""
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate mono_s2s 2>/dev/null || {
    echo "WARNING: conda environment not found. Run: bash bootstrap_azure.sh"
}

echo ""
echo "======================================================================"
echo "RECOVERY COMPLETE"
echo "======================================================================"
echo ""
echo "To resume training:"
echo "  cd ~/mono-s2s/foundation_llm_experiments"
echo "  tmux new-session -s sprint"
echo "  ./run_azure_persistent.sh"
echo ""
echo "Training will automatically resume from the latest checkpoint."
echo ""
