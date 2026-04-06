#!/bin/bash
################################################################################
# Setup Persistent Disk for Azure Spot VM
#
# Run ONCE after attaching a managed data disk in Azure Portal.
# This formats and mounts the disk at /persist.
#
# PREREQUISITE: Attach a managed data disk (256GB+) in Azure Portal:
#   VM -> Disks -> + Create and attach a new disk
################################################################################

set -euo pipefail

echo "======================================================================"
echo "PERSISTENT DISK SETUP"
echo "======================================================================"

# Find the new disk (not sda=OS, not sdb=temp)
echo ""
echo "Current disk layout:"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
echo ""

# Look for unformatted/unmounted disks
CANDIDATES=$(lsblk -dno NAME,SIZE,TYPE | grep disk | grep -v -E "^sda |^sdb " | awk '{print "/dev/"$1, $2}')

if [ -z "$CANDIDATES" ]; then
    echo "ERROR: No candidate data disk found."
    echo "Please attach a managed data disk in Azure Portal first."
    echo "  VM -> Disks -> + Create and attach a new disk"
    exit 1
fi

echo "Candidate disk(s):"
echo "$CANDIDATES"
echo ""

# Use the first candidate
DISK=$(echo "$CANDIDATES" | head -1 | awk '{print $1}')
echo "Using disk: $DISK"

# Check if it already has a filesystem
if blkid "$DISK" 2>/dev/null | grep -q "TYPE="; then
    echo "Disk already formatted: $(blkid $DISK)"
    read -p "Reformat? This will ERASE all data. (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping format. Mounting existing filesystem..."
    else
        echo "Formatting $DISK as ext4..."
        sudo mkfs.ext4 -F "$DISK"
    fi
else
    echo "Formatting $DISK as ext4..."
    sudo mkfs.ext4 "$DISK"
fi

# Mount
sudo mkdir -p /persist
sudo mount "$DISK" /persist
sudo chown $USER:$USER /persist

# Add to fstab for auto-mount on reboot
DISK_UUID=$(blkid -s UUID -o value "$DISK")
if ! grep -q "$DISK_UUID" /etc/fstab 2>/dev/null; then
    echo "UUID=$DISK_UUID /persist ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
    echo "Added to /etc/fstab for auto-mount"
fi

echo ""
echo "======================================================================"
echo "PERSISTENT DISK READY"
echo "======================================================================"
echo ""
echo "  Mount point: /persist"
echo "  Size:        $(df -h /persist | awk 'NR==2{print $2}')"
echo "  Free:        $(df -h /persist | awk 'NR==2{print $4}')"
echo "  UUID:        $DISK_UUID"
echo ""
echo "This disk SURVIVES spot deallocation."
echo "Checkpoints and results will be saved here."
echo ""
echo "Next steps:"
echo "  bash bootstrap_azure.sh    # if not already done"
echo "  tmux new-session -s sprint"
echo "  ./run_azure_persistent.sh"
echo ""
