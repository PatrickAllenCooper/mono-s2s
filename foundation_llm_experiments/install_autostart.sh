#!/bin/bash
################################################################################
# Install Automatic Experiment Resume on Boot
#
# Creates a systemd service that automatically:
#   1. Mounts NVMe fast storage
#   2. Resumes training from the latest checkpoint
#   3. Runs inside a persistent tmux session
#
# Run ONCE after initial setup:
#   bash install_autostart.sh
#
# After this, every time the VM starts (including spot deallocation recovery),
# training resumes automatically. Just SSH in and:
#   tmux attach -t experiment
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_NAME="$(whoami)"
HOME_DIR="$HOME"

echo "======================================================================"
echo "INSTALLING AUTOMATIC EXPERIMENT RESUME"
echo "======================================================================"
echo ""
echo "User:       $USER_NAME"
echo "Home:       $HOME_DIR"
echo "Script dir: $SCRIPT_DIR"
echo ""

# ============================================================================
# CREATE THE BOOT SCRIPT
# ============================================================================

BOOT_SCRIPT="${SCRIPT_DIR}/on_boot_resume.sh"

cat > "$BOOT_SCRIPT" << BOOTEOF
#!/bin/bash
################################################################################
# Boot-time resume script (called by systemd)
# Do not run manually - use: sudo systemctl start mono-experiment
################################################################################

LOG="/var/log/mono-experiment-boot.log"
exec >> "\$LOG" 2>&1

echo ""
echo "============================================"
echo "[\$(date)] VM started. Resuming experiments."
echo "============================================"

# Wait for GPU driver to initialize
sleep 15

# Check if nvidia-smi works
if ! nvidia-smi > /dev/null 2>&1; then
    echo "[\$(date)] ERROR: nvidia-smi not available. Waiting..."
    sleep 30
    nvidia-smi > /dev/null 2>&1 || { echo "[\$(date)] GPU still not ready. Exiting."; exit 1; }
fi
echo "[\$(date)] GPU ready."

# Mount NVMe (ephemeral - needs reformat after deallocation)
echo "[\$(date)] Setting up NVMe..."
for dev in /dev/nvme0n1 /dev/nvme1n1; do
    if [ -b "\$dev" ]; then
        # Check if already mounted
        if ! mountpoint -q /data 2>/dev/null; then
            mkfs.ext4 -F "\$dev" 2>/dev/null || true
            mkdir -p /data
            mount "\$dev" /data 2>/dev/null || true
            chown ${USER_NAME}:${USER_NAME} /data
            echo "[\$(date)] NVMe mounted at /data"
        else
            echo "[\$(date)] /data already mounted"
        fi
        break
    fi
done

# Verify persistent disk
if mountpoint -q /persist 2>/dev/null; then
    echo "[\$(date)] /persist mounted."
else
    echo "[\$(date)] Mounting /persist..."
    mount /persist 2>/dev/null || mount -a 2>/dev/null || true
    if mountpoint -q /persist 2>/dev/null; then
        echo "[\$(date)] /persist mounted successfully."
    else
        echo "[\$(date)] ERROR: /persist not available. Exiting."
        exit 1
    fi
fi

# Show checkpoint status
echo "[\$(date)] Checkpoint status:"
for f in /persist/foundation_llm_work_seed*/stage_*_complete.flag; do
    [ -f "\$f" ] && echo "  DONE: \$(basename \$f)"
done 2>/dev/null
for d in /persist/foundation_llm_work_seed*/checkpoints/*/; do
    CKPTS=\$(find "\$d" -name "checkpoint_epoch_*.pt" 2>/dev/null | wc -l)
    [ "\$CKPTS" -gt 0 ] && echo "  PARTIAL: \$(basename \$(dirname \$d))/\$(basename \$d) - \$CKPTS epochs"
done 2>/dev/null

# Start the experiment in a tmux session as the actual user
echo "[\$(date)] Starting experiment in tmux session 'experiment'..."
su - ${USER_NAME} -c '
    export PATH="${HOME_DIR}/miniconda3/bin:\$PATH"
    source ${HOME_DIR}/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate mono_s2s 2>/dev/null

    # Kill any existing experiment session
    tmux kill-session -t experiment 2>/dev/null || true

    # Start new tmux session running the experiment
    tmux new-session -d -s experiment -c ${SCRIPT_DIR} \
        "source ${SCRIPT_DIR}/.azure_env 2>/dev/null; ${SCRIPT_DIR}/run_azure_persistent.sh; echo EXPERIMENT COMPLETE; sleep 999999"
'

echo "[\$(date)] Experiment launched in tmux session 'experiment'."
echo "[\$(date)] SSH in and run: tmux attach -t experiment"
echo "============================================"
BOOTEOF

chmod +x "$BOOT_SCRIPT"
echo "Boot script created: $BOOT_SCRIPT"

# ============================================================================
# CREATE SYSTEMD SERVICE
# ============================================================================

SERVICE_FILE="/etc/systemd/system/mono-experiment.service"

sudo tee "$SERVICE_FILE" > /dev/null << SVCEOF
[Unit]
Description=Mono-S2S Foundation LLM Experiment Auto-Resume
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=${BOOT_SCRIPT}
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
SVCEOF

echo "Systemd service created: $SERVICE_FILE"

# ============================================================================
# ENABLE THE SERVICE
# ============================================================================

sudo systemctl daemon-reload
sudo systemctl enable mono-experiment.service
echo "Service enabled for boot."

# ============================================================================
# TEST
# ============================================================================

echo ""
echo "======================================================================"
echo "AUTOSTART INSTALLED"
echo "======================================================================"
echo ""
echo "What happens now on every VM start:"
echo "  1. systemd runs on_boot_resume.sh"
echo "  2. NVMe is formatted and mounted at /data"
echo "  3. /persist is verified (your checkpoints)"
echo "  4. Training resumes in tmux session 'experiment'"
echo ""
echo "After SSH-ing in:"
echo "  tmux attach -t experiment     # see training progress"
echo "  Ctrl+B D                      # detach without stopping"
echo ""
echo "Manual controls:"
echo "  sudo systemctl status mono-experiment    # check service status"
echo "  sudo systemctl start mono-experiment     # manually trigger resume"
echo "  sudo systemctl stop mono-experiment      # stop (does not kill training)"
echo "  sudo systemctl disable mono-experiment   # disable auto-start"
echo "  cat /var/log/mono-experiment-boot.log    # check boot log"
echo ""
echo "To test right now:"
echo "  sudo systemctl start mono-experiment"
echo "  tmux attach -t experiment"
echo ""
