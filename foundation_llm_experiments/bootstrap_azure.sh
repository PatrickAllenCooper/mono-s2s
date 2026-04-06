#!/bin/bash
################################################################################
# Bootstrap Script for Azure Spot VM (2x A100 80GB)
#
# Sets up the Python environment on an Azure NC-series spot instance.
# Handles CUDA compatibility and configures for spot deallocation resilience.
#
# USAGE:
#   bash bootstrap_azure.sh
################################################################################

set -euo pipefail

echo "======================================================================"
echo "FOUNDATION LLM EXPERIMENTS - AZURE SPOT VM BOOTSTRAP"
echo "======================================================================"
echo ""
echo "Host:   $(hostname)"
echo "User:   $(whoami)"
echo "Date:   $(date)"
echo ""

WORK_ROOT="${HOME}"
CONDA_BASE="${WORK_ROOT}/miniconda3"
ENV_NAME="mono_s2s"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# GPU INFO
# ============================================================================

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null
echo ""

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "GPUs available: $GPU_COUNT"

# Detect CUDA version from driver
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver version: $DRIVER_CUDA"

# PyTorch CUDA compatibility: use cu124 for CUDA 12.x+ drivers
TORCH_CUDA_TAG="cu124"
echo "PyTorch CUDA build: $TORCH_CUDA_TAG"
echo ""

# Disk space
echo "Disk space:"
df -h / | tail -1
echo ""

# ============================================================================
# INSTALL MINICONDA
# ============================================================================

if [ -d "$CONDA_BASE" ] && [ -f "$CONDA_BASE/bin/conda" ]; then
    echo "Conda already installed at $CONDA_BASE"
else
    echo "Installing Miniconda..."
    cd "$WORK_ROOT"
    wget -q --show-progress \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda_installer.sh
    bash miniconda_installer.sh -b -p "$CONDA_BASE"
    rm miniconda_installer.sh
    echo "Miniconda installed"
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

# Accept ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if ! grep -q "conda initialize" ~/.bashrc 2>/dev/null; then
    "$CONDA_BASE/bin/conda" init bash
fi

# ============================================================================
# CREATE CONDA ENVIRONMENT
# ============================================================================

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists"
else
    echo "Creating conda environment '${ENV_NAME}' (Python 3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
echo "Environment activated: $(python --version)"

# ============================================================================
# INSTALL PYTORCH AND DEPENDENCIES
# ============================================================================

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "Installing PyTorch for ${TORCH_CUDA_TAG}..."
    pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
        --quiet
fi

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
else
    pip install transformers datasets accelerate numpy scipy scikit-learn tqdm pandas --quiet
fi

pip install zstandard --quiet 2>/dev/null

# Verify
python -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'GPU {i}:        {name} ({mem:.0f}GB)')
"

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

echo ""
echo "Creating directory structure..."

AZURE_WORK="${WORK_ROOT}/foundation_llm_work"
AZURE_RESULTS="${WORK_ROOT}/foundation_llm_results"
AZURE_CACHE="${WORK_ROOT}/foundation_llm_cache"

mkdir -p "$AZURE_WORK/checkpoints/baseline_checkpoints"
mkdir -p "$AZURE_WORK/checkpoints/monotonic_checkpoints"
mkdir -p "$AZURE_WORK/stage_logs"
mkdir -p "$AZURE_RESULTS"
mkdir -p "$AZURE_CACHE"

# ============================================================================
# ENVIRONMENT FILE
# ============================================================================

cat > "${SCRIPT_DIR}/.azure_env" << EOF
export AZURE_WORK="${AZURE_WORK}"
export AZURE_RESULTS="${AZURE_RESULTS}"
export AZURE_CACHE="${AZURE_CACHE}"
export HF_HOME="${AZURE_CACHE}/huggingface"
export HF_DATASETS_CACHE="${AZURE_CACHE}/huggingface/datasets"
export TRANSFORMERS_CACHE="${AZURE_CACHE}/huggingface/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:16:8
EOF

source "${SCRIPT_DIR}/.azure_env"
mkdir -p "$HF_HOME"

# ============================================================================
# SPOT DEALLOCATION RESILIENCE
# ============================================================================

echo ""
echo "Setting up spot deallocation resilience..."

# Create a systemd service that auto-restarts training on reboot
RESTART_SCRIPT="${SCRIPT_DIR}/auto_restart_azure.sh"
cat > "$RESTART_SCRIPT" << 'RESTART_EOF'
#!/bin/bash
# Auto-restart script for Azure spot VM
# Called by systemd on boot or manually after spot deallocation recovery

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${SCRIPT_DIR}/logs/auto_restart.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date)] Spot VM recovered. Checking for interrupted training..." >> "$LOG"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mono_s2s
source "${SCRIPT_DIR}/.azure_env"

cd "$SCRIPT_DIR"

# Check if training was interrupted (no completion flag but checkpoints exist)
WORK_DIR="${AZURE_WORK}_seed${EXPERIMENT_SEED:-42}"

if [ ! -f "${WORK_DIR}/stage_2_train_baseline_complete.flag" ] && \
   [ -d "${WORK_DIR}/checkpoints/baseline_checkpoints" ]; then
    echo "[$(date)] Resuming baseline training on GPU 0..." >> "$LOG"
    CUDA_VISIBLE_DEVICES=0 EXPERIMENT_SEED=${EXPERIMENT_SEED:-42} \
        nohup bash -c "cd scripts && python stage_2_train_baseline.py" \
        >> "${SCRIPT_DIR}/logs/baseline_gpu0.log" 2>&1 &
    echo "[$(date)] Baseline PID: $!" >> "$LOG"
fi

if [ ! -f "${WORK_DIR}/stage_3_train_monotonic_complete.flag" ] && \
   [ -d "${WORK_DIR}/checkpoints/monotonic_checkpoints" ]; then
    echo "[$(date)] Resuming monotonic training on GPU 1..." >> "$LOG"
    CUDA_VISIBLE_DEVICES=1 EXPERIMENT_SEED=${EXPERIMENT_SEED:-42} \
        nohup bash -c "cd scripts && python stage_3_train_monotonic.py" \
        >> "${SCRIPT_DIR}/logs/monotonic_gpu1.log" 2>&1 &
    echo "[$(date)] Monotonic PID: $!" >> "$LOG"
fi

echo "[$(date)] Auto-restart complete." >> "$LOG"
RESTART_EOF
chmod +x "$RESTART_SCRIPT"

# Add to crontab for auto-restart on reboot
(crontab -l 2>/dev/null | grep -v "auto_restart_azure" ; \
 echo "@reboot sleep 30 && ${RESTART_SCRIPT}") | crontab -
echo "Crontab configured for auto-restart on reboot"

chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

# ============================================================================
# DONE
# ============================================================================

echo ""
echo "======================================================================"
echo "BOOTSTRAP COMPLETE"
echo "======================================================================"
echo ""
echo "GPUs: $GPU_COUNT x A100 80GB"
echo "Strategy: Baseline on GPU 0, Monotonic on GPU 1 (parallel)"
echo "Spot resilience: Auto-restart on reboot via crontab"
echo ""
echo "To start experiments:"
echo "  tmux new-session -s experiment"
echo "  source .azure_env"
echo "  ./run_azure.sh"
echo ""
echo "======================================================================"
