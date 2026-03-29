#!/bin/bash
################################################################################
# Bootstrap Script for Lambda Cloud
#
# Sets up the Python environment and dependencies on a Lambda Cloud GPU instance.
# Lambda Cloud instances run Ubuntu with CUDA pre-installed; this script adds
# the Python environment on top.
#
# USAGE (run once after launching your instance):
#   bash bootstrap_lambda.sh
#
# WHAT THIS DOES:
#   1. Installs Miniconda to ~/miniconda3
#   2. Creates conda environment 'mono_s2s' with Python 3.10
#   3. Installs PyTorch (detects CUDA version automatically)
#   4. Installs all Python dependencies
#   5. Creates the working directory structure
#   6. Configures HuggingFace cache
#
# LAMBDA CLOUD NOTES:
#   - Home directory is /home/ubuntu (or /home/$USER)
#   - Large NVMe storage is typically at /home/ubuntu (~1.4TB on H100 instances)
#   - CUDA is pre-installed (usually 11.8 or 12.x)
#   - Instances are ephemeral - save results to persistent storage or pull to
#     local machine before terminating
#   - Use tmux to keep sessions alive across SSH disconnects
################################################################################

set -euo pipefail

echo "======================================================================"
echo "FOUNDATION LLM EXPERIMENTS - LAMBDA CLOUD BOOTSTRAP"
echo "======================================================================"
echo ""
echo "Host:   $(hostname)"
echo "User:   $(whoami)"
echo "Date:   $(date)"
echo ""

# ============================================================================
# DETECT ENVIRONMENT
# ============================================================================

WORK_ROOT="${HOME}"
CONDA_BASE="${WORK_ROOT}/miniconda3"
ENV_NAME="mono_s2s"

# Detect CUDA version
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    echo "CUDA version detected: $CUDA_VER"
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VER=$(cat /usr/local/cuda/version.txt | awk '{print $3}')
    echo "CUDA version detected: $CUDA_VER"
else
    CUDA_VER="12.1"
    echo "Could not detect CUDA version, defaulting to $CUDA_VER"
fi

# Determine PyTorch CUDA tag
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_CUDA_TAG="cu121"
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "${CUDA_MINOR:-0}" -ge 8 ]; then
    TORCH_CUDA_TAG="cu118"
else
    TORCH_CUDA_TAG="cu118"
    echo "Warning: Unexpected CUDA version $CUDA_VER, using cu118 PyTorch build"
fi

echo "PyTorch CUDA tag: $TORCH_CUDA_TAG"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Disk space check
echo "Disk space:"
df -h "$WORK_ROOT" | tail -1
echo ""

# ============================================================================
# STEP 1: Install Miniconda
# ============================================================================

if [ -d "$CONDA_BASE" ] && [ -f "$CONDA_BASE/bin/conda" ]; then
    echo "✓ Conda already installed at $CONDA_BASE"
else
    echo "Installing Miniconda to $CONDA_BASE..."

    cd "$WORK_ROOT"
    wget -q --show-progress \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O miniconda_installer.sh

    bash miniconda_installer.sh -b -p "$CONDA_BASE"
    rm miniconda_installer.sh

    echo "✓ Miniconda installed"
fi

# Initialise conda for this script session
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Add to .bashrc for future sessions
if ! grep -q "conda initialize" ~/.bashrc 2>/dev/null; then
    "$CONDA_BASE/bin/conda" init bash
    echo "✓ conda added to ~/.bashrc"
else
    echo "✓ conda already in ~/.bashrc"
fi

# ============================================================================
# STEP 2: Create conda environment
# ============================================================================

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✓ Conda environment '${ENV_NAME}' already exists"
else
    echo "Creating conda environment '${ENV_NAME}' (Python 3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -y
    echo "✓ Environment created"
fi

conda activate "$ENV_NAME"
echo "✓ Environment '${ENV_NAME}' activated"
echo "  Python: $(which python) ($(python --version))"
echo ""

# ============================================================================
# STEP 3: Install PyTorch
# ============================================================================

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    EXISTING_TORCH=$(python -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch already installed: $EXISTING_TORCH"
else
    echo "Installing PyTorch for $TORCH_CUDA_TAG..."
    pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
        --quiet
    echo "✓ PyTorch installed"
fi

# Verify GPU access
python -c "
import torch
print(f'  PyTorch version:  {torch.__version__}')
print(f'  CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:     {torch.version.cuda}')
    print(f'  GPU:              {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU memory:       {mem_gb:.1f} GB')
"
echo ""

# ============================================================================
# STEP 4: Install Python dependencies
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
    echo "✓ All dependencies installed"
else
    echo "requirements.txt not found at $SCRIPT_DIR, installing core packages..."
    pip install \
        transformers>=4.30.0 \
        datasets>=2.14.0 \
        accelerate>=0.21.0 \
        numpy>=1.24.0 \
        scipy>=1.10.0 \
        scikit-learn>=1.3.0 \
        tqdm>=4.65.0 \
        pandas>=2.0.0 \
        zstandard \
        --quiet
    echo "✓ Core dependencies installed"
fi

# zstandard is required for monology/pile-uncopyrighted dataset
python -c "import zstandard" 2>/dev/null || pip install zstandard --quiet
echo "✓ zstandard (Pile dataset decompression) available"
echo ""

# ============================================================================
# STEP 5: Create directory structure
# ============================================================================

echo "Creating working directory structure..."

# Lambda Cloud: use home directory (large NVMe storage)
LAMBDA_WORK="${WORK_ROOT}/foundation_llm_work"
LAMBDA_RESULTS="${WORK_ROOT}/foundation_llm_results"
LAMBDA_CACHE="${WORK_ROOT}/foundation_llm_cache"

mkdir -p "$LAMBDA_WORK/checkpoints/baseline_checkpoints"
mkdir -p "$LAMBDA_WORK/checkpoints/monotonic_checkpoints"
mkdir -p "$LAMBDA_WORK/stage_logs"
mkdir -p "$LAMBDA_RESULTS"
mkdir -p "$LAMBDA_CACHE"

echo "✓ Directories created:"
echo "  Work:    $LAMBDA_WORK"
echo "  Results: $LAMBDA_RESULTS"
echo "  Cache:   $LAMBDA_CACHE"
echo ""

# ============================================================================
# STEP 6: Configure HuggingFace cache
# ============================================================================

HF_CACHE="${LAMBDA_CACHE}/huggingface"
mkdir -p "$HF_CACHE"

# Write environment variables to a sourceable file
cat > "${SCRIPT_DIR}/.lambda_env" << EOF
# Lambda Cloud environment variables for mono-s2s experiments
# Source this file before running: source .lambda_env

export LAMBDA_WORK="${LAMBDA_WORK}"
export LAMBDA_RESULTS="${LAMBDA_RESULTS}"
export LAMBDA_CACHE="${LAMBDA_CACHE}"

export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:16:8
EOF

echo "✓ Environment config written to .lambda_env"
source "${SCRIPT_DIR}/.lambda_env"
echo ""

# ============================================================================
# STEP 7: Make scripts executable
# ============================================================================

chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR"/scripts/*.py 2>/dev/null || true
echo "✓ Scripts made executable"
echo ""

# ============================================================================
# STEP 8: Verify setup
# ============================================================================

echo "Verifying full setup..."
cd "$SCRIPT_DIR"
python -c "
import sys
sys.path.insert(0, '.')
from configs.experiment_config import FoundationExperimentConfig as C
print(f'  Model:        {C.MODEL_NAME}')
print(f'  Work dir:     {C.WORK_DIR}')
print(f'  Results dir:  {C.RESULTS_DIR}')
print(f'  Data cache:   {C.DATA_CACHE_DIR}')
print(f'  Seed:         {C.CURRENT_SEED}')
print(f'  Epochs (mono):{C.MONOTONIC_RECOVERY_EPOCHS}')
" || echo "Warning: config validation failed (non-fatal)"
echo ""

# ============================================================================
# DONE
# ============================================================================

echo "======================================================================"
echo "✓ BOOTSTRAP COMPLETE"
echo "======================================================================"
echo ""
echo "To start the experiment:"
echo ""
echo "  # Option 1: Run in tmux (RECOMMENDED - survives SSH disconnects)"
echo "  tmux new-session -s experiment"
echo "  source .lambda_env"
echo "  ./run_lambda.sh"
echo "  # Detach with Ctrl+B then D; reconnect with: tmux attach -t experiment"
echo ""
echo "  # Option 2: Run directly (will stop if SSH disconnects)"
echo "  source .lambda_env"
echo "  ./run_lambda.sh"
echo ""
echo "  # Single seed:"
echo "  EXPERIMENT_SEED=42 ./run_lambda.sh"
echo ""
echo "  # Multiple seeds sequentially:"
echo "  SEEDS='42 1337 2024' ./run_lambda.sh"
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "======================================================================"
