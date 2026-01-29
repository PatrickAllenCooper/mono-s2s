#!/bin/bash
################################################################################
# Bootstrap Script for CURC Alpine
#
# Automatically handles:
# - Conda installation to /projects/$USER (not $HOME to avoid quota issues)
# - Conda environment creation
# - All Python dependencies
# - Script permissions
#
# Run this ONCE before submitting jobs:
#   bash bootstrap_curc.sh
################################################################################

set -euo pipefail

echo "======================================================================"
echo "FOUNDATION LLM EXPERIMENTS - CURC ALPINE BOOTSTRAP"
echo "======================================================================"
echo ""

# Detect if running on CURC Alpine
if [[ ! $(hostname) =~ login.* ]] && [[ ! $(hostname) =~ colostate.edu ]]; then
    echo "WARNING: This script is designed for CURC Alpine"
    echo "Current hostname: $(hostname)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Check if we have $USER
if [ -z "${USER:-}" ]; then
    echo "ERROR: \$USER environment variable not set"
    exit 1
fi

echo "User: $USER"
echo ""

# ======================================================================
# STEP 1: Install Conda to /projects (if not already installed)
# ======================================================================

CONDA_BASE="/projects/$USER/miniconda3"

if [ -d "$CONDA_BASE" ] && [ -f "$CONDA_BASE/bin/conda" ]; then
    echo "✓ Conda already installed at $CONDA_BASE"
else
    echo "Installing Miniconda to $CONDA_BASE..."
    echo "This avoids \$HOME quota issues on Alpine"
    echo ""
    
    # Create projects directory if it doesn't exist
    mkdir -p "/projects/$USER"
    cd "/projects/$USER"
    
    # Download installer
    echo "Downloading Miniconda installer..."
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_installer.sh
    
    # Install
    echo "Installing Miniconda (this takes ~2 minutes)..."
    bash miniconda_installer.sh -b -p "$CONDA_BASE"
    rm miniconda_installer.sh
    
    echo "✓ Conda installed successfully"
fi

# Initialize conda for this script
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Update .bashrc if not already done
if ! grep -q "conda initialize" ~/.bashrc; then
    echo ""
    echo "Adding conda to .bashrc..."
    "$CONDA_BASE/bin/conda" init bash
    echo "✓ .bashrc updated (will take effect in new shells)"
else
    echo "✓ Conda already configured in .bashrc"
fi

echo ""

# ======================================================================
# STEP 2: Create conda environment (if not exists)
# ======================================================================

ENV_NAME="mono_s2s"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "✓ Conda environment '$ENV_NAME' already exists"
else
    echo "Creating conda environment '$ENV_NAME' with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
    echo "✓ Environment created"
fi

# Activate environment
echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"
echo "✓ Environment activated"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo ""

# ======================================================================
# STEP 3: Install dependencies
# ======================================================================

echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet

echo "Installing other dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
else
    echo "WARNING: requirements.txt not found, installing core dependencies..."
    pip install transformers datasets accelerate scipy scikit-learn pandas tqdm pytest pytest-cov --quiet
fi

echo "✓ All dependencies installed"
echo ""

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# ======================================================================
# STEP 4: Make scripts executable
# ======================================================================

echo "Making scripts executable..."
chmod +x run_all.sh 2>/dev/null || true
chmod +x jobs/*.sh 2>/dev/null || true
chmod +x scripts/*.py 2>/dev/null || true
echo "✓ Scripts are executable"
echo ""

# ======================================================================
# STEP 5: Set up environment variables
# ======================================================================

echo "Setting up HuggingFace cache locations..."

# Use SCRATCH for cache (not home directory)
SCRATCH_DIR="${SCRATCH:-/scratch/alpine/$USER}"

export HF_HOME="$SCRATCH_DIR/huggingface_cache"
export HF_DATASETS_CACHE="$SCRATCH_DIR/huggingface_cache/datasets"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/huggingface_cache/transformers"

mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"

echo "✓ Cache directories created:"
echo "  HF_HOME: $HF_HOME"
echo ""

# ======================================================================
# STEP 6: Validate configuration
# ======================================================================

echo "Validating experiment configuration..."
if [ -f "configs/experiment_config.py" ]; then
    python -c "
from configs.experiment_config import FoundationExperimentConfig as Config
print('  Model:', Config.MODEL_NAME)
print('  Partition:', Config.SLURM_PARTITION)
print('  Scratch dir:', Config.SCRATCH_DIR)
print('  Project dir:', Config.PROJECT_DIR)
" || echo "WARNING: Configuration validation failed (non-fatal)"
else
    echo "WARNING: configs/experiment_config.py not found"
fi
echo ""

# ======================================================================
# STEP 7: Print next steps
# ======================================================================

echo "======================================================================"
echo "✓ BOOTSTRAP COMPLETE"
echo "======================================================================"
echo ""
echo "Your environment is ready for Foundation LLM experiments!"
echo ""
echo "Next steps:"
echo ""
echo "1. (Optional) Test the setup:"
echo "   conda activate $ENV_NAME"
echo "   python -m configs.experiment_config"
echo ""
echo "2. Submit all jobs:"
echo "   ./run_all.sh"
echo ""
echo "3. Or submit individual stages:"
echo "   sbatch jobs/job_0_setup.sh"
echo ""
echo "4. Monitor progress:"
echo "   squeue -u \$USER | grep foundation"
echo "   tail -f logs/job_0_setup_*.out"
echo ""
echo "Environment details:"
echo "  Conda base: $CONDA_BASE"
echo "  Environment: $ENV_NAME"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Important: Source your .bashrc in new shells to enable conda:"
echo "  source ~/.bashrc"
echo ""
echo "======================================================================"
