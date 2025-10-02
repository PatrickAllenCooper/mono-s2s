#!/bin/bash
# MONO-S2S Conda Environment Setup Script
# Compatible with Linux and Windows (Git Bash/WSL)
# Requires: Anaconda or Miniconda installed

set -e  # Exit on error

echo "=========================================="
echo "MONO-S2S Environment Setup"
echo "=========================================="

# Environment name
ENV_NAME="mono-s2s"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing ${ENV_NAME} environment..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment with Python 3.10
echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install PyTorch with CUDA support
# This will install CUDA-enabled PyTorch on Linux/Windows, CPU-only on systems without CUDA
echo "Installing PyTorch with CUDA support..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    echo "Detected Windows system"
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Detected Linux system"
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    echo "WARNING: Unsupported OS for CUDA. Installing CPU-only PyTorch."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install core scientific computing packages
echo "Installing core scientific packages..."
conda install -y \
    numpy \
    scipy \
    pandas \
    matplotlib

# Install Jupyter and related tools
echo "Installing Jupyter..."
conda install -y \
    jupyter \
    jupyterlab \
    ipykernel \
    notebook

# Install additional packages via pip
echo "Installing additional packages via pip..."
pip install --upgrade pip

# Hugging Face datasets and related
pip install datasets
pip install transformers  # Often useful with datasets
pip install accelerate    # For optimized training

# Additional utilities
pip install tqdm          # Progress bars
pip install scikit-learn  # For additional ML utilities

# Install ipywidgets for better notebook experience
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Verify PyTorch CUDA availability
echo ""
echo "=========================================="
echo "Verifying PyTorch Installation..."
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Create a local data directory structure
echo ""
echo "Creating project directory structure..."
mkdir -p data/checkpoints
mkdir -p data/tokenizer
mkdir -p results
mkdir -p logs

# Create a local config file to replace Google Drive paths
echo "Creating local configuration..."
cat > local_config.py << 'EOF'
"""
Local configuration to replace Google Colab Drive paths.
Import this at the start of your notebook with:
    from local_config import *
"""
import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_PATH = os.path.join(DATA_PATH, 'checkpoints')
TOKENIZER_PATH = os.path.join(DATA_PATH, 'tokenizer', 'tokenizer_v4.json')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
LOGS_PATH = os.path.join(PROJECT_ROOT, 'logs')

# Create directories if they don't exist
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

print(f"✓ Local paths configured:")
print(f"  - Checkpoints: {CHECKPOINT_PATH}")
print(f"  - Tokenizer: {TOKENIZER_PATH}")
print(f"  - Results: {RESULTS_PATH}")
print(f"  - Logs: {LOGS_PATH}")
EOF

# Create a requirements.txt for reference
echo "Creating requirements.txt for reference..."
cat > requirements.txt << 'EOF'
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
matplotlib>=3.7.0
jupyter>=1.0.0
jupyterlab>=4.0.0
datasets>=2.14.0
transformers>=4.30.0
accelerate>=0.20.0
tqdm>=4.65.0
scikit-learn>=1.3.0
ipywidgets>=8.0.0
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use this environment:"
echo "  1. Activate: conda activate ${ENV_NAME}"
echo "  2. Start Jupyter Lab: jupyter lab"
echo "  3. Open Mono_S2S_v1_2.ipynb"
echo ""
echo "IMPORTANT NOTES:"
echo "  - Replace Google Colab drive.mount() calls with:"
echo "      from local_config import *"
echo "  - Replace DRIVE_PATH with DATA_PATH"
echo "  - The notebook expects CUDA. If unavailable, training will be slow."
echo ""
echo "To verify CUDA is working:"
echo "  conda activate ${ENV_NAME}"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "=========================================="

