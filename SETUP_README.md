# MONO-S2S Environment Setup Guide

This guide will help you set up a local conda environment to run the `Mono_S2S_v1_2.ipynb` notebook with CUDA support on Windows or Linux systems.

## Prerequisites

- **Anaconda or Miniconda** installed
  - Download from: https://docs.conda.io/en/latest/miniconda.html
- **NVIDIA GPU** with CUDA support (optional but highly recommended)
  - For CUDA 12.1 support, ensure drivers are up to date
  - Check compatibility: https://developer.nvidia.com/cuda-gpus

## Quick Start

### Linux / macOS (via Terminal)

```bash
# Make the script executable
chmod +x setup_conda_env.sh

# Run the setup script
./setup_conda_env.sh

# Activate the environment
conda activate mono-s2s

# Start Jupyter Lab
jupyter lab
```

### Windows (via Command Prompt or PowerShell)

```batch
# Run the setup script
setup_conda_env.bat

# Activate the environment
conda activate mono-s2s

# Start Jupyter Lab
jupyter lab
```

### Windows (via Git Bash / WSL)

```bash
# Make the script executable
chmod +x setup_conda_env.sh

# Run the setup script
bash setup_conda_env.sh

# Activate the environment
conda activate mono-s2s

# Start Jupyter Lab
jupyter lab
```

## What Gets Installed

The setup script installs:

### Core Dependencies
- **Python 3.10**
- **PyTorch 2.1+** with CUDA 12.1 support
- **NumPy, SciPy, Pandas** - Scientific computing
- **Matplotlib** - Visualization
- **Jupyter Lab** - Notebook interface

### ML/NLP Packages
- **Hugging Face Datasets** - Dataset loading
- **Transformers** - Model utilities
- **Accelerate** - Training optimization

### Additional Utilities
- **tqdm** - Progress bars
- **scikit-learn** - ML utilities
- **ipywidgets** - Interactive notebook widgets

## Project Structure

After setup, your project will have:

```
mono-s2s/
├── Mono_S2S_v1_2.ipynb       # Main notebook
├── local_config.py            # Local path configuration
├── setup_conda_env.sh         # Linux/Mac setup script
├── setup_conda_env.bat        # Windows setup script
├── requirements.txt           # Package dependencies
├── data/
│   ├── checkpoints/          # Model checkpoints
│   └── tokenizer/            # Tokenizer files
├── results/                   # Output results
└── logs/                      # Training logs
```

## Adapting the Notebook for Local Use

The original notebook is designed for Google Colab. Make these changes:

### 1. Replace Google Drive Mounting

**Original (Colab):**
```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE_PATH = '/content/drive/MyDrive/transformer_summarization_v4'
```

**Replace with (Local):**
```python
from local_config import *
# This imports:
# - DATA_PATH (replaces DRIVE_PATH)
# - CHECKPOINT_PATH
# - TOKENIZER_PATH
# - RESULTS_PATH
# - LOGS_PATH
```

### 2. Update Path References

Find and replace throughout the notebook:
- `DRIVE_PATH` → `DATA_PATH`
- `/content/drive/MyDrive/...` → Use the paths from `local_config.py`

### 3. Device Configuration

The notebook already handles device selection:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To verify CUDA is working:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## Verifying Installation

After setup, verify your environment:

### 1. Check Conda Environment

```bash
conda activate mono-s2s
conda list | grep torch
```

Should show PyTorch with CUDA support.

### 2. Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3. Test Dataset Loading

```python
from datasets import load_dataset
ds = load_dataset("knkarthick/dialogsum", split="test[:10]")
print(f"Loaded {len(ds)} samples")
```

## Troubleshooting

### CUDA Not Available

**Symptoms:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. **Check GPU drivers:**
   ```bash
   nvidia-smi  # Linux/WSL
   ```
   
2. **Reinstall PyTorch with CUDA:**
   ```bash
   conda activate mono-s2s
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```

3. **Check CUDA compatibility:**
   - Visit: https://pytorch.org/get-started/locally/
   - Select your CUDA version and install accordingly

### Out of Memory Errors

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. **Reduce batch size** in the notebook:
   ```python
   batch_size = 2  # Reduce from 4
   ```

2. **Enable gradient accumulation** (already in notebook):
   ```python
   accumulation_steps = 8  # Increase to reduce memory per step
   ```

3. **Use mixed precision training:**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### Dataset Download Issues

**Symptoms:** Errors loading Hugging Face datasets

**Solutions:**
1. **Check internet connection**
2. **Clear cache:**
   ```bash
   rm -rf ~/.cache/huggingface/datasets/
   ```
3. **Set proxy if needed:**
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   ```

### Windows-Specific Issues

**Long path errors:**
- Enable long paths: Run PowerShell as admin:
  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```

**Conda not recognized:**
- Ensure Anaconda/Miniconda is in your PATH
- Restart terminal after installation

## Performance Tips

### For Training

1. **Use GPU if available** (automatically detected)
2. **Adjust batch size** based on GPU memory:
   - 4GB VRAM: batch_size=2
   - 8GB VRAM: batch_size=4
   - 16GB+ VRAM: batch_size=8+

3. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi  # Linux
   # Or use Windows Task Manager > Performance > GPU
   ```

### For Development

1. **Use Jupyter Lab extensions:**
   ```bash
   conda activate mono-s2s
   jupyter labextension list
   ```

2. **Enable autoreload for development:**
   ```python
   %load_ext autoreload
   %autoreload 2
   ```

## Updating the Environment

To update packages:

```bash
conda activate mono-s2s
conda update --all
pip install --upgrade datasets transformers
```

To recreate from scratch:

```bash
# Remove old environment
conda env remove -n mono-s2s

# Run setup script again
./setup_conda_env.sh  # Linux/Mac
# or
setup_conda_env.bat   # Windows
```

## Additional Resources

- **PyTorch Documentation:** https://pytorch.org/docs/
- **Hugging Face Datasets:** https://huggingface.co/docs/datasets/
- **Jupyter Lab Guide:** https://jupyterlab.readthedocs.io/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-toolkit

## Support

For issues specific to:
- **PyTorch/CUDA:** https://discuss.pytorch.org/
- **Datasets:** https://discuss.huggingface.co/
- **Notebook code:** Check the project's issue tracker

## License

See LICENSE file in the project root.

