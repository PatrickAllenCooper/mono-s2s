# MONO-S2S Quick Start Guide

## TL;DR

```bash
# Linux/Mac
./setup_conda_env.sh
conda activate mono-s2s
jupyter lab

# Windows
setup_conda_env.bat
conda activate mono-s2s
jupyter lab
```

Then modify the notebook to use `local_config.py` instead of Google Drive paths.

---

## One-Time Setup (5-10 minutes)

### Step 1: Run Setup Script

**Linux/Mac:**
```bash
chmod +x setup_conda_env.sh
./setup_conda_env.sh
```

**Windows (Command Prompt):**
```batch
setup_conda_env.bat
```

**Windows (Git Bash):**
```bash
bash setup_conda_env.sh
```

### Step 2: Verify Installation

```bash
conda activate mono-s2s
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Expected output: `CUDA: True` (if you have an NVIDIA GPU)

---

## Every Time You Work

### Start Jupyter

```bash
conda activate mono-s2s
jupyter lab
```

This opens Jupyter Lab in your browser at `http://localhost:8888`

### Open the Notebook

1. In Jupyter Lab, navigate to `Mono_S2S_v1_2.ipynb`
2. Click to open

---

## Notebook Modifications Required

### Change 1: Import Local Config

**Find this cell (near the top):**
```python
from google.colab import drive
print("Mounting Google Drive...")
drive.mount('/content/drive')
```

**Replace with:**
```python
from local_config import *
print("Using local configuration...")
```

### Change 2: Remove Colab-specific Code

**Delete or comment out:**
```python
# Any lines with:
# - drive.mount()
# - /content/drive/...
# - from google.colab import ...
```

### Change 3: Path Updates (Automatic)

The `local_config.py` provides these variables:
- `DATA_PATH` - replaces `/content/drive/MyDrive/...`
- `CHECKPOINT_PATH` - for model checkpoints
- `TOKENIZER_PATH` - for tokenizer files
- `RESULTS_PATH` - for outputs
- `LOGS_PATH` - for training logs

These are automatically set when you import `local_config`.

---

## Common Commands

### Activate Environment
```bash
conda activate mono-s2s
```

### Deactivate Environment
```bash
conda deactivate
```

### Check GPU Status (Linux/Windows with WSL)
```bash
nvidia-smi
```

### Check GPU in Python
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Monitor GPU Usage During Training
```bash
# Linux/WSL - run in separate terminal
watch -n 1 nvidia-smi

# Windows - use Task Manager
# Performance tab > GPU section
```

---

## Troubleshooting Quick Fixes

### "CUDA not available"

```bash
conda activate mono-s2s
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### "Out of memory"

In notebook, reduce batch size:
```python
batch_size = 2  # Change from 4
```

### "Dataset download fails"

```bash
rm -rf ~/.cache/huggingface/datasets/  # Clear cache
```

### "Kernel dies during training"

1. Reduce batch size
2. Increase swap space (Linux)
3. Close other applications
4. Check GPU memory with `nvidia-smi`

---

## Key Files

| File | Purpose |
|------|---------|
| `Mono_S2S_v1_2.ipynb` | Main notebook (modify this) |
| `local_config.py` | Path configuration (auto-generated) |
| `setup_conda_env.sh` | Setup script (Linux/Mac) |
| `setup_conda_env.bat` | Setup script (Windows) |
| `requirements.txt` | Package list (reference) |
| `data/checkpoints/` | Saved models go here |
| `results/` | Output files go here |

---

## Training Time Estimates

With NVIDIA GPU:
- **Non-monotonic model:** ~2-4 hours (10 epochs)
- **Monotonic model:** ~2-4 hours (10 epochs)
- **Total (both models):** ~4-8 hours

Without GPU (CPU only):
- **Each model:** ~20-40 hours
- **Not recommended** for full training
- Consider using a cloud GPU service

---

## Need More Help?

See the full guide: [SETUP_README.md](SETUP_README.md)

## System Requirements

### Minimum
- **RAM:** 8GB
- **Storage:** 10GB free
- **CPU:** 4 cores
- **GPU:** Optional (but recommended)

### Recommended
- **RAM:** 16GB+
- **Storage:** 20GB+ free
- **CPU:** 8+ cores
- **GPU:** NVIDIA with 8GB+ VRAM (e.g., RTX 3060, RTX 3070, etc.)

---

**Ready to start?** Run the setup script and open Jupyter Lab! 🚀

