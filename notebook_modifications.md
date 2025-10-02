# Notebook Automatic Environment Detection

**IMPORTANT:** The notebook has been updated to **automatically detect** whether it's running in Google Colab or locally. **No manual changes are required!**

## ✅ How It Works

The notebook now automatically:
1. Detects if it's running in Google Colab or locally
2. Mounts Google Drive (Colab only)
3. Configures appropriate file paths
4. Creates necessary directories
5. Reports the environment configuration

## 🔵 Google Colab Behavior

When running in Colab, the notebook automatically:
- Imports `google.colab.drive`
- Mounts Google Drive to `/content/drive`
- Uses paths like `/content/drive/MyDrive/transformer_summarization_v4/`
- Saves checkpoints, logs, and results to Google Drive

## 🟢 Local Execution Behavior

When running locally, the notebook automatically:
- Loads configuration from `local_config.py` (if available)
- Falls back to default local paths if `local_config.py` doesn't exist
- Uses local directories: `data/`, `results/`, `logs/`
- Creates all necessary directories automatically

## 📁 Path Variables

The notebook defines these paths for both environments:

| Variable | Description | Colab Path | Local Path |
|----------|-------------|------------|------------|
| `DRIVE_PATH` | Base data directory | `/content/drive/MyDrive/...` | `./data/` |
| `CHECKPOINT_PATH` | Model checkpoints | `<DRIVE_PATH>/checkpoints/` | `./data/checkpoints/` |
| `TOKENIZER_PATH` | Tokenizer files | `<DRIVE_PATH>/tokenizer_v4.json` | `./data/tokenizer/tokenizer_v4.json` |
| `RESULTS_PATH` | Results and analysis | `<DRIVE_PATH>/results/` | `./results/` |
| `LOGS_PATH` | Training logs | `<DRIVE_PATH>/logs/` | `./logs/` |

## 🚀 Quick Start

### Running in Google Colab
1. Upload notebook to Colab
2. Run all cells
3. When prompted, authorize Google Drive access
4. Everything else is automatic!

### Running Locally
1. Set up environment: `./setup_conda_env.sh` or `setup_conda_env.bat`
2. Activate environment: `conda activate mono-s2s`
3. Start Jupyter: `jupyter lab`
4. Open and run the notebook
5. Everything else is automatic!

## 🔍 Environment Detection Code

The detection happens in Cell 7 with this code:

```python
import sys

# Detect if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    # Google Colab setup
    from google.colab import drive
    drive.mount('/content/drive')
    # ... Colab paths ...
else:
    # Local setup
    try:
        from local_config import *
    except ImportError:
        # Fallback to default paths
        # ... Local paths ...
```

## ⚙️ Advanced: Customizing Local Paths

If you want to customize where data is stored locally, create/edit `local_config.py`:

```python
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/path/to/your/data"  # Customize this
CHECKPOINT_PATH = os.path.join(DATA_PATH, 'checkpoints')
TOKENIZER_PATH = os.path.join(DATA_PATH, 'tokenizer', 'tokenizer_v4.json')
RESULTS_PATH = "/path/to/results"  # Customize this
LOGS_PATH = "/path/to/logs"  # Customize this

# Create directories
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
```

## 📊 What Gets Saved Where

### Checkpoints
- **Location:** `CHECKPOINT_PATH`
- **Files:**
  - `best_model_nonmono.pt` - Best non-monotonic model
  - `latest_model_nonmono.pt` - Latest non-monotonic checkpoint
  - `best_model_mono.pt` - Best monotonic model
  - `latest_model_mono.pt` - Latest monotonic checkpoint

### Training Logs
- **Location:** `LOGS_PATH`
- **Files:**
  - `training_log_nonmono.json` - Non-monotonic training history
  - `training_log_mono.json` - Monotonic training history

### Results
- **Location:** `RESULTS_PATH`
- **Files:**
  - `adversarial_results.json` - Attack analysis results
  - `adversarial_analysis.png` - Visualization plots

### Tokenizer
- **Location:** `TOKENIZER_PATH`
- **File:** `tokenizer_v4.json` - Tokenizer vocabulary

## 🔧 Troubleshooting

### "No module named 'local_config'"

This is **normal** and **not an error**! The notebook will automatically:
1. Try to import `local_config.py`
2. If it doesn't exist, use default paths
3. Print: `⚠ local_config.py not found, using default local paths`

This fallback ensures the notebook works even without the setup script.

### Google Drive Not Mounting

In Colab, if Drive doesn't mount:
1. Check you granted authorization
2. Restart the runtime
3. Run the cell again

### CUDA Not Available Locally

If you see `⚠ CUDA not available`:
1. Check GPU drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA: See [SETUP_README.md](SETUP_README.md)
3. Training will work on CPU but be slower

### Permission Errors

If you get permission errors writing files:
1. Check directory permissions
2. On Linux: `chmod -R 755 data/ results/ logs/`
3. Or customize paths in `local_config.py` to a writable location

## 📝 Key Changes from Original

The original notebook required:
- Manual path changes when switching between Colab and local
- Commenting/uncommenting Drive mount code
- Updating multiple path variables

The updated notebook:
- ✅ **Automatically detects** the environment
- ✅ **No manual changes** needed
- ✅ **Works seamlessly** in both Colab and locally
- ✅ **Creates directories** automatically
- ✅ **Clear status messages** about configuration

## 🎯 Benefits

1. **Portability:** Same notebook works everywhere
2. **Simplicity:** No manual configuration needed
3. **Flexibility:** Easy to customize via `local_config.py`
4. **Robustness:** Fallback paths if config is missing
5. **Clarity:** Clear messages about where files are saved

## 🆘 Need Help?

- **Setup issues:** See [QUICKSTART.md](QUICKSTART.md)
- **Environment setup:** See [SETUP_README.md](SETUP_README.md)
- **CUDA problems:** Check PyTorch documentation
- **General questions:** Check notebook output messages - they tell you exactly what's happening!

---

**Bottom line:** Just run the notebook. It figures everything out automatically! 🚀
