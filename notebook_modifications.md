# Notebook Modifications for Local Execution

This document shows the exact changes needed to run `Mono_S2S_v1_2.ipynb` locally instead of in Google Colab.

## Cell-by-Cell Modifications

### Cell 3: Setup and Imports (Lines 172-210)

**ORIGINAL CODE:**
```python
from google.colab import drive

print("Mounting Google Drive...")
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Setup paths and device
DRIVE_PATH = '/content/drive/MyDrive/transformer_summarization_v4'
CHECKPOINT_PATH = os.path.join(DRIVE_PATH, 'checkpoints')
TOKENIZER_PATH = os.path.join(DRIVE_PATH, 'tokenizer_v4.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

**REPLACE WITH:**
```python
# Import local configuration instead of mounting Google Drive
from local_config import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Local paths are now set via local_config.py:
# - DATA_PATH (replaces DRIVE_PATH)
# - CHECKPOINT_PATH
# - TOKENIZER_PATH
# - RESULTS_PATH
# - LOGS_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Verify CUDA setup
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available. Training will be very slow on CPU.")
```

### Cell 5: Dataset Loading (Lines 213-371)

**ORIGINAL CODE:**
```python
from google.colab import drive
from datasets import load_dataset
```

**REPLACE WITH:**
```python
from datasets import load_dataset
# Google Colab drive import removed - not needed locally
```

**No other changes needed in this cell** - the dataset loading code works the same locally.

### Cell 13: Non-Monotonic Training Setup (Lines 1283-1387)

**ORIGINAL CODE:**
```python
# Mount Google Drive and setup
print("Mounting Google Drive...")
drive.mount('/content/drive')
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
```

**REPLACE WITH:**
```python
# Setup directories (already created by local_config.py)
print("Using local storage...")
print(f"Checkpoints will be saved to: {CHECKPOINT_PATH}")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)  # Ensure it exists
```

**Also update path variables:**
```python
# Update paths for non-monotonic
BEST_MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'best_model_nonmono.pt')
LATEST_MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'latest_model_nonmono.pt')
TRAINING_LOG_PATH = os.path.join(DATA_PATH, 'training_log_nonmono.json')  # Changed from DRIVE_PATH
```

### Cell 14: Monotonic Training Setup (Lines 1401-1490)

**Update path variables:**
```python
# Update paths for monotonic
BEST_MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'best_model_mono.pt')
LATEST_MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'latest_model_mono.pt')
TRAINING_LOG_PATH = os.path.join(DATA_PATH, 'training_log_mono.json')  # Changed from DRIVE_PATH
```

### Cell 15: Adversarial Attack Loading (Lines 1542-1652)

**Update paths when loading models:**
```python
# Results storage - use RESULTS_PATH instead of DRIVE_PATH
results_path = os.path.join(RESULTS_PATH, 'adversarial_results.json')
```

### Cell 17: Results Saving (Lines 2350-2456)

**Update path for saving results:**
```python
# Save results - use RESULTS_PATH
results_path = os.path.join(RESULTS_PATH, 'adversarial_results.json')
with open(results_path, 'w') as f:
    json.dump({...}, f, indent=2)
```

### Cell 18: Visualization Saving (Lines 2459-2613)

**Update plot save path:**
```python
# Save visualization - use RESULTS_PATH
plot_path = os.path.join(RESULTS_PATH, 'adversarial_analysis.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
```

---

## Global Find/Replace Operations

For convenience, you can do these global replacements:

1. **Remove Colab-specific imports:**
   - Find: `from google.colab import drive`
   - Replace: `# Google Colab not needed`

2. **Remove drive mounting:**
   - Find: `drive.mount('/content/drive')`
   - Replace: `# Using local storage`

3. **Update path references:**
   - Find: `DRIVE_PATH`
   - Replace: `DATA_PATH`

---

## Additional Recommendations

### 1. Add Progress Tracking

At the start of training cells, add:
```python
from tqdm.auto import tqdm
import time

start_time = time.time()
print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
```

At the end:
```python
elapsed = time.time() - start_time
print(f"\nTraining completed in: {elapsed/3600:.2f} hours")
```

### 2. Add Checkpointing Info

```python
# Print checkpoint info
if os.path.exists(BEST_MODEL_PATH):
    checkpoint = torch.load(BEST_MODEL_PATH, map_location='cpu')
    print(f"Checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint['val_loss']:.4f}")
```

### 3. Add Memory Management

```python
# Clear GPU cache between models
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory cleared. Available: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
```

### 4. Add Error Handling

Wrap training loops:
```python
try:
    # Training code here
    train_loss = train_stable(...)
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    save_checkpoint(...)  # Save progress
    raise
except Exception as e:
    print(f"\nError during training: {e}")
    save_checkpoint(...)  # Save progress
    raise
```

---

## Verification Checklist

After making modifications, verify:

- [ ] No `google.colab` imports remain
- [ ] No `/content/drive/...` paths remain
- [ ] All paths use variables from `local_config.py`
- [ ] Device is correctly set to CUDA/CPU
- [ ] Checkpoint directories exist
- [ ] Dataset loading works (test with small subset)
- [ ] First training epoch completes successfully

---

## Testing Your Changes

Run this minimal test before full training:

```python
# Test cell - add at the top of notebook
print("Testing local setup...")

# 1. Test imports
try:
    from local_config import *
    print("✓ Local config imported")
except Exception as e:
    print(f"✗ Config import failed: {e}")

# 2. Test paths
import os
for name, path in [
    ("DATA_PATH", DATA_PATH),
    ("CHECKPOINT_PATH", CHECKPOINT_PATH),
    ("RESULTS_PATH", RESULTS_PATH)
]:
    if os.path.exists(path):
        print(f"✓ {name} exists: {path}")
    else:
        print(f"✗ {name} missing: {path}")

# 3. Test CUDA
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA not available (will use CPU)")

# 4. Test dataset loading
try:
    from datasets import load_dataset
    ds_test = load_dataset("knkarthick/dialogsum", split="test[:5]")
    print(f"✓ Dataset loading works: {len(ds_test)} samples loaded")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")

print("\nAll tests completed!")
```

If all tests pass, you're ready to run the full notebook! 🎉

---

## Quick Reference

**Before (Colab):**
```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE_PATH = '/content/drive/MyDrive/...'
```

**After (Local):**
```python
from local_config import *
# DATA_PATH, CHECKPOINT_PATH, etc. are now available
```

---

## Need Help?

- See [QUICKSTART.md](QUICKSTART.md) for basics
- See [SETUP_README.md](SETUP_README.md) for detailed troubleshooting

