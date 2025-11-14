# Alpine-Specific Setup Instructions

**CRITICAL:** Alpine's `$HOME` directory has a very small quota (~5GB). You MUST install everything to `/projects/$USER/` where you have TBs of space.

---

## ⚠️ Common Issue: "No space left on device"

If you see this error, your home directory is full. Follow these steps to fix it:

### Step 1: Clean Up Home Directory

```bash
# Remove any conda installations from $HOME
rm -rf ~/miniconda3
rm -rf ~/anaconda3

# Remove broken conda environments
conda env remove -n mono_s2s 2>/dev/null || true

# Clean conda initialization from .bashrc
sed -i '/>>> conda initialize >>>/,/<<< conda initialize <<</d' ~/.bashrc

# Reload shell
source ~/.bashrc

# Verify conda is gone from home
which conda  # Should show "not found" or nothing
```

### Step 2: Check Disk Space

```bash
# Check home directory quota (usually only 2-5 GB!)
df -h ~

# Check projects directory (should show TBs available)
df -h /projects/$USER

# See what's using space in home
du -sh ~/* | sort -h
```

### Step 3: Install Everything to /projects

```bash
# Navigate to projects
cd /projects/$USER

# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install to /projects/$USER/miniconda3 (NOT $HOME!)
bash Miniconda3-latest-Linux-x86_64.sh -b -p /projects/$USER/miniconda3

# Initialize conda (updates .bashrc to use /projects/paco0228/miniconda3)
/projects/$USER/miniconda3/bin/conda init bash

# Reload shell
source ~/.bashrc

# CRITICAL: Verify conda is in /projects (not $HOME)
which conda
# Must show: /projects/$USER/miniconda3/bin/conda
# If it shows ~/miniconda3, something went wrong - redo cleanup step
```

### Step 4: Create Environment and Install Packages

```bash
# Create environment (will go to /projects/$USER/miniconda3/envs/mono_s2s)
conda create -n mono_s2s python=3.10 -y

# Activate
conda activate mono_s2s

# Verify environment location (should be in /projects, not ~)
conda info --envs
# mono_s2s should show: /projects/$USER/miniconda3/envs/mono_s2s

# Install PyTorch (~3GB - needs space!)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
pip install transformers datasets rouge-score scipy pandas tqdm

# Verify everything works
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 5: Run Your Experiment

```bash
# Navigate to code
cd /projects/$USER/mono-s2s/hpc_version

# Make scripts executable
chmod +x run_all.sh jobs/*.sh scripts/*.py

# Validate
./validate_setup.sh

# Edit configuration
nano configs/experiment_config.py
# Verify line 147: SLURM_PARTITION = "aa100"

# Submit!
./run_all.sh
```

---

## 📋 Storage Layout Summary

After proper setup:

```
/projects/$USER/
├── miniconda3/                    ← Conda installed here (~5-10 GB)
│   ├── bin/conda
│   ├── envs/mono_s2s/            ← Python environment here (~3-4 GB)
│   └── pkgs/                     ← Package cache here (~3-5 GB)
├── mono-s2s/                     ← Your code here
│   └── hpc_version/
└── mono_s2s_final_results/       ← Final results copied here (created by pipeline)

$HOME/                            ← Keep this minimal!
└── .bashrc                       ← Only config files

$SCRATCH/                         ← Temporary work files
├── mono_s2s_work/               ← Checkpoints, data cache (created by pipeline)
└── mono_s2s_results/            ← Temporary results (created by pipeline)
```

---

## 💡 Key Points

1. **$HOME quota:** ~2-5 GB (too small for conda!)
2. **$SCRATCH quota:** ~10 TB (auto-cleaned after 90 days)
3. **$PROJECT quota:** ~1-5 TB (persistent, backed up)

**Solution:** Install conda to `/projects/$USER/miniconda3`

All job scripts are configured to use `/projects/$USER/miniconda3` automatically.

---

**Created:** 2025-11-14  
**Platform:** CURC Alpine  
**Issue:** Home directory quota too small for conda/PyTorch

