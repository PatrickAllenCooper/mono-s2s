# QUICKSTART - HPC Mono-S2S in 5 Minutes

Get your fair comparison experiments running on HPC in 5 minutes!

---

## 🚀 Step 1: Configure (2 minutes)

```bash
cd hpc_version
nano configs/experiment_config.py
```

**Edit these 3 lines:**
```python
# Line 19-20: Your HPC paths
SCRATCH_DIR = "/scratch/summit/YOUR_USERNAME"  # ← Change this
PROJECT_DIR = "/projects/YOUR_PROJECT"          # ← Change this

# Line 81: Your partition
SLURM_PARTITION = "shas"  # ← Change if needed (shas, aa100, blanca-ics, etc.)
```

Save and exit (Ctrl+O, Enter, Ctrl+X)

---

## 🚀 Step 2: Setup Environment (2 minutes)

```bash
# Load modules
module load python/3.10.0
module load cuda/11.8

# Install dependencies (one-time)
pip install --user transformers datasets torch rouge-score pandas scipy matplotlib tqdm

# Or use conda (recommended)
module load anaconda
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers datasets rouge-score pandas scipy matplotlib tqdm
```

---

## 🚀 Step 3: Submit Jobs (1 minute)

```bash
# Make scripts executable (one-time)
chmod +x run_all.sh jobs/*.sh

# Submit all jobs with automatic dependencies
./run_all.sh

# That's it! Jobs are now queued.
```

---

## 📊 Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch logs in real-time
tail -f logs/job_*_*.out

# Check completion
ls -la $SCRATCH/mono_s2s_work/*.flag
```

---

## ⏱️ Timeline

| Stage | Time | Description |
|-------|------|-------------|
| 0. Setup | 10 min | Download model |
| 1. Data | 1 hr | Load datasets |
| 2-3. Training | 8-16 hrs | **Train both models** (parallel) |
| 4. Evaluation | 2-4 hrs | Evaluate all 3 models |
| 5-6. Attacks | 3-4 hrs | UAT + HotFlip (parallel) |
| 7. Aggregate | 5 min | Final analysis |
| **TOTAL** | **12-25 hrs** | **Fully automated** |

---

## ✅ When Complete

Results will be in:
```
$SCRATCH/mono_s2s_results/
├── evaluation_results.json    # ← PRIMARY RESULTS with bootstrap CIs
├── transfer_matrix.json       # ← Cross-model attack analysis
├── experiment_metadata.json   # ← Complete configuration
└── final_results.json         # ← Aggregated analysis
```

---

## 🎯 Quick Test Mode (Fast)

Before running full experiment, test with quick mode:

```bash
# Edit config
nano configs/experiment_config.py

# Set this line to False:
USE_FULL_TEST_SETS = False  # Uses 200 samples instead of 11k

# Run (completes in ~4-6 hours instead of 12-25)
./run_all.sh
```

---

## 🆘 Common Issues

**"Invalid partition"**
→ Run `sinfo`, use that partition name in config

**"Out of memory"**
→ Set `BATCH_SIZE = 2` in config

**"Timeout"**
→ Increase `#SBATCH --time` in job_2 and job_3

**"Module not found"**
→ Run `module avail python`, use exact version

---

## 📖 Full Documentation

- **README.md** - Complete overview
- **CURC_SETUP_GUIDE.md** - Detailed HPC setup
- **configs/experiment_config.py** - All settings explained

---

**That's it!** Your experiments are now running on HPC. ✨

**Estimated completion:** 12-25 hours (automated, with email notifications if configured)

