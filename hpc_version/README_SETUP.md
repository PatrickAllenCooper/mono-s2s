# Mono-S2S HPC Setup Guide

**Version:** 1.7 - HPC Edition  
**Status:** ✅ **READY FOR EXECUTION**  
**Platform:** CURC (Summit, Alpine, Blanca) and other SLURM-based HPC clusters

---

## 🎉 Implementation Complete!

All 7 stages are now fully implemented:

### ✅ **Completed Components**

**Python Stage Scripts:**
- ✅ `stage_0_setup.py` - Environment setup and model download
- ✅ `stage_1_prepare_data.py` - Dataset loading and caching
- ✅ `stage_2_train_baseline.py` - Train baseline T5 (unconstrained)
- ✅ `stage_3_train_monotonic.py` - Train monotonic T5 (W≥0 FFN)
- ✅ `stage_4_evaluate.py` - Comprehensive evaluation with bootstrap CIs
- ✅ `stage_5_uat_attacks.py` - UAT attacks with transfer matrix
- ✅ `stage_6_hotflip_attacks.py` - Gradient-based HotFlip attacks
- ✅ `stage_7_aggregate.py` - Result aggregation and analysis

**SLURM Job Scripts:**
- ✅ `job_0_setup.sh` - SLURM wrapper for stage 0
- ✅ `job_1_data.sh` - SLURM wrapper for stage 1
- ✅ `job_2_baseline.sh` - SLURM wrapper for stage 2
- ✅ `job_3_monotonic.sh` - SLURM wrapper for stage 3
- ✅ `job_4_evaluate.sh` - SLURM wrapper for stage 4
- ✅ `job_5_uat.sh` - SLURM wrapper for stage 5
- ✅ `job_6_hotflip.sh` - SLURM wrapper for stage 6
- ✅ `job_7_aggregate.sh` - SLURM wrapper for stage 7

**Infrastructure:**
- ✅ `configs/experiment_config.py` - Centralized configuration
- ✅ `utils/common_utils.py` - Shared utility functions
- ✅ `run_all.sh` - Master orchestration script
- ✅ `validate_setup.sh` - Setup validation script
- ✅ `QUICKSTART.md` - Detailed user guide
- ✅ This `README_SETUP.md` - Setup documentation

---

## 🚀 Quick Start (3 Steps)

### Step 1: Validate Your Setup

```bash
# Install Miniconda to /projects (NOT $HOME - home directory is too small!)
cd /projects/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /projects/$USER/miniconda3
/projects/$USER/miniconda3/bin/conda init bash
source ~/.bashrc

# Create conda environment
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s

# Clone to /projects for larger disk space
cd /projects/$USER
git clone https://github.com/PatrickAllenCooper/mono-s2s.git
cd mono-s2s/hpc_version

# Install dependencies
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers datasets rouge-score scipy pandas tqdm

# Validate environment
./validate_setup.sh
```

This checks:
- Python version and packages
- CUDA availability
- SLURM commands
- File structure
- Configuration placeholders

### Step 2: Configure for Your Cluster

Edit `configs/experiment_config.py`:

```python
# Lines 23-24: Set YOUR paths
SCRATCH_DIR = os.environ.get("SCRATCH", "/scratch/summit/YOUR_USERNAME")
PROJECT_DIR = os.environ.get("PROJECT", "/projects/YOUR_PROJECT")

# Line 147: Set YOUR partition
SLURM_PARTITION = "shas"  # Change to your GPU partition
```

**CURC Cluster-Specific Settings:**

| Cluster | SCRATCH | PROJECT | Partition |
|---------|---------|---------|-----------|
| **Summit** | `/scratch/summit/$USER` | `/projects/$USER` | `shas` or `sgpu` |
| **Alpine** | `/scratch/alpine/$USER` | `/pl/active/$USER` | `aa100` or `ami100` |
| **Blanca** | `/rc_scratch/$USER` | `/projects/your_group` | `blanca-ics` |

### Step 3: Submit Jobs

```bash
# Make scripts executable (one-time)
chmod +x run_all.sh jobs/*.sh scripts/*.py

# Submit all jobs with automatic dependencies
./run_all.sh

# OR specify a custom seed
./run_all.sh 42
```

That's it! Jobs will run automatically with proper dependencies.

---

## 📊 What Gets Executed

### Pipeline Overview

```
Stage 0: Setup (30 min, no GPU)
    ↓
Stage 1: Data Preparation (2 hr, no GPU)
    ↓
    ├─→ Stage 2: Train Baseline (12 hr, GPU)    ← Run in parallel
    └─→ Stage 3: Train Monotonic (12 hr, GPU)   ← Run in parallel
         ↓
Stage 4: Comprehensive Evaluation (4 hr, GPU)
    ↓
    ├─→ Stage 5: UAT Attacks (3 hr, GPU)        ← Run in parallel
    └─→ Stage 6: HotFlip Attacks (2 hr, GPU)    ← Run in parallel
         ↓
Stage 7: Aggregate Results (15 min, no GPU)
```

**Total Wall-Clock Time:** ~20 hours (with parallelization)  
**Total GPU Hours:** ~31 hours per seed  
**Total Compute (SUs):** ~400-600 (cluster-dependent)

### Three-Model Fair Comparison

1. **Standard T5** (pre-trained, not fine-tuned) - Reference
2. **Baseline T5** (fine-tuned, unconstrained) - Fair baseline
3. **Monotonic T5** (fine-tuned, W≥0 FFN constraints) - Treatment

All use **identical**:
- Training data and hyperparameters
- Tokenization and max lengths
- Optimizer (AdamW), LR schedule, batch size
- Decoding parameters (num_beams, length_penalty, etc.)
- Evaluation metrics (ROUGE with bootstrap CIs)

---

## 📦 Dependencies

### Required Python Packages

**For CURC Alpine (Python 3.6.8 is too old - use Miniconda):**

```bash
# 1. Install Miniconda to /projects (NOT $HOME - home dir is too small!)
cd /projects/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /projects/$USER/miniconda3
/projects/$USER/miniconda3/bin/conda init bash
source ~/.bashrc

# 2. Create conda environment with Python 3.10
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s

# 3. Install PyTorch with CUDA support
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install other packages
pip install transformers datasets rouge-score scipy pandas tqdm matplotlib

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note:** Alpine's system Python 3.6.8 is too old for modern PyTorch. Miniconda provides a local Python 3.10 installation. CUDA will be available on GPU compute nodes automatically.

### Alternative: Other HPC Systems

If your cluster has module system with Python/Conda:

```bash
# Find available modules
module spider python
module avail

# Load required modules
module load anaconda  # or python
module load cuda

# Create conda environment
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers datasets rouge-score scipy pandas tqdm
```

---

## 🔍 Monitoring Progress

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Watch in real-time
watch -n 10 squeue -u $USER

# Check specific job
scontrol show job <job_id>
```

### View Logs

```bash
# Latest outputs
tail -f logs/job_*.out

# Specific stage
tail -f logs/job_2_baseline_*.out

# Stage-specific detailed logs
cat $SCRATCH/mono_s2s_work/stage_logs/stage_2_train_baseline.log
```

### Check Completion Flags

```bash
# List all completion flags
ls -la $SCRATCH/mono_s2s_work/*.flag

# Should see 8 flags when complete:
# - stage_0_setup_complete.flag
# - stage_1_data_prep_complete.flag
# - stage_2_train_baseline_complete.flag
# - stage_3_train_monotonic_complete.flag
# - stage_4_evaluate_complete.flag
# - stage_5_uat_complete.flag
# - stage_6_hotflip_complete.flag
# - stage_7_aggregate_complete.flag
```

---

## 📂 Storage Best Practices

**Always clone to `/projects/$USER/` not `$HOME`:**
- `/projects/` has much larger quota (typically TBs vs GBs)
- Persistent and backed up
- Good for code, environments, and permanent files

**Use `$SCRATCH` for temporary work:**
- Fast I/O, not backed up
- Auto-cleaned after 90 days of inactivity
- Perfect for checkpoints, data cache, intermediate results

**Copy final results to `$PROJECT`:**
- Long-term persistent storage
- Backed up regularly
- The pipeline automatically copies results here

## 📂 Output Structure

### Code Location (persistent, backed up)

```
/projects/$USER/mono-s2s/hpc_version/
└── (Your code and scripts - cloned here)
```

### Working Directory (Scratch - temporary, auto-cleaned after 90 days)

```
$SCRATCH/mono_s2s_work/
├── checkpoints/
│   ├── baseline_checkpoints/
│   │   ├── best_model.pt
│   │   └── checkpoint_epoch_*.pt
│   └── monotonic_checkpoints/
│       ├── best_model.pt
│       └── checkpoint_epoch_*.pt
├── data_cache/
│   ├── train_data.pt
│   ├── val_data.pt
│   ├── test_data.pt
│   └── attack_data.pt
├── stage_logs/
│   └── stage_*.log
└── *.flag (completion markers)

$SCRATCH/mono_s2s_results/
├── evaluation_results.json       ★ PRIMARY RESULTS
├── uat_results.json
├── hotflip_results.json
├── baseline_training_history.json
├── monotonic_training_history.json
├── learned_triggers.csv
├── final_results.json             ★ AGGREGATED RESULTS
└── experiment_summary.txt         ★ HUMAN-READABLE
```

### Permanent Storage (Project - persistent, backed up)

```
$PROJECT/mono_s2s_final_results/
└── (Copy of all results from scratch - kept permanently)
```

**Important:** Always work from `/projects/$USER/` for code to ensure you have enough disk space.

---

## ⚙️ Configuration Options

### Key Settings in `experiment_config.py`

```python
# Model selection
MODEL_NAME = "t5-small"  # Options: t5-small, t5-base, t5-large

# Testing mode (set False for quick testing)
USE_FULL_TEST_SETS = True  # False = 200 samples, True = full datasets

# Training hyperparameters
NUM_EPOCHS = 3        # Reduce to 2 for faster testing
BATCH_SIZE = 4        # Reduce to 2 if OOM
LEARNING_RATE = 3e-5  # Standard T5 fine-tuning LR

# Attack configuration
ATTACK_TRIGGER_LENGTH = 5         # UAT trigger length
ATTACK_NUM_ITERATIONS = 50        # UAT optimization iterations
ATTACK_NUM_RESTARTS = 3           # UAT random restarts
ATTACK_NUM_CANDIDATES = 100       # HotFlip candidate tokens

# Multi-seed experiments
RANDOM_SEEDS = [42, 1337, 2024, 8888, 12345]
CURRENT_SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))
```

### Resource Requests (in job scripts)

```bash
# Adjust in jobs/*.sh based on your cluster
#SBATCH --mem=64G           # Memory
#SBATCH --time=12:00:00     # Time limit
#SBATCH --gres=gpu:1        # Number of GPUs
#SBATCH --partition=shas    # GPU partition
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Job Fails Immediately**

```bash
# Check module availability
module avail python
module avail cuda

# Verify paths
echo $SCRATCH
echo $PROJECT

# Review error log
cat logs/job_*_<jobid>.err
```

#### 2. **Out of Memory (OOM)**

```python
# In experiment_config.py
BATCH_SIZE = 2              # Reduce from 4
EVAL_BATCH_SIZE = 4         # Reduce from 8
```

```bash
# In job scripts
#SBATCH --mem=128G  # Increase from 64G
```

#### 3. **Wrong Partition**

```bash
# Check available partitions
sinfo
sacctmgr show assoc where user=$USER format=partition,qos%50

# Update in experiment_config.py
SLURM_PARTITION = "your_correct_partition"
```

#### 4. **Dataset Download Fails**

```bash
# Pre-download on login node (has internet)
export HF_DATASETS_CACHE=$PROJECT/hf_cache
python -c "from datasets import load_dataset; load_dataset('samsum')"

# Then add to job scripts
export HF_DATASETS_CACHE=$PROJECT/hf_cache
```

---

## 📈 Expected Results

### After All Stages Complete

You will have:

1. **evaluation_results.json** - Three-model ROUGE comparison with bootstrap 95% CIs
2. **uat_results.json** - UAT attack results + transfer matrix
3. **hotflip_results.json** - Gradient attack results with statistics
4. **final_results.json** - Comprehensive aggregated analysis
5. **experiment_summary.txt** - Human-readable summary tables

### Key Metrics

- **ROUGE-1, ROUGE-2, ROUGE-L** (with confidence intervals)
- **Attack robustness** (ROUGE degradation, loss increase)
- **Transfer attack matrix** (cross-model vulnerability)
- **Training statistics** (loss curves, training time)
- **Length statistics** (brevity penalty, length ratios)

---

## 🎯 Next Steps After Completion

### 1. Review Results

```bash
# Primary results with CIs
cat $SCRATCH/mono_s2s_results/evaluation_results.json | jq .

# Human-readable summary
cat $SCRATCH/mono_s2s_results/experiment_summary.txt

# Final aggregated results
cat $SCRATCH/mono_s2s_results/final_results.json | jq .
```

### 2. Run Multiple Seeds

```bash
# For robust statistics, run with all 5 seeds
for seed in 42 1337 2024 8888 12345; do
    ./run_all.sh $seed
    # Wait for completion, then rename results
    mv $SCRATCH/mono_s2s_results $SCRATCH/mono_s2s_results_seed_$seed
done
```

### 3. Clean Up Scratch

```bash
# After copying results to project
rm -rf $SCRATCH/mono_s2s_work/data_cache  # Can re-download
rm -rf $SCRATCH/mono_s2s_work/checkpoints/*/checkpoint_epoch_*.pt  # Keep best only
```

---

## 📚 Additional Resources

- **QUICKSTART.md** - Detailed step-by-step guide
- **CURC Documentation** - https://curc.readthedocs.io/
- **Experiment Config** - `configs/experiment_config.py` (extensive comments)
- **Common Utils** - `utils/common_utils.py` (function documentation)

### CURC-Specific Links

- **Alpine Quick Start** - https://curc.readthedocs.io/en/latest/clusters/alpine/quick-start.html
- **Slurm Guide** - https://curc.readthedocs.io/en/latest/running-jobs/batch-jobs.html
- **Support** - rc-help@colorado.edu

---

## ✅ Success Criteria

Your experiment is successful if:

- [x] All 8 completion flags exist in `$SCRATCH/mono_s2s_work/`
- [x] Both checkpoint files exist (baseline + monotonic)
- [x] `evaluation_results.json` has bootstrap CIs for all models
- [x] `uat_results.json` has transfer matrix
- [x] `hotflip_results.json` has attack statistics
- [x] No "FAILED" in any `logs/job_*.out`
- [x] Results copied to `$PROJECT/mono_s2s_final_results/`

---

## 🎉 Ready to Run!

Your HPC implementation is **complete and ready for execution**!

### Final Checklist

- [ ] Run `./validate_setup.sh` to check environment
- [ ] Edit `configs/experiment_config.py` with your paths and partition
- [ ] Make scripts executable: `chmod +x run_all.sh jobs/*.sh scripts/*.py`
- [ ] Submit jobs: `./run_all.sh`
- [ ] Monitor progress: `squeue -u $USER`
- [ ] Check results in 12-25 hours

**Questions?** Check `QUICKSTART.md` or contact CURC support (rc-help@colorado.edu)

---

**Good luck with your experiments!** 🚀

---

*Last updated: 2025-11-13*  
*Version: HPC Edition v1.7*  
*Status: Production Ready*

