# HPC Pipeline Guide

**Pipeline Version:** Production (ICML 2025)  
**Last Updated:** 2026-01-29  
**Status:** Ready for deployment on SLURM clusters

---

## Overview

This directory contains the production SLURM/HPC pipeline for running fair-comparison experiments testing whether local monotonic constraints in T5 feed-forward sublayers (FFNs) improve adversarial robustness in seq2seq summarization.

**Key Features:**
- Fully automated 7-stage pipeline with dependency management
- Fair experimental design (both models train 7 epochs)
- Full test sets enabled (11,490 examples for CNN/DM)
- Timestamp-labeled results for tracking
- Robust error handling and recovery

---

## Quick Start (CURC Alpine)

### Prerequisites

Alpine `$HOME` quota is tiny (~2-5GB). Install conda under `/projects/$USER/`:

```bash
ssh your_username@login.rc.colorado.edu

# Remove old conda installation if in $HOME
rm -rf ~/miniconda3 ~/anaconda3
sed -i '/>>> conda initialize >>>/,/<<< conda initialize <<</d' ~/.bashrc
source ~/.bashrc

# Install Miniconda to /projects
cd /projects/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /projects/$USER/miniconda3
/projects/$USER/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Setup

```bash
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s

cd /projects/$USER
git clone https://github.com/PatrickAllenCooper/mono-s2s.git
cd mono-s2s/hpc_version

# PyTorch (prefer pip on Alpine to avoid conda/MKL conflicts)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install transformers datasets rouge-score scipy pandas tqdm sentencepiece protobuf

# Make scripts executable
chmod +x run_all.sh jobs/*.sh scripts/*.py

# Validate setup
./validate_setup.sh
```

### Configuration

Edit `configs/experiment_config.py`:

```python
# Set your HPC paths
SCRATCH_DIR = "/scratch/$USER"  # or use $SCRATCH environment variable
PROJECT_DIR = "/projects/$USER"  # or use $PROJECT environment variable

# Set your partition
SLURM_PARTITION = "aa100"  # For A100 GPUs on Alpine
```

### Run

```bash
cd /projects/$USER/mono-s2s/hpc_version
conda activate mono_s2s

./run_all.sh          # Default seed (42)
./run_all.sh 1337     # Custom seed
```

### Monitor

```bash
# Check job queue
squeue -u $USER

# Watch logs
tail -f logs/job_*.out

# Check completion flags
ls $SCRATCH/mono_s2s_work/*.flag

# View stage logs
tail -f $SCRATCH/mono_s2s_work/stage_logs/stage_*.log
```

---

## Pipeline Stages

The pipeline runs 7 stages with automatic dependency management:

```
Stage 0: Setup
    ↓
Stage 1: Data Preparation
    ↓
    ├── Stage 2: Train Baseline (parallel)
    └── Stage 3: Train Monotonic (parallel)
         ↓
    Stage 4: Evaluation
         ↓
         ├── Stage 5: UAT Attacks (parallel)
         └── Stage 6: HotFlip Attacks (parallel)
              ↓
         Stage 7: Aggregate Results
```

### Stage Details

**Stage 0: Setup**
- Downloads and validates T5-small model (~240MB)
- Creates working directories
- Verifies GPU availability and library versions
- Time: ~30 minutes | Memory: 16 GB | GPU: No

**Stage 1: Data Preparation**
- Loads training datasets (DialogSum, HighlightSum, arXiv)
- Loads evaluation datasets (CNN/DM, XSUM, SAMSum)
- Computes dataset statistics
- Creates preprocessed cache
- Time: ~2 hours | Memory: 32 GB | GPU: No

**Stage 2: Train Baseline**
- Fine-tunes T5-small without constraints
- 7 epochs, learning rate 5e-5, batch size 4
- Saves checkpoints and training history
- Time: ~10-12 hours | Memory: 64 GB | GPU: Yes (A100 recommended)

**Stage 3: Train Monotonic**
- Fine-tunes T5-small with FFN non-negativity constraints
- 7 epochs (same as baseline for fair comparison)
- Extended warmup (15% vs 10%) for softplus stability
- Saves checkpoints and training history
- Time: ~10-12 hours | Memory: 64 GB | GPU: Yes (A100 recommended)

**Stage 4: Evaluation**
- Evaluates both models on CNN/DM, XSUM, SAMSum test sets
- Computes ROUGE-1/2/L with bootstrap 95% confidence intervals
- Uses full test sets (11,490 examples for CNN/DM)
- Time: ~2-4 hours | Memory: 32 GB | GPU: Yes

**Stage 5: UAT Attacks**
- Learns universal adversarial triggers via coordinate ascent
- 5 trigger types, 100 iterations, 5 restarts
- Evaluates on 1,500 examples
- Computes cross-model transfer attack matrix
- Time: ~2-3 hours | Memory: 16 GB | GPU: Yes

**Stage 6: HotFlip Attacks**
- Gradient-based token flipping attacks
- Up to 5 flips per example
- Measures attack success rate (15% ROUGE-L degradation threshold)
- Evaluates on 1,500 examples
- Time: ~1-2 hours | Memory: 16 GB | GPU: Yes

**Stage 7: Aggregate Results**
- Combines all results into final_results.json
- Generates human-readable experiment_summary.txt
- Copies results to permanent storage
- Adds timestamp metadata to all outputs
- Time: ~15 minutes | Memory: 8 GB | GPU: No

---

## Resource Requirements

### Per-Stage Requirements (Alpine aa100)

| Stage | GPU | Memory | Time | Partition |
|------:|:---:|:------:|:----:|-----------|
| 0 | No | 16 GB | 30 min | amilan |
| 1 | No | 32 GB | 2 hr | amilan |
| 2 | Yes | 64 GB | 10-12 hr | aa100 |
| 3 | Yes | 64 GB | 10-12 hr | aa100 |
| 4 | Yes | 32 GB | 2-4 hr | aa100 |
| 5 | Yes | 16 GB | 2-3 hr | aa100 |
| 6 | Yes | 16 GB | 1-2 hr | aa100 |
| 7 | No | 8 GB | 15 min | amilan |

### Total Resource Usage

- **Wall Time:** ~30-35 hours (with parallelization)
- **GPU Hours:** ~25-30 hours (A100)
- **Storage:** ~50 GB (checkpoints + results)
- **Scratch Space:** ~100 GB (temporary working files)

---

## Configuration Guide

### Main Configuration File

**Location:** `configs/experiment_config.py`

### Critical Settings (Fair Comparison)

```python
# Training Configuration (BOTH MODELS IDENTICAL)
NUM_EPOCHS = 7                    # Baseline epochs
MONOTONIC_NUM_EPOCHS = 7          # Monotonic epochs (MUST match)
LEARNING_RATE = 5e-5              # Both models
BATCH_SIZE = 4                    # Both models
GRADIENT_CLIP = 1.0               # Both models
WEIGHT_DECAY = 0.01               # Both models

# Only Difference: Warmup (for softplus stability)
WARMUP_RATIO = 0.10               # Baseline: 10%
MONOTONIC_WARMUP_RATIO = 0.15     # Monotonic: 15%
```

### Evaluation Settings

```python
# Use full test sets for adequate statistical power
USE_FULL_TEST_SETS = True         # 11,490 examples (CNN/DM)

# Bootstrap confidence intervals
BOOTSTRAP_SAMPLES = 1000          # 95% CIs

# Attack evaluation
TRIGGER_EVAL_SIZE_FULL = 1500     # UAT/HotFlip sample size
```

### Analysis Tracking

```python
# Enable for paper analysis (no performance impact)
TRACK_TRAINING_TIME = True        # Computational cost analysis
TRACK_INFERENCE_TIME = True       # Overhead measurement
COMPUTE_GRADIENT_NORMS = True     # Mechanistic understanding
```

### Path Configuration

```python
# Customize for your HPC environment
SCRATCH_DIR = "/scratch/$USER"    # Fast temporary storage
PROJECT_DIR = "/projects/$USER"   # Permanent storage
SLURM_PARTITION = "aa100"         # GPU partition name
SLURM_ACCOUNT = "your_account"    # Optional: accounting group
```

---

## Directory Structure

```
hpc_version/
├── configs/
│   └── experiment_config.py      # All configuration settings
├── jobs/
│   ├── job_0_setup.sh            # SLURM wrapper for Stage 0
│   ├── job_1_data.sh             # Stage 1
│   ├── job_2_baseline.sh         # Stage 2
│   ├── job_3_monotonic.sh        # Stage 3
│   ├── job_4_evaluate.sh         # Stage 4
│   ├── job_5_uat.sh              # Stage 5
│   ├── job_6_hotflip.sh          # Stage 6
│   └── job_7_aggregate.sh        # Stage 7
├── scripts/
│   ├── stage_0_setup.py          # Setup implementation
│   ├── stage_1_prepare_data.py   # Data loading
│   ├── stage_2_train_baseline.py # Baseline training
│   ├── stage_3_train_monotonic.py # Monotonic training
│   ├── stage_4_evaluate.py       # Evaluation
│   ├── stage_5_uat_attacks.py    # UAT attacks
│   ├── stage_6_hotflip_attacks.py # HotFlip attacks
│   ├── stage_7_aggregate.py      # Results aggregation
│   └── aggregate_multi_seed.py   # Multi-seed analysis
├── utils/
│   └── common_utils.py           # Shared utilities
├── run_all.sh                    # Master pipeline script
├── run_multi_seed.sh             # Multi-seed experiments
├── validate_setup.sh             # Environment validation
├── check_training_status.sh      # Status monitoring
├── diagnose_and_recover.sh       # Error recovery
├── clean_all.sh                  # Cleanup utility
└── clean_checkpoints.sh          # Checkpoint cleanup
```

---

## Key Improvements

### Fair Comparison Implementation

**Problem (Original):**
- Baseline: 5 epochs
- Monotonic: 7 epochs
- MAJOR VALIDITY THREAT

**Fix (Current):**
- Both models: 7 epochs
- Fair comparison achieved
- Results now valid for publication

### Adequate Statistical Power

**Problem (Original):**
- Evaluation on only 200 examples
- Insufficient for detecting effects
- Confidence intervals too wide

**Fix (Current):**
- USE_FULL_TEST_SETS = True
- 11,490 examples for CNN/DM
- Statistical power adequate

### Improved Initialization

**Problem (Original):**
```python
# Random initialization destroyed pretrained knowledge
V = torch.randn_like(W)
```

**Fix (Current):**
```python
# Preserve pretrained features with inverse softplus
W_abs = torch.abs(W_pretrained) + eps
V = torch.log(torch.exp(W_abs) - 1.0 + eps)
```

**Impact:** Better starting point, faster convergence, less performance degradation

### Robust Dataset Loading

**Problem (Original):**
- Single load attempt
- Pipeline fails on transient HuggingFace errors
- XSUM and SAMSum disabled

**Fix (Current):**
```python
# Retry logic with exponential backoff
load_dataset_split(..., max_retries=3, retry_delay=10)
# Graceful fallback if partial datasets unavailable
```

**Impact:** All 3 evaluation datasets (CNN/DM, XSUM, SAMSum) now working

### Timestamp Labeling

**Problem (Original):**
- No metadata in result files
- Can't track which run produced which results

**Fix (Current):**
```json
{
  "results": {...},
  "_metadata": {
    "timestamp": "2026-01-29 10:15:30",
    "run_id": "20260129_101530_seed42",
    "seed": 42,
    "unix_timestamp": 1738150530
  }
}
```

**Impact:** Full traceability and reproducibility

---

## Output Files

### Location

- **Working Directory:** `$SCRATCH/mono_s2s_work/`
- **Results (scratch):** `$SCRATCH/mono_s2s_results/`
- **Results (permanent):** `$PROJECT/mono_s2s_final_results/`

### Key Result Files

```
$SCRATCH/mono_s2s_results/
├── setup_complete.json                # Environment validation
├── data_statistics.json               # Dataset splits and statistics
├── baseline_training_history.json     # Baseline training curves
├── monotonic_training_history.json    # Monotonic training curves
├── evaluation_results.json            # ROUGE scores + 95% CIs
├── uat_results.json                   # UAT attacks + transfer matrix
├── hotflip_results.json               # HotFlip attack results
├── final_results.json                 # Aggregated results
├── experiment_summary.txt             # Human-readable summary
└── learned_triggers.csv               # UAT triggers (for inspection)
```

### Checkpoint Files

```
$SCRATCH/mono_s2s_work/
├── baseline_model/                    # Baseline checkpoints
│   ├── epoch_1/
│   ├── epoch_2/
│   └── ...
└── monotonic_model/                   # Monotonic checkpoints
    ├── epoch_1/
    ├── epoch_2/
    └── ...
```

---

## Troubleshooting

### Common Issues

**1. Disk Quota Exceeded (Alpine)**

Problem: "No space left on device"

Solution:
```bash
# Check quota
df -h $HOME
df -h /projects/$USER

# Move conda to /projects
rm -rf ~/miniconda3
# Follow installation instructions above
```

**2. CUDA Out of Memory**

Problem: "CUDA out of memory" during training

Solution:
```python
# Edit configs/experiment_config.py
BATCH_SIZE = 2              # Reduce from 4
EVAL_BATCH_SIZE = 4         # Reduce from 8
```

**3. Dataset Download Failures**

Problem: "Connection error" or "Dataset not found"

Solution:
```bash
# Pre-download on login node (has internet)
python -c "from datasets import load_dataset; load_dataset('cnn_dailymail', '3.0.0')"

# Set cache to project directory
export HF_DATASETS_CACHE=/projects/$USER/.cache/huggingface
```

**4. Job Fails Immediately**

Problem: Job exits with error code before running

Solution:
```bash
# Check error log
cat hpc_version/logs/job_*.err

# Validate environment
cd hpc_version
./validate_setup.sh

# Check configuration
python -c "from configs.experiment_config import *; print('Config OK')"
```

**5. Training Stuck or Slow**

Problem: Training not progressing or very slow

Solution:
```bash
# Check if actually running
squeue -u $USER

# Check GPU utilization
srun --pty --partition=aa100 --gres=gpu:1 nvidia-smi

# Monitor stage log
tail -f $SCRATCH/mono_s2s_work/stage_logs/stage_2_*.log
```

### Recovery Tools

**Check Status:**
```bash
cd hpc_version
./check_training_status.sh
```

**Diagnose and Recover:**
```bash
cd hpc_version
./diagnose_and_recover.sh
```

**Start Fresh:**
```bash
cd hpc_version
./clean_all.sh --force        # Remove all working files
./clean_all.sh --keep-cache   # Keep dataset cache
```

**Clean Checkpoints Only:**
```bash
cd hpc_version
./clean_checkpoints.sh
```

---

## Multi-Seed Experiments

For robust statistical analysis, run multiple seeds:

```bash
cd hpc_version

# Run 5 seeds in sequence (recommended for paper)
./run_multi_seed.sh 42 1337 2024 9999 12345

# Results aggregated automatically
# See: $PROJECT/mono_s2s_multi_seed_results/
```

Multi-seed aggregation includes:
- Mean ± standard deviation for all metrics
- Statistical significance testing across seeds
- Combined results table for paper

---

## Advanced Usage

### Running Individual Stages

```bash
# Run specific stage manually
sbatch jobs/job_4_evaluate.sh

# Run with custom parameters
sbatch --time=6:00:00 jobs/job_2_baseline.sh
```

### Testing Before Full Run

```bash
cd hpc_version

# Test dataset loading
python test_dataset_loading.py

# Test all improvements
python test_improvements.py

# Validate configuration
./validate_setup.sh
```

### Monitoring Training

```bash
# Watch training progress
tail -f $SCRATCH/mono_s2s_work/stage_logs/stage_2_train_baseline.log

# Check GPU usage
squeue -u $USER --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"

# View checkpoint sizes
du -sh $SCRATCH/mono_s2s_work/*/epoch_*
```

---

## Expected Results (ICML Configuration)

### Clean Performance (ROUGE-L on CNN/DM)

| Model | ROUGE-L | 95% CI | vs Baseline |
|-------|---------|--------|-------------|
| Standard T5 | 0.2683 | [0.2510, 0.2842] | Reference |
| Baseline T5 | 0.2577 | [0.2427, 0.2726] | Control |
| Monotonic T5 | 0.2540-0.2565 | TBD | -0.5% to -1.5% |

### Adversarial Robustness (HotFlip)

| Model | Degradation | Success Rate | vs Baseline |
|-------|-------------|--------------|-------------|
| Baseline T5 | 16.35% | 61.0% | Control |
| Monotonic T5 | ~7-9% | ~23-28% | ~50% reduction |

### Dataset Coverage

- CNN/DailyMail: 11,490 test examples
- XSUM: 11,334 test examples  
- SAMSum: 819 test examples

---

## Next Steps

### For Running Experiments

1. Configure paths in `configs/experiment_config.py`
2. Run: `./run_all.sh`
3. Monitor: `squeue -u $USER`
4. Collect results from `$SCRATCH/mono_s2s_results/`

### For Paper Development

1. Run pipeline with current configuration
2. Extract results for tables (evaluation_results.json, uat_results.json, hotflip_results.json)
3. Add missing tables to paper (see `/documentation/README.md`)
4. Expand Methods section with implementation details

### For Multi-Seed Robustness

1. Run: `./run_multi_seed.sh 42 1337 2024 9999 12345`
2. Collect aggregated results
3. Report mean ± std in paper tables

---

## Documentation

- **Main README:** `/README.md` - Project overview
- **Documentation Guide:** `/documentation/README.md` - Paper development
- **Test Guide:** `/tests/README.md` - Test suite
- **Configuration:** `configs/experiment_config.py` - All settings

---

## Contact

For questions or issues:
- Check troubleshooting section above
- Review main README at `/README.md`
- See documentation at `/documentation/README.md`

---

**Last Updated:** 2026-01-29  
**Status:** Production-ready ICML 2025 pipeline
