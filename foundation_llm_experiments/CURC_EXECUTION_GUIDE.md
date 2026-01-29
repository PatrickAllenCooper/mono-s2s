# Foundation LLM Experiments - CURC Alpine Execution Guide

**Target Cluster:** CURC Alpine (University of Colorado Boulder)  
**Model:** Pythia-1.4B (EleutherAI)  
**Total Runtime:** ~60-70 hours per seed  
**GPU Required:** 1x A100 (40GB)

---

## Overview

This pipeline extends the main mono-s2s experiments to general-purpose foundation models. Instead of testing monotonicity constraints on T5 for summarization, we test on Pythia-1.4B (a decoder-only LLM) for general language modeling and diverse NLP benchmarks.

**Key Differences from Main Pipeline:**
- Uses Pythia-1.4B (1.4B params) instead of T5-small (60M params)
- Evaluates perplexity instead of ROUGE scores
- Includes "recovery training" phase to restore performance after applying constraints
- Tests on diverse benchmarks (LAMBADA, HellaSwag, Winogrande, TruthfulQA)
- Longer runtime due to larger model and dataset

---

## Quick Start (2 Steps)

```bash
# 1. SSH to Alpine and clone repository
ssh your_username@login.rc.colorado.edu
cd /projects/$USER
git clone https://github.com/PatrickAllenCooper/mono-s2s.git
cd mono-s2s/foundation_llm_experiments

# 2. Run the pipeline (handles ALL setup automatically)
./run_all.sh
```

**That's literally it!** The `run_all.sh` script automatically:

**First-time setup (if needed):**
- ✓ Checks for conda/environment
- ✓ Runs bootstrap if not found
- ✓ Installs Miniconda to `/projects` (avoids home quota issues)
- ✓ Creates Python 3.10 environment
- ✓ Installs PyTorch + CUDA 11.8
- ✓ Installs all dependencies
- ✓ Sets up HuggingFace cache in `$SCRATCH`

**Then automatically:**
- ✓ Submits all 7 experimental stages
- ✓ Sets up job dependencies
- ✓ Provides monitoring commands

**Total setup time:** ~5-10 minutes (first run only)  
**Total experiment time:** ~60-70 hours per seed  
**Subsequent runs:** Instant (environment already exists)

---

## Prerequisites

### Required

- **CURC Alpine account** with access to `aa100` partition (A100 GPUs)
- **Allocation hours:** ~70 GPU-hours per seed
- **Internet access** on login node (for downloading model/data)

### Important Notes

- **Storage:** Alpine `$HOME` quota is tiny (~2-5GB)
  - The bootstrap script automatically installs conda to `/projects/$USER/`
  - HuggingFace cache goes to `$SCRATCH` (not `$HOME`)
  
- **No manual setup required:** The bootstrap script handles everything

---

## Setup

### Automatic Setup (Default)

**You don't need to do anything special!** Just run `./run_all.sh` and it handles all setup automatically.

The script will:
1. Check if conda/environment exists
2. If not found, automatically run the bootstrap script
3. Bootstrap installs conda, creates environment, installs dependencies
4. Then submit all jobs

**Total setup time (first run):** ~5-10 minutes  
**Subsequent runs:** Instant (environment already exists)

---

### Manual Bootstrap (Optional)

If you want to run setup separately before submitting jobs:

```bash
cd /projects/$USER/mono-s2s/foundation_llm_experiments
bash bootstrap_curc.sh
```

This is the same bootstrap that `run_all.sh` runs automatically, but you can run it manually if you want to verify setup before submitting jobs.

---

### Advanced: Manual Setup

If you prefer complete manual control or have an existing conda installation:

<details>
<summary>Click to expand manual setup instructions</summary>

#### 1. Ensure Conda is in /projects (not $HOME)

```bash
# Check conda location
which conda

# Should be: /projects/<username>/miniconda3/bin/conda
# If in $HOME, follow bootstrap script or move it:
rm -rf ~/miniconda3
# Then reinstall to /projects
```

#### 2. Create Environment

```bash
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s
```

#### 3. Install Dependencies

```bash
# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install -r requirements.txt
```

#### 4. Make Executable

```bash
chmod +x run_all.sh jobs/*.sh scripts/*.py
```

#### 5. Verify

```bash
python -m configs.experiment_config
```

</details>

---

### Configuration (Optional)

The default configuration should work for most users. To customize:

Edit `configs/experiment_config.py`:

```python
# For production runs (full evaluation)
USE_FULL_EVAL_SETS = True  # 60-70 hours

# For quick testing (reduced data)
USE_FULL_EVAL_SETS = False  # 5 hours
TRAINING_SAMPLES = QUICK_TRAINING_SAMPLES
```

---

## Running the Pipeline

### Option 1: Full Pipeline (Recommended)

Submit all stages with automatic dependencies:

```bash
cd /projects/$USER/mono-s2s/foundation_llm_experiments

# Activate environment
conda activate mono_s2s

# Submit all jobs
./run_all.sh

# The script will prompt for confirmation and display job IDs
```

**What happens:**
1. Stage 0 downloads Pythia-1.4B (~6GB) - 1 hour
2. Stage 1 applies monotonicity constraints - 30 min
3. Stages 2 & 3 train baseline and monotonic models in parallel - 24-32 hours each
4. Stage 4 evaluates both models on benchmarks - 8 hours
5. Stages 5 & 6 run UAT and HotFlip attacks in parallel - 6-10 hours total
6. Stage 7 aggregates all results - 30 min

**Total wall time:** ~60-70 hours (with parallelization)

### Option 2: Individual Stages

For debugging or resuming failed stages:

```bash
# Submit individual stage
sbatch jobs/job_0_setup.sh
sbatch jobs/job_1_monotonicity.sh
# etc.

# With custom seed
EXPERIMENT_SEED=1337 sbatch jobs/job_0_setup.sh
```

### Option 3: Quick Testing Mode

For rapid testing before full run:

```bash
# Edit configs/experiment_config.py first:
# Set USE_FULL_EVAL_SETS = False
# Set TRAINING_SAMPLES = QUICK_TRAINING_SAMPLES

# Then run pipeline
./run_all.sh

# Quick mode runtime: ~5 hours total
```

---

## Monitoring Progress

### Check Job Queue

```bash
# View your jobs
squeue -u $USER

# Filter to foundation experiments
squeue -u $USER | grep foundation

# Detailed view
squeue -u $USER --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
```

### Watch Logs

```bash
# Live log viewing (replace JOBID with actual job ID)
tail -f logs/job_0_setup_JOBID.out
tail -f logs/job_2_baseline_JOBID.out

# Check for errors
tail -20 logs/job_2_baseline_JOBID.err

# View all active logs
ls -lht logs/*.out | head -5
```

### Check Completion Flags

```bash
# Check which stages completed
ls -lh $SCRATCH/foundation_llm_work/*.flag

# Expected flags:
# - setup_complete.flag
# - monotonicity_applied.flag
# - baseline_trained.flag
# - monotonic_trained.flag
# - evaluation_complete.flag
# - uat_complete.flag
# - hotflip_complete.flag
# - aggregation_complete.flag
```

### Monitor GPU Usage

For active jobs:

```bash
# Get node name from squeue
squeue -u $USER

# SSH to compute node (replace nodename)
ssh nodename

# Check GPU utilization
nvidia-smi

# Watch GPU in real-time
watch -n 1 nvidia-smi
```

### Check Intermediate Results

```bash
# View experiment progress
cat $SCRATCH/foundation_llm_work/experiment_summary.txt

# Check training progress (updated periodically)
tail -50 $SCRATCH/foundation_llm_work/baseline_training.log
tail -50 $SCRATCH/foundation_llm_work/monotonic_training.log

# Check disk usage
du -sh $SCRATCH/foundation_llm_work
du -sh $SCRATCH/huggingface_cache
```

---

## Expected Resource Usage

### Per-Stage Requirements

| Stage | GPU | Memory | Time | Storage |
|-------|-----|--------|------|---------|
| 0: Setup | Yes | 80GB | 1h | ~6GB |
| 1: Apply Monotonicity | Yes | 80GB | 30min | ~6GB |
| 2: Baseline Training | Yes | 80GB | 24h | ~25GB |
| 3: Monotonic Training | Yes | 80GB | 32h | ~25GB |
| 4: Evaluation | Yes | 80GB | 8h | ~5GB |
| 5: UAT Attacks | Yes | 80GB | 6h | ~2GB |
| 6: HotFlip Attacks | Yes | 80GB | 4h | ~2GB |
| 7: Aggregate | No | 16GB | 30min | ~1GB |

### Total Resource Summary

- **GPU Hours:** ~75 hours per seed (A100)
- **Peak Storage:** ~500GB (SCRATCH)
- **Permanent Storage:** ~50GB (PROJECT)
- **Total Wall Time:** ~60-70 hours (with parallelization)

### Storage Breakdown

```
$SCRATCH/foundation_llm_work/          (~100GB)
├── checkpoints/
│   ├── baseline_checkpoints/          (~25GB)
│   └── monotonic_checkpoints/         (~25GB)
├── data_cache/                        (~50GB Pile data)
└── logs/                              (~100MB)

$SCRATCH/huggingface_cache/            (~400GB)
├── datasets/                          (~350GB Pile)
└── transformers/                      (~50GB models)

$SCRATCH/foundation_llm_results/       (~5GB)
└── final_results.json, etc.

$PROJECT/foundation_llm_final_results/ (~5GB permanent copy)
```

---

## Output Files

### Main Results Directory

```bash
$SCRATCH/foundation_llm_results/
```

Contains:
- `setup_validation.json` - Environment and model info
- `baseline_training_history.json` - Baseline training curves
- `monotonic_training_history.json` - Monotonic training curves
- `evaluation_results.json` - Perplexity and benchmark scores
- `uat_results.json` - UAT attack results
- `hotflip_results.json` - HotFlip attack results
- `final_results.json` - Aggregated results
- `experiment_summary.txt` - Human-readable summary

### Key Metrics to Check

**Perplexity (lower is better):**
```bash
# View evaluation results
cat $SCRATCH/foundation_llm_results/evaluation_results.json | grep -A 5 "pile_perplexity"

# Expected values:
# Baseline: ~10.2
# Monotonic: ~10.9 (6-7% degradation)
```

**HotFlip Attack Success (lower is better):**
```bash
cat $SCRATCH/foundation_llm_results/hotflip_results.json | grep "attack_success_rate"

# Expected values:
# Baseline: ~55% (vulnerable)
# Monotonic: ~18% (67% reduction, robust!)
```

---

## Troubleshooting

### Common Issues

#### 1. "No space left on device"

**Cause:** Home directory quota exceeded

**Solution:**
```bash
# Check quotas
df -h $HOME
df -h /projects/$USER

# Move conda if in $HOME
rm -rf ~/miniconda3
# Follow "Install Conda to /projects" section above

# Clean HuggingFace cache if needed
rm -rf ~/.cache/huggingface
```

#### 2. "CUDA out of memory"

**Cause:** Batch size too large for A100

**Solution:**
```python
# Edit configs/experiment_config.py
BATCH_SIZE = 4  # Reduce from 8
GRADIENT_ACCUMULATION_STEPS = 8  # Increase from 4
EVAL_BATCH_SIZE = 8  # Reduce from 16
```

#### 3. "Failed to download Pile dataset"

**Cause:** Network timeout or HuggingFace API issues

**Solution:**
```bash
# Pre-download on login node (has internet)
conda activate mono_s2s
python -c "from datasets import load_dataset; load_dataset('EleutherAI/pile', split='test', streaming=False)"

# Set cache location
export HF_DATASETS_CACHE=/scratch/alpine/$USER/huggingface_cache/datasets
```

#### 4. Job fails immediately

**Cause:** Environment not activated or paths incorrect

**Solution:**
```bash
# Verify conda environment
conda activate mono_s2s
which python
# Should be: /projects/<your_username>/miniconda3/envs/mono_s2s/bin/python

# Test configuration
cd /projects/$USER/mono-s2s/foundation_llm_experiments
python -m configs.experiment_config
```

#### 5. Job stuck in queue

**Cause:** aa100 partition busy or insufficient priority

**Solution:**
```bash
# Check partition status
sinfo -p aa100

# Check job queue position
squeue -u $USER --start

# Use different partition if available
# Edit all jobs/*.sh files: #SBATCH --partition=amilan
```

### Recovery Procedures

#### Resume Failed Training

If a job fails mid-training, checkpoints are saved every 5000 steps:

```bash
# Check available checkpoints
ls -lh $SCRATCH/foundation_llm_work/checkpoints/baseline_checkpoints/
ls -lh $SCRATCH/foundation_llm_work/checkpoints/monotonic_checkpoints/

# Resubmit the failed stage
sbatch jobs/job_2_baseline.sh  # Will auto-resume from latest checkpoint
```

#### Clean and Restart

To start completely fresh:

```bash
# WARNING: This deletes all intermediate files and checkpoints
rm -rf $SCRATCH/foundation_llm_work
rm -rf $SCRATCH/foundation_llm_results

# Keep HuggingFace cache to avoid re-downloading (optional)
# rm -rf $SCRATCH/huggingface_cache

# Resubmit pipeline
cd /projects/$USER/mono-s2s/foundation_llm_experiments
./run_all.sh
```

#### Check Logs for Errors

```bash
# View recent errors
tail -50 logs/job_2_baseline_JOBID.err

# Search for specific errors
grep -i "error\|failed\|exception" logs/job_2_baseline_JOBID.err

# View full log
less logs/job_2_baseline_JOBID.out
```

---

## Multi-Seed Experiments

For robust statistical analysis, run with multiple seeds:

```bash
cd /projects/$USER/mono-s2s/foundation_llm_experiments

# Run seed 42
EXPERIMENT_SEED=42 ./run_all.sh

# After completion, run seed 1337
EXPERIMENT_SEED=1337 ./run_all.sh

# Recommended seeds: 42, 1337, 2024, 8888, 12345
```

Results for each seed will be stored separately and can be aggregated later.

---

## Estimated Costs

### Computational Costs

**Single seed run:**
- GPU hours: ~75 hours (A100)
- Core hours: ~600 hours (8 cores × 75 hours)
- Storage: ~500GB (temporary)

**Five seeds (for paper):**
- GPU hours: ~375 hours (A100)
- Core hours: ~3000 hours
- Storage: ~500GB (reused between runs)

### Timeline

- **Quick test (reduced data):** ~5 hours
- **Single full run:** ~3 days
- **Five seeds (sequential):** ~15 days
- **Five seeds (parallel):** ~3 days (if enough allocation)

---

## After Completion

### Retrieve Results

```bash
# Copy results to your local machine
scp -r your_username@login.rc.colorado.edu:$PROJECT/foundation_llm_final_results/ .

# Or create archive
cd $PROJECT/foundation_llm_final_results
tar -czf foundation_results_seed42.tar.gz *
```

### View Summary

```bash
# Human-readable summary
cat $PROJECT/foundation_llm_final_results/experiment_summary.txt

# JSON results
cat $PROJECT/foundation_llm_final_results/final_results.json | python -m json.tool
```

### Clean Up Scratch Space

```bash
# After copying results to PROJECT, clean SCRATCH
rm -rf $SCRATCH/foundation_llm_work
rm -rf $SCRATCH/foundation_llm_results

# Keep HuggingFace cache for future runs (optional)
# rm -rf $SCRATCH/huggingface_cache
```

---

## Expected Results

Based on extrapolation from T5-small findings:

### Clean Performance
- **Baseline Perplexity:** ~10.2 (standard Pythia-1.4B)
- **Monotonic Perplexity:** ~10.9 (+6.8% degradation)
- **Trade-off:** Small performance cost for robustness

### Adversarial Robustness
- **Baseline HotFlip Success:** ~55% (vulnerable)
- **Monotonic HotFlip Success:** ~18% (67% reduction!)
- **UAT Impact:** Minimal across both models

### Benchmark Performance
- **LAMBADA:** ~70% accuracy (both models similar)
- **HellaSwag:** ~45% accuracy (both models similar)
- **Winogrande:** ~60% accuracy (both models similar)
- **TruthfulQA:** ~40% accuracy (both models similar)

---

## Contact and Support

### CURC Resources

- **Documentation:** https://curc.readthedocs.io/
- **Help Desk:** rc-help@colorado.edu
- **Office Hours:** Check CURC website

### Project-Specific Help

- See main repo: `/README.md`
- Foundation LLM README: This directory's `README.md`
- Configuration reference: `configs/experiment_config.py`

---

## Quick Reference

### Essential Commands

```bash
# First time: Clone and run (handles all setup automatically)
cd /projects/$USER
git clone https://github.com/PatrickAllenCooper/mono-s2s.git
cd mono-s2s/foundation_llm_experiments
./run_all.sh

# Subsequent runs: Just run (environment already exists)
cd /projects/$USER/mono-s2s/foundation_llm_experiments
./run_all.sh

# Check jobs
squeue -u $USER

# Watch logs
tail -f logs/job_2_baseline_*.out

# Check results
cat $SCRATCH/foundation_llm_results/experiment_summary.txt

# Cancel all jobs
scancel -u $USER -n foundation
```

### Important Paths

```bash
# Source code
/projects/$USER/mono-s2s/foundation_llm_experiments/

# Working directory
$SCRATCH/foundation_llm_work/

# Results
$SCRATCH/foundation_llm_results/

# Permanent results
$PROJECT/foundation_llm_final_results/

# Conda environment
/projects/$USER/miniconda3/envs/mono_s2s/
```

---

**Last Updated:** 2026-01-29  
**Status:** Production-ready for CURC Alpine
