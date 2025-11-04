# CURC/HPC Setup Guide - Mono-S2S Fair Comparison Experiments

This guide helps you set up and run the mono-s2s experiments on SLURM-based HPC clusters (CURC, Summit, Alpine, etc.).

---

## 🚀 Quick Start (3 Steps)

### 1. Configure Your Environment

Edit `configs/experiment_config.py`:
```python
# Line 19-20: Set your HPC paths
SCRATCH_DIR = "/scratch/summit/your_username"  # Your scratch space
PROJECT_DIR = "/projects/your_project"          # Your project space

# Line 81-82: Choose partition
SLURM_PARTITION = "shas"  # or "blanca-ics", "aa100", etc.
SLURM_QOS = "normal"      # or "long", "blanca-ics", etc.
```

### 2. Set Up Python Environment

```bash
# Option A: Using module system
module load python/3.10.0
module load cuda/11.8
pip install --user transformers datasets torch rouge-score pandas scipy matplotlib tqdm

# Option B: Using conda (recommended)
module load anaconda
conda create -n mono_s2s python=3.10
conda activate mono_s2s
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers datasets rouge-score pandas scipy matplotlib tqdm
```

### 3. Run Experiments

```bash
# Make scripts executable
chmod +x run_all.sh
chmod +x jobs/*.sh

# Submit all jobs (with automatic dependencies)
./run_all.sh

# Or submit individual stages manually:
sbatch jobs/job_0_setup.sh
# Wait, then:
sbatch jobs/job_1_data.sh
# Wait, then:
sbatch jobs/job_2_baseline.sh
sbatch jobs/job_3_monotonic.sh
# etc.
```

---

## 📁 File Structure

```
hpc_version/
├── configs/
│   └── experiment_config.py    # EDIT THIS: paths, partition, resources
├── utils/
│   └── common_utils.py         # Shared utilities (don't edit)
├── scripts/
│   ├── stage_0_setup.py        # Environment setup
│   ├── stage_1_prepare_data.py # Data loading
│   ├── stage_2_train_baseline.py    # Train baseline
│   ├── stage_3_train_monotonic.py   # Train monotonic
│   ├── stage_4_evaluate.py     # Comprehensive evaluation
│   ├── stage_5_uat_attacks.py  # UAT attacks
│   ├── stage_6_hotflip_attacks.py   # HotFlip attacks
│   └── stage_7_aggregate.py    # Final analysis
├── jobs/
│   ├── job_0_setup.sh          # SLURM script for setup
│   ├── job_1_data.sh           # SLURM script for data
│   ├── job_2_baseline.sh       # SLURM script for baseline training
│   ├── job_3_monotonic.sh      # SLURM script for monotonic training
│   ├── job_4_evaluate.sh       # SLURM script for evaluation
│   ├── job_5_uat.sh            # SLURM script for UAT
│   ├── job_6_hotflip.sh        # SLURM script for HotFlip
│   └── job_7_aggregate.sh      # SLURM script for aggregation
├── run_all.sh                  # Master orchestrator
└── README.md                   # Main documentation
```

---

## ⚙️ Configuration

### CURC-Specific Settings

**For Summit (CU Boulder):**
```python
# In configs/experiment_config.py
SCRATCH_DIR = "/scratch/summit/your_username"
PROJECT_DIR = "/projects/your_project"
SLURM_PARTITION = "shas"  # GPU partition
SLURM_QOS = "normal"
```

**For Alpine (CU Boulder):**
```python
SCRATCH_DIR = "/scratch/alpine/your_username"
PROJECT_DIR = "/pl/active/your_project"
SLURM_PARTITION = "aa100"  # A100 GPUs
SLURM_QOS = "normal"
```

**For Blanca (CU Boulder):**
```python
SCRATCH_DIR = "/rc_scratch/your_username"
PROJECT_DIR = "/projects/your_group"
SLURM_PARTITION = "blanca-ics"  # Your Blanca partition
SLURM_QOS = "blanca-ics"
```

### Resource Tuning

**For Quick Testing (small datasets):**
```python
USE_FULL_TEST_SETS = False  # Uses 200 samples
BATCH_SIZE = 4
NUM_EPOCHS = 2
```

**For Full Evaluation:**
```python
USE_FULL_TEST_SETS = True  # Uses full test sets (~11k samples)
BATCH_SIZE = 4
NUM_EPOCHS = 3
```

**If you get OOM (Out of Memory):**
```python
BATCH_SIZE = 2  # Reduce batch size
EVAL_BATCH_SIZE = 4  # Reduce eval batch
```

---

## 📊 Stage Pipeline

```
┌─────────────────┐
│  Stage 0: Setup │  (10 min, no GPU)
│  Download model │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stage 1: Data   │  (1 hr, no GPU)
│ Load datasets   │
└────────┬────────┘
         │
         ├──────────────────┬────────────────────┐
         ▼                  ▼                    ▼
┌─────────────────┐  ┌─────────────────┐   (Parallel)
│Stage 2: Baseline│  │Stage 3:Monotonic│
│   Train (GPU)   │  │   Train (GPU)   │
│   4-12 hours    │  │   4-12 hours    │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └──────────┬─────────┘
                    ▼
           ┌─────────────────┐
           │ Stage 4: Eval   │  (2-4 hr, GPU)
           │ All 3 models    │
           │ Bootstrap CIs   │
           └────────┬────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐  (Parallel)
│ Stage 5: UAT    │   │Stage 6: HotFlip │
│ Transfer matrix │   │   Attacks       │
│   2-3 hours     │   │   1-2 hours     │
└────────┬────────┘   └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
           ┌─────────────────┐
           │Stage 7:Aggregate│  (5 min, no GPU)
           │ Final results   │
           └─────────────────┘
```

---

## 🔍 Monitoring Jobs

### Check Job Queue
```bash
# Your jobs
squeue -u $USER

# Specific job details
squeue -j <job_id>

# All mono_s2s jobs
squeue -u $USER | grep mono_s2s
```

### Monitor Logs in Real-Time
```bash
# Follow specific job log
tail -f logs/job_2_baseline_*.out

# Check for errors
tail -f logs/job_2_baseline_*.err

# List all logs
ls -lht logs/
```

### Check Stage Completion
```bash
# Check completion flags
ls -la $SCRATCH/mono_s2s_work/*.flag

# View completion details
cat $SCRATCH/mono_s2s_work/stage_0_setup_complete.flag
```

### View Progress
```bash
# Stage-specific logs
cat $SCRATCH/mono_s2s_work/stage_logs/stage_2_train_baseline.log

# Check GPU usage
ssh <node> nvidia-smi

# Check disk usage
du -sh $SCRATCH/mono_s2s_work/
```

---

## 🛠️ Troubleshooting

### Job Fails Immediately
**Symptom:** Job exits within seconds

**Solutions:**
1. Check module names: `module avail python`, `module avail cuda`
2. Verify paths in `experiment_config.py`
3. Check Python environment: `which python`, `python --version`
4. Review error log: `logs/job_*_<id>.err`

### Out of Memory (OOM)
**Symptom:** Job killed, "out of memory" in logs

**Solutions:**
1. Reduce batch size: `BATCH_SIZE = 2` in config
2. Request more memory in job script: `#SBATCH --mem=128G`
3. Use smaller model: `MODEL_NAME = "t5-small"` (not t5-base)
4. Enable quick mode: `USE_FULL_TEST_SETS = False`

### Job Times Out
**Symptom:** Job reaches time limit

**Solutions:**
1. Increase time limit in job script: `#SBATCH --time=24:00:00`
2. Reduce epochs: `NUM_EPOCHS = 2`
3. Enable checkpointing (already implemented)
4. Use quick mode for testing first

### Dataset Download Fails
**Symptom:** "Error loading dataset" in logs

**Solutions:**
1. Check internet connectivity from compute nodes
2. Pre-download datasets: `HF_DATASETS_CACHE=/projects/your_cache`
3. Increase timeout in stage_1 script
4. Use compute node with internet access

### Checkpoint Not Found
**Symptom:** "Checkpoint not found" error

**Solutions:**
1. Verify previous stage completed: `ls $SCRATCH/mono_s2s_work/*.flag`
2. Check checkpoint directory: `ls $SCRATCH/mono_s2s_work/checkpoints/`
3. Review previous job logs for training errors
4. Rerun failed stage

### Wrong Partition/QOS
**Symptom:** "Invalid partition" or "Invalid QOS"

**Solutions:**
1. Check available partitions: `sinfo`
2. Check your QOS: `sacctmgr show assoc where user=$USER format=account,qos%50`
3. Update `experiment_config.py` with correct partition/QOS

---

## 📊 Expected Outputs

### After Stage 0 (Setup)
```
$SCRATCH/mono_s2s_results/
└── setup_complete.json

$SCRATCH/mono_s2s_work/
└── stage_0_setup_complete.flag
```

### After Stage 1 (Data)
```
$SCRATCH/mono_s2s_work/data_cache/
├── train_data.pt
├── val_data.pt
├── test_data.pt
└── attack_data.pt

$SCRATCH/mono_s2s_results/
└── data_statistics.json
```

### After Stage 2 & 3 (Training)
```
$SCRATCH/mono_s2s_work/checkpoints/
├── baseline_checkpoints/
│   ├── best_model.pt
│   └── checkpoint_epoch_*.pt
└── monotonic_checkpoints/
    ├── best_model.pt
    └── checkpoint_epoch_*.pt

$SCRATCH/mono_s2s_results/
├── baseline_training_history.json
└── monotonic_training_history.json
```

### After Stage 4 (Evaluation)
```
$SCRATCH/mono_s2s_results/
└── evaluation_results.json  # Primary results with bootstrap CIs
```

### After Stage 5 & 6 (Attacks)
```
$SCRATCH/mono_s2s_results/
├── uat_results.json
├── transfer_matrix.json
├── learned_triggers.csv
└── hotflip_results.json
```

### After Stage 7 (Final)
```
$SCRATCH/mono_s2s_results/
├── final_results.json
├── experiment_metadata.json
└── plots/
    ├── comparison_table.png
    ├── transfer_matrix.png
    └── attack_robustness.png

$PROJECT/mono_s2s_final_results/  # Persistent copy
└── (same files)
```

---

## 🔄 Multi-Seed Execution

To run with multiple seeds (for robust statistics):

### Method 1: Sequential
```bash
for seed in 42 1337 2024 8888 12345; do
    echo "Running with seed $seed..."
    ./run_all.sh $seed
    # Rename results directory
    mv $SCRATCH/mono_s2s_results $SCRATCH/mono_s2s_results_seed_$seed
done

# Aggregate across seeds
python aggregate_multi_seed.py
```

### Method 2: Parallel (Different Scratch Dirs)
```bash
# Run each seed in a separate directory
for seed in 42 1337 2024 8888 12345; do
    export EXPERIMENT_SEED=$seed
    export SCRATCH=/scratch/summit/$USER/seed_$seed
    ./run_all.sh $seed &
done

wait  # Wait for all to complete
```

---

## 💾 Storage Management

### Disk Space Requirements

| Component | Size | Location |
|-----------|------|----------|
| Model cache | 5GB | HuggingFace cache |
| Datasets | 15-20GB | data_cache/ |
| Checkpoints | 10GB | checkpoints/ |
| Results | 2GB | results/ |
| **Total** | **~35GB** | Scratch |

### Clean Up After Completion

```bash
# Copy final results to project (persistent)
cp -r $SCRATCH/mono_s2s_results $PROJECT/mono_s2s_final_results

# Clean up scratch (temporary)
rm -rf $SCRATCH/mono_s2s_work/data_cache  # Can re-download
rm -rf $SCRATCH/mono_s2s_work/checkpoints/*/checkpoint_epoch_*.pt  # Keep best_model.pt only

# Archive and compress
cd $PROJECT
tar -czf mono_s2s_results_seed_42.tar.gz mono_s2s_final_results/
```

---

## 📧 Email Notifications

Add to job scripts to get email updates:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@colorado.edu
```

---

## 🔧 Advanced: Job Array for Multi-Seed

Create `run_multiseed.sh`:
```bash
#!/bin/bash
#SBATCH --array=0-4  # 5 seeds
#SBATCH --job-name=mono_s2s_array
# ... other SBATCH directives ...

# Map array index to seed
SEEDS=(42 1337 2024 8888 12345)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

export EXPERIMENT_SEED=$SEED
export SCRATCH=/scratch/summit/$USER/seed_$SEED

# Run pipeline for this seed
./run_all.sh $SEED
```

---

## 📊 Estimating Costs

### SU (Service Unit) Calculation

For CURC:
```
SUs = walltime_hours × cores × GPU_multiplier

Example (Stage 2 - Baseline Training):
- Walltime: 8 hours
- Cores: 8
- GPU: 1 (multiplier = varies by partition)
- SUs ≈ 8 × 8 × 2 = 128 SUs

Total for all stages: ~400-600 SUs per seed
```

Check your allocation:
```bash
curc-quota
```

---

## 🐛 Debugging Failed Jobs

### 1. Check SLURM Output
```bash
# Standard output
cat logs/job_2_baseline_<jobid>.out

# Error output
cat logs/job_2_baseline_<jobid>.err
```

### 2. Check Stage Log
```bash
cat $SCRATCH/mono_s2s_work/stage_logs/stage_2_train_baseline.log
```

### 3. Check Completion Flags
```bash
ls -la $SCRATCH/mono_s2s_work/*.flag
# Missing flag = stage didn't complete
```

### 4. Interactive Debugging
```bash
# Request interactive GPU node
sinteractive --partition=gpu --gres=gpu:1 --time=01:00:00

# Load modules
module load python/3.10.0 cuda/11.8

# Run stage manually
cd hpc_version/scripts
python stage_2_train_baseline.py
```

### 5. Check Resource Usage
```bash
# After job completes
sacct -j <job_id> --format=JobID,JobName,MaxRSS,Elapsed,State

# Check if OOM
sacct -j <job_id> | grep -i "out of memory"
```

---

## 📈 Performance Optimization

### For Faster Training
```python
# In experiment_config.py
BATCH_SIZE = 8  # Increase if you have memory
NUM_EPOCHS = 2  # Reduce for quick tests
USE_FULL_TEST_SETS = False  # Quick mode
```

### For More Accurate Results
```python
USE_FULL_TEST_SETS = True  # Full test sets
ROUGE_BOOTSTRAP_SAMPLES = 2000  # More bootstrap samples
# Run with all 5 seeds
```

### GPU Selection
```bash
# In job scripts, request specific GPU type:
#SBATCH --gres=gpu:v100:1  # V100
#SBATCH --gres=gpu:a100:1  # A100 (faster)
```

---

## ✅ Verification Checklist

After running all stages:

- [ ] All 7 completion flags exist
- [ ] baseline_checkpoints/best_model.pt exists
- [ ] monotonic_checkpoints/best_model.pt exists
- [ ] evaluation_results.json has bootstrap CIs
- [ ] transfer_matrix.json shows 3×3 grid
- [ ] final_results.json contains aggregated analysis
- [ ] No "FAILED" in job logs
- [ ] Copied results to project directory

---

## 🆘 Getting Help

### CURC Resources
- **Documentation:** https://curc.readthedocs.io/
- **Help:** rc-help@colorado.edu
- **Office Hours:** Check CURC website

### Common Issues
- **Account/allocation:** Contact your PI or rc-help
- **Partition access:** Check with `sacctmgr show assoc where user=$USER`
- **Software issues:** Check `module spider <package>`
- **Quota exceeded:** Check `curc-quota`, clean up scratch

---

## 📝 Citation

If you use this code, please cite:
```
@software{mono_s2s_v1_7,
  title={Mono-S2S v1.7: Fair Comparison Edition for HPC},
  author={Your Name},
  year={2025},
  url={https://github.com/PatrickAllenCooper/mono-s2s}
}
```

---

**Happy Computing!** 🚀

For questions specific to this pipeline, check the main README.md or review the code comments in each stage script.

