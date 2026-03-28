# Foundation LLM Monotonicity Experiments - Setup and Execution Guide

## Overview

This guide covers how to reproduce the Pythia-1.4B monotonicity constraint
experiments on the CURC Alpine HPC cluster. The experiment trains and evaluates
two models:

- **Baseline**: Standard Pythia-1.4B fine-tuned on 100K Pile samples
- **Monotonic**: Pythia-1.4B with non-negative FFN weight constraints

The key finding: softplus parametrization (W = softplus(V) >= 0) applied to
FFN input projections reduces adversarial vulnerability while maintaining
reasonable language modeling performance.

---

## Prerequisites

- CURC Alpine account with access to the `aa100` partition (A100 80GB GPUs)
- `/projects/<username>` allocation of at least 10GB (for conda environment)
- `/scratch/alpine/<username>` allocation of at least 300GB per seed

---

## Step 1: Clone the Repository

```bash
cd /projects/$USER
git clone https://github.com/PatrickAllenCooper/mono-s2s.git
cd mono-s2s/foundation_llm_experiments
```

---

## Step 2: Set Up the Environment

The bootstrap script handles everything automatically:

```bash
bash bootstrap_curc.sh
```

This will:
1. Install Miniconda to `/projects/$USER/miniconda3`
2. Create a conda environment named `mono_s2s` with Python 3.10
3. Install PyTorch 2.x with CUDA 11.8
4. Install all dependencies (transformers, datasets, tqdm, etc.)
5. Configure HuggingFace cache to point to `/scratch`

**Why `/projects` for conda?** The home directory has a 2GB quota on Alpine.
Conda environments can be several GB, so they must go in `/projects`.

Verify the environment:
```bash
conda activate mono_s2s
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

---

## Step 3: Run the Experiment

### Quick Start (Single Seed)

```bash
SEEDS="42" ./run_expanded_experiment.sh
```

### Full Multi-Seed Run (For Publication)

```bash
./run_expanded_experiment.sh
# Runs seeds: 42, 1337, 2024
```

### Custom Seeds

```bash
SEEDS="42 1337 2024 8888 12345" ./run_expanded_experiment.sh
```

### Preview Without Submitting

```bash
DRY_RUN=1 ./run_expanded_experiment.sh
```

---

## Step 4: Monitor Progress

```bash
# See all your queued and running jobs
squeue -u $USER

# See estimated start times
squeue -u $USER --start

# Monitor Stage 3 (longest stage) - replace 42 with your seed
tail -f /scratch/alpine/$USER/foundation_llm_work_seed42/stage_logs/stage_3_train_monotonic.log

# See live training progress bar (in SLURM stderr)
tail -f foundation_llm_experiments/logs/job_3_monotonic_<JOBID>.err

# Check completion flags
ls /scratch/alpine/$USER/foundation_llm_work_seed42/stage_*_complete.flag

# Check your quota
curc-quota
```

---

## Step 5: Handle Stage 3 Timeouts

Stage 3 (Monotonic Training) requires ~10 days total but SLURM limits jobs to
24 hours. The training is designed to checkpoint frequently and resume
automatically.

**When a Stage 3 job times out:**

```bash
# Check that a checkpoint was saved
ls -lh /scratch/alpine/$USER/foundation_llm_work_seed42/checkpoints/monotonic_checkpoints/

# Resubmit - will automatically resume from latest checkpoint
sbatch --export=ALL,EXPERIMENT_SEED=42 jobs/job_3_monotonic.sh
```

**Verify resume worked:**
```bash
grep "Resuming from epoch" /scratch/alpine/$USER/foundation_llm_work_seed42/stage_logs/stage_3_train_monotonic.log
```

**Stage 3 is done when:**
```bash
ls /scratch/alpine/$USER/foundation_llm_work_seed42/stage_3_train_monotonic_complete.flag
# File exists = done
```

**After Stage 3 completes**, resubmit downstream stages:
```bash
sbatch --export=ALL,EXPERIMENT_SEED=42 jobs/job_4_evaluate.sh
sbatch --export=ALL,EXPERIMENT_SEED=42 jobs/job_5_uat.sh
sbatch --export=ALL,EXPERIMENT_SEED=42 jobs/job_6_hotflip.sh
```

---

## Results

Results are saved to `/scratch/alpine/$USER/foundation_llm_results_seed<SEED>/`:

```
evaluation_results.json        # Perplexity comparison
uat_results.json               # Universal Adversarial Trigger attack results
hotflip_results.json           # HotFlip gradient attack results
baseline_training_history.json # Training curve for baseline
monotonic_training_history.json# Training curve for monotonic model
```

### Expected Results (from seed 42 pilot)

| Model | Perplexity (Pile test) |
|-------|----------------------|
| Baseline Pythia-1.4B | ~6.9 |
| Monotonic Pythia-1.4B | ~27.6 |

The 4x perplexity gap reflects the expressiveness cost of the non-negativity
constraint. Adversarial robustness metrics (UAT, HotFlip) are expected to
show improved resistance in the monotonic model.

---

## Troubleshooting

### "No space left on device" during git pull

```bash
# Check quota
curc-quota

# Clean conda cache
conda clean --all -y

# Set git temp dir to scratch (has much more space)
export TMPDIR=/scratch/alpine/$USER/tmp
mkdir -p $TMPDIR
git pull origin main
```

### CUDA out of memory

This is handled automatically via bfloat16 precision and gradient checkpointing.
If you still encounter OOM, reduce the eval batch size in the config:

```python
# In configs/experiment_config.py
EVAL_BATCH_SIZE = 2  # Reduce from 4
```

### DependencyNeverSatisfied

An upstream job failed. Check which stage failed:

```bash
sacct -u $USER --starttime=$(date -d '7 days ago' '+%Y-%m-%d') \
    --format=JobID,JobName,State,ExitCode,Elapsed | grep foundation
```

Then check the relevant stage log:
```bash
cat /scratch/alpine/$USER/foundation_llm_work_seed42/stage_logs/stage_2_train_baseline.log | tail -50
```

Fix the issue (check this repository's issue tracker or contact the authors)
and resubmit from the failed stage.

### Stage 5 (UAT) times out at 6 hours

UAT optimization is compute-intensive. The current config (50 iterations,
3 restarts, 100 candidates) should complete in ~4-5 hours. If it still times
out, reduce further in `configs/experiment_config.py`:

```python
ATTACK_NUM_ITERATIONS = 30
ATTACK_NUM_RESTARTS = 2
ATTACK_NUM_CANDIDATES = 50
```

---

## File Structure

```
foundation_llm_experiments/
    bootstrap_curc.sh           # Environment setup (run once)
    run_expanded_experiment.sh  # Main entry point (run this)
    run_all.sh                  # Single-seed submission (alternative)
    jobs/                       # SLURM job scripts (stages 0-7)
    scripts/                    # Python stage scripts
    configs/experiment_config.py# All hyperparameters
    utils/                      # Shared utilities
    logs/                       # SLURM output/error logs
```

---

## Contact

For questions about the experiments, please open an issue on the repository
or contact the authors through the paper submission system.
