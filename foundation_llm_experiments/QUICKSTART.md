# Foundation LLM Experiments - Quick Start Guide

This guide will help you quickly set up and run the foundation model monotonicity experiments.

## Prerequisites

1. **HPC Access**: Access to an HPC cluster with A100 GPUs (CURC Alpine recommended)
2. **Conda Environment**: The `mono_s2s` conda environment from the main project
3. **Storage**: ~500GB scratch space, ~100GB project space
4. **Time**: ~60-70 hours compute time per seed

## Setup (5 minutes)

### 1. Navigate to Experiment Directory

```bash
cd /path/to/mono-s2s/foundation_llm_experiments
```

### 2. Verify Conda Environment

```bash
conda activate mono_s2s
python -c "import torch, transformers; print('✓ Environment OK')"
```

If this fails, install dependencies:

```bash
conda activate mono_s2s
pip install torch transformers datasets accelerate lm-eval
```

### 3. Configure Paths

Edit `configs/experiment_config.py` if needed:
- `SCRATCH_DIR`: Should auto-detect from `$SCRATCH`
- `PROJECT_DIR`: Should auto-detect from `$PROJECT`
- `SLURM_PARTITION`: Change if not using `aa100`

Most users can skip this step (defaults work for CURC Alpine).

### 4. Test Configuration

```bash
python configs/experiment_config.py
```

Should output:
```
✓ Configuration validated successfully
Estimated Training Time: ...
```

## Running Experiments

### Option A: Full Pipeline (Recommended)

Submit all stages with automatic dependencies:

```bash
bash run_all.sh
```

This will:
1. Download Pythia-1.4B (~6GB)
2. Apply monotonicity constraints
3. Run baseline and monotonic recovery training
4. Evaluate on benchmarks
5. Run adversarial attacks
6. Aggregate results

**Total time**: ~60-70 hours per seed

### Option B: Individual Stages

Run specific stages manually:

```bash
# Stage 0: Setup
sbatch jobs/job_0_setup.sh

# Check when finished, then:
sbatch jobs/job_1_monotonicity.sh

# etc...
```

### Option C: Quick Test Run

For testing the pipeline without full training:

```bash
# Edit config to use quick mode
sed -i 's/USE_FULL_EVAL_SETS = True/USE_FULL_EVAL_SETS = False/' configs/experiment_config.py
sed -i 's/TRAINING_SAMPLES = None/TRAINING_SAMPLES = 10000/' configs/experiment_config.py

# Submit
bash run_all.sh
```

**Total time**: ~5-8 hours (quick testing only)

## Monitoring Progress

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View foundation jobs only
squeue -u $USER | grep foundation

# Check specific job
scontrol show job <JOB_ID>
```

### View Logs

```bash
# Live tail of setup log
tail -f logs/job_0_setup_<JOB_ID>.out

# Check for errors
grep -i error logs/*.err

# View stage completion flags
ls -lh $SCRATCH/foundation_llm_work/*.flag
```

### Check Results

```bash
# Quick summary
cat $SCRATCH/foundation_llm_work/experiment_summary.txt

# Detailed results
cat $SCRATCH/foundation_llm_results/*.json | jq .

# Final aggregated results
cat $PROJECT/foundation_llm_final_results/final_results.json
```

## Expected Outputs

After all jobs complete, you should have:

### Checkpoints
- `$SCRATCH/foundation_llm_work/checkpoints/baseline_checkpoints/best_model.pt`
- `$SCRATCH/foundation_llm_work/checkpoints/monotonic_checkpoints/best_model.pt`
- `$SCRATCH/foundation_llm_work/checkpoints/monotonic_initialized.pt`

### Results
- `$SCRATCH/foundation_llm_results/evaluation_results.json` (perplexity, benchmarks)
- `$SCRATCH/foundation_llm_results/uat_results.json` (UAT attacks)
- `$SCRATCH/foundation_llm_results/hotflip_results.json` (HotFlip attacks)
- `$SCRATCH/foundation_llm_results/final_results.json` (aggregated)

### Logs
- `$SCRATCH/foundation_llm_work/stage_logs/*.log` (detailed stage logs)
- `logs/job_*_<JOB_ID>.out` (SLURM output)

## Expected Results

Based on extrapolation from T5-small experiments:

| Metric | Baseline | Monotonic | Change |
|---|---|---|---|
| **Perplexity (Pile)** | ~10.2 | ~10.9 | +6.8% |
| **HotFlip Success Rate** | ~55% | ~18% | -67% |
| **LAMBADA Accuracy** | TBD | TBD | TBD |

## Troubleshooting

### Job Fails Immediately

**Check**: Does conda environment exist?
```bash
conda env list | grep mono_s2s
```

**Fix**: Create environment:
```bash
conda create -n mono_s2s python=3.10 -y
conda activate mono_s2s
pip install torch transformers datasets accelerate
```

### Out of Memory (OOM)

**Symptom**: Job killed with exit code 137

**Fix 1**: Reduce batch size in `configs/experiment_config.py`:
```python
BATCH_SIZE = 4  # Reduce from 8
```

**Fix 2**: Use gradient checkpointing (add to training scripts):
```python
model.gradient_checkpointing_enable()
```

### Model Download Fails

**Symptom**: "Connection timeout" or "403 Forbidden"

**Fix**: Set Hugging Face token:
```bash
export HF_TOKEN=your_token_here
# Or store in ~/.huggingface/token
```

### Jobs Stuck in Queue

**Check**: Partition availability
```bash
sinfo -p aa100
```

**Alternative**: Use different partition
```bash
# Edit all job scripts:
#SBATCH --partition=ami100  # Instead of aa100
```

## Multi-Seed Runs

To run multiple seeds in parallel:

```bash
# Submit seed 42
EXPERIMENT_SEED=42 bash run_all.sh

# Submit seed 1337 (parallel)
EXPERIMENT_SEED=1337 bash run_all.sh

# etc. for seeds 2024, 8888, 12345
```

Each seed runs independently (~60 hours each).

## Stopping Experiments

### Cancel All Jobs

```bash
# List your job IDs
squeue -u $USER -o "%.18i" -h

# Cancel all
scancel $(squeue -u $USER -o "%.18i" -h | tr '\n' ' ')
```

### Cancel Specific Pipeline

If you saved the job IDs from `run_all.sh` output:
```bash
scancel JOB_ID_1 JOB_ID_2 JOB_ID_3 ...
```

## Next Steps

After experiments complete:

1. **Update Paper**: Replace red placeholder values in `documentation/monotone_llms_paper.tex`
2. **Generate Figures**: Use results to create plots for Section 4.3
3. **Scale Up**: Try Pythia-2.8B or Pythia-6.9B
4. **Different Models**: Test on OPT, GPT-Neo, or Llama variants

## Getting Help

- **Main Project**: See `../README.md` and `../START_HERE.md`
- **HPC Issues**: Consult `../hpc_version/TROUBLESHOOTING.md`
- **Questions**: Contact research group or check documentation

---

**Estimated Setup Time**: 5 minutes
**Estimated First-Run Time**: 60-70 hours
**Storage Required**: ~500GB scratch, ~100GB project

**Ready to start?** Run `bash run_all.sh`
