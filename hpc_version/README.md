# HPC Version - CURC-Compatible SLURM Jobs

This directory contains a modular, HPC-compatible version of the mono-s2s fair comparison experiment, designed for execution on SLURM-based clusters (e.g., CURC).

## Directory Structure

```
hpc_version/
├── configs/
│   └── experiment_config.py    # Centralized configuration
├── utils/
│   └── common_utils.py         # Shared utility functions
├── scripts/
│   ├── stage_0_setup.py        # Environment setup, download models
│   ├── stage_1_prepare_data.py # Load and prepare datasets
│   ├── stage_2_train_baseline.py    # Train baseline model
│   ├── stage_3_train_monotonic.py   # Train monotonic model
│   ├── stage_4_evaluate.py     # Comprehensive evaluation
│   ├── stage_5_uat_attacks.py  # UAT attacks + transfer matrix
│   ├── stage_6_hotflip_attacks.py   # HotFlip attacks
│   └── stage_7_aggregate.py    # Final analysis and results
├── jobs/
│   ├── job_0_setup.sh          # SLURM script for stage 0
│   ├── job_1_data.sh           # SLURM script for stage 1
│   ├── job_2_baseline.sh       # SLURM script for stage 2
│   ├── job_3_monotonic.sh      # SLURM script for stage 3
│   ├── job_4_evaluate.sh       # SLURM script for stage 4
│   ├── job_5_uat.sh            # SLURM script for stage 5
│   ├── job_6_hotflip.sh        # SLURM script for stage 6
│   └── job_7_aggregate.sh      # SLURM script for stage 7
├── logs/                       # Job output logs (created at runtime)
├── run_all.sh                  # Master orchestration script
└── README.md                   # This file
```

## Quick Start

### 1. Configure Your Environment

Edit `configs/experiment_config.py`:
```python
# Set your paths
SCRATCH_DIR = "/scratch/your_username"  # Your scratch directory
PROJECT_DIR = "/projects/your_project"   # Your project directory
```

### 2. Submit Jobs Sequentially

```bash
# Stage 0: Setup (download models, verify environment)
sbatch jobs/job_0_setup.sh
# Wait for completion, check logs/job_0_*.out

# Stage 1: Prepare data (load all datasets)
sbatch jobs/job_1_data.sh
# Wait for completion, check logs/job_1_*.out

# Stage 2: Train baseline model (~4-8 hours)
sbatch jobs/job_2_baseline.sh
# Wait for completion, check logs/job_2_*.out

# Stage 3: Train monotonic model (~4-8 hours)
sbatch jobs/job_3_monotonic.sh
# Wait for completion, check logs/job_3_*.out

# Stage 4: Comprehensive evaluation (~2-4 hours)
sbatch jobs/job_4_evaluate.sh
# Wait for completion, check logs/job_4_*.out

# Stage 5: UAT attacks + transfer matrix (~2-3 hours)
sbatch jobs/job_5_uat.sh
# Wait for completion, check logs/job_5_*.out

# Stage 6: HotFlip attacks (~1-2 hours)
sbatch jobs/job_6_hotflip.sh
# Wait for completion, check logs/job_6_*.out

# Stage 7: Aggregate results
sbatch jobs/job_7_aggregate.sh
# Check final results
```

### 3. Or Use Master Script (Automated)

```bash
./run_all.sh
# This submits all jobs with dependencies
# Each stage waits for previous to complete
# Automatically checks for errors before proceeding
```

## Stage Descriptions

### Stage 0: Setup (~10 minutes)
- Verify Python environment
- Download t5-small model
- Create directory structure
- Log environment details
- **Output:** `setup_complete.json`

### Stage 1: Data Preparation (~30-60 minutes)
- Load all 7 training datasets
- Load all 3 test datasets
- Load attack datasets (validation + test splits)
- Prepare DataLoaders
- **Output:** `data_prepared.json`, cached datasets

### Stage 2: Train Baseline (~4-8 hours)
- Initialize baseline T5 model (unconstrained)
- Train on mixed dataset (7 sources, ~100k examples)
- Save checkpoints every epoch
- Early stopping based on validation loss
- **Output:** `baseline_checkpoints/best_model.pt`, `baseline_training_history.json`

### Stage 3: Train Monotonic (~4-8 hours)
- Initialize monotonic T5 model (W≥0 FFN constraints)
- Train with IDENTICAL settings as baseline
- Save checkpoints every epoch
- Early stopping based on validation loss
- **Output:** `monotonic_checkpoints/best_model.pt`, `monotonic_training_history.json`

### Stage 4: Comprehensive Evaluation (~2-4 hours)
- Load all three models (Standard, Baseline, Monotonic)
- Evaluate on all three test sets (CNN/DM, XSUM, SAMSum)
- Compute ROUGE with bootstrap 95% CIs
- Token-level length statistics
- Brevity penalties
- **Output:** `evaluation_results.json` with all metrics + CIs

### Stage 5: UAT Attacks (~2-3 hours)
- Learn triggers for each model (on validation set)
- Evaluate on held-out test set
- Compute transfer attack matrix (3×3)
- **Output:** `uat_results.json`, `transfer_matrix.json`, `learned_triggers.csv`

### Stage 6: HotFlip Attacks (~1-2 hours)
- Run HotFlip on all three models
- Evaluate attack effectiveness
- Statistical significance tests
- **Output:** `hotflip_results.json`

### Stage 7: Aggregate Results (~5 minutes)
- Combine all results
- Generate comparison tables
- Create visualizations
- Final statistical analysis
- **Output:** `final_results.json`, `experiment_metadata.json`, plots

## Resource Requirements

### Minimum (Quick Testing)
- **GPU:** 1× V100 or A100 (16GB+ VRAM)
- **Memory:** 32GB RAM
- **Time:** ~12-20 hours total
- **Storage:** 50GB scratch space

### Recommended (Full Evaluation)
- **GPU:** 1× A100 (40GB VRAM) 
- **Memory:** 64GB RAM
- **Time:** ~20-30 hours total
- **Storage:** 100GB scratch space

### Per-Stage Requirements

| Stage | GPU | RAM | Time | Storage |
|-------|-----|-----|------|---------|
| 0. Setup | No | 8GB | 10 min | 5GB |
| 1. Data | No | 16GB | 1 hr | 20GB |
| 2. Baseline | Yes | 32GB | 4-8 hr | 10GB |
| 3. Monotonic | Yes | 32GB | 4-8 hr | 10GB |
| 4. Evaluate | Yes | 32GB | 2-4 hr | 5GB |
| 5. UAT | Yes | 16GB | 2-3 hr | 2GB |
| 6. HotFlip | Yes | 16GB | 1-2 hr | 2GB |
| 7. Aggregate | No | 8GB | 5 min | 1GB |

## Error Handling

Each stage:
1. **Checks dependencies** from previous stages
2. **Validates inputs** before processing
3. **Logs progress** to stage-specific files
4. **Creates completion marker** (`stage_N_complete.flag`)
5. **Exits with error code** if problems occur

**The master script stops automatically if any stage fails.**

## Multi-Seed Support

To run with multiple seeds (for robust statistics):

```bash
# Edit configs/experiment_config.py
CURRENT_SEED = 42  # or 1337, 2024, 8888, 12345

# Run all stages
./run_all.sh

# After completion, repeat with different seed
# Aggregate across seeds manually or use aggregation script
```

## Checkpointing & Resume

Each training stage:
- **Saves checkpoints** every epoch
- **Can resume** from last checkpoint if interrupted
- **Automatic resume** if resubmitted

To force restart:
```bash
rm -rf baseline_checkpoints/
rm -rf monotonic_checkpoints/
```

## Monitoring Progress

```bash
# Check job status
squeue -u $USER

# Monitor logs in real-time
tail -f logs/job_2_*.out

# Check completion flags
ls -la *.flag

# View results
cat evaluation_results.json
```

## Troubleshooting

### Job Fails with OOM
- Reduce batch size in `configs/experiment_config.py`
- Set `USE_FULL_TEST_SETS = False` for quick testing
- Request more memory in job script

### Job Times Out
- Increase `#SBATCH --time` in job script
- Enable checkpointing (already implemented)
- Split into more stages if needed

### Dataset Download Fails
- Check network connectivity
- Increase timeout in stage 1
- Pre-download datasets to `/projects` directory

### Model Checkpoint Not Found
- Verify previous stage completed
- Check `stage_N_complete.flag` files
- Review logs for errors

## Outputs

All results saved to `$SCRATCH_DIR/mono_s2s_results/`:
```
results/
├── experiment_metadata.json       # Complete configuration
├── baseline_checkpoints/
│   └── best_model.pt              # Best baseline model
├── monotonic_checkpoints/
│   └── best_model.pt              # Best monotonic model
├── evaluation_results.json        # Primary comparison with CIs
├── transfer_matrix.json           # Cross-model attack results
├── uat_results.json               # UAT attack details
├── hotflip_results.json           # HotFlip attack details
├── final_results.json             # Aggregated analysis
└── plots/                         # Visualizations
    ├── comparison_table.png
    ├── transfer_matrix.png
    └── attack_robustness.png
```

## Best Practices

1. **Always check logs** after each stage before proceeding
2. **Verify completion flags** exist before next stage
3. **Use scratch space** for intermediate files (faster I/O)
4. **Save final results** to project directory (persistent storage)
5. **Set proper permissions** on output directories
6. **Use job dependencies** in run_all.sh (automatic chaining)

## Contact

For issues with HPC execution:
- Check CURC documentation
- Review SLURM logs in `logs/`
- Verify GPU availability: `sinfo -o "%20P %5a %.10l %16F"`

---

**Status:** Ready for HPC execution  
**Tested On:** SLURM-based clusters  
**Estimated Total Time:** 12-30 hours (depending on configuration)

