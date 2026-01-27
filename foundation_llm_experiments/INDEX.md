# Foundation LLM Experiments - Directory Index

Complete reference for all files in this experimental pipeline.

## Quick Navigation

- **Getting Started**: `QUICKSTART.md`
- **Paper Integration**: `PAPER_INTEGRATION.md`
- **Full Documentation**: `README.md`
- **Run Experiments**: `bash run_all.sh`

## Directory Structure

```
foundation_llm_experiments/
├── README.md                    # Full project documentation
├── QUICKSTART.md               # 5-minute getting started guide
├── PAPER_INTEGRATION.md        # How to update paper with results
├── INDEX.md                    # This file
│
├── run_all.sh                  # Master job submission script
│
├── configs/
│   └── experiment_config.py    # All hyperparameters and paths
│
├── scripts/                    # Python execution scripts
│   ├── stage_0_setup.py                 # Download Pythia-1.4B
│   ├── stage_1_apply_monotonicity.py    # Apply constraints
│   ├── stage_2_train_baseline.py        # [TODO] Baseline recovery
│   ├── stage_3_train_monotonic.py       # [TODO] Monotonic recovery
│   ├── stage_4_evaluate.py              # [TODO] Benchmark evaluation
│   ├── stage_5_uat_attacks.py           # [TODO] UAT attacks
│   ├── stage_6_hotflip_attacks.py       # [TODO] HotFlip attacks
│   └── stage_7_aggregate.py             # [TODO] Aggregate results
│
├── jobs/                       # SLURM job scripts
│   ├── job_0_setup.sh                   # Setup job (1 hour)
│   ├── job_1_monotonicity.sh            # Monotonicity job (30 min)
│   ├── job_2_baseline.sh                # [TODO] Baseline job (24 hours)
│   ├── job_3_monotonic.sh               # [TODO] Monotonic job (32 hours)
│   ├── job_4_evaluate.sh                # [TODO] Evaluation job (8 hours)
│   ├── job_5_uat.sh                     # [TODO] UAT job (6 hours)
│   ├── job_6_hotflip.sh                 # [TODO] HotFlip job (4 hours)
│   └── job_7_aggregate.sh               # [TODO] Aggregate job (30 min)
│
└── utils/
    └── common_utils.py         # Shared utilities (adapted from main project)
```

## File Purposes

### Documentation Files

| File | Purpose | When to Read |
|---|---|---|
| `README.md` | Complete project overview | First time setup |
| `QUICKSTART.md` | Get running in 5 minutes | When in a hurry |
| `PAPER_INTEGRATION.md` | Update paper with results | After experiments complete |
| `INDEX.md` | This file - directory reference | Finding specific files |

### Configuration Files

| File | Purpose | When to Edit |
|---|---|---|
| `configs/experiment_config.py` | All hyperparameters, paths, model selection | Before first run |

**Key Variables to Review**:
- `MODEL_NAME`: Change to test different models
- `BATCH_SIZE`: Adjust if OOM errors
- `USE_FULL_EVAL_SETS`: Set `False` for quick testing
- `SLURM_PARTITION`: Match your HPC cluster

### Execution Scripts

**Stage 0: Setup**
- **File**: `scripts/stage_0_setup.py`
- **Job**: `jobs/job_0_setup.sh`
- **Runtime**: ~1 hour
- **Purpose**: Download Pythia-1.4B, verify environment
- **Outputs**: `setup_complete.json`, model cache

**Stage 1: Apply Monotonicity**
- **File**: `scripts/stage_1_apply_monotonicity.py`
- **Job**: `jobs/job_1_monotonicity.sh`
- **Runtime**: ~30 minutes
- **Purpose**: Apply softplus constraints to FFN layers
- **Outputs**: `monotonic_initialized.pt`, `monotonicity_application_log.json`

**Stage 2: Baseline Training** [TODO]
- **File**: `scripts/stage_2_train_baseline.py`
- **Job**: `jobs/job_2_baseline.sh`
- **Runtime**: ~24 hours
- **Purpose**: Finetune standard Pythia on Pile (1 epoch)
- **Outputs**: `baseline_checkpoints/best_model.pt`, `baseline_training_history.json`

**Stage 3: Monotonic Training** [TODO]
- **File**: `scripts/stage_3_train_monotonic.py`
- **Job**: `jobs/job_3_monotonic.sh`
- **Runtime**: ~32 hours
- **Purpose**: Finetune monotonic Pythia on Pile (1 epoch, extended warmup)
- **Outputs**: `monotonic_checkpoints/best_model.pt`, `monotonic_training_history.json`

**Stage 4: Evaluation** [TODO]
- **File**: `scripts/stage_4_evaluate.py`
- **Job**: `jobs/job_4_evaluate.sh`
- **Runtime**: ~8 hours
- **Purpose**: Evaluate on Pile test, LAMBADA, HellaSwag, etc.
- **Outputs**: `evaluation_results.json`

**Stage 5: UAT Attacks** [TODO]
- **File**: `scripts/stage_5_uat_attacks.py`
- **Job**: `jobs/job_5_uat.sh`
- **Runtime**: ~6 hours
- **Purpose**: Learn universal adversarial triggers
- **Outputs**: `uat_results.json`, `learned_triggers.csv`

**Stage 6: HotFlip Attacks** [TODO]
- **File**: `scripts/stage_6_hotflip_attacks.py`
- **Job**: `jobs/job_6_hotflip.sh`
- **Runtime**: ~4 hours
- **Purpose**: Gradient-based token flipping attacks
- **Outputs**: `hotflip_results.json`

**Stage 7: Aggregate** [TODO]
- **File**: `scripts/stage_7_aggregate.py`
- **Job**: `jobs/job_7_aggregate.sh`
- **Runtime**: ~30 minutes
- **Purpose**: Combine all results, generate summary
- **Outputs**: `final_results.json`, `experiment_summary.txt`

### Utility Files

| File | Purpose |
|---|---|
| `utils/common_utils.py` | Shared functions (adapted from `../hpc_version/utils/common_utils.py`) |

**Key Functions**:
- `set_all_seeds()`: Reproducibility
- `make_model_monotonic()`: Apply constraints
- `compute_perplexity()`: Evaluate LM performance
- `StageLogger`: Track progress

### Master Scripts

| File | Purpose | Usage |
|---|---|---|
| `run_all.sh` | Submit all jobs with dependencies | `bash run_all.sh` |

## Status Tracking

### Completion Flags

After each stage completes successfully, a flag file is created in `$SCRATCH/foundation_llm_work/`:

- `stage_0_setup_complete.flag`
- `stage_1_apply_monotonicity_complete.flag`
- `stage_2_train_baseline_complete.flag`
- `stage_3_train_monotonic_complete.flag`
- `stage_4_evaluate_complete.flag`
- `stage_5_uat_complete.flag`
- `stage_6_hotflip_complete.flag`
- `stage_7_aggregate_complete.flag`

**Check Progress**:
```bash
ls -lh $SCRATCH/foundation_llm_work/*.flag
```

### Log Files

SLURM output logs are written to `logs/`:

- `logs/job_0_setup_<JOB_ID>.out` - stdout
- `logs/job_0_setup_<JOB_ID>.err` - stderr
- (repeat for jobs 1-7)

Stage-specific detailed logs in `$SCRATCH/foundation_llm_work/stage_logs/`:

- `stage_logs/stage_0_setup.log`
- `stage_logs/stage_1_apply_monotonicity.log`
- (repeat for stages 2-7)

## Results Files

### Intermediate Results

Located in `$SCRATCH/foundation_llm_results/`:

| File | Content |
|---|---|
| `setup_complete.json` | Environment info, model details |
| `monotonicity_application_log.json` | Constraint statistics |
| `baseline_training_history.json` | Training losses (baseline) |
| `monotonic_training_history.json` | Training losses (monotonic) |
| `evaluation_results.json` | Perplexity, benchmark scores |
| `uat_results.json` | Universal trigger attack results |
| `hotflip_results.json` | HotFlip attack results |

### Final Results

Located in `$PROJECT/foundation_llm_final_results/`:

| File | Content |
|---|---|
| `final_results.json` | Aggregated metrics across all stages |
| `experiment_summary.txt` | Human-readable summary |

## Common Tasks

### Running Experiments

```bash
# Full pipeline
bash run_all.sh

# Individual stage
sbatch jobs/job_0_setup.sh

# Quick test
# (Edit config first to set USE_FULL_EVAL_SETS=False)
bash run_all.sh
```

### Monitoring

```bash
# Check jobs
squeue -u $USER | grep foundation

# View logs
tail -f logs/job_0_setup_*.out

# Check completion
ls $SCRATCH/foundation_llm_work/*.flag
```

### Debugging

```bash
# Check stage log
cat $SCRATCH/foundation_llm_work/stage_logs/stage_0_setup.log

# Check for errors
grep -i error logs/*.err

# Verify results
cat $SCRATCH/foundation_llm_results/*.json | jq .
```

### Updating Paper

See `PAPER_INTEGRATION.md` for detailed instructions.

Quick reference:
1. Extract perplexity: `jq '.pile_test' $SCRATCH/foundation_llm_results/evaluation_results.json`
2. Extract attack success: `jq '.results' $SCRATCH/foundation_llm_results/hotflip_results.json`
3. Update Table 7 in `../documentation/monotone_llms_paper.tex`
4. Remove red text from Section 4.3

## Development Status

**Completed**:
- ✅ Configuration system
- ✅ Utilities (adapted from main project)
- ✅ Stage 0: Setup script
- ✅ Stage 1: Apply monotonicity script
- ✅ Job scripts (stages 0-1)
- ✅ Master run script
- ✅ Documentation

**TODO** (for full implementation):
- ⏳ Stage 2: Baseline training script
- ⏳ Stage 3: Monotonic training script
- ⏳ Stage 4: Evaluation script
- ⏳ Stage 5: UAT attack script
- ⏳ Stage 6: HotFlip attack script
- ⏳ Stage 7: Aggregation script
- ⏳ Job scripts (stages 2-7)
- ⏳ Multi-seed aggregation script

**Note**: Scripts 2-7 can be adapted from `../hpc_version/scripts/` with modifications for:
- Decoder-only architecture (vs encoder-decoder)
- Perplexity evaluation (vs ROUGE)
- General LM tasks (vs summarization)

## Relation to Main Project

This directory is **separate but parallel** to the main `mono-s2s` project:

| Aspect | Main Project | This Project |
|---|---|---|
| **Location** | `hpc_version/` | `foundation_llm_experiments/` |
| **Model** | T5-small (encoder-decoder) | Pythia-1.4B (decoder-only) |
| **Task** | Summarization | General language modeling |
| **Metrics** | ROUGE scores | Perplexity, benchmark accuracy |
| **Paper Section** | Sections 4.1-4.2 (core results) | Section 4.3 (scaling) |
| **Status** | ✅ Complete | ⏳ In Progress |

**Shared Components**:
- Monotonicity implementation (softplus parametrization)
- Attack methodology (UAT, HotFlip)
- Experimental design (baseline vs monotonic)
- Statistical analysis (multi-seed, significance tests)

**Key Differences**:
- **Architecture**: Decoder-only requires different model handling
- **Evaluation**: Perplexity instead of ROUGE
- **Training**: Recovery phase to restore perplexity post-constraints
- **Benchmarks**: LAMBADA, HellaSwag vs CNN/DM

## Getting Help

1. **Quick Start**: Read `QUICKSTART.md`
2. **Full Docs**: Read `README.md`
3. **Paper Updates**: Read `PAPER_INTEGRATION.md`
4. **Main Project**: See `../README.md` and `../START_HERE.md`
5. **HPC Issues**: See `../hpc_version/TROUBLESHOOTING.md`
6. **Contact**: Research group or paper authors

---

**Last Updated**: 2026-01-27
**Version**: 0.1.0 (Initial scaffold)
**Status**: Core infrastructure complete, training/evaluation scripts pending
