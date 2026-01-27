# Experiment Tracking and Version Control System

**Created**: January 27, 2026
**Purpose**: Ensure all experimental evidence is tracked, organized, and version controlled

## Overview

This document defines the system for tracking all experimental results from both:
1. **Main T5 experiments** (`hpc_version/` and results in `mono_s2s_results/`)
2. **Foundation LLM experiments** (`foundation_llm_experiments/`)

## Directory Structure for Results

```
mono-s2s/
├── experiment_results/                    # NEW: Centralized results tracking
│   ├── t5_experiments/                    # T5 summarization experiments
│   │   ├── seed_42/                       # Results by seed
│   │   │   ├── metadata.json              # Experiment metadata
│   │   │   ├── baseline_training.json
│   │   │   ├── monotonic_training.json
│   │   │   ├── evaluation_results.json
│   │   │   ├── uat_results.json
│   │   │   ├── hotflip_results.json
│   │   │   ├── final_results.json
│   │   │   └── experiment_log.txt
│   │   ├── seed_1337/
│   │   ├── seed_2024/
│   │   ├── seed_8888/
│   │   ├── seed_12345/
│   │   └── aggregated/                    # Multi-seed aggregated results
│   │       ├── all_seeds_summary.json
│   │       ├── statistical_tests.json
│   │       └── paper_ready_tables.tex
│   │
│   ├── foundation_llm_experiments/        # Pythia experiments
│   │   ├── seed_42/
│   │   ├── seed_1337/
│   │   ├── (etc.)
│   │   └── aggregated/
│   │
│   ├── experiment_index.json              # Master index of all experiments
│   └── RESULTS_README.md                  # Documentation
│
├── mono_s2s_results/                      # EXISTING: Keep for now (legacy)
│   └── (current results)
│
└── paper_evidence/                        # NEW: Results used in paper
    ├── table_1_training_dynamics.json
    ├── table_2_rouge_scores.json
    ├── table_3_multiseed_training.json
    ├── table_4_multiseed_rouge.json
    ├── table_5_hotflip_attacks.json
    ├── table_6_uat_attacks.json
    ├── table_7_foundation_models.json
    ├── provenance.json                    # Which experiments produced which tables
    └── PAPER_EVIDENCE_README.md
```

## Git Tracking Strategy

### What TO Track (Version Control)

```gitignore
# Track these in git:
experiment_results/**/*.json          # All JSON results (small)
experiment_results/**/*.csv           # All CSV data
experiment_results/**/*.txt           # All text summaries
experiment_results/**/metadata.json   # Experiment metadata
experiment_results/**/provenance.json # Data provenance
paper_evidence/**/*                   # All paper evidence
EXPERIMENT_TRACKING_SYSTEM.md         # This document
*.md                                  # All documentation
```

### What NOT to Track (Too Large)

```gitignore
# DO NOT track these (too large):
*.pt                                  # Model checkpoints (GBs)
*.pth                                 # PyTorch weights
*.ckpt                                # Checkpoints
**/checkpoints/**/*.pt                # Checkpoint directories
logs/*.out                            # SLURM output (large)
logs/*.err                            # SLURM errors (large)

# But DO track:
# - Metadata about checkpoints (which epoch, what loss, where stored)
# - Checkpoint index files
# - Final model summaries (without weights)
```

### Git LFS for Large Results (Optional)

For very large result files (>100MB):

```bash
# Initialize Git LFS (one time)
git lfs install

# Track large result files
git lfs track "experiment_results/**/*.tar.gz"
git lfs track "paper_evidence/**/*.pkl"

# Add to git
git add .gitattributes
git commit -m "Configure Git LFS for large results"
```

## Metadata Schema

Every experiment run must have `metadata.json`:

```json
{
  "experiment_id": "t5_small_seed42_20260127",
  "experiment_type": "t5_summarization",
  "timestamp_start": "2026-01-27T10:30:00",
  "timestamp_end": "2026-01-29T14:45:00",
  "seed": 42,
  "model": {
    "name": "t5-small",
    "parameters": 60000000,
    "architecture": "encoder-decoder"
  },
  "hardware": {
    "gpu": "NVIDIA A100-SXM4-40GB",
    "node": "c3gpu-a9-u33-1",
    "slurm_job_ids": [23293371, 23293373, 23293358]
  },
  "hyperparameters": {
    "baseline_epochs": 5,
    "monotonic_epochs": 7,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "(...)": "from experiment_config.py"
  },
  "results_files": {
    "baseline_training": "baseline_training.json",
    "monotonic_training": "monotonic_training.json",
    "evaluation": "evaluation_results.json",
    "uat": "uat_results.json",
    "hotflip": "hotflip_results.json",
    "final": "final_results.json"
  },
  "checkpoints": {
    "baseline_best": "$SCRATCH/mono_s2s_work/checkpoints/baseline_checkpoints/best_model.pt",
    "monotonic_best": "$SCRATCH/mono_s2s_work/checkpoints/monotonic_checkpoints/best_model.pt",
    "archived_to": "$PROJECT/mono_s2s_final_results/checkpoints_seed42.tar.gz"
  },
  "status": "complete",
  "verification": {
    "all_stages_completed": true,
    "results_validated": true,
    "used_in_paper": true,
    "paper_tables": ["table_2", "table_5"]
  }
}
```

## Provenance Tracking

Track which experimental runs produced which paper results:

```json
{
  "paper": "documentation/monotone_llms_paper.tex",
  "version": "2026-01-27",
  "tables": {
    "table_1_training_dynamics": {
      "source_experiment": "t5_small_seed42_20260127",
      "source_files": [
        "experiment_results/t5_experiments/seed_42/baseline_training.json",
        "experiment_results/t5_experiments/seed_42/monotonic_training.json"
      ],
      "extracted_values": {
        "baseline_initial_loss": 2.90,
        "monotonic_initial_loss": 4.97,
        "baseline_final_loss": 2.47,
        "monotonic_final_loss": 2.69
      },
      "extraction_date": "2026-01-27",
      "verified_by": "researcher_initials"
    },
    "table_2_rouge_scores": {
      "source_experiment": "t5_small_seed42_20260127",
      "source_files": [
        "experiment_results/t5_experiments/seed_42/evaluation_results.json"
      ],
      "extracted_values": {
        "baseline_rouge_l": 0.250,
        "monotonic_rouge_l": 0.242
      }
    },
    "(...)": "similar for all tables"
  }
}
```

## Automated Result Organization Script

I'll create a script to organize results automatically.
