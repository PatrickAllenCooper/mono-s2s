# Experimental Results Workflow

**Purpose**: Ensure all experimental evidence is tracked, organized, and version controlled

## Quick Reference

### After Experiment Completes on HPC

```bash
# 1. Archive from SCRATCH to PROJECT (on HPC)
bash scripts/archive_experiment.sh 42

# 2. Organize and commit (on local machine or HPC)
bash scripts/commit_experiment_results.sh 42 t5_summarization

# 3. Link to paper tables
python scripts/link_results_to_paper.py --auto

# 4. Push to git
git push origin main
```

**All experimental evidence now version controlled** ✅

## Detailed Workflow

### Step 1: While Experiments Run (On HPC)

**Create periodic snapshots**:

```bash
# Run daily while experiments are active
bash scripts/snapshot_running_experiments.sh --commit
```

**Purpose**:
- Records which jobs were running when
- Tracks progress over time
- Helps diagnose issues if something fails

**Creates**: `experiment_snapshots/snapshot_YYYYMMDD_HHMMSS.json`

**Commits automatically** with `--commit` flag

---

### Step 2: When Experiment Completes (On HPC)

**Archive results from SCRATCH to PROJECT**:

```bash
# Basic archive (results only, ~100MB)
bash scripts/archive_experiment.sh 42

# Include checkpoints (larger, ~5GB)
bash scripts/archive_experiment.sh 42 --checkpoints
```

**Purpose**:
- Preserves results (SCRATCH may be purged)
- Moves to persistent storage (PROJECT)
- Prepares for version control

**Location**: `$PROJECT/mono_s2s_final_results/seed_42/`

---

### Step 3: Organize for Version Control (Local or HPC)

**Organize into structured format**:

```bash
# For T5 experiments
bash scripts/commit_experiment_results.sh 42 t5_summarization

# For Foundation LLM experiments
bash scripts/commit_experiment_results.sh 42 pythia_foundation
```

**What this does**:
1. Copies results from PROJECT to `experiment_results/`
2. Creates `metadata.json` with full experiment details
3. Updates `experiment_index.json`
4. Stages files for git commit
5. Prompts for confirmation
6. Commits with detailed message

**Location**: `experiment_results/t5_experiments/seed_42/`

**Git Status**: Files staged and committed

---

### Step 4: Link to Paper (Local)

**Create provenance linking results to paper tables**:

```bash
# Automatic linking (recommended)
python scripts/link_results_to_paper.py --auto

# Manual linking
python scripts/link_results_to_paper.py \
    --experiment seed_42 \
    --table table_2_rouge_scores \
    --values baseline_rouge_l=0.250,monotonic_rouge_l=0.242
```

**Purpose**:
- Records which experiments produced which paper values
- Enables reproducibility
- Helps reviewers verify claims

**Creates**: `paper_evidence/provenance.json`

---

### Step 5: Push to Remote (Local)

**Push to git repository**:

```bash
# Push commits
git push origin main

# Create tag for important experiments
git tag exp-t5-seed42-final
git push origin exp-t5-seed42-final
```

**Purpose**:
- Backs up experimental evidence
- Shares with collaborators
- Permanent record

---

## Directory Structure

### After Following Workflow

```
mono-s2s/
├── experiment_results/
│   ├── t5_experiments/
│   │   ├── seed_42/                      # ✅ Version controlled
│   │   │   ├── metadata.json
│   │   │   ├── baseline_training.json
│   │   │   ├── monotonic_training.json
│   │   │   ├── evaluation_results.json
│   │   │   ├── uat_results.json
│   │   │   ├── hotflip_results.json
│   │   │   ├── final_results.json
│   │   │   └── experiment_log.txt
│   │   ├── seed_1337/                    # ✅ Version controlled
│   │   ├── (other seeds...)
│   │   └── aggregated/
│   │       └── all_seeds_summary.json    # ✅ Version controlled
│   │
│   ├── foundation_llm_experiments/
│   │   └── seed_42/                      # ✅ Version controlled
│   │
│   └── experiment_index.json             # ✅ Version controlled
│
├── paper_evidence/
│   ├── provenance.json                   # ✅ Version controlled
│   ├── table_1_training_dynamics.json    # ✅ Version controlled
│   ├── table_2_rouge_scores.json         # ✅ Version controlled
│   └── (all paper tables)                # ✅ Version controlled
│
├── experiment_snapshots/
│   ├── snapshot_20260127_100000.json     # ✅ Version controlled
│   ├── snapshot_20260127_180000.json     # ✅ Version controlled
│   └── (daily snapshots)                 # ✅ Version controlled
│
└── .gitignore                            # ✅ Configured properly
```

**All small result files**: ✅ Tracked in git
**Large checkpoints**: ⚠️ Archived to PROJECT (referenced in metadata)
**Provenance**: ✅ Complete audit trail

---

## What Gets Tracked

### ✅ Tracked in Git (Small Files)

- `experiment_results/**/*.json` - All JSON results
- `experiment_results/**/*.csv` - Data tables
- `experiment_results/**/*.txt` - Text summaries
- `experiment_results/**/metadata.json` - Experiment metadata
- `paper_evidence/**/*.json` - Paper table data
- `paper_evidence/provenance.json` - Audit trail
- `experiment_snapshots/*.json` - Progress snapshots
- All `.md` documentation
- All scripts

**Total size**: ~100-500 MB per experiment (small enough for git)

### ❌ NOT Tracked in Git (Large Files)

- `*.pt`, `*.pth` - Model checkpoints (GBs)
- `logs/*.out`, `logs/*.err` - SLURM logs (large)
- HuggingFace cache - Downloaded data
- Temporary work directories

**But**:
- Checkpoint locations recorded in `metadata.json`
- Log summaries included in `experiment_log.txt`
- Cache info in `metadata.json`

**Storage**: Archived to `$PROJECT` (persistent but not in git)

---

## Automated Daily Tracking

### Cron Job Setup (On HPC)

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 6 PM)
0 18 * * * cd ~/mono-s2s && bash scripts/snapshot_running_experiments.sh --commit
```

**Effect**: Automatically creates daily snapshots of experiment status

### Manual Tracking

```bash
# Run whenever you want to record current state
bash scripts/snapshot_running_experiments.sh --commit
```

---

## Verifying Experimental Evidence

### Check What's Tracked

```bash
# List all tracked experiments
cat experiment_results/experiment_index.json | jq '.experiments[] | {id: .experiment_id, seed: .seed, status: .status}'

# Check specific experiment
cat experiment_results/t5_experiments/seed_42/metadata.json | jq .

# View provenance (which experiments → which paper tables)
cat paper_evidence/provenance.json | jq '.tables'
```

### Check Git Status

```bash
# See what's being tracked
git ls-files experiment_results/
git ls-files paper_evidence/
git ls-files experiment_snapshots/

# See what's ignored (shouldn't include small .json files)
git status --ignored
```

---

## Recovering Experimental Evidence

### If You Need Results from Seed 42

```bash
# Results are in git
git log --all --oneline -- experiment_results/t5_experiments/seed_42/

# Checkout specific version
git checkout <commit-hash> -- experiment_results/t5_experiments/seed_42/

# View results
cat experiment_results/t5_experiments/seed_42/final_results.json | jq .
```

### If You Need to Reproduce

```bash
# Check metadata for exact configuration
cat experiment_results/t5_experiments/seed_42/metadata.json

# Shows:
# - Exact hyperparameters used
# - Model version
# - Dataset versions
# - Hardware used
# - SLURM job IDs
# - Git commit hash of code

# Rerun with same settings
cd hpc_version
# Use hyperparameters from metadata.json
```

---

## Multi-Seed Aggregation

### After All Seeds Complete

```bash
# Aggregate results across seeds
python hpc_version/scripts/aggregate_multi_seed.py \
    --seeds 42,1337,2024,8888,12345 \
    --output experiment_results/t5_experiments/aggregated/

# Commit aggregated results
git add experiment_results/t5_experiments/aggregated/
git commit -m "Add multi-seed aggregated results for T5 experiments

Aggregated across 5 seeds: 42, 1337, 2024, 8888, 12345

Includes:
- Mean and std for all metrics
- Statistical significance tests
- Cross-seed consistency analysis

See experiment_results/t5_experiments/aggregated/ for details."

git push origin main
```

---

## Paper Integration

### Workflow for Updating Paper

1. **Extract values from experimental results**:
   ```bash
   # Values are in version control
   cat experiment_results/t5_experiments/seed_42/evaluation_results.json | \
       jq '.cnn_dm.baseline_t5.rouge_scores.rougeLsum.mean'
   ```

2. **Update paper table**:
   ```bash
   # Edit documentation/monotone_llms_paper.tex
   # Replace values in tables
   ```

3. **Record provenance**:
   ```bash
   python scripts/link_results_to_paper.py \
       --experiment seed_42 \
       --table table_2_rouge_scores \
       --values baseline_rouge_l=0.250,monotonic_rouge_l=0.242
   ```

4. **Commit paper update**:
   ```bash
   git add documentation/monotone_llms_paper.tex
   git add paper_evidence/provenance.json
   git commit -m "Update paper Table 2 with experimental results from seed 42

Values extracted from experiment_results/t5_experiments/seed_42/
Provenance recorded in paper_evidence/provenance.json

Changes:
- Table 2: Added baseline and monotonic ROUGE-L scores
- Values verified against experimental results
- Provenance link created

See paper_evidence/provenance.json for full audit trail."
   ```

---

## Best Practices

### DO:

- ✅ Commit results immediately after experiments complete
- ✅ Use descriptive commit messages
- ✅ Link results to paper tables via provenance
- ✅ Create daily snapshots during long runs
- ✅ Verify metadata is complete before committing
- ✅ Tag important experimental milestones

### DON'T:

- ❌ Commit large checkpoint files to git (archive to PROJECT instead)
- ❌ Commit raw SLURM logs (summarize in experiment_log.txt)
- ❌ Mix results from different experiments in same commit
- ❌ Update paper without recording provenance
- ❌ Delete results from SCRATCH before archiving

---

## Troubleshooting

### Results Not Committing

**Problem**: `git add experiment_results/` does nothing

**Check**: Make sure not in .gitignore
```bash
git check-ignore experiment_results/t5_experiments/seed_42/*.json
# Should return nothing (files not ignored)
```

### Commit Too Large

**Problem**: Commit rejected (>100MB)

**Solution**: Remove large files
```bash
# Find large files
find experiment_results/ -type f -size +50M

# Don't commit large files, reference in metadata instead
```

### Lost Provenance

**Problem**: Don't know which experiment produced paper values

**Solution**: Check provenance file
```bash
cat paper_evidence/provenance.json | jq '.tables.table_2_rouge_scores'
```

---

## Maintenance

### Clean Up Old Snapshots

```bash
# Keep only last 30 days
find experiment_snapshots/ -name "snapshot_*.json" -mtime +30 -delete

# Or keep only monthly
bash scripts/prune_old_snapshots.sh --keep-monthly
```

### Verify Index Consistency

```bash
# Check all experiments in index actually exist
python scripts/verify_experiment_index.py

# Fix any inconsistencies
python scripts/verify_experiment_index.py --fix
```

---

## Summary Commands

### Complete Workflow (After Experiment)

```bash
# On HPC (after experiment completes)
bash scripts/archive_experiment.sh 42

# On local machine or HPC
bash scripts/commit_experiment_results.sh 42 t5_summarization
python scripts/link_results_to_paper.py --auto
git push origin main
```

**Time**: ~5 minutes
**Result**: All evidence version controlled and linked to paper

---

## Git Commit Message Template

```
Add experimental results: [experiment_type], seed [seed]

Experiment Details:
- Type: [t5_summarization | pythia_foundation]
- Seed: [42 | 1337 | ...]
- Date: [YYYY-MM-DD]
- Status: [complete | partial]

Results Include:
- Training dynamics (baseline + monotonic)
- Evaluation on [datasets]
- Attack results (UAT + HotFlip)
- Final aggregated metrics

Key Findings:
- Perplexity gap: [X%]
- Attack reduction: [Y%]
- [Other notable results]

Files Added:
- experiment_results/[type]/seed_[seed]/metadata.json
- experiment_results/[type]/seed_[seed]/final_results.json
- [other files]

Provenance:
- Used in paper tables: [table_1, table_2, ...]
- Linked via paper_evidence/provenance.json

This commit preserves experimental evidence for reproducibility.
```

---

**All experimental evidence from this point forward will be systematically tracked and version controlled using this workflow.**
