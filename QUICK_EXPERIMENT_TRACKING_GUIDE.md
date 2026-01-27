# Quick Guide: Tracking Your Experiments

**TL;DR**: Simple commands to ensure all experimental evidence is version controlled

## For Your Current Running T5 Experiments

### Right Now (5 minutes)

```bash
cd ~/code/mono-s2s

# Create snapshot of current running experiments
bash scripts/snapshot_running_experiments.sh --commit

# This captures:
# - Which jobs are running (23293371, 23293373, 23293358)
# - What stage they're in
# - How long they've been running
# - Disk usage

# Commits snapshot to git automatically
git push origin main
```

**Done**: Current state preserved ✅

### Tomorrow and Daily (1 minute)

```bash
# Create another snapshot
bash scripts/snapshot_running_experiments.sh --commit
git push
```

**Purpose**: Track progress over time

### After Jobs Complete (~60-70 hours from now)

```bash
# On HPC - Archive results from SCRATCH to PROJECT
bash scripts/archive_experiment.sh 42

# On local machine - Organize and commit to git
cd ~/code/mono-s2s
bash scripts/commit_experiment_results.sh 42 t5_summarization

# Link results to paper tables
python scripts/link_results_to_paper.py --auto

# Push to git
git push origin main
```

**Done**: All experimental evidence version controlled ✅

---

## For Future Experiments

**Same simple workflow**:

```bash
# Daily while running
bash scripts/snapshot_running_experiments.sh --commit

# After complete
bash scripts/archive_experiment.sh <seed>
bash scripts/commit_experiment_results.sh <seed> <type>
python scripts/link_results_to_paper.py --auto
git push
```

---

## What Gets Tracked

### ✅ In Git (Small Files)

- All `.json` results (~100-500 MB per experiment)
- All `.csv` data tables
- All `.txt` summaries
- `metadata.json` for each experiment
- `provenance.json` linking experiments → paper
- Daily experiment snapshots

### ❌ Not in Git (Large Files)

- Model checkpoints (*.pt files, GBs)
- SLURM output logs (too large)
- Cached datasets

**But**: Archived to `$PROJECT` and referenced in metadata

---

## Verification

### Check Everything Is Tracked

```bash
# Verify tracking system is working
bash scripts/verify_results_organized.sh

# Should output:
# ✓ ALL CHECKS PASSED
```

### Check What's in Git

```bash
# See tracked experiments
git ls-files experiment_results/

# See paper evidence
git ls-files paper_evidence/

# See snapshots
git ls-files experiment_snapshots/
```

---

## Quick Commands Reference

```bash
# === DAILY (while experiments run) ===
bash scripts/snapshot_running_experiments.sh --commit
git push

# === AFTER EXPERIMENT COMPLETES ===
bash scripts/archive_experiment.sh <seed>
bash scripts/commit_experiment_results.sh <seed> <type>
python scripts/link_results_to_paper.py --auto
git push

# === VERIFICATION ===
bash scripts/verify_results_organized.sh
```

---

## What This Achieves

From this point forward:

- ✅ **Nothing lost** - All results preserved
- ✅ **Everything tracked** - Complete history in git
- ✅ **Fully auditable** - Provenance for all paper claims
- ✅ **Easily reproducible** - Complete metadata
- ✅ **Collaboration-friendly** - Shared via git

**All experimental evidence systematically managed** ✅

---

**Next Action**: `bash scripts/snapshot_running_experiments.sh --commit`

**Time Required**: 1 minute

**Result**: Current experiment state captured and version controlled
