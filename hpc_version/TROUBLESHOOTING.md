# HPC Troubleshooting Guide

## Quick Recovery Steps

If you encountered the "DependencyNeverSatisfied" error with all jobs stuck, follow these steps:

### 1. Cancel Stuck Jobs
```bash
scancel -u $USER
```

### 2. Pull Latest Fixes
```bash
cd ~/code/mono-s2s  # or wherever you cloned the repo
git pull origin main
```

### 3. Run Diagnostic Script
```bash
cd hpc_version
./diagnose_and_recover.sh
```

This script will:
- Check if SCRATCH/PROJECT variables are set correctly
- Find any completed work that's missing completion flags
- Auto-create missing flags if the work actually completed
- Show you what needs to be rerun

### 4. Rerun Multi-Seed Experiment
```bash
cd hpc_version
./run_multi_seed.sh
```

The fixed scripts now properly set environment variables, so this should work correctly.

---

## Common Issues

### Issue 1: `SCRATCH` or `PROJECT` not set

**Symptom:**
```bash
echo $SCRATCH
# (empty output)
```

**Fix:**
The updated `run_all.sh` and `run_multi_seed.sh` now automatically set these variables to Alpine defaults. No action needed.

If you want to set them manually:
```bash
export SCRATCH=/scratch/alpine/$USER
export PROJECT=/projects/$USER
```

---

### Issue 2: Jobs show "DependencyNeverSatisfied"

**Symptom:**
```bash
squeue -u $USER
# Shows: DependencyNeverSatisfied
```

**Cause:**
A job in the dependency chain failed or was cancelled, causing all downstream jobs to be stuck.

**Fix:**
1. Cancel all jobs: `scancel -u $USER`
2. Run `./diagnose_and_recover.sh` to check what actually completed
3. Rerun: `./run_multi_seed.sh`

---

### Issue 3: No log files for failed jobs

**Symptom:**
```bash
cat logs/job_2_baseline_23270661.out
# cat: ... No such file or directory
```

**Cause:**
Jobs were submitted but never ran (stuck in dependency chain).

**Fix:**
The job never executed on a compute node. Check earlier jobs in the chain (job_0, job_1) to see what actually failed.

---

### Issue 4: Completion flags missing but work completed

**Symptom:**
- Job logs show "COMPLETED SUCCESSFULLY"
- Flag file doesn't exist
- Subsequent jobs failed with dependency errors

**Cause:**
Path mismatch between where flags were written (compute node) and checked (login node).

**Fix:**
Run `./diagnose_and_recover.sh` - it will auto-create missing flags if it can verify the work actually completed.

---

## Checking Job Status

### View Queue
```bash
squeue -u $USER
```

### View Completed Jobs (today)
```bash
sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed
```

### View Job Details
```bash
# For a specific job ID
sacct -j 23270661 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
```

### View Job Logs
```bash
# Find logs for a specific job
ls -lt logs/job_*_23270661.*

# View output
cat logs/job_2_baseline_23270661.out

# View errors
cat logs/job_2_baseline_23270661.err
```

---

## Disk Space Issues

### Check Quota
```bash
# Check scratch space
curc-quota

# Or manually check
df -h $SCRATCH
df -h $PROJECT
```

### Clean Up Old Runs
```bash
# Remove old checkpoints (use with caution!)
./clean_checkpoints.sh

# Remove all work (WARNING: deletes everything!)
./clean_all.sh
```

---

## Manual Recovery

If `diagnose_and_recover.sh` doesn't auto-fix the issue, you can manually create flags:

### Create Setup Flag
```bash
mkdir -p $SCRATCH/mono_s2s_work
cat > $SCRATCH/mono_s2s_work/stage_0_setup_complete.flag <<EOF
Completed at: $(date '+%Y-%m-%d %H:%M:%S')
Seed: 42
Note: Manually created
EOF
```

### Create Data Prep Flag
```bash
cat > $SCRATCH/mono_s2s_work/stage_1_data_prep_complete.flag <<EOF
Completed at: $(date '+%Y-%m-%d %H:%M:%S')
Seed: 42
Note: Manually created
EOF
```

### Create Seed-Specific Flag (e.g., baseline training)
```bash
SEED=42
mkdir -p $SCRATCH/mono_s2s_work/seed_$SEED
cat > $SCRATCH/mono_s2s_work/seed_$SEED/stage_2_train_baseline_complete.flag <<EOF
Completed at: $(date '+%Y-%m-%d %H:%M:%S')
Seed: $SEED
Note: Manually created
EOF
```

**Warning:** Only create flags if you've verified the corresponding work actually completed!

---

## Getting Help

If issues persist:

1. Check the main README: `../README.md`
2. Review job logs in `logs/`
3. Check CURC documentation: https://curc.readthedocs.io/
4. Contact CURC support: rc-help@colorado.edu

---

## Prevention

To avoid these issues in future runs:

1. Always use the latest version from git
2. Run on a compute node (not login node) when testing
3. Use `sbatch` for all jobs, never run training directly on login nodes
4. Monitor with `squeue -u $USER` and `watch squeue -u $USER`
5. Check logs regularly during long runs
