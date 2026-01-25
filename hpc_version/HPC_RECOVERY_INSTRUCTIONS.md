# HPC Multi-Seed Run Recovery Instructions

## What Happened

Your multi-seed experiment failed with all 5 seeds showing the same error:

```
[FAILED] stage_2_train_baseline FAILED
Expected flag: /scratch/alpine/paco0228/mono_s2s_work/seed_12345/stage_2_train_baseline_complete.flag
```

All jobs showed `DependencyNeverSatisfied` status in the queue.

## Root Cause

**Critical path mismatch:**
- On Alpine's **login nodes**, the `$SCRATCH` and `$PROJECT` environment variables are **not automatically set**
- On Alpine's **compute nodes**, these variables **are automatically set** by SLURM
- This created a mismatch:
  - `run_all.sh` (on login node): Looked for flags in `/mono_s2s_work/` (invalid - `$SCRATCH` was empty)
  - Python scripts (on compute nodes): Wrote flags to `/scratch/alpine/paco0228/mono_s2s_work/` (correct)
- Result: Scripts thought Stage 0 and 1 failed (couldn't find flags), so all downstream jobs were cancelled

## What Was Fixed

### 1. Updated `run_all.sh`
- Now explicitly sets `SCRATCH=/scratch/alpine/$USER` if not already set
- Validates the directory exists before proceeding
- Provides clear error messages

### 2. Updated `run_multi_seed.sh`
- Same fixes as `run_all.sh`
- Ensures consistent environment across all seed runs

### 3. Created `diagnose_and_recover.sh`
- Diagnostic tool to check what actually completed
- Auto-recovers missing completion flags when work is done
- Shows detailed status of all seeds

### 4. Created `TROUBLESHOOTING.md`
- Comprehensive guide for common HPC issues
- Quick reference for recovery procedures

## Next Steps

### On Your HPC Login Node:

```bash
# 1. Cancel all stuck jobs
scancel -u paco0228

# 2. Navigate to the project directory
cd ~/code/mono-s2s  # or wherever you cloned the repo

# 3. Pull the latest fixes
git pull origin main

# 4. Navigate to HPC version
cd hpc_version

# 5. Run the diagnostic script
./diagnose_and_recover.sh

# This will:
# - Set SCRATCH and PROJECT variables correctly
# - Check if any previous work actually completed
# - Auto-create missing completion flags
# - Show you what needs to be rerun

# 6. After reviewing the diagnostic output, rerun the multi-seed experiment
./run_multi_seed.sh
```

### What to Expect

The diagnostic script will likely show:
- **Stage 0 (setup)**: May have completed successfully, just missing the flag
- **Stage 1 (data prep)**: May have completed successfully, just missing the flag
- **Stage 2+ (training/evaluation)**: Likely never ran due to dependency failures

If Stages 0 and 1 completed, the script will auto-create their flags, and the rerun will skip directly to training.

## Monitoring the New Run

```bash
# Check queue status
squeue -u paco0228

# Watch queue in real-time (Ctrl+C to exit)
watch squeue -u paco0228

# Check recent job history
sacct -u paco0228 --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed

# View logs for a specific job
cat logs/job_2_baseline_<JOBID>.out
```

## Expected Timeline

With the fixes in place:
1. **Stage 0 (setup)**: ~2 minutes (or skip if already done)
2. **Stage 1 (data prep)**: ~2 minutes (or skip if already done)
3. **Stages 2-3 (training)**: 4-12 hours per seed (depends on queue wait time + actual training)
4. **Stages 4-6 (evaluation/attacks)**: 6-8 hours per seed
5. **Stage 7 (aggregation)**: 5-15 minutes

Total: Approximately 10-20 hours per seed (mostly queue wait time)

For 5 seeds running sequentially: 50-100 hours total

## Verification Checklist

After running `diagnose_and_recover.sh`, verify:

- [ ] `SCRATCH=/scratch/alpine/paco0228` (correct path)
- [ ] `PROJECT=/projects/paco0228` (correct path)
- [ ] Work directory exists: `$SCRATCH/mono_s2s_work/`
- [ ] If Stage 0 ran: Flag auto-created or setup needs rerun
- [ ] If Stage 1 ran: Flag auto-created or data prep needs rerun
- [ ] No stuck jobs: `squeue -u paco0228` shows empty or only new jobs

## If Issues Persist

1. **Check disk quota**: Run `curc-quota` to ensure you have space
2. **Review TROUBLESHOOTING.md**: Covers common issues and fixes
3. **Check job logs**: `ls -lt logs/` and review recent failures
4. **Contact CURC support**: rc-help@colorado.edu with job IDs

## Questions?

- Main README: `../README.md`
- Troubleshooting: `TROUBLESHOOTING.md`
- HPC changes summary: `CHANGES_AT_A_GLANCE.md`

---

**Bottom line**: Pull the latest code, run `./diagnose_and_recover.sh`, then `./run_multi_seed.sh`. The path issue is now fixed.
